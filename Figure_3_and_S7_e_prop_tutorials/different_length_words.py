import datetime
import socket
from time import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from tools import update_plot, generate_phoneme_data, generate_phoneme_data_dif_length
from models import EligALIF, exp_convolve

FLAGS = tf.app.flags.FLAGS
start_time = datetime.datetime.now()
# training parameters
tf.app.flags.DEFINE_integer('n_batch', 4, 'batch size')
tf.app.flags.DEFINE_integer('n_iter', 100, 'total number of iterations')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Base learning rate.')
tf.app.flags.DEFINE_float('stop_crit', -0.001, 'Stopping criterion. Stops training if error goes below this value')#if negative just disables it
tf.app.flags.DEFINE_integer('print_every', 1, 'Print every')
tf.app.flags.DEFINE_integer('validate_every', 1, 'validate every')

# training algorithm
tf.app.flags.DEFINE_bool('eprop', True, 'Use e-prop to train network (BPTT if false)')
tf.app.flags.DEFINE_string('eprop_impl', 'hardcoded', '["autodiff", "hardcoded"] Use tensorflow for computing e-prop '
                                                     'updates or implement equations directly')
tf.app.flags.DEFINE_string('feedback', 'symmetric', '["random", "symmetric"] Use random or symmetric e-prop')
tf.app.flags.DEFINE_string('f_regularization_type', 'simple', '["simple", "online"] Twos types of firing rate regularization.')

# neuron model and simulation parameters
tf.app.flags.DEFINE_float('tau_a', 2000, 'model alpha - threshold decay [ms]')
tf.app.flags.DEFINE_float('thr', 0.6, 'threshold at which the LSNN neurons spike')
tf.app.flags.DEFINE_float('tau_v', 20, 'tau for filtered_z decay in LSNN  neurons [ms]')
tf.app.flags.DEFINE_float('tau_out', 20, 'tau for filtered_z decay in output neurons [ms]')
tf.app.flags.DEFINE_float('reg_f', 1, 'regularization coefficient for firing rate')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target firing rate for regularization [Hz]')
tf.app.flags.DEFINE_integer('n_ref', 5, 'Number of refractory steps [ms]')
tf.app.flags.DEFINE_integer('dt', 1, 'Simulation time step [ms]')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, 'factor that controls amplitude of pseudoderivative')

# other settings
tf.app.flags.DEFINE_bool('do_plot', False, 'Perform plots')
tf.app.flags.DEFINE_bool('device_placement', False, '')

assert FLAGS.eprop_impl in ['autodiff', 'hardcoded']
assert FLAGS.feedback in ['random', 'symmetric']

# Experiment parameters
t_cue_spacing = 15  # distance between two consecutive cues in ms

# Frequencies
input_f0 = 40. / 1000.  # poisson firing rate of input neurons in khz
regularization_f0 = FLAGS.reg_rate / 1000.  # mean target network firing frequency

# Network parameters
tau_v = FLAGS.tau_v
thr = FLAGS.thr
n_adaptive = 10
n_regular = 10
n_neurons = n_adaptive + n_regular
decay = np.exp(-FLAGS.dt / FLAGS.tau_out)  # output layer filtered_z decay, chose value between 15 and 30ms as for tau_v




phoneme_sequences = [[0, 1, 2], [0, 2, 1], [1, 2, 3], [1,2,1,2]]
word_mapping = {(0, 1, 2): 0, (0, 2, 1): 1, (1, 2, 3): 2, (1, 2, 1,2): 3}
n_words = len(word_mapping)
n_in = 10 * n_words #i think this is spike input??? instead of first layer of "true"" NN? not sure


def get_data_dict(batch_size):
    # Generate data according to the phoneme recognition task
    spk_data, in_nums, target_data = generate_phoneme_data_dif_length(batch_size, phoneme_sequences, word_mapping, num_neurons_per_phoneme=int(n_in/(n_words)), cue_spacing=t_cue_spacing)
    return {
        input_spikes: spk_data,
        input_nums: in_nums,  # Now including input nums in the data dictionary
        target_nums: target_data.reshape(batch_size, 1)  # reshape target to match placeholder
    }

input_spikes = tf.placeholder(dtype=tf.float32, shape=(FLAGS.n_batch, None, n_in), name='InputSpikes')
input_nums = tf.placeholder(dtype=tf.int32, shape=(FLAGS.n_batch, None), name='InputNums')  # Corrected name for clarity
target_nums = tf.placeholder(dtype=tf.int64, shape=(FLAGS.n_batch, 1), name='TargetNums')  # Shape changed to (batch_size, 1)

# build computational graph
with tf.variable_scope('CellDefinition'):
    # generate threshold decay time constants
    tau_a = FLAGS.tau_a
    rhos = np.exp(- FLAGS.dt / tau_a)  # decay factors for adaptive threshold
    beta_a = 1.7 * (1 - rhos) / (1 - np.exp(-1 / FLAGS.tau_v))  # this is a heuristic value
    beta = np.concatenate([np.zeros(n_regular), beta_a * np.ones(n_adaptive)])  # multiplicative factors for adaptive threshold
    # Generate the cell
    cell = EligALIF(n_in=n_in, n_rec=n_regular + n_adaptive, tau=tau_v, beta=beta, thr=thr,
                    dt=FLAGS.dt, tau_adaptation=tau_a, dampening_factor=FLAGS.dampening_factor,
                    stop_z_gradients=FLAGS.eprop, n_refractory=FLAGS.n_ref)

with tf.name_scope('SimulateNetwork'):
    outputs, final_state = tf.nn.dynamic_rnn(cell, input_spikes, dtype=tf.float32)
    # z - spikes, v - membrane potentials, b - threshold adaptation variables
    z, s = outputs
    v, b = s[..., 0], s[..., 1]

with tf.name_scope('OutputComputation'):
    # Adjust the shape of the output weight matrix to accommodate four classes
    W_out = tf.get_variable(name='out_weight', shape=[n_regular + n_adaptive, n_words])
    filtered_z = exp_convolve(z, decay)

    if FLAGS.eprop and FLAGS.feedback == 'random':
        # Function to generate the random feedback path with adjustments for four classes
        b_out_vals = rd.randn(n_regular + n_adaptive, n_words)  # Adjusted for 4 classes
        B_out = tf.constant(b_out_vals, dtype=tf.float32, name='feedback_weights')

        @tf.custom_gradient
        def matmul_random_feedback(filtered_z, W_out_arg, B_out_arg):
            logits = tf.einsum('btj,jk->btk', filtered_z, W_out_arg)
            def grad(dy):
                dloss_dW_out = tf.einsum('bij,bik->jk', filtered_z, dy)
                dloss_dfiltered_z = tf.einsum('bik,jk->bij', dy, B_out_arg)
                dloss_db_out = tf.zeros_like(B_out_arg)
                return [dloss_dfiltered_z, dloss_dW_out, dloss_db_out]

            return logits, grad

        out = matmul_random_feedback(filtered_z, W_out, B_out)
    else:
        # Standard output calculation adjusted for 4 classes
        out = tf.einsum('btj,jk->btk', filtered_z, W_out)

    # Use network output at the end for classification
    output_logits = out[:, -t_cue_spacing:]
    print("Defined output logits, shape: ", np.shape(output_logits))
    print("Shape of output logits", np.shape(output_logits))

with tf.name_scope('TaskLoss'):
    tiled_targets = tf.tile(target_nums[:, np.newaxis, -1], (1, t_cue_spacing))
    loss_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tiled_targets, logits=output_logits))
    y_predict = tf.argmax(tf.reduce_mean(output_logits, axis=1), axis=1)

    # Define accuracy based on the last label in the sequence
    accuracy = tf.reduce_mean(tf.cast(tf.equal(target_nums[:, -1], y_predict), dtype=tf.float32))
    recall_errors = 1 - accuracy

    with tf.name_scope('PlotNodes'):
        out_plot = tf.nn.softmax(out, axis=-1)

# Target regularization
with tf.name_scope('RegularizationLoss'):
    # Firing rate regularization
    av = tf.reduce_mean(z, axis=(0, 1)) / FLAGS.dt
    regularization_coeff = tf.Variable(np.ones(n_neurons) * FLAGS.reg_f, dtype=tf.float32, trainable=False)

    if(FLAGS.f_regularization_type == "simple"):
        # For historical reason we often use this regularization,
        # but the other one is easier to implement in an "online" fashion by a single agent.
        loss_reg_f = tf.reduce_sum(tf.square(av - regularization_f0) * regularization_coeff)
    else:
        # Basically, we need to replace the average firing rate by a running average:
        shp = tf.shape(z)
        z_single_agent = tf.concat(tf.unstack(z,axis=0),axis=0)
        spike_count_single_agent = tf.cumsum(z_single_agent,axis=0)
        timeline_single_agent = tf.cast(tf.range(shp[0] * shp[1]),tf.float32)
        running_av = spike_count_single_agent / (timeline_single_agent + 1)[:,None] / FLAGS.dt
        running_av = tf.stack(tf.split(running_av,FLAGS.n_batch),axis=0)

        # otherwise nothing changed:
        loss_reg_f = tf.square(running_av - regularization_f0)
        loss_reg_f = tf.reduce_sum(tf.reduce_mean(loss_reg_f,axis=1) * regularization_coeff)

# Aggregate the losses
with tf.name_scope('OptimizationScheme'):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False)
    loss = loss_reg_f + loss_cls
    loss = loss_cls
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    var_list = [cell.w_in_var, cell.w_rec_var, W_out]

    if FLAGS.eprop and FLAGS.eprop_impl == 'hardcoded':

        # This compute the partial derivative dE/dz_bar where z_bar is the filtered spike train
        # For Cross entropy loss as in classification:
        # learning_signal_j(t) = sum_k B_kj (pi*_k(t) - pi_k(t))
        learning_signal_classification = tf.gradients(loss_cls, filtered_z)[0]
        learning_signal_regularization = tf.gradients(loss_reg_f, filtered_z)[0]
        print("learning_signal_classification:", learning_signal_classification)
        print("learning_signal_regularization:", learning_signal_regularization)
        print("Details of loss_reg_f computation:")
        print("log_reg_f:",loss_reg_f)
        print("Filtered Z:", filtered_z)

        if learning_signal_regularization is None:
            learning_signal_regularization = tf.zeros_like(learning_signal_classification)
        learning_signal = learning_signal_classification + learning_signal_regularization

        # e-traces for input synapses
        grad_in, e_trace, _, epsilon_a = cell.compute_loss_gradient(learning_signal, input_spikes, z, v, b,
                                                                    zero_on_diagonal=False, decay_out=decay)
        # e-traces for recurrent synapses
        z_previous_step = tf.concat([tf.zeros_like(z[:, 0])[:, None], z[:, :-1]], axis=1)
        grad_rec, _, _, _ = cell.compute_loss_gradient(learning_signal, z_previous_step, z, v, b,
                                                       zero_on_diagonal=True,decay_out=decay)
        # gradients for output weights
        grad_out = tf.gradients(loss, W_out)[0]
        # concatenate all gradients
        gradient_list = [grad_in, grad_rec, grad_out]
        # for comparision with auto-diff version
        true_gradient_list = tf.gradients(loss, var_list)
        # check that tensorflows auto-diff produces same result
        g_name = ['in', 'rec', 'out']

        grad_error_assertions = []
        grad_error_prints = []
        for g1, g2, nn in zip(gradient_list, true_gradient_list, g_name):
            NN = tf.reduce_max(tf.square(g2))
            max_gradient_error = tf.reduce_max(tf.square(g1 - g2) / NN)

            gradient_error_print = tf.print(nn + " gradient error: ",max_gradient_error)
            #gradient_error_assertion = tf.debugging.Assert(max_gradient_error < 1.0, data=[max_gradient_error],name=nn)
            grad_error_prints.append(gradient_error_print)
            #grad_error_assertions.append(gradient_error_assertion)
    else:
        # This automatically computes the correct gradients in tensor flow
        learning_signal = tf.zeros_like(z)
        grad_error_prints = []
        grad_error_assertions = []
        gradient_list = tf.gradients(loss, var_list)

    grads_and_vars = [(g, v) for g, v in zip(gradient_list, var_list)]

    with tf.control_dependencies(grad_error_prints + grad_error_assertions):
        train_step = opt.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

# create session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.device_placement))
sess.run(tf.global_variables_initializer())

# variables for storing results and plots
if FLAGS.do_plot:
    plt.ion()

    if FLAGS.eprop and FLAGS.eprop_impl == 'hardcoded':
        # plot learning signal and traces - only works in hardcoded mode!
        n_subplots = 7 - int(n_regular == 0) - int(n_adaptive == 0)
    else:
        n_subplots = 4
    fig, ax_list = plt.subplots(n_subplots, figsize=(5.9, 6))
    # re-name the window with the name of the cluster to track relate to the terminal window
    fig.canvas.manager.set_window_title(socket.gethostname())

validation_loss_list = []
validation_error_list = []
training_time_list = []
n_iter_list = []

# Initialize arrays to store losses over time
train_loss_list = []
validation_reg_loss_list = []
validation_total_loss_list = []

results_tensors = {
    'loss_recall': loss,
    'loss_reg': loss_reg_f,
    'recall_errors': recall_errors,
    'av': av,
    'regularization_coeff': regularization_coeff,
}

plot_result_tensors = {'input_spikes': input_spikes,
                       'input_nums': input_nums,
                       'z': z,
                       'thr': tf.constant(thr),
                       'target_nums': target_nums,
                       'out_plot': out_plot  # Include out_plot in tensors to fetch
                       }
try:
    flag_dict = FLAGS.flag_values_dict()
except:
    print('Deprecation WARNING: with tensorflow >= 1.5 we should use FLAGS.flag_values_dict() to transform to dict')
    flag_dict = FLAGS.__flags

# fill flag dict for plot
flag_dict['n_regular'] = n_regular
flag_dict['n_adaptive'] = n_adaptive
flag_dict['recall_cue'] = True

# training loop
t_train = 0
t_ref = time()
for k_iter in range(FLAGS.n_iter):
    # Monitor the training with a validation set
    if np.mod(k_iter, FLAGS.validate_every) == 0:
        t0 = time()
        val_dict = get_data_dict(FLAGS.n_batch)
        results_values = sess.run(results_tensors, feed_dict=val_dict)
        validation_loss_list.append(results_values['loss_recall'])
        validation_error_list.append(results_values['recall_errors'])
        validation_reg_loss_list.append(results_values['loss_reg'])  # Track validation regularization loss
        validation_total_loss_list.append(results_values['loss_recall'] + results_values['loss_reg'])  # Track total loss
        t_run = time() - t0

    if np.mod(k_iter, FLAGS.print_every) == 0:
        print('''Iteration {}, statistics on the validation set average error {:.2g} +- {:.2g} (trial averaged)'''
              .format(k_iter, np.mean(validation_error_list[-FLAGS.print_every:]),
                      np.std(validation_error_list[-FLAGS.print_every:])))

        def get_stats(v):
            if np.size(v) == 0:
                return np.nan, np.nan, np.nan, np.nan
            min_val = np.min(v)
            max_val = np.max(v)

            k_min = np.sum(v == min_val)
            k_max = np.sum(v == max_val)

            return np.min(v), np.max(v), np.mean(v), np.std(v), k_min, k_max

        firing_rate_stats = get_stats(results_values['av'] * 1000)
        reg_coeff_stats = get_stats(results_values['regularization_coeff'])

        print('''
        firing rate (Hz)  min {:.0f} ({}) \t max {:.0f} ({}) \t
        average {:.0f} +- std {:.0f} (averaged over batches and time)
        comput. time (s)  training {:.2g} \t validation {:.2g}
        loss              classif. {:.2g} \t reg. loss  {:.2g}
        '''.format(
            firing_rate_stats[0], firing_rate_stats[4], firing_rate_stats[1], firing_rate_stats[5],
            firing_rate_stats[2], firing_rate_stats[3],
            t_train, t_run,
            results_values['loss_recall'], results_values['loss_reg']
        ))

        if FLAGS.do_plot:
            print(np.shape(out_plot), "shape out_plot")
            plot_result_tensors['out_plot'] = out_plot
            plot_result_tensors['y_predict'] = y_predict
            plot_result_tensors['thr'] = FLAGS.thr + b * beta
            if FLAGS.eprop_impl == 'hardcoded':
                plot_result_tensors['e_trace'] = e_trace
                plot_result_tensors['learning_signal_cls'] = learning_signal_classification
                plot_result_tensors['learning_signal_reg'] = learning_signal_regularization
                plot_result_tensors['epsilon_a'] = epsilon_a

            plot_results_values = sess.run(plot_result_tensors, feed_dict=val_dict)
            plot_results_values['flags'] = flag_dict

            plot_trace = True if FLAGS.eprop_impl == 'hardcoded' else False
            update_plot(plot_results_values, ax_list, plot_traces=plot_trace, n_max_neuron_per_raster=20,
                        title='Training at iteration ' + str(k_iter),t_cue_spacing=t_cue_spacing)

            plt.draw()
            plt.pause(1)
            plt.savefig(f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_plot_iteration_{k_iter}.png')

    # do early stopping check if single batch validation error under stop_crit
    if (k_iter > 0 and validation_error_list[-1] < FLAGS.stop_crit):
        early_stopping_list = []
        t_es_0 = time()
        for i in range(8):
            val_dict = get_data_dict(FLAGS.n_batch)
            early_stopping_list.append(sess.run(results_tensors['recall_errors'], feed_dict=val_dict))
        t_es = time() - t_es_0
        print("comput. time (s): early stopping: " + str(t_es))
        if np.mean(early_stopping_list) < FLAGS.stop_crit:
            n_iter_list.append(k_iter)
            print('less than ' + str(FLAGS.stop_crit) + ' - stopping training at iteration ' + str(k_iter))
            break
        else:
            print('early stopping error: ' + str(np.mean(early_stopping_list)) + ' higher than stop crit of: '
                  + str(FLAGS.stop_crit) + ' CONTINUE TRAINING')

    if k_iter == FLAGS.n_iter - 1:
        n_iter_list.append(k_iter)
        break

    # do train step
    train_dict = get_data_dict(FLAGS.n_batch)
    t0 = time()
    results_values_train = sess.run([train_step, loss, loss_reg_f], feed_dict=train_dict)
    train_loss_list.append(results_values_train[1])  # Track training loss
    t_train = time() - t0
    training_time_list.append(t_train)

print('FINISHED IN {:.2g} s'.format(time() - t_ref))

# Save training progress
results = {
    'iterations': n_iter_list,
    'final_loss': validation_loss_list[-1],
    'val_errors': validation_error_list,
    'val_losses': validation_loss_list,
    'training_time': training_time_list,
    'flags': flag_dict,
}

# Save sample trajectory (input, output, etc. for plotting) and test final performance
test_errors = []
for i in range(FLAGS.n_batch):
    test_dict = get_data_dict(FLAGS.n_batch)
    results_values, plot_results_values, in_spk, spk, target_nums_np = sess.run(
        [results_tensors, plot_result_tensors, input_spikes, z, target_nums],
        feed_dict=test_dict)
    test_errors.append(results_values['recall_errors'])
    flag_dict['n_regular'] = n_regular
    plot_results_values['flags'] = flag_dict

    # Print output probabilities for each timestep
    output_probs = plot_results_values['out_plot']
    #for timestep in range(output_probs.shape[1]):
     #   print(f"Timestep {timestep}: {output_probs[:, timestep, :]}")

    if True:#override FLAGS.do_plot for final test plot
        fig, ax_list = plt.subplots(4, figsize=(5.9, 6))
        fig.canvas.manager.set_window_title(f'Test Case {i}')
        update_plot(plot_results_values, ax_list, n_max_neuron_per_raster=20, title=f'Test Case {i}', batch=i,t_cue_spacing=t_cue_spacing)

        # Show the correct decision for the test case
        correct_decision = plot_results_values['target_nums'][i, -1]  # Only taking the current batch item for simplicity
        ax_list[3].axvline(x=len(output_probs[0]) - 1, color='k', linestyle='--', label='Correct Decision')
        ax_list[3].text(len(output_probs[0]) - 1, 0.5, f'Correct: {correct_decision}', verticalalignment='center', fontsize=8)

        plt.draw()
        plt.pause(1)
        plt.savefig(f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_plot_test_case_{i}.png')

print('''Statistics on the test set average error {:.2g} +- {:.2g} (averaged over 16 test batches of size {})'''
      .format(np.mean(test_errors), np.std(test_errors), FLAGS.n_batch))

# Plot the loss over time
plt.figure()
plt.plot(train_loss_list, label='Training Loss')
plt.plot(validation_total_loss_list, label='Validation Total Loss')
plt.plot(validation_reg_loss_list, label='Validation Regularization Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()

# List hyperparameters
hyperparams_text = f"Batch size: {FLAGS.n_batch}\n" \
                   f"Learning rate: {FLAGS.learning_rate}\n" \
                   f"Regular neurons: {n_regular}\n" \
                   f"Adaptive neurons: {n_adaptive}\n" \
                   f"Total neurons: {n_neurons}\n" \
                   f"Threshold: {thr}\n" \
                   f"Tau_v: {tau_v}\n" \
                   f"Tau_a: {tau_a}\n" \
                   f"Number of phonemes: {len(phoneme_sequences)}\n" \
                   f"Total number of words: {n_words}"

plt.gcf().text(0.02, 0.98, hyperparams_text, fontsize=9, verticalalignment='top')

plt.savefig(f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_loss_plot_{k_iter}.png')
print("about to show plot??")
plt.show()
print("plot is shown???")

del sess