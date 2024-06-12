import datetime
import socket
from time import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from tools import update_plot, generate_phoneme_data, generate_phoneme_data_dif_length
from models import EligALIF, exp_convolve
import random
import json

FLAGS = tf.app.flags.FLAGS
start_time = datetime.datetime.now()

n_adaptive = 5
n_regular = 5
n_neurons = n_adaptive + n_regular

# Training parameters
tf.app.flags.DEFINE_integer('n_batch', 500, 'Batch size')
tf.app.flags.DEFINE_integer('n_iter', 500, 'Total number of iterations')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Base learning rate.')
tf.app.flags.DEFINE_float('stop_crit', -0.001, 'Stopping criterion. Stops training if error goes below this value')
tf.app.flags.DEFINE_integer('print_every', 1, 'Print every')
tf.app.flags.DEFINE_integer('validate_every', 1, 'Validate every')

# Training algorithm
tf.app.flags.DEFINE_bool('eprop', False, 'Use e-prop to train network (BPTT if false)')
tf.app.flags.DEFINE_string('eprop_impl', 'autodiff', '["autodiff", "hardcoded"] Use tensorflow for computing e-prop updates or implement equations directly')
tf.app.flags.DEFINE_string('feedback', 'symmetric', '["random", "symmetric"] Use random or symmetric e-prop')
tf.app.flags.DEFINE_string('f_regularization_type', 'simple', '["simple", "online"] Two types of firing rate regularization.')

# Neuron model and simulation parameters
tf.app.flags.DEFINE_float('tau_a', 2000, 'Model alpha - threshold decay [ms]')
tf.app.flags.DEFINE_float('thr', 0.6, 'Threshold at which the LSNN neurons spike')
tf.app.flags.DEFINE_float('tau_v', 40, 'Tau for filtered_z decay in LSNN neurons [ms]')
tf.app.flags.DEFINE_float('tau_out', 20, 'Tau for filtered_z decay in output neurons [ms]')
tf.app.flags.DEFINE_float('reg_f', 1, 'Regularization coefficient for firing rate')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'Target firing rate for regularization [Hz]')
tf.app.flags.DEFINE_integer('n_ref', 5, 'Number of refractory steps [ms]')
tf.app.flags.DEFINE_integer('dt', 1, 'Simulation time step [ms]')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, 'Factor that controls amplitude of pseudoderivative')

# Other settings
tf.app.flags.DEFINE_bool('do_plot', False, 'Perform plots')
tf.app.flags.DEFINE_bool('device_placement', False, '')

assert FLAGS.eprop_impl in ['autodiff', 'hardcoded']
assert FLAGS.feedback in ['random', 'symmetric']

# Experiment parameters
t_cue_spacing = 17 # Distance between two consecutive cues in ms

# Frequencies
input_f0 = 40. / 1000.  # Poisson firing rate of input neurons in kHz
regularization_f0 = FLAGS.reg_rate / 1000.  # Mean target network firing frequency

# Network parameters
tau_v = FLAGS.tau_v
thr = FLAGS.thr

decay = np.exp(-FLAGS.dt / FLAGS.tau_out)  # Output layer filtered_z decay, chose value between 15 and 30ms as for tau_v

# Load the data from the JSON file
with open("phoneme_data.json", "r") as f:
    data = json.load(f)

phoneme_sequences = data["phoneme_sequences"]
word_mapping = {eval(k): v for k, v in data["word_mapping"].items()}
word_to_index = data["word_to_index"]
phoneme_to_id = data["phoneme_to_id"]

print(len(word_to_index),"AMOUNT OF WORDS")

n_in = data["n_in"]
n_words = len(word_mapping)
n_phonemes = len(phoneme_to_id)
print(n_phonemes, "amount unique phonemes")
print("n_in:", n_in)
input_encoding_type = "single, low cue spacing"
active_share = 0.2


# Define the neural network placeholders and the rest of the computational graph
FLAGS = tf.app.flags.FLAGS



num_neurons_per_phoneme=int(n_in / len(phoneme_to_id))
# Function to generate unique spike patterns for each phoneme with a specified share of active neurons
def generate_spike_pattern(phoneme_id, num_neurons, active_share):
    np.random.seed(phoneme_id)
    pattern = np.zeros(num_neurons)
    active_neurons = int(active_share * num_neurons)
    active_indices = np.random.choice(num_neurons, active_neurons, replace=False)
    pattern[active_indices] = np.random.rand(active_neurons)
    return pattern

# Function to generate phoneme data with unique patterns
def generate_phoneme_data_unique_patterns(batch_size, phoneme_sequences, word_mapping, num_neurons, cue_spacing, active_share):
    max_length = max(len(seq) for seq in phoneme_sequences)
    spk_data = np.zeros((batch_size, max_length * cue_spacing, n_in))
    in_nums = np.zeros((batch_size, max_length), dtype=np.int32)
    target_data = np.zeros((batch_size,), dtype=np.int32)

    for i in range(batch_size):
        seq_idx = random.randint(0, len(phoneme_sequences) - 1)
        seq = phoneme_sequences[seq_idx]
        for t, phoneme_id in enumerate(seq):
            pattern = generate_spike_pattern(phoneme_id, num_neurons, active_share)
            spk_data[i, t * cue_spacing:(t + 1) * cue_spacing] = pattern
        in_nums[i, :len(seq)] = seq
        
        target_data[i] = word_mapping[tuple(seq)]
    
    return spk_data, in_nums, target_data


spk_data, in_nums, target_data = generate_phoneme_data_unique_patterns(FLAGS.n_batch, phoneme_sequences, word_mapping, n_in, t_cue_spacing,active_share)

print("Spike Data Shape:", spk_data.shape)
print("Input Numbers Shape:", in_nums.shape)
print("Target Data Shape:", target_data.shape)

def get_data_dict(batch_size):
    spk_data, in_nums, target_data = generate_phoneme_data_dif_length(batch_size, phoneme_sequences, word_mapping, num_neurons_per_phoneme=int(n_in / len(phoneme_to_id)), cue_spacing=2)
    return {
        input_spikes: spk_data,
        input_nums: in_nums,
        target_nums: target_data.reshape(batch_size, 1)
    }

input_spikes = tf.placeholder(dtype=tf.float32, shape=(FLAGS.n_batch, None, n_in), name='InputSpikes')
input_nums = tf.placeholder(dtype=tf.int32, shape=(FLAGS.n_batch, None), name='InputNums')
target_nums = tf.placeholder(dtype=tf.int64, shape=(FLAGS.n_batch, 1), name='TargetNums')



# Build and train your model using the data
# ...

# Build computational graph
with tf.variable_scope('CellDefinition'):
    tau_a = FLAGS.tau_a
    rhos = np.exp(- FLAGS.dt / tau_a)  # Decay factors for adaptive threshold
    beta_a = 1.7 * (1 - rhos) / (1 - np.exp(-1 / FLAGS.tau_v))  # This is a heuristic value
    beta = np.concatenate([np.zeros(n_regular), beta_a * np.ones(n_adaptive)])  # Multiplicative factors for adaptive threshold
    cell = EligALIF(n_in=n_in, n_rec=n_regular + n_adaptive, tau=tau_v, beta=beta, thr=thr, dt=FLAGS.dt, tau_adaptation=tau_a, dampening_factor=FLAGS.dampening_factor, stop_z_gradients=FLAGS.eprop, n_refractory=FLAGS.n_ref)

with tf.name_scope('SimulateNetwork'):
    outputs, final_state = tf.nn.dynamic_rnn(cell, input_spikes, dtype=tf.float32)
    z, s = outputs
    v, b = s[..., 0], s[..., 1]

with tf.name_scope('OutputComputation'):
    W_out = tf.get_variable(name='out_weight', shape=[n_regular + n_adaptive, n_words])
    filtered_z = exp_convolve(z, decay)

    if FLAGS.eprop and FLAGS.feedback == 'random':
        b_out_vals = rd.randn(n_regular + n_adaptive, n_words)
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
        out = tf.einsum('btj,jk->btk', filtered_z, W_out)

    output_logits = out[:, -t_cue_spacing:]
    print("Defined output logits, shape: ", np.shape(output_logits))

    temperature = 0.05
    output_probs = tf.nn.softmax(output_logits / temperature, axis=-1)
    print("Output probabilities shape: ", output_probs.shape)

with tf.name_scope('TaskLoss'):
    time_weights = tf.constant([.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,10], dtype=tf.float32, shape=(t_cue_spacing, 1))

    weighted_output_logits = tf.reduce_sum(out[:, -t_cue_spacing:] * time_weights[None, :, :], axis=1)
    temperature = 0.5
    scaled_logits = out[:, -t_cue_spacing:] / temperature
    output_probs = tf.nn.softmax(scaled_logits, axis=-1)
    tiled_targets = tf.tile(target_nums[:, -1:], (1, 1))
    loss_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(tiled_targets, axis=-1), logits=weighted_output_logits))
    y_predict = tf.argmax(weighted_output_logits, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(tiled_targets, axis=-1), y_predict), dtype=tf.float32))
    recall_errors = 1 - accuracy

    with tf.name_scope('PlotNodes'):
        out_plot = output_probs

with tf.name_scope('RegularizationLoss'):
    av = tf.reduce_mean(z, axis=(0, 1)) / FLAGS.dt
    regularization_coeff = tf.Variable(np.ones(n_neurons) * FLAGS.reg_f, dtype=tf.float32, trainable=False)

    if(FLAGS.f_regularization_type == "simple"):
        loss_reg_f = tf.reduce_sum(tf.square(av - regularization_f0) * regularization_coeff)
    else:
        shp = tf.shape(z)
        z_single_agent = tf.concat(tf.unstack(z,axis=0),axis=0)
        spike_count_single_agent = tf.cumsum(z_single_agent,axis=0)
        timeline_single_agent = tf.cast(tf.range(shp[0] * shp[1]),tf.float32)
        running_av = spike_count_single_agent / (timeline_single_agent + 1)[:,None] / FLAGS.dt
        running_av = tf.stack(tf.split(running_av,FLAGS.n_batch),axis=0)
        loss_reg_f = tf.square(running_av - regularization_f0)
        loss_reg_f = tf.reduce_sum(tf.reduce_mean(loss_reg_f,axis=1) * regularization_coeff)

with tf.name_scope('OptimizationScheme'):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False)
    loss = loss_reg_f + loss_cls
    loss = loss_cls
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    var_list = [cell.w_in_var, cell.w_rec_var, W_out]

    if FLAGS.eprop and FLAGS.eprop_impl == 'hardcoded':
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

        grad_in, e_trace, _, epsilon_a = cell.compute_loss_gradient(learning_signal, input_spikes, z, v, b, zero_on_diagonal=False, decay_out=decay)
        z_previous_step = tf.concat([tf.zeros_like(z[:, 0])[:, None], z[:, :-1]], axis=1)
        grad_rec, _, _, _ = cell.compute_loss_gradient(learning_signal, z_previous_step, z, v, b, zero_on_diagonal=True,decay_out=decay)
        grad_out = tf.gradients(loss, W_out)[0]
        gradient_list = [grad_in, grad_rec, grad_out]
        true_gradient_list = tf.gradients(loss, var_list)
        g_name = ['in', 'rec', 'out']

        grad_error_assertions = []
        grad_error_prints = []
        for g1, g2, nn in zip(gradient_list, true_gradient_list, g_name):
            NN = tf.reduce_max(tf.square(g2))
            max_gradient_error = tf.reduce_max(tf.square(g1 - g2) / NN)

            gradient_error_print = tf.print(nn + " gradient error: ",max_gradient_error)
            grad_error_prints.append(gradient_error_print)
    else:
        learning_signal = tf.zeros_like(z)
        grad_error_prints = []
        grad_error_assertions = []
        gradient_list = tf.gradients(loss, var_list)

    grads_and_vars = [(g, v) for g, v in zip(gradient_list, var_list)]

    with tf.control_dependencies(grad_error_prints + grad_error_assertions):
        train_step = opt.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.device_placement))
sess.run(tf.global_variables_initializer())

if FLAGS.do_plot:
    plt.ion()

    if FLAGS.eprop and FLAGS.eprop_impl == 'hardcoded':
        n_subplots = 7 - int(n_regular == 0) - int(n_adaptive == 0)
    else:
        n_subplots = 4
    fig, ax_list = plt.subplots(n_subplots, figsize=(5.9, 6))
    fig.canvas.manager.set_window_title(socket.gethostname())

validation_loss_list = []
validation_error_list = []
training_time_list = []
n_iter_list = []

train_loss_list = []
val_accuracy_list = []
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
                       'out_plot': out_plot
                       }

if FLAGS.eprop_impl == 'hardcoded' and FLAGS.eprop == True:
    plot_result_tensors.update({
        'e_trace': e_trace,
        'learning_signal_cls': learning_signal_classification,
        'learning_signal_reg': learning_signal_regularization,
        'epsilon_a': epsilon_a
    })

flag_dict = FLAGS.flag_values_dict()
flag_dict['n_regular'] = n_regular
flag_dict['n_adaptive'] = n_adaptive
flag_dict['recall_cue'] = True

# Initialize arrays to store information for visualization
neuron_activations_per_word = {word: [] for word in word_mapping.values()}
firing_rates_per_word = {word: [] for word in word_mapping.values()}
firing_rate_total = []
complete_firing_rate_data = []

t_train = 0
t_ref = time()
for k_iter in range(FLAGS.n_iter):
    if np.mod(k_iter, FLAGS.validate_every) == 0:
        t0 = time()
        val_dict = get_data_dict(FLAGS.n_batch)
        results_values = sess.run(results_tensors, feed_dict=val_dict)
        validation_loss_list.append(results_values['loss_recall'])
        validation_error_list.append(results_values['recall_errors'])
        validation_reg_loss_list.append(results_values['loss_reg'])
        validation_total_loss_list.append(results_values['loss_recall'] + results_values['loss_reg'])
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
        Firing rate (Hz)  min {:.0f} ({}) \t max {:.0f} ({}) \t
        average {:.0f} +- std {:.0f} (averaged over batches and time)
        Comput. time (s)  training {:.2g} \t validation {:.2g}
        Loss              classif. {:.2g} \t reg. loss  {:.2g}
        '''.format(
            firing_rate_stats[0], firing_rate_stats[4], firing_rate_stats[1], firing_rate_stats[5],
            firing_rate_stats[2], firing_rate_stats[3],
            t_train, t_run,
            results_values['loss_recall'], results_values['loss_reg']
        ))

        #save firing rate stats to array HEREEEE. then to json outside loop
        firing_rate_total.append(firing_rate_stats)

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
    
    # Save neuron activations and parameters for visualization per word
    # Save neuron activations and parameters for visualization per word
    # Save neuron activations and parameters for visualization per word
    for i in range(FLAGS.n_batch):
        sequence = val_dict[input_nums][i]
        unique_phonemes = sorted(set(seq for seq in sequence if seq != 0))  # Ensure unique and sorted phonemes
        non_zero_sequence = tuple(unique_phonemes)

        if non_zero_sequence in word_mapping:
            word = word_mapping[non_zero_sequence]
            neuron_activations_per_word[word].append(results_values['av'].tolist())
            firing_rates_per_word[word].append(results_values['av'].tolist())  # Save the average firing rates



    # Do early stopping check if single batch validation error under stop_crit
    if (k_iter > 0 and validation_error_list[-1] < FLAGS.stop_crit):
        early_stopping_list = []
        t_es_0 = time()
        for i in range(8):
            val_dict = get_data_dict(FLAGS.n_batch)
            early_stopping_list.append(sess.run(results_tensors['recall_errors'], feed_dict=val_dict))
        t_es = time() - t_es_0
        print("Comput. time (s): early stopping: " + str(t_es))
        if np.mean(early_stopping_list) < FLAGS.stop_crit:
            n_iter_list.append(k_iter)
            print('Less than ' + str(FLAGS.stop_crit) + ' - stopping training at iteration ' + str(k_iter))
            break
        else:
            print('Early stopping error: ' + str(np.mean(early_stopping_list)) + ' higher than stop crit of: '
                  + str(FLAGS.stop_crit) + ' CONTINUE TRAINING')

    if k_iter == FLAGS.n_iter - 1:
        n_iter_list.append(k_iter)
        break

    # Do train step
    train_dict = get_data_dict(FLAGS.n_batch)
    t0 = time()
    results_values_train = sess.run([train_step, loss, loss_reg_f], feed_dict=train_dict)
    train_loss_list.append(results_values_train[1])
    t_train = time() - t0
    training_time_list.append(t_train)



    #save data here, still in training loop
    complete_firing_rate_data.append(results_values['av'])



training_time = time() - t_ref
print('FINISHED IN {:.2g} s'.format(training_time))

# Save neuron activation data to a file per word
with open("neuron_activations_per_word.txt", "w") as f:
    json.dump({word: acts for word, acts in neuron_activations_per_word.items()}, f)

# Save firing rates per word
with open("firing_rates_per_word.txt", "w") as f:
    json.dump({word: rates for word, rates in firing_rates_per_word.items()}, f)




#save total
def convert_and_save_list_of_lists(data, filename):
  """Saves a list of lists to a JSON file after converting float32 to float.

  Args:
    data: The list of lists to save.
    filename: The name of the file to save to.
  """
  # Iterate through the list of lists and convert float32 to float
  converted_data = []
  for inner_list in data:
    converted_inner_list = [float(item) for item in inner_list]
    converted_data.append(converted_inner_list)

  with open(filename, 'w') as f:
    json.dump(converted_data, f)


convert_and_save_list_of_lists(firing_rate_total,"firing_rate_total.txt")
convert_and_save_list_of_lists(complete_firing_rate_data,"complete_firing_rate_data.txt")

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
    test_dict = get_data_dict(FLAGS.n_batch)#FLAGS.n_batch to 
    results_values, plot_results_values, in_spk, spk, target_nums_np = sess.run(
        [results_tensors, plot_result_tensors, input_spikes, z, target_nums],
        feed_dict=test_dict)
    test_errors.append(results_values['recall_errors'])
    flag_dict['n_regular'] = n_regular
    plot_results_values['flags'] = flag_dict

    output_probs = plot_results_values['out_plot']

    if False:  # Override FLAGS.do_plot for final test plot
        fig, ax_list = plt.subplots(4, figsize=(5.9, 6))
        fig.canvas.manager.set_window_title(f'Test Case {i}')
        update_plot(plot_results_values, ax_list, n_max_neuron_per_raster=20, title=f'Test Case {i}', batch=i, t_cue_spacing=t_cue_spacing)

        correct_decision = plot_results_values['target_nums'][i, -1]  # Only taking the current batch item for simplicity
        ax_list[3].axvline(x=len(output_probs[0]) - 1, color='k', linestyle='--', label='Correct Decision')
        ax_list[3].text(len(output_probs[0]) - 1, 0.5, f'Correct: {correct_decision}', verticalalignment='center', fontsize=8)
        
        # Add vertical lines at each "time between cues" interval
        for t in range(0, len(output_probs[0]), t_cue_spacing):
            ax_list[3].axvline(x=t, color='r', linestyle='--', linewidth=0.5, label='Cue Interval' if t == 0 else "")

        plt.draw()
        plt.pause(1)
        plt.savefig(f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_plot_test_case_{i}.png')


print('''Statistics on the test set average error {:.2g} +- {:.2g} (averaged over 16 test batches of size {})'''
      .format(np.mean(test_errors), np.std(test_errors), FLAGS.n_batch))

# Plot the loss over time
#plt.figure()
fig, ax1 = plt.subplots()

# Plotting training loss
ax1.plot(train_loss_list, label='Training Loss', color='b')
ax1.set_ylim(0, 10)
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Training Loss')
ax1.axhline(y=0, color='r', linestyle='-')
ax1.legend(loc='upper left')

# Creating a second y-axis for validation error
#extend val error list by validate_every
def repeat_elements(lst, x):
    return [item for item in lst for _ in range(x)]

validation_error_list_extend = repeat_elements(validation_error_list,FLAGS.validate_every)

ax2 = ax1.twinx()
ax2.plot(validation_error_list_extend, label='Validation Error', color='g', linestyle='--')
ax2.set_ylim(0, 1)
ax2.set_ylabel('Validation Error')
ax2.legend(loc='upper right')

plt.title('Training Loss and Validation Error Over Time')




# List hyperparameters
hyperparams_text = f"Batch size: {FLAGS.n_batch}\n" \
                   f"Learning rate: {FLAGS.learning_rate}\n" \
                   f"Regular neurons: {n_regular}\n" \
                   f"Adaptive neurons: {n_adaptive}\n" \
                   f"Total neurons: {n_neurons}\n" \
                   f"Threshold: {thr}\n" \
                   f"Tau_v: {tau_v}\n" \
                   f"Tau_a: {tau_a}\n" \
                   f"Number of phonemes: {n_phonemes}\n" \
                   f"eprop: {FLAGS.eprop}\n" \
                   f"training time: {int(training_time)}\n" \
                   f"t_cue_spacing: {t_cue_spacing}\n" \
                   f"softmax temperature: {temperature}\n" \
                   f"input_encoding: {input_encoding_type}\n" \
                   f"Total number of words: {n_words}"

plt.gcf().text(0.02, 0.98, hyperparams_text, fontsize=9, verticalalignment='top')

plt.savefig(f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_loss_plot_{k_iter}.png')
print("about to show plot??")
plt.show()
print("plot is shown???")

def normalize_array(arr):
    total = np.sum(arr)
    if total == 0:
        raise ValueError("Sum of array elements is zero, cannot normalize")
    normalized_arr = arr / total
    return normalized_arr

#plot_result_values
print(np.shape(plot_results_values))
output_probs = plot_results_values['out_plot']
print(output_probs)
print(np.shape(output_probs))
print(type(output_probs))
#4 batches, 200 timesteps, 500 "words"
#interested in one batch, last timestep, first 5 words
output_probs_final = output_probs[0][-1]
output_probs_selected_words = output_probs[0][-1][:5]
predicted_word_prob = max(output_probs_final)
print(predicted_word_prob)

output_probs_selected_words_normal = normalize_array(output_probs_selected_words)
print(output_probs_selected_words_normal)


#run network for 100 words, find accuracy

del sess
