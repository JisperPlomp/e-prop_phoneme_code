import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import numpy.random as rd
import json

class NumpyAwareEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyAwareEncoder, self).default(obj)

def raster_plot(ax, spikes, linewidth=0.8, **kwargs):
    n_t, n_n = spikes.shape
    event_times, event_ids = np.where(spikes)
    max_spike = 10000
    event_times = event_times[:max_spike]
    event_ids = event_ids[:max_spike]
    for n, t in zip(event_ids, event_times):
        ax.vlines(t, n + 0., n + 1., linewidth=linewidth, **kwargs)
    ax.set_ylim([0 + .5, n_n + .5])
    ax.set_xlim([0, n_t])
    ax.set_yticks([0, n_n])

def strip_right_top_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def generate_poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern, list):
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]
    shp = prob_pattern.shape
    if not(freezing_seed is None): rng = rd.RandomState(freezing_seed)
    else: rng = rd.RandomState()
    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return spikes

def generate_phoneme_data(batch_size, phoneme_sequences, word_mapping, num_neurons_per_phoneme=10, cue_spacing=150):
    num_phonemes = max(max(seq) for seq in phoneme_sequences) + 1
    seq_len = sum((len(seq) * cue_spacing for seq in phoneme_sequences)) // len(phoneme_sequences)
    input_spikes = np.zeros((batch_size, seq_len, num_neurons_per_phoneme * num_phonemes))
    input_nums = np.zeros((batch_size, seq_len), dtype=int)
    target_words = np.zeros(batch_size, dtype=int)
    for i in range(batch_size):
        seq = phoneme_sequences[i % len(phoneme_sequences)]
        time_index = 0
        for phoneme in seq:
            start_idx = phoneme * num_neurons_per_phoneme
            input_spikes[i, time_index:time_index + cue_spacing, start_idx:start_idx + num_neurons_per_phoneme] = 1
            input_nums[i, time_index:time_index + cue_spacing] = phoneme
            time_index += cue_spacing

        target_words[i] = word_mapping[tuple(seq)]
    return input_spikes, input_nums, target_words


def generate_phoneme_data_dif_length(batch_size, phoneme_sequences, word_mapping, num_neurons_per_phoneme=10, cue_spacing=150):
    num_phonemes = max(max(seq) for seq in phoneme_sequences) + 1
    max_seq_len = max(len(seq) for seq in phoneme_sequences) * cue_spacing
    input_spikes = np.zeros((batch_size, max_seq_len, num_neurons_per_phoneme * num_phonemes))
    input_nums = np.zeros((batch_size, max_seq_len), dtype=int)
    target_words = np.zeros(batch_size, dtype=int)
    for i in range(batch_size):
        seq = phoneme_sequences[i % len(phoneme_sequences)]
        time_index = 0
        for phoneme in seq:
            start_idx = phoneme * num_neurons_per_phoneme
            input_spikes[i, time_index:time_index + cue_spacing, start_idx:start_idx + num_neurons_per_phoneme] = 1
            input_nums[i, time_index:time_index + cue_spacing] = phoneme
            time_index += cue_spacing

        target_words[i] = word_mapping[tuple(seq)]
    return input_spikes, input_nums, target_words


import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

def update_plot(plot_result_values, ax_list, plot_traces=False, batch=0, n_max_neuron_per_raster=10, title=None,
                eps_sel=None, trace_sel=None,t_cue_spacing = 150):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """
    flags = plot_result_values['flags']
    n_rec = flags['n_regular']
    n_con = flags['n_adaptive']
    n_tot = n_rec + n_con



    #phoneme_sequences = [[0, 1, 2], [0, 2, 1], [1, 2, 3]]
    #word_mapping = {(0, 1, 2): 0, (0, 2, 1): 1, (1, 2, 3): 2}
    
    phoneme_sequences = [[0, 1, 2], [0, 2, 1], [1, 2, 3], [1,2,1,2]]
    word_mapping = {(0, 1, 2): 0, (0, 2, 1): 1, (1, 2, 3): 2, (1, 2, 1,2): 3}
    phoneme_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}
    n_input_symbols = len(word_mapping)


    # Clear the axis to print new plots
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        strip_right_top_axis(ax)
        ax_list[0].set_title(title)
    k_ax = 0

    # Plot the data, from top to bottom each axe represents: inputs, recurrent and controller
    for data, d_name in zip([plot_result_values['input_spikes'], plot_result_values['z'], plot_result_values['z']],
                            ['Input', 'LIF', 'ALIF']):
        if np.size(data) > 0:
            ax = ax_list[k_ax]
            data = data[batch]

            if d_name == 'Input':
                n_max = data.shape[1]
                cell_select = np.linspace(start=0, stop=data.shape[1] - 1, num=n_max, dtype=int)

            elif d_name == 'LIF':
                n_max = min(n_rec, n_max_neuron_per_raster)
                cell_select = np.linspace(start=0, stop=n_rec - 1, num=n_max, dtype=int)

            elif d_name == 'ALIF':
                n_max = min(n_con, n_max_neuron_per_raster)
                cell_select = np.linspace(start=n_rec, stop=n_rec + n_con - 1, num=n_max, dtype=int)

            if cell_select.size != 0:
                k_ax += 1

                data = data[:, cell_select]  # select a maximum of n_max_neuron_per_raster neurons to plot
                if d_name == 'Input':
                    n_channel = data.shape[1] // n_input_symbols
                    # insert empty row
                    zero_fill = np.zeros((data.shape[0], int(n_channel / 2)))
                    data = np.concatenate((data[:, 3 * n_channel:], zero_fill,
                                           data[:, 2 * n_channel:3 * n_channel], zero_fill,
                                           data[:, :n_channel], zero_fill,
                                           data[:, n_channel:2 * n_channel]), axis=1)
                    ax.set_yticklabels([])

                    # Highlight active phonemes
                    for i, phoneme_seq in enumerate(phoneme_sequences):
                        time_index = 0
                        for phoneme in phoneme_seq:
                            ax.add_patch(
                                patches.Rectangle((time_index, 0), 150, data.shape[1], facecolor=phoneme_colors[phoneme], alpha=0.3)
                            )
                            time_index += 150

                    # Add legend for phoneme colors
                    handles = [patches.Patch(color=phoneme_colors[phoneme], label=f'Phoneme {phoneme}') for phoneme in phoneme_colors]
                    ax.legend(handles=handles, loc='upper right', fontsize=7)

                    ax.add_patch(  # Value 0 row
                        patches.Rectangle((0, 2 * n_channel + 2 * int(n_channel / 2)), data.shape[0], n_channel,
                                          facecolor="red", alpha=0.1))
                    ax.add_patch(  # Value 1 row
                        patches.Rectangle((0, 3 * n_channel + 3 * int(n_channel / 2)), data.shape[0], n_channel,
                                          facecolor="blue", alpha=0.1))

                raster_plot(ax, data, linewidth=0.4)
                ax.set_ylabel(d_name)
                ax.set_xticklabels([])
                ax.set_xticks([])

    ax = ax_list[k_ax]
    output_probs = plot_result_values['out_plot'][batch]

    presentation_steps = np.arange(output_probs.shape[0])

    for class_idx in range(output_probs.shape[1]):
        ax.plot(presentation_steps, output_probs[:, class_idx], label=f'Class {class_idx}', alpha=0.7)

    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel('Output Probabilities')
    ax.legend(loc='lower center', fontsize=7, bbox_to_anchor=(0.5, -0.05), ncol=3)
    ax.axis([0, presentation_steps[-1] + 1, -0.3, 1.1])
    ax.set_xticklabels([])
    ax.set_xticks([])

    # plot learning signal
    if plot_traces:
        k_ax += 1
        ax = ax_list[k_ax]
        ax.set_ylabel('$L_j$')
        sub_data = plot_result_values['learning_signal_cls'][batch]

        presentation_steps = np.arange(sub_data.shape[0])
        cell_select = np.linspace(start=0, stop= n_tot , num=1, dtype=int)#this should be n_neurons?maybe also change num
        v = np.maximum(abs(np.min(sub_data[:, cell_select].T)), abs(np.max(sub_data[:, cell_select].T)))
        ax.pcolor(sub_data[:, cell_select].T, label='learning_signal', cmap='seismic', alpha=0.4, linewidth=0.3,
                  vmin=-v, vmax=v)
        ax.set_xticklabels([])
        ax.set_xticks([])

        # plot eligibility traces
        k_ax += 1
        ax = ax_list[k_ax]
        ax.set_xticklabels([])
        ax.set_xticks([])

        e_trace = plot_result_values['e_trace'][batch]
        epsilon = plot_result_values['epsilon_a'][batch]

        presentation_steps = np.arange(e_trace.shape[0])
        if trace_sel is None:
            trace_sel = np.linspace(start=0, stop=e_trace.shape[1], num=n_max)
        if eps_sel is None:
            eps_sel = np.linspace(start=0, stop=epsilon.shape[1], num=n_max)

        colors = plt.get_cmap("tab10")

        for k in range(e_trace.shape[1]):
            if k in trace_sel:
                ax.plot(e_trace[:, k], alpha=0.8, linewidth=1, label=str(k), color=colors(k))

        ax.axis([0, presentation_steps[-1], 1.2 * np.min(e_trace), np.max(e_trace)])
        ax.set_ylabel('e-trace')

        # plot epsilon
        k_ax += 1
        ax = ax_list[k_ax]

        for k in range(epsilon.shape[1]):
            if k in eps_sel:
                ax.plot(epsilon[:, k], label='slow e-trace component', alpha=0.8, linewidth=1, color=colors(k))

        ax.axis([0, presentation_steps[-1], np.min(epsilon), np.max(epsilon)])
        ax.set_ylabel('slow factor')

    ax.set_xlabel('Time in ms')
    plt.subplots_adjust(hspace=0.3)

