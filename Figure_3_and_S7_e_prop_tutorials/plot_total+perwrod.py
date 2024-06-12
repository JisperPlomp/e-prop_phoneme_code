import matplotlib.pyplot as plt
import numpy as np
import json
import random

# Function to read data from JSON files
def read_json_data(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    return {k: np.array(v) for k, v in data.items()}

# Function to read data from text files
def read_data(file_name):
    with open(file_name, "r") as f:
        data = [eval(line.strip()) for line in f]
    return data

# Read the saved data
neuron_activations_per_word = read_json_data("neuron_activations_per_word.txt")
firing_rates_per_word = read_json_data("firing_rates_per_word.txt")
neuron_activations = read_data("neuron_activations.txt")
firing_rates = read_data("firing_rates.txt")
learned_params = read_data("learned_params.txt")

# Function to plot firing rate distribution per word for the first and last iterations
def plot_firing_rate_distribution_per_word(neuron_activations_per_word, words, title, bins=10):
    plt.figure(figsize=(10, 6))
    for word in words:
        if word in neuron_activations_per_word and len(neuron_activations_per_word[word]) > 0:
            first_iter_avg_firing_rates = np.mean(neuron_activations_per_word[word][0], axis=0)
            last_iter_avg_firing_rates = np.mean(neuron_activations_per_word[word][-1], axis=0)
            if not np.isnan(first_iter_avg_firing_rates).all() and not np.isnan(last_iter_avg_firing_rates).all():
                plt.hist(first_iter_avg_firing_rates, bins=bins, alpha=0.5, edgecolor='black', label=f'{word} (First Iter)', density=True)
                plt.hist(last_iter_avg_firing_rates, bins=bins, alpha=0.5, edgecolor='red', label=f'{word} (Last Iter)', density=True)
    plt.xlabel('Average Firing Rate')
    plt.ylabel('Number of Neurons')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# Function to plot total firing rate distribution for the first and last iterations
def plot_firing_rate_distribution(first_activations, last_activations, title, bins=10):
    # If activations are 1D, they already represent the average firing rates
    if len(first_activations.shape) == 1:
        avg_firing_rates_first = first_activations
    else:
        avg_firing_rates_first = np.mean(first_activations, axis=0)
    
    if len(last_activations.shape) == 1:
        avg_firing_rates_last = last_activations
    else:
        avg_firing_rates_last = np.mean(last_activations, axis=0)
    
    # Create histograms
    plt.figure(figsize=(10, 6))
    plt.hist(avg_firing_rates_first, bins=bins, alpha=0.5, label='First Iteration', edgecolor='black')
    plt.hist(avg_firing_rates_last, bins=bins, alpha=0.5, label='Last Iteration', edgecolor='red')
    plt.xlabel('Average Firing Rate')
    plt.ylabel('Number of Neurons')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

plot_firing_rate_distribution(np.array(neuron_activations[0]), np.array(neuron_activations[-1]), "Firing Rate Distribution (First vs Last Iteration)", bins=10)


# Function to plot average firing rate over iterations per word
def plot_firing_rates_over_iterations_per_word(firing_rates_per_word, words, title):
    plt.figure()
    for word in words:
        if word in firing_rates_per_word and len(firing_rates_per_word[word]) > 0:
            avg_firing_rates = np.mean(firing_rates_per_word[word], axis=1)
            if not np.isnan(avg_firing_rates).all():
                plt.plot(avg_firing_rates, label=word)
    plt.xlabel('Iteration')
    plt.ylabel('Average Firing Rate')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# Function to plot average firing rate over iterations
def plot_firing_rates_over_iterations(firing_rates, title):
    plt.figure()
    plt.plot(firing_rates)
    plt.xlabel('Iteration')
    plt.ylabel('Average Firing Rate')
    plt.title(title)
    plt.grid(True)
    plt.show()

# Function to plot learned parameters
def plot_learned_params(params, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    axes[0].imshow(params[0], aspect='auto', cmap='viridis')
    axes[0].set_title('Input Weights')
    axes[0].set_xlabel('Input Neurons')
    axes[0].set_ylabel('Hidden Neurons')

    axes[1].imshow(params[1], aspect='auto', cmap='viridis')
    axes[1].set_title('Recurrent Weights')
    axes[1].set_xlabel('Hidden Neurons')
    axes[1].set_ylabel('Hidden Neurons')

    axes[2].imshow(params[2], aspect='auto', cmap='viridis')
    axes[2].set_title('Output Weights')
    axes[2].set_xlabel('Hidden Neurons')
    axes[2].set_ylabel('Output Neurons')

    plt.show()

# Plotting for specific words or 3 random words
word_to_plot = None  # Replace with the word you want to plot, or set to None for random selection

if word_to_plot:
    words_to_plot = [word_to_plot]
else:
    # Filter out words that do not have activation data
    valid_words = [word for word, activations in neuron_activations_per_word.items() if len(activations) > 0]
    words_to_plot = random.sample(valid_words, min(3, len(valid_words)))

# Plot firing rate distribution for the specified words
print(f"Plotting firing rate distribution for the words: {', '.join(words_to_plot)}")
plot_firing_rate_distribution_per_word(neuron_activations_per_word, words_to_plot, "Firing Rate Distribution (First vs Last Iteration)", bins=10)

# Aggregate first and last activations across all words for total firing rate distribution
first_iter_total_firing_rates = []
last_iter_total_firing_rates = []
for activations in neuron_activations_per_word.values():
    if len(activations) > 0 and isinstance(activations[0], np.ndarray):
        first_iter_avg = np.mean(activations[0], axis=0)
        last_iter_avg = np.mean(activations[-1], axis=0)
        if first_iter_avg.ndim > 0:
            first_iter_total_firing_rates.append(first_iter_avg)
        if last_iter_avg.ndim > 0:
            last_iter_total_firing_rates.append(last_iter_avg)

if first_iter_total_firing_rates and last_iter_total_firing_rates:
    first_iter_total_firing_rates = np.concatenate(first_iter_total_firing_rates)
    last_iter_total_firing_rates = np.concatenate(last_iter_total_firing_rates)

    # Plot total firing rate distribution for the first and last iterations
    print("Plotting total firing rate distribution")
    plot_total_firing_rate_distribution(first_iter_total_firing_rates, last_iter_total_firing_rates, "Total Firing Rate Distribution (First vs Last Iteration)", bins=10)
else:
    print("No valid data to plot for total firing rate distribution.")

# Plotting the first iteration learned parameters
print("Plotting the first iteration learned parameters:")
plot_learned_params([np.array(param) for param in learned_params[0]], "Learned Parameters (First Iteration)")

# Plotting the last iteration learned parameters
print("Plotting the last iteration learned parameters:")
plot_learned_params([np.array(param) for param in learned_params[-1]], "Learned Parameters (Last Iteration)")

# Plotting average firing rate over all iterations
print("Plotting average firing rate over all iterations:")
plot_firing_rates_over_iterations(firing_rates, "Average Firing Rate Over Iterations")
