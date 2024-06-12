import matplotlib.pyplot as plt
import numpy as np
import json
import random

# Function to read data from JSON files
def read_json_data(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    return {k: np.array(v) for k, v in data.items()}

# Read the saved data
neuron_activations_per_word = read_json_data("neuron_activations_per_word.txt")
firing_rates_per_word = read_json_data("firing_rates_per_word.txt")

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

# Function to plot firing rate distribution
def plot_total_firing_rate_distribution(first_activations, last_activations, title, bins=10):
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

# Plot total firing rate distribution for the first and last iterations
print("Plotting total firing rate distribution")
plot_total_firing_rate_distribution(neuron_activations_per_word, "Total Firing Rate Distribution (First vs Last Iteration)", bins=10,title = "total firing rate distribution")

# Plot average firing rate over iterations for the specified words
print(f"Plotting average firing rate over iterations for the words: {', '.join(words_to_plot)}")
plot_firing_rates_over_iterations_per_word(firing_rates_per_word, words_to_plot, "Average Firing Rate Over Iterations")
    