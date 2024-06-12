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
learned_params = read_data("learned_params.txt")


# Function to plot firing rate distribution per word for the first and last iterations
def plot_firing_rate_distribution_per_word(neuron_activations_per_word, words, title, bins=20, range=(0, 0.2)):
    plt.figure(figsize=(10, 6))
    for word in words:
        if word in neuron_activations_per_word and len(neuron_activations_per_word[word]) > 0:
            first_iter_firing_rates = neuron_activations_per_word[word][0].flatten()
            last_iter_firing_rates = neuron_activations_per_word[word][-1].flatten()
            if not np.isnan(first_iter_firing_rates).all() and not np.isnan(last_iter_firing_rates).all():
                plt.hist(first_iter_firing_rates, bins=bins, range=range, alpha=0.5, edgecolor='black', label=f'{word} (First Iter)', density=True)
                plt.hist(last_iter_firing_rates, bins=bins, range=range, alpha=0.5, edgecolor='red', label=f'{word} (Last Iter)', density=True)
    plt.xlabel('Firing Rate')
    plt.ylabel('Number of Neurons')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# Function to plot total firing rate distribution for the first and last iterations
def plot_firing_rate_distribution(first_activations, midlle_activations, last_activations, title, bins=20, range=(0, 0.2)):
    first_firing_rates = first_activations.flatten()
    last_firing_rates = last_activations.flatten()
    middel_firing_rates = midlle_activations.flatten()
    # Create histograms
    plt.figure(figsize=(10, 6))
    plt.hist(first_firing_rates, bins=bins, range=range, alpha=0.5, label='First Iteration', edgecolor='black', density=True)
    plt.hist(last_firing_rates, bins=bins, range=range, alpha=0.5, label='Last Iteration', edgecolor='red', density=True)
    plt.hist(middel_firing_rates, bins=bins, range=range, alpha=0.5, label='middle Iteration', edgecolor='red', density=True)
    plt.xlabel('Firing Rate(Khz)')
    plt.ylabel('Number of Neurons')
    titlee = title + ", BPTT"
    plt.title(titlee)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()





def load_list_of_lists(filename):
  """Loads a list of lists from a JSON file.

  Args:
    filename: The name of the file to load from.

  Returns:
    The list of lists loaded from the file.
  """
  with open(filename, 'r') as f:
    return json.load(f)
  

firing_rate_total_data = load_list_of_lists("firing_rate_total.txt")
complete_firing_rate_data = load_list_of_lists("complete_firing_rate_data.txt")


# Sample data (replace with your actual list of lists)
data = firing_rate_total_data
iterations = range(len(data))

# Separate lists for each value
#this is avg and std and min and max ....
value1 = [inner_list[0] for inner_list in data]
value2 = [inner_list[1] for inner_list in data]#avg
value3 = [inner_list[2] for inner_list in data]
value4 = [inner_list[3] for inner_list in data]
value5 = [inner_list[4] for inner_list in data]
value6 = [inner_list[5] for inner_list in data]

# Create the plot
#plt.plot(iterations, value1, label='Value 1', marker='o', linestyle='-')
plt.plot(iterations, value2, label='average firing rate', marker='s', linestyle='--')
#plt.plot(iterations, value3, label='Value 3', marker='^', linestyle = ':')
#plt.plot(iterations, value4, label='Value 4', marker='*', linestyle='-.')  # Add markers and linestyles for all values
#plt.plot(iterations, value5, label='Value 5', marker='x', linestyle='solid')
#plt.plot(iterations, value6, label='Value 6', marker='d', linestyle='dashed')

# Customize the plot
plt.xlabel("Iterations")
plt.ylabel("average firing rate")
titlee = "average firing rate across iterations"
titlee = titlee + ", BPTT"
plt.title(titlee)
plt.grid(True)
plt.legend()  # Add legend to identify each line

# Show the plot
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


#find point of least firing rate, plot that as well
interesting_firing_rate_index = complete_firing_rate_data.index(min(complete_firing_rate_data))
interesting_firing_rate_index = int(len(complete_firing_rate_data)/2)

plot_firing_rate_distribution(np.array(complete_firing_rate_data[0]),np.array(complete_firing_rate_data[interesting_firing_rate_index]),np.array(complete_firing_rate_data[-1]),"network firing rate distribution: first iteration, middel iteration, and  last iteratiom",bins = 50)
print("done")