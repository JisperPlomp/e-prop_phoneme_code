import matplotlib.pyplot as plt
import numpy as np
import json
import random

# Function to read data from JSON files
def read_json_data(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    return {k: np.array(v) for k, v in data.items()}

# Function to read multiple JSON objects from a text file
def read_multiple_json_objects(file_name):
    with open(file_name, "r") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    return data

# Read the saved data
neuron_activations_per_word = read_json_data("neuron_activations_per_word.txt")
firing_rates_per_word = read_json_data("firing_rates_per_word.txt")
learned_params = read_multiple_json_objects("learned_params.txt")
firing_rate_total = read_json_data("firing_rate_total.txt")


print (np.shape(firing_rates_per_word.items()))

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
def plot_firing_rate_distribution(first_activations, last_activations, title, bins=20, range=(0, 0.2)):
    first_firing_rates = first_activations.flatten()
    last_firing_rates = last_activations.flatten()
    
    # Create histograms
    plt.figure(figsize=(10, 6))
    plt.hist(first_firing_rates, bins=bins, range=range, alpha=0.5, label='First Iteration', edgecolor='black', density=True)
    plt.hist(last_firing_rates, bins=bins, range=range, alpha=0.5, label='Last Iteration', edgecolor='red', density=True)
    plt.xlabel('Firing Rate')
    plt.ylabel('Number of Neurons')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# Aggregate first and last activations across all words for total firing rate distribution
first_iter_total_firing_rates = []
last_iter_total_firing_rates = []
for activations in neuron_activations_per_word.values():
    if len(activations) > 0 and isinstance(activations[0], np.ndarray):
        first_iter_total_firing_rates.append(activations[0].flatten())
        last_iter_total_firing_rates.append(activations[-1].flatten())

if first_iter_total_firing_rates and last_iter_total_firing_rates:
    first_iter_total_firing_rates = np.concatenate(first_iter_total_firing_rates)
    last_iter_total_firing_rates = np.concatenate(last_iter_total_firing_rates)

    # Plot total firing rate distribution for the first and last iterations
    print("Plotting total firing rate distribution")
    plot_firing_rate_distribution(first_iter_total_firing_rates, last_iter_total_firing_rates, "Total Firing Rate Distribution (First vs Last Iteration)", bins=20, range=(0, 0.2))
else:
    print("No valid data to plot for total firing rate distribution.")

# Function to calculate and plot average firing rate over iterations
def calculate_and_plot_average_firing_rate(firing_rates_per_word, title):
    total_firing_rates = []

    # Determine the maximum number of iterations
    iteration_count = 200
    print(f"Total number of iterations found: {iteration_count}")

    # Calculate average firing rate for each iteration
    for iteration in range(iteration_count):
        iter_firing_rates = []
        for word, rates in firing_rates_per_word.items():
            iter_firing_rates.append(rates[iteration])
        if iter_firing_rates:
            avg_firing_rate = np.mean(np.concatenate(iter_firing_rates))  # Combine all data for the iteration and calculate the mean
            total_firing_rates.append(avg_firing_rate)
        else:
            total_firing_rates.append(0)  # If there are no firing rates for this iteration, set average to 0
        print(f"Iteration {iteration}: Avg firing rate = {avg_firing_rate} (based on {len(iter_firing_rates)} words)")

    # Debug print to check the calculated average firing rates
    print("Average firing rates over iterations:", total_firing_rates)

    # Plot average firing rate over iterations
    plt.figure()
    plt.plot(total_firing_rates)
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
    valid_words = [word for word, activations in neuron_activations_per_word.items()]# if len(activations) > 0
    words_to_plot = random.sample(valid_words, min(3, len(valid_words)))

# Plot firing rate distribution for the specified words
print(f"Plotting firing rate distribution for the words: {', '.join(words_to_plot)}")
plot_firing_rate_distribution_per_word(neuron_activations_per_word, words_to_plot, "Firing Rate Distribution (First vs Last Iteration)", bins=20, range=(0, 0.2))

# Plotting the first iteration learned parameters
print("Plotting the first iteration learned parameters:")
plot_learned_params(learned_params[0], "Learned Parameters (First Iteration)")

# Plotting the last iteration learned parameters
print("Plotting the last iteration learned parameters:")
plot_learned_params(learned_params[-1], "Learned Parameters (Last Iteration)")

# Calculate and plot average firing rate over all iterations
print("Plotting average firing rate over all iterations:")
calculate_and_plot_average_firing_rate(firing_rates_per_word, "Average Firing Rate Over Iterations")
