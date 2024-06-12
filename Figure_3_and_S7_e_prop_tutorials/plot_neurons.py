import matplotlib.pyplot as plt
import numpy as np

# Function to read data from text files
def read_data(file_name):
    with open(file_name, "r") as f:
        data = [eval(line.strip()) for line in f]
    return data

# Read the saved data
neuron_activations = read_data("neuron_activations.txt")
firing_rates = read_data("firing_rates.txt")
learned_params = read_data("learned_params.txt")

# Function to plot firing rate distribution
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

# Plotting firing rate distribution for the first and last iteration
print("Plotting firing rate distribution for the first and last iteration:")
plot_firing_rate_distribution(np.array(neuron_activations[0]), np.array(neuron_activations[-1]), "Firing Rate Distribution (First vs Last Iteration)", bins=10)

# Plotting learned parameters for the first iteration
print("Plotting the first iteration learned parameters:")
plot_learned_params([np.array(param) for param in learned_params[0]], "Learned Parameters (First Iteration)")

# Plotting learned parameters for the last iteration
print("Plotting the last iteration learned parameters:")
plot_learned_params([np.array(param) for param in learned_params[-1]], "Learned Parameters (Last Iteration)")

# Plotting average firing rate over all iterations
print("Plotting average firing rate over all iterations:")
plot_firing_rates_over_iterations(firing_rates, "Average Firing Rate Over Iterations")
