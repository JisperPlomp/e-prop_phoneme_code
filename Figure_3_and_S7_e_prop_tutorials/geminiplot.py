import matplotlib.pyplot as plt
import json
import matplotlib.pyplot as plt
import numpy as np
import json
import random

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
#print(firing_rate_total_data)
print(np.shape(firing_rate_total_data))
print("rgette")

import matplotlib.pyplot as plt

# Sample data (replace with your actual list of lists)
data = firing_rate_total_data

iterations = range(len(data))

# Separate lists for each value
value1 = [inner_list[0] for inner_list in data]
value2 = [inner_list[1] for inner_list in data]
value3 = [inner_list[2] for inner_list in data]
value4 = [inner_list[3] for inner_list in data]
value5 = [inner_list[4] for inner_list in data]
value6 = [inner_list[5] for inner_list in data]

# Create the plot
#plt.plot(iterations, value1, label='Value 1', marker='o', linestyle='-')
plt.plot(iterations, value2, label='Value 2', marker='s', linestyle='--')
#plt.plot(iterations, value3, label='Value 3', marker='^', linestyle = ':')
#plt.plot(iterations, value4, label='Value 4', marker='*', linestyle='-.')  # Add markers and linestyles for all values
#plt.plot(iterations, value5, label='Value 5', marker='x', linestyle='solid')
#plt.plot(iterations, value6, label='Value 6', marker='d', linestyle='dashed')

# Customize the plot
plt.xlabel("Iterations")
plt.ylabel("average firing rate")
plt.title("average firing rate across iterations")
plt.grid(True)
plt.legend()  # Add legend to identify each line

# Show the plot
plt.show()