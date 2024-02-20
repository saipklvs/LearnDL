import matplotlib.pyplot as plt
import numpy as np

# Define your lists
lists = {
    'A': [1, 2, 3, 5],
    'B': [1],
    'C': [5, 4],
    'D': [9, 10],
    'E': [1, 3]
    # Add more lists if needed
}

# Convert lists to a matrix where rows represent lists and columns represent elements
max_len = max(len(lst) for lst in lists.values())
matrix = np.zeros((len(lists), max_len))
for i, key in enumerate(lists):
    matrix[i, :len(lists[key])] = lists[key]

# Plot the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(matrix, cmap='viridis', aspect='auto')
plt.colorbar(label='Value')
plt.xlabel('Element Index')
plt.ylabel('List')
plt.title('Heatmap of Lists')
plt.xticks(np.arange(max_len))
plt.yticks(np.arange(len(lists)), lists.keys())
plt.grid(False)
plt.show()
