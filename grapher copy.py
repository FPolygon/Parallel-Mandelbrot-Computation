import matplotlib.pyplot as plt

# Data
nodes = [1, 2, 4, 8]
times = [649.9698414, 325.7037568, 33.36293, 20.3279278]

# Create a new figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data
ax.plot(nodes, times, marker='o', linestyle='-', linewidth=2, markersize=8)

# Set the title and labels
ax.set_title('Mandelbrot Set Generation Performance', fontsize=16)
ax.set_xlabel('Number of Nodes', fontsize=12)
ax.set_ylabel('Execution Time (seconds)', fontsize=12)

# Set the tick positions and labels
ax.set_xticks(nodes)
ax.set_xticklabels(nodes)

# Add a grid
ax.grid(True, linestyle='--', alpha=0.7)

# Add a legend
ax.legend(['Execution Time'], loc='best', fontsize=12)

# Add annotations with the exact time values
for i, time in enumerate(times):
    ax.annotate(f'{time:.2f}s', (nodes[i], time), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()