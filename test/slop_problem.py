import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Line with slope 1 in data coordinates
x = [0, 1]
y = [0, 1]  # Slope = 1 (45 degrees in data space)

# Left plot: Default aspect ratio
ax1.plot(x, y, 'r-', linewidth=3)
ax1.set_xlim(0, 2)
ax1.set_ylim(0, 2)
ax1.set_aspect('auto')  # Default
ax1.set_title(f"Slope = 1 (data units)\nAspect: 'auto'")
ax1.grid(True)
ax1.axhline(0, color='gray', alpha=0.3)
ax1.axvline(0, color='gray', alpha=0.3)

# Right plot: Equal aspect ratio
ax2.plot(x, y, 'b-', linewidth=3)
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 2)
ax2.set_aspect('equal')  # Makes 1 data unit = same length on both axes
ax2.set_title(f"Slope = 1 (data units)\nAspect: 'equal'")
ax2.grid(True)
ax2.axhline(0, color='gray', alpha=0.3)
ax2.axvline(0, color='gray', alpha=0.3)

plt.show()