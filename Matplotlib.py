import matplotlib.pyplot as plt
import numpy as np

#xpoints = np.array([0, 10])
#ypoints = np.array([0, 10])

class Grid():
    xaxis = 0
    yaxis = 0
    Matrix = [[]]
    fig, ax = plt.subplots(figsize=(10, 10))

    def __init__(self, xsize, ysize):
        self.xaxis = xsize
        self.yaxis = ysize
        self.Matrix = [[0 for x in range(xsize)] for y in range(ysize)] 
        self.ax.set(xlim=(0, self.xaxis), ylim=(0, self.xaxis), aspect='equal')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['left'].set_position('zero')
        x_ticks = np.arange(self.xaxis + 1)
        y_ticks = np.arange(self.yaxis + 1)
        self.ax.set_xticks(x_ticks[x_ticks != 0])
        self.ax.set_yticks(y_ticks[y_ticks != 0])
        self.ax.grid()
        
    def generate_Obstacle(self):
        for x in range(15 + 1):
            self.ax.axhline(x, lw=2, color='k', zorder=5)
            self.ax.axvline(x, lw=2, color='k', zorder=5)

#plt.plot(xpoints, ypoints, "o")



grid1 = Grid(5, 6)
print(grid1.Matrix)
plt.show()