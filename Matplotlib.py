import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random as rand

#xpoints = np.array([0, 10])
#ypoints = np.array([0, 10])

class Grid():
    xsize = 0
    ysize = 0
    startpoint =[1, 1]
    endgoal = []
    Matrix = [[]]
    obstacles = []
    fig, ax = plt.subplots(figsize=(10, 10))

    def __init__(self, yaxis, xaxis):
        self.xsize = xaxis
        self.ysize = yaxis
        self.Matrix = [[0 for y in range(self.xsize)] for x in range(self.ysize)] 
        self.ax.set(xlim=(0, self.xsize), ylim=(0, self.ysize), aspect='equal')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['left'].set_position('zero')
        x_ticks = np.arange(xaxis + 1)
        y_ticks = np.arange(yaxis + 1)
        self.ax.set_xticks(x_ticks[x_ticks])
        self.ax.set_yticks(y_ticks[y_ticks])
        self.ax.grid()
        self.ax.plot(self.startpoint[0] - 0.5, self.startpoint[1] - 0.5, 'o')

        self.Matrix[self.ysize-1][0] = 1
        

    def graphToTable(self, xcor, ycor):
        newY = xcor -1
        newX = self.ysize - ycor 
        return([newX, newY]) 

    def generate_Obstacle(self, xcor, ycor):
        newx = xcor-1
        newy = ycor-1
        obst = Obstacle(xcor, ycor)
        self.obstacles.append(obst)
        self.ax.add_patch(patches.Rectangle(xy=(newx, newy), width=1, height=1, linewidth=1, color='red', fill=True))
        self.Matrix[self.graphToTable(xcor, ycor)[0]][self.graphToTable(xcor, ycor)[1]] = 3
        print(self.Matrix)

    def createEndGoal(self):
        while (True):
            xcor = rand.randint(2, self.xsize)
            ycor = rand.randint(2, self.ysize)
            unique = True
            for i in range(0, len(self.obstacles)):
                if (xcor == self.obstacles[i].getXcor() and ycor == self.obstacles[i].getYcor()):
                    unique = False
                    break
            if (unique == True):
                break

        self.ax.plot(xcor - 0.5, ycor - 0.5, 'o', color = "green")
        self.Matrix[self.graphToTable(xcor, ycor)[0]][self.graphToTable(xcor, ycor)[1]] = 2
        print(self.Matrix)
        self.endgoal = [xcor, ycor]
    
    def showGraph(self):
        plt.show()

    
class Obstacle():
    xcor = 0
    ycor = 0
    def __init__(self, xpos, ypos):
        self.xcor = xpos
        self.ycor = ypos
    
    def getXcor(self):
        return(self.xcor)
    
    def getYcor(self):
        return(self.ycor)

grid1 = Grid(5, 5)
grid1.generate_Obstacle(2,2)
grid1.generate_Obstacle(3,3)
grid1.generate_Obstacle(4,4)



grid1.createEndGoal()


grid1.showGraph()