import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random as rand
import math
import copy

#xpoints = np.array([0, 10])
#ypoints = np.array([0, 10])

# Grid class
class Grid():
    xsize = 0
    ysize = 0
    startpoint =[]
    endgoal = []
    addedNeighbors = False
    currPos = [1,1]
    Matrix = [[]]
    obstacles = []
    fig, ax = plt.subplots(figsize=(10, 10))

    # Creates grid and start point 

    def __init__(self, xaxis, yaxis, num_obstacles, start, goal):
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
        self.startpoint = start
        self.ax.plot(self.startpoint[0] - 0.5, self.startpoint[1] - 0.5, 'o')
        startcoor = self.graphToTable(self.startpoint[0], self.startpoint[1])
        self.Matrix[startcoor[0]][startcoor[1]] = 1
        self.chooseEndGoal(goal)
        for i in range(num_obstacles):
            self.gen_rand_obstacle()
        #print(self.Matrix)
        #self.showGraph()



    # Converts the graph coordinates to the 2D array coordinates
    def graphToTable(self, xcor, ycor):
        newY = xcor -1
        newX = self.ysize - ycor 
        return([newX, newY]) 

    def tableToGraph(self, xcor, ycor):
        newX = ycor + 1
        newY = self.ysize - xcor
        return([newX, newY])
    # Generates an obstacle object given the xcoordinate and y coordinate

    def generate_Cup(self, xcor, ycor):
        self.ax.add_patch(patches.Rectangle(xy=(xcor-1, ycor-1), width=1, height=1, linewidth=1, color='red', fill=True))
        self.Matrix[self.graphToTable(xcor, ycor)[0]][self.graphToTable(xcor, ycor)[1]] = 3
        self.ax.add_patch(patches.Rectangle(xy=(xcor, ycor-1), width=1, height=1, linewidth=1, color='red', fill=True))
        self.Matrix[self.graphToTable(xcor+1, ycor)[0]][self.graphToTable(xcor+1, ycor)[1]] = 3
        self.ax.add_patch(patches.Rectangle(xy=(xcor+1, ycor-1), width=1, height=1, linewidth=1, color='red', fill=True))
        self.Matrix[self.graphToTable(xcor+2, ycor)[0]][self.graphToTable(xcor+2, ycor)[1]] = 3


        self.ax.add_patch(patches.Rectangle(xy=(xcor+2, ycor), width=1, height=1, linewidth=1, color='red', fill=True))
        self.Matrix[self.graphToTable(xcor+3, ycor+1)[0]][self.graphToTable(xcor+3, ycor+1)[1]] = 3
        self.ax.add_patch(patches.Rectangle(xy=(xcor+2, ycor+1), width=1, height=1, linewidth=1, color='red', fill=True))
        self.Matrix[self.graphToTable(xcor+3, ycor+2)[0]][self.graphToTable(xcor+3, ycor+2)[1]] = 3

        self.ax.add_patch(patches.Rectangle(xy=(xcor+1, ycor+2), width=1, height=1, linewidth=1, color='red', fill=True))
        self.Matrix[self.graphToTable(xcor+2, ycor+3)[0]][self.graphToTable(xcor+2, ycor+3)[1]] = 3
        self.ax.add_patch(patches.Rectangle(xy=(xcor-1, ycor+2), width=1, height=1, linewidth=1, color='red', fill=True))
        self.Matrix[self.graphToTable(xcor, ycor+3)[0]][self.graphToTable(xcor, ycor+3)[1]] = 3
        self.ax.add_patch(patches.Rectangle(xy=(xcor, ycor+2), width=1, height=1, linewidth=1, color='red', fill=True))
        self.Matrix[self.graphToTable(xcor+1, ycor+3)[0]][self.graphToTable(xcor+1, ycor+3)[1]] = 3
 

    def gen_rand_obstacle(self):
        while True:
            xcor = rand.randint(1, self.xsize)
            ycor = rand.randint(1, self.ysize)
            if xcor == 1 and ycor ==1:
                continue
            elif xcor == self.startpoint[0] and ycor == self.startpoint[1]:
                continue
            elif xcor == self.endgoal[0] and ycor == self.endgoal[1]:
                continue
            obst = Obstacle(xcor, ycor)
            self.obstacles.append(obst)
            self.ax.add_patch(patches.Rectangle(xy=(xcor-1, ycor-1), width=1, height=1, linewidth=1, color='red', fill=True))
            self.Matrix[self.graphToTable(xcor, ycor)[0]][self.graphToTable(xcor, ycor)[1]] = 3
            return



    def generate_Obstacle(self, xcor, ycor):
        if xcor == 1:
            if ycor == 1:
                return
        obst = Obstacle(xcor, ycor)
        self.obstacles.append(obst)
        self.ax.add_patch(patches.Rectangle(xy=(xcor-1, ycor-1), width=1, height=1, linewidth=1, color='red', fill=True))
        self.Matrix[self.graphToTable(xcor, ycor)[0]][self.graphToTable(xcor, ycor)[1]] = 3
        #print(self.Matrix)

    # Creates a random end goal that is not on an object
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

        self.ax.add_patch(patches.Rectangle(xy=(xcor-1, ycor-1), width=1, height=1, linewidth=1, color='green', fill=True))
        self.Matrix[self.graphToTable(xcor, ycor)[0]][self.graphToTable(xcor, ycor)[1]] = 2
        #print(self.Matrix)
        self.endgoal = [xcor, ycor]
    
    def chooseEndGoal(self, coordinates):
        
        self.ax.add_patch(patches.Rectangle(xy=(coordinates[0]-1, coordinates[1]-1), width=1, height=1, linewidth=1, color='green', fill=True))
        #print(self.Matrix)
        self.endgoal = [coordinates[0], coordinates[1]]
        endcoor = self.graphToTable(self.endgoal[0], self.endgoal[1])
        self.Matrix[endcoor[0]][endcoor[1]] = 2

    def checkEndGoal(self):
        tableEndX = self.graphToTable(self.endgoal[0], self.endgoal[1])[0]
        tableEndY = self.graphToTable(self.endgoal[0], self.endgoal[1])[1]
        blocked = True
        if tableEndY != 0:
            if self.Matrix[tableEndX][tableEndY-1] != 3:
                blocked = False

        if tableEndX != (self.ysize - 1):
            if self.Matrix[tableEndX+1][tableEndY] != 3:        
                blocked = False

        if tableEndX != 0:
            if self.Matrix[tableEndX-1][tableEndY] != 3:
                blocked = False

        if tableEndY != (self.xsize -1):
            if self.Matrix[tableEndX][tableEndY + 1] != 3:
                blocked = False
        
        if blocked == True:
            print("Enforced")
            self.showGraph()
##########################################################################################################################################################################
    def getDIJNeighbors(self, currNode, visited, queue):
        tableX = self.graphToTable(self.currPos[0], self.currPos[1])[0]
        tableY = self.graphToTable(self.currPos[0], self.currPos[1])[1]
        neighbors = []
        reached = False
        currpathway = currNode[1]

        # Left neighbor
        if tableY != 0:
            if self.Matrix[tableX][tableY-1] != 3:
                graphX = self.tableToGraph(tableX, tableY-1)[0]
                graphY = self.tableToGraph(tableX, tableY-1)[1]
                for i in visited:
                    if i[0] == graphX and i[1] == graphY:
                        reached = True
                for j in queue:
                    if j[0][0] == graphX and j[0][1] == graphY:
                        reached = True
                if reached == False:    
                    lpathway = copy.deepcopy(currpathway)
                    lpathway.append([graphX, graphY])
                    neighbors.append([[graphX, graphY], lpathway])
                    
        
        
        reached = False
        # Botton neighbor
        if tableX != (self.ysize - 1):
            if self.Matrix[tableX+1][tableY] != 3:
                graphX = self.tableToGraph(tableX+1, tableY)[0]
                graphY = self.tableToGraph(tableX+1, tableY)[1]
                for i in visited:
                    if i[0] == graphX and i[1] == graphY:
                        reached = True
                for j in queue:
                    if j[0][0] == graphX and j[0][1] == graphY:
                        reached = True

                if reached == False:
                    bpathway = copy.deepcopy(currpathway)
                    bpathway.append([graphX, graphY])
                    neighbors.append([[graphX, graphY], bpathway])

        
        
        # Top neighbor
        reached = False
        if tableX != 0:
            if self.Matrix[tableX-1][tableY] != 3:
                graphX = self.tableToGraph(tableX-1, tableY)[0]
                graphY = self.tableToGraph(tableX-1, tableY)[1]
                for i in visited:
                    if i[0] == graphX and i[1] == graphY:
                        reached = True
                for j in queue:
                    if j[0][0] == graphX and j[0][1] == graphY:
                        reached = True
                if reached == False:
                    tpathway = copy.deepcopy(currpathway)
                    tpathway.append([graphX, graphY])
                    #print(currNode[1])
                    neighbors.append([[graphX, graphY], tpathway])
                    

        # Right neighbor
        reached = False
        if tableY != (self.xsize -1):
            if self.Matrix[tableX][tableY + 1] != 3:
                graphX = self.tableToGraph(tableX, tableY+1)[0]
                graphY = self.tableToGraph(tableX, tableY+1)[1]
                for i in visited:
                    if i[0] == graphX and i[1] == graphY:
                        reached = True
                for j in queue:
                    if j[0][0] == graphX and j[0][1] == graphY:
                        reached = True

                if reached == False:
                    rpathway = copy.deepcopy(currpathway)
                    rpathway.append([graphX, graphY])
                    #print(currNode[1])
                    neighbors.append([[graphX, graphY], rpathway])


        #print(neighbors)
        add = True

        for node in neighbors:
            for i in range(0, len(queue)):
                if node[0] == queue[i][0]:
                    if len(node[1]) > len(queue[i][1]):
                        add = False

            if add == True:
                queue.append(node)
            add = True
        
        if len(neighbors) != 0:
            self.addedNeighbors = True

        #print(stack)
        return(queue)

##########################################################################################################################################################################
    def getBFSNeighbors(self, currNode, visited, queue):
        tableX = self.graphToTable(self.currPos[0], self.currPos[1])[0]
        tableY = self.graphToTable(self.currPos[0], self.currPos[1])[1]
        neighbors = []
        reached = False
        currpathway = currNode[1]

        # Left neighbor
        if tableY != 0:
            if self.Matrix[tableX][tableY-1] != 3:
                graphX = self.tableToGraph(tableX, tableY-1)[0]
                graphY = self.tableToGraph(tableX, tableY-1)[1]
                for i in visited:
                    if i[0] == graphX and i[1] == graphY:
                        reached = True
                for j in queue:
                    if j[0][0] == graphX and j[0][1] == graphY:
                        reached = True
                if reached == False:    
                    lpathway = copy.deepcopy(currpathway)
                    lpathway.append([graphX, graphY])
                    neighbors.append([[graphX, graphY], lpathway])
                    
        
        
        reached = False
        # Botton neighbor
        if tableX != (self.ysize - 1):
            if self.Matrix[tableX+1][tableY] != 3:
                graphX = self.tableToGraph(tableX+1, tableY)[0]
                graphY = self.tableToGraph(tableX+1, tableY)[1]
                for i in visited:
                    if i[0] == graphX and i[1] == graphY:
                        reached = True
                for j in queue:
                    if j[0][0] == graphX and j[0][1] == graphY:
                        reached = True

                if reached == False:
                    bpathway = copy.deepcopy(currpathway)
                    bpathway.append([graphX, graphY])
                    neighbors.append([[graphX, graphY], bpathway])

        
        
        # Top neighbor
        reached = False
        if tableX != 0:
            if self.Matrix[tableX-1][tableY] != 3:
                graphX = self.tableToGraph(tableX-1, tableY)[0]
                graphY = self.tableToGraph(tableX-1, tableY)[1]
                for i in visited:
                    if i[0] == graphX and i[1] == graphY:
                        reached = True
                for j in queue:
                    if j[0][0] == graphX and j[0][1] == graphY:
                        reached = True
                if reached == False:
                    tpathway = copy.deepcopy(currpathway)
                    tpathway.append([graphX, graphY])
                    #print(currNode[1])
                    neighbors.append([[graphX, graphY], tpathway])
                    

        # Right neighbor
        reached = False
        if tableY != (self.xsize -1):
            if self.Matrix[tableX][tableY + 1] != 3:
                graphX = self.tableToGraph(tableX, tableY+1)[0]
                graphY = self.tableToGraph(tableX, tableY+1)[1]
                for i in visited:
                    if i[0] == graphX and i[1] == graphY:
                        reached = True
                for j in queue:
                    if j[0][0] == graphX and j[0][1] == graphY:
                        reached = True

                if reached == False:
                    rpathway = copy.deepcopy(currpathway)
                    rpathway.append([graphX, graphY])
                    #print(currNode[1])
                    neighbors.append([[graphX, graphY], rpathway])

        #print(neighbors)
        for node in neighbors:
            queue.append(node)
        
        if len(neighbors) != 0:
            self.addedNeighbors = True

        #print(stack)
        return(queue)

#####################################################################################################################################################
    def goto(self, coordinates):
        #print(self.currPos)
        '''graphX = self.graphToTable(self.currPos[0], self.currPos[1])[0]
        graphY = self.graphToTable(self.currPos[0], self.currPos[1])[1]
        self.Matrix[graphX][graphY] = 0
        
        newX = self.graphToTable(coordinates[0], coordinates[1])[0]
        newY = self.graphToTable(coordinates[0], coordinates[1])[1]
        #print(coordinates)
        #print(newX)
        #print(newY)
        self.Matrix[newX][newY] = 1

        #xchange = [self.currPos[0] -0.5, coordinates[0] -0.5]
        #ychange = [self.currPos[1] -0.5, coordinates[1] -0.5]
        '''
        self.currPos = coordinates

        #self.ax.plot(xchange, ychange, color= "blue")
        #self.ax.plot(self.currPos[0]-0.5, self.currPos[1]-0.5, "o", color = "blue")
###################################################################################################################################################
    def getNeighbors(self, currNode, visited, stack):
        tableX = self.graphToTable(self.currPos[0], self.currPos[1])[0]
        tableY = self.graphToTable(self.currPos[0], self.currPos[1])[1]
        neighbors = []
        reached = False
        currpathway = currNode[1]

        # Left neighbor
        if tableY != 0:
            if self.Matrix[tableX][tableY-1] != 3:
                graphX = self.tableToGraph(tableX, tableY-1)[0]
                graphY = self.tableToGraph(tableX, tableY-1)[1]
                for i in visited:
                    if i[0] == graphX and i[1] == graphY:
                        reached = True
                if reached == False:    
                    lpathway = copy.deepcopy(currpathway)
                    lpathway.append([graphX, graphY])
                    neighbors.append([[graphX, graphY], lpathway])
                    
        
        
        reached = False
        # Botton neighbor
        if tableX != (self.ysize - 1):
            if self.Matrix[tableX+1][tableY] != 3:
                graphX = self.tableToGraph(tableX+1, tableY)[0]
                graphY = self.tableToGraph(tableX+1, tableY)[1]
                for i in visited:
                    if i[0] == graphX and i[1] == graphY:
                        reached = True
                if reached == False:
                    bpathway = copy.deepcopy(currpathway)
                    bpathway.append([graphX, graphY])
                    neighbors.append([[graphX, graphY], bpathway])

        
        
        # Top neighbor
        reached = False
        if tableX != 0:
            if self.Matrix[tableX-1][tableY] != 3:
                graphX = self.tableToGraph(tableX-1, tableY)[0]
                graphY = self.tableToGraph(tableX-1, tableY)[1]
                for i in visited:
                    if i[0] == graphX and i[1] == graphY:
                        reached = True
                if reached == False:
                    tpathway = copy.deepcopy(currpathway)
                    tpathway.append([graphX, graphY])
                    #print(currNode[1])
                    neighbors.append([[graphX, graphY], tpathway])
                    

        # Right neighbor
        reached = False
        if tableY != (self.xsize -1):
            if self.Matrix[tableX][tableY + 1] != 3:
                graphX = self.tableToGraph(tableX, tableY+1)[0]
                graphY = self.tableToGraph(tableX, tableY+1)[1]
                for i in visited:
                    if i[0] == graphX and i[1] == graphY:
                        reached = True
                if reached == False:
                    rpathway = copy.deepcopy(currpathway)
                    rpathway.append([graphX, graphY])
                    #print(currNode[1])
                    neighbors.append([[graphX, graphY], rpathway])

        #print(neighbors)
        for node in neighbors:
            stack.append(node)
        
        if len(neighbors) != 0:
            self.addedNeighbors= True

        #print(stack)
        return(stack)


    def updateShortestPath(self, currNode, curr, shortestPath):
        updateStart = 0
        tempCurrNode = copy.deepcopy(currNode)
        for pos in range(0, len(shortestPath)):
            if shortestPath[pos] == curr:
                updateStart = pos + 1
        
        for i in range(updateStart, len(shortestPath)):
            tempCurrNode[1].append(shortestPath[updateStart])
        
        if len(tempCurrNode[1]) < len(shortestPath):
            return(tempCurrNode[1])
        else:
            return(shortestPath)

    def printStack(self, stack):
        newStack = []
        for i in stack:
            newStack.append(i[0])
        print(newStack)

####################################################################### Prints the shortest path in red
    def getShortestPath(self, shortestPath):
        for i in range(0, len(shortestPath)):
            counter = len(shortestPath) -1
            while counter > i:
                try:
                    if shortestPath[i] == shortestPath[counter]:
                        print(i)
                        print(counter)
                        currPos = i
                        for j in range(i, counter):
                            del shortestPath[currPos]
                except:
                    break
                    #pdb.set_trace()

                    
                counter -= 1

        for i in range(0, len(shortestPath)-1):
            self.ax.plot([shortestPath[i][0]-0.5, shortestPath[i+1][0]-0.5] , [shortestPath[i][1]-0.5, shortestPath[i+1][1]-0.5], color= "red")
            self.ax.plot(shortestPath[i+1][0]-0.5, shortestPath[i+1][1]-0.5, "o", color = "red")
#################################################################################################################################################
    def MHueristics(self, coordinates):
        xpoints = abs(coordinates[0] - self.endgoal[0])
        ypoints = abs(coordinates[1] - self.endgoal[1])
        return(xpoints + ypoints+xpoints + ypoints)
    
    def EHueristics(self, coordinates):
        xpoints = abs(coordinates[0] - self.endgoal[0])
        ypoints = abs(coordinates[1] - self.endgoal[1])
        distance = (xpoints*xpoints) + (ypoints * ypoints)
        return(distance)


#################################################################################################################################################
    def DFS(self):
        #currPath = [[1,1]]
        shortestPath = []
        stack = [ [[1,1], [[1,1]] ] ]
        visited = []
        #Worked = False
        #self.checkEndGoal()
        FirstTime = True
#        print(stack[-1])
        #print(self.getNeighbors(visited))
        newvisited = []
        while (len(stack) > 0):
            currNode = stack[-1]
            curr = stack[-1][0]
            stack.pop()         
            #print("Visited: " + str(visited))

            if (curr in visited) == False:
                self.goto(curr)

                # Reach Destination
                if self.currPos == self.endgoal:
                    if FirstTime == True or len(shortestPath) > len(currNode[1]):
                        shortestPath = copy.deepcopy(currNode[1])
                        print("Shortest Path: " + str(shortestPath))
                        FirstTime = False
                    #print("Shortest Path: " + str(shortestPath))
                    if len(stack) > 0:
                        newvisited = copy.deepcopy(stack[-1][1])
                        del newvisited[-1]
                        visited = copy.deepcopy(newvisited)
                    continue

                visited.append(curr)
                stack = self.getNeighbors(currNode, visited, stack)
                if self.addedNeighbors:
                    #self.printStack(stack)
                    #print("Enter1")
                    self.addedNeighbors = False
                else:
                    #print("Enter 2")
                    if len(stack) > 0:
                        newvisited = copy.deepcopy(stack[-1][1])
                        del newvisited[-1]
                        visited = copy.deepcopy(newvisited)
                    continue
                
            else:
                if curr in shortestPath:
                    shortestPath = self.updateShortestPath(currNode, curr, shortestPath)

        print("Final Shortest Path: " + str(shortestPath))
        for i in range(0, len(shortestPath)-1):
            self.ax.plot([shortestPath[i][0]-0.5, shortestPath[i+1][0]-0.5] , [shortestPath[i][1]-0.5, shortestPath[i+1][1]-0.5], color= "red")
            self.ax.plot(shortestPath[i+1][0]-0.5, shortestPath[i+1][1]-0.5, "o", color = "red")



############################################################################################################################################       
    def BFS(self):
        queue = [ [self.startpoint, [self.startpoint] ] ]
        visited = []
        shortestPath = []
        counter = 0
        #Worked = False
        self.checkEndGoal()
        while len(queue) > 0:
            if counter > self.xsize*self.ysize*self.xsize:
                print("Did not finish")
                print(counter)
                break
            currNode = queue[0]
            curr = queue[0][0]
            del queue[0]

            if (curr in visited) == False:
                self.goto(curr)

                if self.currPos == self.endgoal:
                    shortestPath = currNode[1]
                    break
                else:
                    visited.append(curr)
                    queue = self.getBFSNeighbors(currNode, visited, queue)
            #print(queue)
            counter += 1
        
        print("Final Shortest Path: " + str(shortestPath))
        for i in range(0, len(shortestPath)-1):
            self.ax.plot([shortestPath[i][0]-0.5, shortestPath[i+1][0]-0.5] , [shortestPath[i][1]-0.5, shortestPath[i+1][1]-0.5], color= "green")
            self.ax.plot(shortestPath[i+1][0]-0.5, shortestPath[i+1][1]-0.5, "o", color = "green")

#####################################################################################################################################################   

    def Dijkstra(self):
        queue = [ [self.startpoint, [self.startpoint] ] ]
        visited = []
        shortestPath =[]
        self.checkEndGoal()
        while len(queue) > 0:
            shortestLength = len(queue[0][1])
            shortestPos = 0

            for i in range(1, len(queue)):
                if len(queue[i][1]) < shortestLength:
                    shortestPos = i
                    shortestLength = len(queue[i][1])
            
            currNode = queue[shortestPos]
            del queue[shortestPos]
            self.goto(currNode[0])

            if self.currPos == self.endgoal:
                shortestPath = currNode[1]
                break
            else:
                visited.append(currNode[0])
                queue = self.getDIJNeighbors(currNode, visited, queue)

        #print("Final Shortest Path: " + str(shortestPath))
        for i in range(0, len(shortestPath)-1):
            self.ax.plot([shortestPath[i][0]-0.5, shortestPath[i+1][0]-0.5] , [shortestPath[i][1]-0.5, shortestPath[i+1][1]-0.5], color= "green")
            self.ax.plot(shortestPath[i+1][0]-0.5, shortestPath[i+1][1]-0.5, "o", color = "green")

    

    def Dijkstra_ML(self):
        queue = [ [self.startpoint, [self.startpoint] ] ]
        visited = []
        shortestPath =[]
        self.checkEndGoal()
        while len(queue) > 0:
            shortestLength = len(queue[0][1])
            shortestPos = 0

            for i in range(1, len(queue)):
                if len(queue[i][1]) < shortestLength:
                    shortestPos = i
                    shortestLength = len(queue[i][1])
            
            currNode = queue[shortestPos]
            del queue[shortestPos]
            self.goto(currNode[0])

            if self.currPos == self.endgoal:
                shortestPath = currNode[1]
                break
            else:
                visited.append(currNode[0])
                queue = self.getDIJNeighbors(currNode, visited, queue)

        #print("Final Shortest Path: " + str(shortestPath))
        #del shortestPath[-1]
        return(shortestPath)

######################################################################################################################################################xx``


    def Manhattan(self):
        queue = [ [[1,1], [[1,1]] ] ]
        visited = []
        shortestPath = []
        currPath = []
        self.checkEndGoal()
        while len(queue) > 0:
            shortestLength = self.MHueristics(queue[0][0]) + int(len(queue[0][1]))
            shortestPos = 0

            for i in range(1, len(queue)):
                distance = self.MHueristics(queue[i][0]) + int(len(queue[i][1]))
                if distance <= shortestLength:
                    shortestPos = i
                    shortestLength = distance
            
            currNode = queue[shortestPos]
            del queue[shortestPos]
            self.goto(currNode[0])
            currPath.append(currNode[0])

            if self.currPos == self.endgoal:
                shortestPath = currNode[1]
                break
            else:
                visited.append(currNode[0])
                queue = self.getDIJNeighbors(currNode, visited, queue)

        #print("Final Shortest Path: " + str(shortestPath))
        '''for i in range(0, len(shortestPath)-1):
            self.ax.plot([shortestPath[i][0]-0.5, shortestPath[i+1][0]-0.5] , [shortestPath[i][1]-0.5, shortestPath[i+1][1]-0.5], color= "green")
            self.ax.plot(shortestPath[i+1][0]-0.5, shortestPath[i+1][1]-0.5, "o", color = "green")
        '''
        for i in range(0, len(currPath)-1):
            self.ax.plot([currPath[i][0]-0.25, currPath[i+1][0]-0.25] , [currPath[i][1]-0.25, currPath[i+1][1]-0.25], color= "black")
            self.ax.plot(currPath[i+1][0]-0.25, currPath[i+1][1]-0.25, "o", color = "black")

        #self.Dijkstra()

    def Euclidian(self):
        queue = [ [[1,1], [[1,1]] ] ]
        visited = []
        shortestPath = []
        currPath = []
        self.checkEndGoal()
        while len(queue) > 0:
            shortestLength = int(self.EHueristics(queue[0][0])) + int(len(queue[0][1]))
            shortestPos = 0

            for i in range(1, len(queue)):
                distance = int(self.EHueristics(queue[i][0])) + int(len(queue[i][1]))
                if distance <= shortestLength:
                    shortestPos = i
                    shortestLength = distance
            
            currNode = queue[shortestPos]
            del queue[shortestPos]
            self.goto(currNode[0])
            currPath.append(currNode[0])

            if self.currPos == self.endgoal:
                shortestPath = currNode[1]
                break
            else:
                visited.append(currNode[0])
                queue = self.getDIJNeighbors(currNode, visited, queue)

        #print("Final Shortest Path: " + str(shortestPath))
        '''for i in range(0, len(shortestPath)-1):
            self.ax.plot([shortestPath[i][0]-0.5, shortestPath[i+1][0]-0.5] , [shortestPath[i][1]-0.5, shortestPath[i+1][1]-0.5], color= "green")
            self.ax.plot(shortestPath[i+1][0]-0.5, shortestPath[i+1][1]-0.5, "o", color = "green")
        '''
        for i in range(0, len(currPath)-1):
            self.ax.plot([currPath[i][0]-0.75, currPath[i+1][0]-0.75] , [currPath[i][1]-0.75, currPath[i+1][1]-0.75], color= "green")
            self.ax.plot(currPath[i+1][0]-0.75, currPath[i+1][1]-0.75, "o", color = "green")

        #self.Dijkstra()


    # Shows the graph
    def showGraph(self):
        plt.show()
    
    def returnGraph(self):
        return(self.Matrix)
    
    def returnGraphObstacles(self):
        startcoor = self.graphToTable(self.startpoint[0], self.startpoint[1])
        endcoor = self.graphToTable(self.endgoal[0], self.endgoal[1])
        tempMatrix = copy.deepcopy(self.Matrix)
        #print(startcoor)
        #print(endcoor)
        tempMatrix[startcoor[0]][startcoor[1]] = 0
        tempMatrix[endcoor[0]][endcoor[1]] = 0
        return(tempMatrix)        
 
# Obstacle class
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


import pdb
import pickle

num_points_desired = 1
dataset = []
for i in range(num_points_desired):
    xaxis = 6
    yaxis = 6
    start = [rand.randint(1, xaxis), rand.randint(1, yaxis)]
    goal = [rand.randint(1, xaxis), rand.randint(1, yaxis)]
    if goal == start:
        goal = [rand.randint(1, xaxis), rand.randint(1, yaxis)]
    num_obstacles = rand.randint(1, 5)
    grid = Grid(xaxis, yaxis, num_obstacles, start, goal)

    pathway = grid.Dijkstra_ML()
    scenario_datasetx = []   
    scenario_datasety = []

    for point in pathway:
        d = [grid.returnGraphObstacles(), start, goal]
        scenario_datasetx.append(d)
        scenario_datasety.append(point)
    dataset.append([scenario_datasetx, scenario_datasety])

print(dataset)

with open("datasetShortestPath", "wb") as fp:   #Pickling
   pickle.dump(dataset, fp)





'''


grid1 = Grid(xsize, ysize)



for i in range(numObstacles):
#    try:
    x = rand.randint(1, xsize)
    y = rand.randint(1, ysize)
    grid1.generate_Obstacle(x,y)
#    except:
#        pdb.set_trace()

#grid1.generate_Cup(2,2)

#grid1.generate_Cup(6,6)
grid1.generate_Obstacle(1,3)
grid1.generate_Obstacle(4,4)
grid1.generate_Obstacle(1,3)
grid1.generate_Obstacle(3,2)
grid1.generate_Obstacle(2, 3)

#grid1.generate_Obstacle(2,2)
#grid1.generate_Obstacle(3, 2)


grid1.createEndGoal()
#grid1.chooseEndGoal([9, 9])
#grid1.generate_Obstacle(2,1)
#grid1.generate_Obstacle(1, 2)
#grid1.test()
grid1.Dijkstra()
#grid1.Manhattan()
grid1.showGraph()'''