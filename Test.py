'''def shortestPath(gridMatrix, startpoint, endpoint):
    pathway = [[]]
    while (True):
        if forward == true: 
            goForward()
            pathway.append(currXY)
        elif forward == true && left == true:
            turnLeft()
            goForward()
            pathway.append(currXY)
        elif forward == true && right == true:
            turnRight()
            goForward()
            pathway.append(currXY)
        else:
            goto(pathway[len{pathway} - 1])
        
1, 1, 2, 3, 5, 8, 13, ...
'''
def recursion(n):
    if n <= 1:
        return(n)
    
    return(recursion(n-1) + recursion (n-2))

print(recursion(5))