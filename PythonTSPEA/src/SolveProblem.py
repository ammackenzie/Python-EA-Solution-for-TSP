'''
Created on 26 May 2020

@author: Andre
'''
# -*- coding: utf-8 -*-


import random, math
import numpy as np
import matplotlib.pyplot as plt
import cma


class Parameters:
    #declare editable variables

    maxIterations = 1000
    populationSize = 100
        
    mutationRate = 0.08
    tournamentSize = 10

    offspring = 50
    #different cities arrays available for algorithm
    cities = np.array([[41, 94], [37, 84], [54, 67], [25, 62], [7, 64], [2, 99], [68, 58], [71, 44], [54, 62], [83, 69], [64, 60], [18, 54], [22, 60], [83, 46], [91, 38], [25, 38], [24, 42], [58, 69], [71, 71], [74, 78], [87, 76], [18, 40], [13, 40], [82, 7], [62, 32], [58, 35], [45, 21], [41, 26], [44, 35], [4, 50]])
    
#     cities = np.array([[565.0, 575.0],
#                     [25.0, 185.0],
#                     [345.0, 750.0],
#                     [945.0, 685.0],
#                     [845.0, 655.0],
#                     [880.0, 660.0],
#                     [25.0, 230.0],
#                     [525.0, 1000.0],
#                     [580.0, 1175.0],
#                     [650.0, 1130.0],
#                     [1605.0, 620.0],
#                     [1220.0, 580.0],
#                     [1465.0, 200.0],
#                     [1530.0, 5.0],
#                     [845.0, 680.0],
#                     [725.0, 370.0],
#                     [145.0, 665.0],
#                     [415.0, 635.0],
#                     [510.0, 875.0],
#                     [560.0, 365.0],
#                     [300.0, 465.0],
#                     [520.0, 585.0],
#                     [480.0, 415.0],
#                     [835.0, 625.0],
#                     [975.0, 580.0],
#                     [1215.0, 245.0],
#                     [1320.0, 315.0],
#                     [1250.0, 400.0],
#                     [660.0, 180.0],
#                     [410.0, 250.0],
#                     [420.0, 555.0],
#                     [575.0, 665.0],
#                     [1150.0, 1160.0],
#                     [700.0, 580.0],
#                     [685.0, 595.0],
#                     [685.0, 610.0],
#                     [770.0, 610.0],
#                     [795.0, 645.0],
#                     [720.0, 635.0],
#                     [760.0, 650.0],
#                     [475.0, 960.0],
#                     [95.0, 260.0],
#                     [875.0, 920.0],
#                     [700.0, 500.0],
#                     [555.0, 815.0],
#                     [830.0, 485.0],
#                     [1170.0, 65.0],
#                     [830.0, 610.0],
#                     [605.0, 625.0],
#                     [595.0, 360.0],
#                     [1340.0, 725.0],
#                     [1740.0, 245.0]])
      
#     cities = np.array([[35, 51],
#                    [113, 213],
#                    [82, 280],
#                    [322, 340],
#                    [256, 352],
#                    [160, 24],
#                    [322, 145],
#                    [12, 349],
#                    [282, 20],
#                    [241, 8],
#                    [398, 153],
#                    [182, 305],
#                    [153, 257],
#                    [275, 190],
#                    [242, 75],
#                    [19, 229],
#                    [303, 352],
#                    [39, 309],
#                    [383, 79],
#                    [226, 343]])
        
class SolveProblem:

    def __init__(self):
        #get variables from the Parameters class and set remaining 
        self.solutionSize = len(Parameters.cities)
        self.offspring = Parameters.offspring
        self.xcoords = [0]* self.solutionSize
        self.ycoords = [0]* self.solutionSize
        self.populationSize = Parameters.populationSize
        self.maxIterations = Parameters.maxIterations
        self.mutationRate = Parameters.mutationRate
        self.tournamentSize = Parameters.tournamentSize
        self.elite = [0]*self.solutionSize
        self.population = np.arange(self.populationSize * self.solutionSize).reshape(self.populationSize,self.solutionSize)
    
        for i in range(self.solutionSize):
            self.xcoords[i] = Parameters.cities[i][0]
            self.ycoords[i] = Parameters.cities[i][1]
        

    def runEA(self):
        self.initialise() #initialise a random population
        bestID, bestFitness = self.findBest()
        progress = []
        progress.append(bestFitness)
        plt.ylabel('Distance')
        plt.xlabel('Generation')
         
        for i in range(self.maxIterations):
            parents = []
            children = [0]*self.offspring
            bestID, bestFitness = self.findBest()
            for j in range(self.offspring):
                p1 = self.tournamentSelect() 
                p2 = self.tournamentSelect() 
                parents.append(p1)
                parents.append(p2)
                child = [0]*self.solutionSize
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                children[j] = child                
                 
            self.mutatePop()
             
            parentIndex = 0
            for k in range(self.offspring):
                 
                self.replaceWorstParent(children[k], self.evaluate(children[k]), parents[parentIndex], parents[parentIndex+1])
                parentIndex +=2
             
            bestID, bestFitness = self.findBest()
            progress.append(bestFitness)
             
            print("Iteration:" + str(i+1) + " best fitness: " + str(bestFitness) + " member elements: " + str(self.population[bestID]))
             
             
        print("Best fitness found: " + str(bestFitness))
        plt.plot(progress) #plot to graph
        plt.show()
    
    
    def initialise(self):
        #have to ensure valid solutions only (no repeating city visits etc.)
        for i in range(self.populationSize):
            self.population[i] = self.generatePermutation()
    
    def generatePermutation(self):
        #generates random arrays with one visit per city, covering all cities
        soln = [0]*self.solutionSize
        for i in range(self.solutionSize):
            soln[i] = i
        
        for j in range(self.solutionSize):
            randIndex = random.randint(0, self.solutionSize-1)
            toSwitch = soln[randIndex]
            soln[randIndex] = soln[j]
            soln[j] = toSwitch
        
        return soln
    
    def mutate(self, child):
        for i in range(self.solutionSize):
            if random.random() < self.mutationRate:
                randIndex = random.randint(0, self.solutionSize-1)
                toSwitch = child[randIndex]
                child[randIndex] = child[i]
                child[i] = toSwitch 
        return child
    
    def mutatePop(self):
        for i in range(self.populationSize):
            if random.random() < self.mutationRate:
                self.population[i] = self.mutate(self.population[i])
                
            
    def tournamentSelect(self):
        tempID = random.randint(0, self.populationSize-1)
        bestID = tempID
        tempFitness = self.evaluate(self.population[tempID])
        bestFitness = tempFitness
        
        for x in range(self.tournamentSize):
            tempID = random.randint(0, self.populationSize-1)
            tempFitness = self.evaluate(self.population[tempID])
            if tempFitness < bestFitness:
                bestFitness = tempFitness
                bestID = tempID
        return bestID
    
    def crossover(self, parentOneIndex, parentTwoIndex):
        #create new child array
        child = []
        childPart1 = []
        childPart2 = []
        crosspointOne = random.randint(0, self.solutionSize-1)
        crosspointTwo = random.randint(0, self.solutionSize-1)
        
        for i in range(crosspointOne, crosspointTwo):
            childPart1.append(self.population[parentOneIndex][i])
        
        childPart2 = [item for item in self.population[parentTwoIndex] if item not in childPart1]
        
        child = childPart1 + childPart2
        return child   
    
    def evaluate(self, populationMember):
        fitness = x1 = x2 = y1 = y2 = j = 0

        for i in range(self.solutionSize):
            id1 = populationMember[i]
            if i == self.solutionSize-1: #on last entry in solution
                j=0 #next city to reach will be first entry
            else:
                j = i + 1
             
            id2 = populationMember[j]
             
            x1 = self.xcoords[id1]
            x2 = self.xcoords[id2]
            y1 = self.ycoords[id1]
            y2 = self.ycoords[id2]
             
            distance = math.sqrt((x1 - x2)**2 + (y1-y2)**2)
            fitness += distance
                
        return fitness
    
    def replaceWorstParent(self, child, childFitness, parentOneIndex, parentTwoIndex):
        worstFitness = self.evaluate(self.population[parentOneIndex])
        worstID = parentOneIndex
        tempFitness = self.evaluate(self.population[parentTwoIndex])
        if tempFitness > worstFitness:
            worstFitness = tempFitness
            worstID = parentTwoIndex
        
        if childFitness < worstFitness:
            for i in range(self.solutionSize):
                self.population[worstID][i] = child[i]                 
        
    def findBest(self):
        #begin with first member of population
        bestFitness = self.evaluate(self.population[0])
        bestID = 0
        
        for i in range(1, self.populationSize):
            tempFitness = self.evaluate(self.population[i])
            if tempFitness < bestFitness:
                bestFitness = tempFitness
                bestID = i
        
        return bestID, bestFitness
    
    def findWorst(self):
        worstFitness = self.evaluate(self.population[0])
        worstID = 0
         
        for i in range(1, self.populationSize):
            tempFitness = self.evaluate(self.population[i])
            if tempFitness > worstFitness:
                worstFitness = tempFitness
                worstID = i
         
        return worstID, worstFitness


