#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from ortools.linear_solver import pywraplp
import itertools
from matplotlib import pyplot as plt
import numpy as np
import time
import random

Point = namedtuple("Point", ['x', 'y'])


class arc:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.length = length(start,end)

class node:
    def __init__(self, x, y, city_name):
        self.x = x
        self.y = y
        self.city_name = city_name

class TSP:
    def __init__(self,input_data):
        self.input_data=input_data
        self.cities = []
        self.tour_length = 0
        self.route = [0]
        self.arcs=[]
        self.output=''

        #populating the tsp with city data 
        lines = input_data.split('\n')
        self.nodeCount = int(lines[0])     
        for i in range(1, self.nodeCount+1):
            line = lines[i]
            parts = line.split()
            self.add_city(node(float(parts[0]), float(parts[1]), i-1))

        self.dist_matrix=self.distance_matrix()

    def route_find(self):
        self.route = [0]
        arcs = 0 
        current = 0
        while True:
            for arc in self.arcs:
                if arc.start.city_name == current:
                    self.route.append(arc.end.city_name)
                    current = arc.end.city_name
                    arcs +=1
                    break 
            if arcs == self.nodeCount:
                break
        # print (self.route)
        return 0


    def add_city(self,node):
        self.cities.append(node)

    def length(self, point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    #p(n) =0 
    def pdf(self, n, x):
        return ((-2/(n**2))*x) + (2/n)

    # p(n/4)=0
    def pdf2(self, n, x):
        return max(0,((-32/(n**2))*x) + (8/n))
        

    def random_arc(self):
        n = self.nodeCount
        u1 = math.ceil(n * random.random()) #discrete uniform random variable from 0-n
        p_x = self.pdf2(n, u1)
        # print(1/n, p_x)

        # envelope function -  f(x) =  1/n, c =  2
        while True:
            u1 = math.ceil(n * random.random()) #discrete uniform random variable from 0-n
            p_x = self.pdf(n, u1)
            # p_x = self.pdf2(n, u1)
            u2 = random.random()
            if u2 <= ((p_x*n)/8):
                return u1
                

    def distance_matrix(self):
        output= np.zeros((len(self.cities),len(self.cities)))
        num_rows, num_cols = output.shape
        for row in range(num_rows):
            for col in range(num_cols):
                if row==col:continue
                output[row][col]=self.length(self.cities[row], self.cities[col])
        return output

    def compute_tour_length(self):
        for arc in self.arcs:
            self.tour_length +=arc.length        
        return self.tour_length
            

    def remove_bad_arcs(self,worst=0):
        temp = sorted(self.arcs, key=lambda x: x.length, reverse=True)
        worst_arc = temp[worst]
        # print (worst_arc.start.city_name, worst_arc.end.city_name)
        temp.remove(worst_arc)
        for arc in temp:
            if worst_arc.start == arc.end or worst_arc.end == arc.start  or worst_arc.start == arc.start or worst_arc.end == arc.end:
                continue
            else:
                second_arc = arc
                break
        self.arcs.remove(worst_arc)
        self.arcs.remove(second_arc)
        return worst_arc.start, worst_arc.end, second_arc.start, second_arc.end


    # def route_fix(self):
    #     # Route_fix is fucked 

    #     # finding which node is affected
    #     # reach =0
    #     for i in range(len(self.arcs)):
    #         for j in range(len(self.arcs)):
    #             if i == j : continue
    #             # if self.arcs[i] in [arc1,arc2] or self.arcs[j] in [arc1,arc2] : continue
    #             if self.arcs[j].start == self.arcs[i].start or self.arcs[j].end == self.arcs[i].end:
    #                 # print("hi")
    #                 self.arcs[j].start, self.arcs[j].end = self.arcs[j].end, self.arcs[j].start
    #                 break

    def route_fix(self, new_arc):

        current = new_arc
        flag = True
        while flag == True:
            no_conflict = 1 #no conflicts with itself
            for arc in self.arcs:
                if current == arc: continue
                if arc.start == current.start or arc.end == current.end:
                    arc.start, arc.end = arc.end, arc.start
                    current = arc
                    break
                else:
                    no_conflict += 1
            if no_conflict == self.nodeCount:
                flag = False
                
                    
    def add_new_arcs(self, node1, node2, node3, node4):
        old = self.tour_length
        new = arc(node1,node3).length + arc(node2, node4).length
        for edge in self.arcs:
            new+=edge.length 
        # print ("Old - ", old)
        # print ("New - ", new)
        if new < old:
            self.arcs.append(arc(node1,node3))
            self.arcs.append(arc(node2, node4))
            # self.arcs.append(arc(node3,node1))
            # self.arcs.append(arc(node4, node2))
            self.route_fix(arc(node1,node3))
            self.tour_length = new
            return 1 #improvement
        else:
            self.arcs.append(arc(node1,node2))
            self.arcs.append(arc(node3, node4))
            return 0 #no improvement



    def visualise_solution(self):
        self.plot_cities()
        arrowprops = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='blue')
        
        for arc in self.arcs:
            plt.annotate('', xy=[arc.end.x, arc.end.y], xytext=[arc.start.x, arc.start.y], arrowprops=arrowprops)

        pass

    def plot_cities(self):
        plt.figure(figsize=(10,10))
        plt.title("City Coordinates")
        x,y=[],[]
        # y=[]
        for city in self.cities:
            x.append(city.x)
            y.append(city.y)
        plt.scatter(np.array(x),np.array(y),c='black')
        plt.scatter(np.array(x[0]),np.array(y[0]),c='red') # Depot
               
    
    def initial_solution_greedy(self):

        #greedy algorithm
        pending_visit= [i for i in range(1,self.nodeCount)]
        current_city = 0
        while len(pending_visit)>0:
            next_city=self.dist_matrix[current_city].tolist().index(sorted(self.dist_matrix[current_city])[1]) #minimum distance excluding distance to itself 
            self.dist_matrix[current_city][next_city]=100000
            # tour+=str(next_city)+" "
            self.route.append(next_city)
            pending_visit.remove(next_city)
        self.route.append(0) #returning to first city
        # self.arcs
        edges = list(zip(self.route[:-1:1], self.route[1::1]))
        for edge in edges:
            self.arcs.append(arc(self.cities[edge[0]],self.cities[edge[1]]))
        # print(self.arcs)
        obj=self.compute_tour_length()

        self.output = '%.2f' % obj + ' ' + str(0) + '\n'
        self.output += ' '.join(map(str, self.route[:-1]))
        # print(self.output)
        return self.arcs

    def two_opt(self):
        # print (self.arcs)
        # prev = self.tour_length
        worst = 0 #the arc to be removed first, 0 represents the first element of the sorted(descending) list of arc costs
        strategy =1
        for i in range(10000000):
            # self.route_fix()
            node1, node2, node3, node4 = self.remove_bad_arcs(worst)
            # self.visualise_solution()
            # print(node1.start.city_name, node2.city_name)
            improvement = self.add_new_arcs(node1, node2, node3, node4)
            # self.route_fix()
            # self.route_fix(node1)
##            if improvement == 0:
                # break
##                worst = self.random_arc()
                # worst =0
                # break
            if improvement == 0:
                strategy = 1
            if strategy == 1: # if we reach an iteration with no improvement then we pick arcs randomly only
                worst = self.random_arc()
        # self.route_fix()  
        # self.visualise_solution()
        self.route_find()

        self.output=''
        obj=self.tour_length
        self.output = '%.2f' % obj + ' ' + str(0) + '\n'
        self.output += ' '.join(map(str, self.route[:-1]))
##        return self.output
        

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def distance_matrix(coordinates):
    output= np.zeros((len(coordinates),len(coordinates)))
    num_rows, num_cols = output.shape
    for row in range(num_rows):
        for col in range(num_cols):
            if row==col:continue
            output[row][col]=length(coordinates[row], coordinates[col])
    return output

def route_finder(tour):
    route=""
    current=0
    route+=str(current)+" "
    for i in range(len(tour)):
        for path in tour:
            if path[0]==current:
                next=path[1]
                route+=str(next)+" "
                current=next
                break
    return route[:-3]

def greedy(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))
    dist_matrix=distance_matrix(points)
    # print(dist_matrix)

    #Greedy Solution
    pending_visit= [i for i in range(1,nodeCount)]
    tour ="0 " #starting from first city
    tour_list=[0]
    current_city = 0
    while len(pending_visit)>0:
        next_city=dist_matrix[current_city].tolist().index(sorted(dist_matrix[current_city])[1]) #minimum distance excluding distance to itself
        dist_matrix[current_city][next_city]=1000000000000000
        tour+=str(next_city)+" "
        tour_list.append(next_city)
        pending_visit.remove(next_city)
    tour=tour[:-1]

    solution = tour_list

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    # output_data += ' '.join(map(str, solution))
    output_data+=tour

    return output_data



def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    if nodeCount>2000:
        return greedy(input_data)
    elif nodeCount > 100:
        tsp = TSP(input_data)
        intial_solution = tsp.initial_solution_greedy()
        tsp.two_opt()
        return (tsp.output)
##        return greedy(input_data)
    else:

        points = []
        for i in range(1, nodeCount + 1):
            line = lines[i]
            parts = line.split()
            points.append(Point(float(parts[0]), float(parts[1])))
        # print (points)
        # plot_fig(points)

        start = time.time()

        solver = pywraplp.Solver.CreateSolver('SCIP')
        # solver = pywraplp.Solver('SolveIntegerProblem',
        #                        pywraplp.Solver.GUROBI_MIXED_INTEGER_PROGRAMMING);

        num_cities = len(points)

        # Decision Variables for route selection - 1 if route i to j is selected, 0 otherwise
        x = {}
        for i in range(num_cities):
            for j in range(num_cities):
                if i == j: continue
                x[i, j] = solver.IntVar(0, 1, f"x[{i},{j}]")

        # print(x)
        # Decision Variables for cities visited before that city
        u = {}
        infinity = solver.infinity()
        for i in range(1, num_cities):
            u[i] = solver.NumVar(0, infinity, 'u[%i]' % i)
        # print (u)

        # Constraints
        # 1. Each city should have only one departure
        for i in range(num_cities):
            expr = []
            for j in range(num_cities):
                if i == j: continue
                expr.append(x[i, j])
            # print(expr)
            solver.Add(sum(expr) <= 1)
            solver.Add(sum(expr) >= 1)

        # 2. Each city should have only one arrival
        for i in range(num_cities):
            expr = []
            for j in range(num_cities):
                if i == j: continue
                expr.append(x[j, i])
            # print(expr)
            solver.Add(sum(expr) <= 1)
            solver.Add(sum(expr) >= 1)

        # 3. Subtour Elimination constraint
        for i in range(1, num_cities):
            for j in range(1, num_cities):
                if i == j: continue
                solver.Add(u[i] + 1 <= u[j] + (num_cities * (1 - x[i, j])))

        # Objective
        objective_terms = []
        for i in range(num_cities):
            for j in range(num_cities):
                if i == j: continue
                objective_terms.append(x[i, j] * length(points[i], points[j]))
                # print(points[i],points[j],length(points[i],points[j]))
        # print(objective_terms)

        solver.Minimize(solver.Sum(objective_terms))
        status = solver.Solve()

        result = []
        for i in range(num_cities):
            for j in range(num_cities):
                if i == j: continue
                if x[i, j].solution_value() == 1:
                    # print (x[i,j].solution_value())
                    result.append((i, j))
        # print(result)

        # Visualising Output
        # arrowprops = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='blue')
        # for i, j in result:
        #     # plt.annotate('', xy=[float(city_data[int(j)][0]), float(city_data[int(j)][1])], xytext=[float(city_data[int(i)][0]), float(city_data[int(i)][1])]
        #     plt.annotate('', xy=[float(points[int(j)].x), float(points[int(j)].y)], xytext=[float(points[int(i)].x), float(points[int(i)].y)], arrowprops=arrowprops)

        # print ('Minimum distance = ', solver.Objective().Value())
        # print("MILP Solution found in: " + str(time.time() - start) + " seconds")
        # # prepare the solution in the specified output format
        obj = solver.Objective().Value()
        output_data = '%.2f' % obj + ' ' + str(1) + '\n'
        # output_data += ' '.join(map(str, solution))
        output_data += route_finder(result)

        return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

