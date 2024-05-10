#Authors
#Final Project 5: ENPM 661 Planning for Autonomous Robots
#1. FNU Koustubh  (dir-id: koustubh@umd.edu)
#2. Keyur Borad   (dir-id: kborad@umd.edu)
#3. Aryan Mishra  (dir-id: amishr17@umd.edu)




import matplotlib.pyplot as plt # Import the matplotlib library
import numpy as np # Import the numpy library
import random # Import the random library
import math # Import the math library
import cv2 # Import the OpenCV library
import copy # Import the copy library

size = (30, 50) # Size of the environment
obstacles = [(5, 5, 10, 15), (20, 5, 25, 15), (5, 30, 10, 40), (20, 30, 25, 40), (3, 20, 12, 25), (18, 20, 27, 25)] # Obstacles in the environment
start = (0, 0) # Start point
plot_num = 0 # Plot number
goal = (29, 35) # Goal point
INITIAL_POPULATION_SIZE = 20 # Initial population size
NUMBER_OF_GENERATIONS = 50 # Number of generations
STEP_SIZE = 3 # Step size for RRT
FPS = 2 # Frames per second
generation_best_fitness = [] # Best fitness of each generation

class Node: # Node class to store the x and y coordinates of the node
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.parent_node = None


def is_within_obstacles(point, obstacles):   
# Check if the point is within the obstacles
  for (x1, y1, x2, y2) in obstacles:
    if x1<=point[0]<=x2 and y1<=point[1]<=y2:
      return True
  return False


def distance(point1, point2):                
# Calculate the Euclidean distance between two points
  return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def nearest_node(nodes, random_point):
# Find the nearest node to the random point
  return min(nodes, key=lambda node: distance((node.x, node.y), random_point))


def steer(from_node, to_point, step_size=1):
# Move from the from_node towards the to_point by step_size
  if distance((from_node.x, from_node.y), to_point)<step_size:
    return Node(to_point[0], to_point[1])
  else:
    theta = np.arctan2(to_point[1]-from_node.y, to_point[0]-from_node.x)
    return Node(from_node.x + step_size*np.cos(theta), from_node.y + step_size*np.sin(theta))


""" Check if the path between node1 and node2 is valid by interpolating points along the way. """
def is_valid_path(node1, node2, obstacles): # Check if the path between node1 and node2 is valid by interpolating points along the way
  steps = int(distance((node1.x, node1.y), (node2.x, node2.y))/0.5)  # Smaller steps for more accuracy

  for i in range(1, steps + 1): # Check if the path between node1 and node2 is valid by interpolating points along the way
    inter_x = node1.x + i*(node2.x-node1.x)/steps # Interpolating x coordinate
    inter_y = node1.y + i*(node2.y-node1.y)/steps # Interpolating y coordinate

    if is_within_obstacles((inter_x, inter_y), obstacles): # Check if the interpolated point is within the obstacles
      return False
  return True


def plot(nodes=None, path=None): # Plot the nodes and path
    global plot_num # Use the global plot_num variable
    fig, ax = plt.subplots() # Create a figure and axis
    if nodes: # If nodes are provided
      for node in nodes: # For each node
          if node.parent_node: # If the node has a parent node
              plt.plot([node.x, node.parent_node.x], [node.y, node.parent_node.y], "g-", linewidth=0.5) # Plot a line between the node and its parent
    for (ox, oy, ex, ey) in obstacles:
        ax.add_patch(plt.Rectangle((ox, oy), ex-ox, ey-oy, color="red")) # Plot the obstacles in red
    if path:
        plt.plot([node.x for node in path], [node.y for node in path], "b-", linewidth=2)  # Highlight path in blue
    plt.plot(start[0], start[1], "bo")  # Start
    plt.plot(goal[0], goal[1], "ro")  # Goal
    plt.grid(True) # Display grid
    plt.savefig(f'plot_{plot_num + 1}.png')  # Save plot as PNG
    # plt.show()
    plot_num += 1 # Increment plot_num


def rrt(step_size=1, max_nodes=10000): # Rapidly-exploring Random Tree (RRT)
    nodes = [Node(start[0], start[1])] # Start with the start node
    while len(nodes) < max_nodes: # Continue until max_nodes is reached
        random_point = (random.randint(0, size[0] - 1), random.randint(0, size[1] - 1)) # Generate a random point
        if is_within_obstacles(random_point, obstacles): # Skip if the random point is within the obstacles
            continue # Skip
        nearest = nearest_node(nodes, random_point) # Find the nearest node to the random point
        new_node = steer(nearest, random_point, step_size) # Steer towards the random point
        if not is_within_obstacles((new_node.x, new_node.y), obstacles) and is_valid_path(nearest, new_node, obstacles): # If the path is valid
            new_node.parent_node = nearest # Set the parent node of the new node
            nodes.append(new_node) # Add the new node to the nodes
            if distance((new_node.x, new_node.y), goal) <= 2:#step_size:
                return nodes, new_node
    return nodes, None  # Return None if max_nodes reached without finding a path


def fitness_function(path):# Calculate the fitness of the path
    coordinates = [(node.x, node.y) for node in path] # Get the coordinates of the path
    w1 = 3
    w2 = 1
    w3 = 2
    euc_dist = 0
    angle_sum = 0
    interference=0
    safe_rad=8
    for i in range(len(coordinates)-1):     # Calculate the Euclidean distance and angle sum of the path
        x1, y1 = coordinates[i] # Getting the coordinates of the current node
        x2, y2 = coordinates[i+1] # Gettin the coordinates of the next node
        dist_ = math.sqrt((x2-x1)**2 +(y2-y1)**2) # Calculating the Euclidean distance between the current and next node
        euc_dist += dist_ # Adding the distance to the total distance
        if i != len(coordinates)-2: # If the current node is not the second last node
          x3, y3 = coordinates[i+2] # Getting the coordinates of the node after the next node
          heading1 = math.degrees(math.atan2((y2-y1),(x2-x1))) # Calculating the heading of the current node
          heading2 = math.degrees(math.atan2((y3-y2),(x3-x2))) # Calculating the heading of the next node
          if heading1 < 0: # If the heading is negative
            heading1 = 360 + heading1 # Add 360 to the heading
          if heading2 < 0: # If the heading is negative
            heading2 = 360 + heading2 # Add 360 to the heading
          angle = abs(heading2-heading1) # Calculating the angle between the current and next node
          angle_sum += angle # Add the angle to the total angle
          
    for center_x,center_y in coordinates: # Calculating the interference of the path
      for (ox , oy, ex, ey) in obstacles: # For each obstacle
        for i in range(ox,ex): # For each x coordinate
          for j in range(oy,ey): # For each y coordinate
            if ((i-center_x)**2+(j-center_y)**2)<safe_rad**2: # If the distance between the obstacle and the path is less than the safe radius
              interference+=1 # Increment the interference

    F = w1*(1/(euc_dist))+ w2*1/(angle_sum)+w3*(1/interference) # Calculate the fitness of the path

    F = w1*(1/(euc_dist))+ w2*1/(angle_sum) # Calculate the fitness of the path
    return euc_dist, F # Return the Euclidean distance and fitness of the path


def calculate_fitness_of_population(population): # Calculate the fitness of the population
  fitness = []
  for index, path in enumerate(population):
    euc_dist, F = fitness_function(path)
    fitness.append(F)
  return fitness


def selection(fitness, population): # Select the best two paths
  P1 = [] # Initialize the selected paths
  P2 = [] # Initialize the selected paths
  while(fitness): # Continue until all the fitness values are used
    max_fitness = max(fitness) # Get the maximum fitness value
    index = fitness.index(max_fitness) #  Get the index of the maximum fitness value
    fitness.pop(index) # Remove the fitness value
    P1.append(population.pop(index)) # Add the path to the selected paths

    l = len(fitness) # Get the length of the fitness values
    i = random.randint(0, l-1) # Generate a random index
    fitness.pop(i)# Remove the fitness value    
    P2.append(population.pop(i)) # Add the path to the selected paths
  return P1, P2# Return the selected paths


def best_selection(fitness, population): # Select the best two paths
  P1 = []# Initialize the selected paths
  P2 = [] # Initialize the selected paths
  while(fitness): # Continue until all the fitness values are used
    max_fitness = max(fitness)  # Get the maximum fitness value
    index = fitness.index(max_fitness)  # Get the index of the maximum fitness value
    fitness.pop(index)  # Remove the fitness value
    P1.append(population.pop(index))    # Add the path to the selected paths

    max_fitness = max(fitness)  # Get the maximum fitness value
    index = fitness.index(max_fitness)      # Get the index of the maximum fitness value
    fitness.pop(index)  # Remove the fitness value
    P2.append(population.pop(index)) # Add the path to the selected paths
  return P1, P2


def plot_best_solution(fitness, population, plot_graph=False):
  # plot the best solution
  max_fitness = max(fitness) # Get the maximum fitness value
  index = fitness.index(max_fitness) # Get the index of the maximum fitness value
  best_path = population[index] # Get the best path
  print(f"Fitness: {max_fitness}") # Print the fitness value
  if plot_graph: # If plot_graph is True
    plot(path=best_path) # Plot the best path
  return max_fitness # Return the fitness value


#CrossOver points
def crossoverpt(parent1,parent2): #CrossOver points
    minval=2 #Minimum value
    offspring1=[] #Offspring 1
    offspring2=[] #Offspring 2
    for pt1 in range(3, len(parent1)-3): #For each point in the parent 1
        for pt2 in range(3, len(parent2)-3): #For each point in the parent 2
            if (math.sqrt((parent1[pt1].x-parent2[pt2].x)**2+(parent1[pt1].y-parent2[pt2].y)**2))<minval: #If the distance between the points is less than the minimum value
                minval=math.sqrt((parent1[pt1].x-parent2[pt2].x)**2+(parent1[pt1].y-parent2[pt2].y)**2) #Set the minimum value to the distance
                parent1_point_idx=pt1 #Set the parent 1 point index
                parent2_point_idx=pt2 #Set the parent 2 point index
    for i in range(parent1_point_idx+1): #For each point in the parent 1
        offspring1.append(Node(parent1[i].x,parent1[i].y)) #Add the point to the offspring 1
    for i in range(parent2_point_idx,len(parent2)):
        offspring1.append(Node(parent2[i].x,parent2[i].y)) #Add the point to the offspring 1

    for i in range(parent2_point_idx+1):
        offspring2.append(Node(parent2[i].x,parent2[i].y)) #Add the point to the offspring 2
    for i in range(parent1_point_idx,len(parent1)):
        offspring2.append(Node(parent1[i].x,parent1[i].y)) #Add the point to the offspring 2


    #Returning offsprings
    return offspring1,offspring2
  
def mutate_path(path): #Mutate the path
  num_nodes = len(path) #Get the number of nodes in the path
  random_node = random.randint(0,num_nodes-1) #Generate a random node
  del_x = random.random()*2 - 1 # Generate a random x coordinate
  del_y = random.random()*2 - 1 # Generate a random y coordinate
  new_path = copy.deepcopy(path) # Copy the path
  new_path[random_node].x += del_x # Add the random x coordinate to the node
  new_path[random_node].y += del_y # Add the random y coordinate to the node
  return new_path


def elimination(fitness, population): # Eliminate the worst two paths
  num_eliminations = 2 # Number of paths to eliminate
  for i in range(0, num_eliminations): # For each path to eliminate
    min_fitness = min(fitness) # Get the minimum fitness value
    index = fitness.index(min_fitness) # Get the index of the minimum fitness value
    fitness.pop(index) # Remove the fitness value
    population.pop(index) # Remove the path
  return fitness, population # Return the fitness values and paths


def create_initial_population(population_size=20, step_size=1, plot_paths = True): # Create the initial population
  population = [] # Initialize the population
  for i in range(population_size): # For each path in the population
    nodes, final_node = rrt(step_size=step_size) # Generate a path using RRT
    path = [] # Initialize the path
    if final_node: # If a path is found
        while final_node.parent_node: # While the node has a parent node
            path.append(final_node) # Add the node to the path
            final_node = final_node.parent_node # Move to the parent node
        path.append(final_node) # Add the start node to the path
        path.reverse() # Reverse the path to get the correct order
    if plot_paths:# If plot_paths is True
      plot(nodes, path) # Plot the path
    population.append(path)# Add the path to the population
  return population # Return the population


def create_opencv_visualisation(parent_gen_size=INITIAL_POPULATION_SIZE, generation_gen_size=100): # Create an OpenCV visualisation
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Define the codec
  video = cv2.VideoWriter("genetic.mp4", fourcc, FPS, (640, 480)) # Create a video writer
  for i in range(0, parent_gen_size+generation_gen_size): # For each generation
    image = cv2.imread(f'plot_{i+1}.png') # Read the image
    if i < parent_gen_size:# If the generation is the parent generation
      fitness = parent_gen_fitness[i] # Get the fitness value
      cv2.putText(image, f"Parent: {i+1} | Fitness: {fitness}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,0,0), 1, cv2.LINE_AA) # Add text to the image
    else:
      fitness = generation_best_fitness[i-parent_gen_size] # Get the fitness value
      cv2.putText(image, f"Generation: {i-parent_gen_size+1} | Fitness: {fitness}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,0,0), 1, cv2.LINE_AA)
    video.write(image)# Writing the image to the video
  for i in range(20):
    video.write(image) # Writing the last image multiple times to increase the video duration

population = create_initial_population(population_size=INITIAL_POPULATION_SIZE, step_size=STEP_SIZE, plot_paths=True) # Creating the initial population
parent_gen_fitness = calculate_fitness_of_population(population) # Calculating the fitness of the parent generation

for gen in range (0,NUMBER_OF_GENERATIONS): # For each generation
  print(f"Generation: {gen+1}")# Print the generation number
  if population:# If the population is not empty
    fitness = calculate_fitness_of_population(population) # Calculating the fitness of the population
    best_fitness = plot_best_solution(fitness=fitness, population=population, plot_graph=True) # Plot the best solution
    generation_best_fitness.append(best_fitness) # Append the best fitness value to the generation best fitness
    P1, P2 = best_selection(fitness, population) # Select the best two paths
    mutation_1 = mutate_path(P1[0]) # Mutate the first path
    mutation_2 = mutate_path(P2[0]) # Mutate the second path
    population.append(mutation_1) # Add the mutated path to the population
    population.append(mutation_2) # Add the mutated path to the population
    for i in range(0,len(P1)): # For each path
      try:
        offspring1 , offspring2 = crossoverpt(P1[i], P2[i]) # Perform crossover
        population.append(offspring1) # Add the offspring to the population
        population.append(offspring2) # Add the offspring to the population
      except:
        pass
  else:
    NUMBER_OF_GENERATIONS = gen # Set the number of generations to the current generation
    print(f"Termination at generation: {gen+1}") # Print the termination message

create_opencv_visualisation(parent_gen_size=INITIAL_POPULATION_SIZE, generation_gen_size=NUMBER_OF_GENERATIONS)# Create the OpenCV visualisation
