import streamlit as st
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap, BoundaryNorm
import time

from Classes.Student import Student
from Classes.PoliceOfficer import PoliceOfficer


def create_layout_grid(height, width):
    layout_grid = np.full((height, width), None)

    street_start = 1
    street_end = width - 2

    # Left bar column
    layout_grid[:, 0] = "X"

    # Right bar column
    layout_grid[:, width - 1] = "X"

    return layout_grid

# Initialize model
def initialize_agents(grid_height, grid_width, num_students = 0, num_police = 0, graph_type="barabasi"):
    
    # Initialize agents and place them on the grid.
    students = []
    police_officers = []
    
    layout_grid = create_layout_grid(grid_height, grid_width)
    grid = np.full((grid_height, grid_width), None) 
    
    # Initialize police officers if any
    for i in range(num_police):
        officer = PoliceOfficer(i)
        placed = False

        # Place the officer in an empty spot on the grid
        while not placed:
            x = random.randint(0, grid_height - 1)
            y = random.randint(0, grid_width - 1)

            if grid[x][y] is None and layout_grid[x][y] == None:
                grid[x][y] = officer
                officer.position = (x, y)
                police_officers.append(officer)
                placed = True
                  
    for i in range(num_students):
        student = Student(i + num_police) 
        placed = False

        while not placed:
            x = random.randint(0, grid_height - 1)
            y = random.randint(0, grid_width - 1)

            if grid[x][y] is None:
                grid[x][y] = student
                student.position = (x, y)
                students.append(student)
                placed = True
                
    # Create friendship graph
    if graph_type == "barabasi":
        G = nx.barabasi_albert_graph(num_students, m=2)
    elif graph_type == "watts":
        G = nx.watts_strogatz_graph(num_students, k=4, p=0.3)
    elif graph_type == "erdos":
        G = nx.erdos_renyi_graph(num_students, p=0.1)
    else:
        G = nx.empty_graph(num_students)  # fallback: no friends

    # Assign friends to each student
    for i, student in enumerate(students):
        friend_ids = list(G.neighbors(i))
        student.friends = [students[j] for j in friend_ids]

    return students, police_officers, grid, layout_grid, G

# Simulate one step of the model
def step(students, police_officers, mode, grid, layout_grid, threshold, fight_spots_grid, bar_discount):
    
    fight_counter = 0  # Initialize fight counter
    
    grid_height, grid_width = len(grid), len(grid[0])
    
    for student in students:
        student.fought_this_step = False  # 🔹 Reset at start of step
    
    # Run one step of the simulation.
    for student in students:
        
        # Move agent to a new position
        new_pos = student.move(grid)
        old_x, old_y = student.position
        new_x, new_y = new_pos

        # Update grid to reflect new positions
        grid[old_x][old_y] = None
        grid[new_x][new_y] = student
        student.position = new_pos  # Update the agent's position

        # Get the neighbors of the agent
        neighbors = student.get_neighbors(grid_height, grid_width)
        
        # If student is in a bar, they buy a drink and no fights occur
        if layout_grid[new_x][new_y] == "X":
            student.buy_drink(bar_discount)
            continue
        
        # If one of the neighbors is a police officer, skip fight check
        if any(isinstance(grid[nx][ny], PoliceOfficer) for nx, ny in neighbors):
            continue
        
        # Check for fights with other students
        for nx, ny in neighbors:
            neighbor_agent = grid[nx][ny]
            if neighbor_agent is not None and student != neighbor_agent and isinstance(neighbor_agent, Student):
                outcome = student.check_for_fight(neighbor_agent, threshold)
                if outcome:
                    fight_counter += 1
                    fight_spots_grid[new_x][new_y] += 1  # Add fight to the grid
                
    # Nonlinear cooling: higher aggressiveness decreases more slowly
    cooling_rate = 0.1  # You can tweak this
    for student in students:
        if not student.fought_this_step:
            reduction = cooling_rate * (1 - student.aggressiveness)
            student.aggressiveness = max(student.aggressiveness - reduction, student.initial_aggressiveness) # student can never go below their initial aggressiveness
        student.fought_this_step = False
        
        
    if mode == "distributed-strategic":
        # Assign officers to nearest hotspots considering fight intensity
        officer_targets = assign_police_to_hotspots(police_officers, fight_spots_grid)
        
        # for officer, target in officer_targets.items():
        #     print(f"Officer {officer.unique_id} assigned to target {target}")

        # Move the officers to their assigned hotspots
        for officer, target in officer_targets.items():
            new_pos = officer.move(grid, mode, fight_spots_grid, target)
            old_x, old_y = officer.position
            new_x, new_y = new_pos

            # Update grid positions
            grid[old_x][old_y] = None
            grid[new_x][new_y] = officer
            officer.position = new_pos
    else:
        for officer in police_officers:
            new_pos = officer.move(grid, mode, fight_spots_grid)
            old_x, old_y = officer.position
            new_x, new_y = new_pos

            grid[old_x][old_y] = None
            grid[new_x][new_y] = officer
            officer.position = new_pos

        
    return fight_counter  # Return the updated fight counter

def calculate_distance_to_bars(fight_spots_grid, grid_width):
    distances = []  # List to store distances of each fight to the nearest bar

    # Iterate over the grid to find the fights
    for x in range(fight_spots_grid.shape[0]):
        for y in range(fight_spots_grid.shape[1]):
            num_fights = fight_spots_grid[x, y]
            
            if num_fights > 0:  # A fight (or multiple fights) occurred here
                # Calculate distance to left bar (y=0)
                distance_to_left_bar = y
                # Calculate distance to right bar (y=grid_width - 1)
                distance_to_right_bar = grid_width - 1 - y
                
                # Take the minimum distance to either bar
                min_distance = min(distance_to_left_bar, distance_to_right_bar)
                
                # Add the distance for each fight at this location
                distances.extend([min_distance] * int(num_fights))  # Repeat the distance for each fight
                
    return distances

def assign_police_to_hotspots(police_officers, fight_spots_grid):
    # Step 1: Get all hotspots with fights (positions with fights > 0)
    hotspots = []
    for i in range(fight_spots_grid.shape[0]):
        for j in range(fight_spots_grid.shape[1]):
            if fight_spots_grid[i][j] > 0:
                hotspots.append((i, j, fight_spots_grid[i][j]))  # (row, col, fight_intensity)
                
    # Sort hotspots by fight intensity (most fights first)
    hotspots.sort(key=lambda x: x[2], reverse=True)  # Sort by intensity in descending order
    
    # Only keep the top n hotspots, where n is the number of police officers
    num_hotspots = min(len(hotspots), len(police_officers))
    hotspots = hotspots[:num_hotspots]
    

    # Step 2: Calculate distance to each hotspot for every officer
    officer_hotspot_assignments = []
    for officer in police_officers:
        officer_pos = officer.position
        distances = []
        for hotspot in hotspots:
            hotspot_pos = hotspot[:2]
            distance = np.abs(officer_pos[0] - hotspot_pos[0]) + np.abs(officer_pos[1] - hotspot_pos[1])  # Manhattan distance
            distances.append((distance, hotspot_pos))
        
        # Sort hotspots by distance (nearest first)
        distances.sort(key=lambda x: x[0])
        officer_hotspot_assignments.append((officer, distances))

    # Step 3: Assign each officer to the nearest hotspot
    assigned_hotspots = set()
    officer_targets = {}
    for officer, distances in officer_hotspot_assignments:
        for distance, hotspot_pos in distances:
            if hotspot_pos not in assigned_hotspots:
                # Assign this officer to the nearest unassigned hotspot
                officer_targets[officer] = hotspot_pos
                assigned_hotspots.add(hotspot_pos)
                break
        else:
            # If no hotspot is available, officer stays in current position
            officer_targets[officer] = officer.position
        

    return officer_targets