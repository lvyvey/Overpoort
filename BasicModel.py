
import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap, BoundaryNorm
import time

# Define the Student Agent
class Student:
    def __init__(self, unique_id):
        self.unique_id = unique_id
        
        self.aggressiveness = np.random.beta(a=2, b=5) # Distribution of aggressiveness follows a beta distribution with a=2, b=5
        self.initial_aggressiveness = self.aggressiveness  # Store initial aggressiveness for later use
        self.budget = np.random.uniform(10, 100)  # Random budget between 10 and 100
        
        self.position = None  # Position will be assigned later
        
        self.fought_this_step = False # Track if the student fought this step

    def check_for_fight(self, other_agent, threshold):
        """Check if two agents are adjacent and fight if they meet aggressiveness condition."""
        if self.unique_id < other_agent.unique_id:  # 🔹 Only one direction allowed
            if self.aggressiveness > threshold and other_agent.aggressiveness > threshold:
                self.aggressiveness = min(self.aggressiveness + 0.2, 1.0)
                other_agent.aggressiveness = min(other_agent.aggressiveness + 0.2, 1.0)
                self.fought_this_step = True
                other_agent.fought_this_step = True
                return True  # Fight occurred
            else:
                return False  # No fight occurred
        else:
            return False
            
    def get_neighbors(self, grid_size):
        """ Get the list of adjacent positions (neighbors) around the agent's position. """
        x, y = self.position
        neighbors = []
        
        #Check all adjacent cells (8 neighbors in total)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # Avoid checking the agent's own position (x, y)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    neighbors.append((nx, ny))
                
        return neighbors
    
    def move(self, grid):
        """ Move the student to a random neighboring empty cell. """
        neighbors = self.get_neighbors(len(grid))  # Get adjacent positions
        available_positions = [pos for pos in neighbors if grid[pos[0]][pos[1]] is None]

        if available_positions:
            new_pos = random.choice(available_positions)  # Pick a random free neighbor
            return new_pos
        
        return self.position  # Stay in place if no available move
    
    def buy_drink(self):
        """ Student buys a drink, if they have enough money, and becomes more aggressive. """
        if self.budget > 0:
            self.budget -= 1  # Drink costs 1 unit of money
            self.aggressiveness = min(self.aggressiveness + (0.1*self.aggressiveness), 1) 
    
    
class PoliceOfficer:
    def __init__(self, unique_id):
        self.unique_id = unique_id
        self.position = None  # Position will be assigned later
        self.response_time = random.uniform(300, 600)  # Random response time (300 to 600 seconds ~ 5 to 10 minutes)
    
    def get_neighbors(self, grid_size):
        x, y = self.position
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    neighbors.append((nx, ny))
        return neighbors
    
    def move(self, grid, mode):
        
        if mode == "random":
            """ Move the police officer to a random neighboring empty cell. """
            neighbors = self.get_neighbors(len(grid))  # Get adjacent positions
            available_positions = [pos for pos in neighbors if grid[pos[0]][pos[1]] is None]

            if available_positions:
                new_pos = random.choice(available_positions)  # Pick a random free neighbor
                return new_pos
            
            return self.position  # Stay in place if no available move

        elif mode == "strategic":
            """ Move police in direction of most aggressive students. Look in entire grid. """
            
        


def create_layout_grid(grid_size):
    layout_grid = np.full((grid_size, grid_size), None)  # Start with walkable grid

    street_width = grid_size-2

    # Add one-column bars on each side of the street
    start_col = grid_size // 2 - street_width // 2
    end_col = start_col + street_width

    # Mark the bars to the left of the street
    if start_col - 1 >= 0:
        layout_grid[:, start_col - 1] = "X"

    # Mark the bars to the right of the street
    if end_col < grid_size:
        layout_grid[:, end_col] = "X"

    return layout_grid

# Initialize model
def initialize_agents(grid_size, num_students = 0, num_police = 0):
    
    # Initialize agents and place them on the grid.
    students = []
    police_officers = []
    
    layout_grid = create_layout_grid(grid_size)
    grid = np.full((grid_size, grid_size), None) 
    
    # Initialize police officers if any
    for i in range(num_police):
        officer = PoliceOfficer(i)
        placed = False

        # Place the officer in an empty spot on the grid
        while not placed:
            x = random.randint(0, grid_size - 1)
            y = random.randint(0, grid_size - 1)

            if grid[x][y] is None and layout_grid[x][y] == None:
                grid[x][y] = officer
                officer.position = (x, y)
                police_officers.append(officer)
                placed = True
                                
    for i in range(num_students):
        student = Student(i + num_police) 
        placed = False

        # Place the student in an empty spot on the grid
        while not placed:
            x = random.randint(0, grid_size - 1)
            y = random.randint(0, grid_size - 1)

            if grid[x][y] is None:
                grid[x][y] = student
                student.position = (x, y)
                students.append(student)
                placed = True

    return students, police_officers, grid, layout_grid

# Simulate one step of the model
def step(students, grid, layout_grid, threshold):
    
    fight_counter = 0  # Initialize fight counter
    
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
        neighbors = student.get_neighbors(len(grid))
        
        # If student is in a bar, they buy a drink and no fights occur
        if layout_grid[new_x][new_y] == "X":
            student.buy_drink()
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
        
    return fight_counter  # Return the updated fight counter

# Streamlit app layout
st.title('Nightlife Simulation')
st.sidebar.header("Simulation Parameters")

# User Inputs (using Streamlit widgets)
num_agents = st.sidebar.slider("Number of Students", 1, 100, 10)
num_police = st.sidebar.slider("Number of Police Officers", 0, 20, 2)
steps = st.sidebar.slider("Simulation Steps", 1, 100, 10)
aggress_threshold = st.sidebar.slider("Aggressiveness Threshold", 0.0, 1.0, 0.5)
grid_size = 10  # Fixed grid size for simplicity

# Initialize agents and run the simulation
students, police_officers, grid, layout_grid = initialize_agents(grid_size, num_agents,num_police)

# Button to start the simulation
run_button = st.button("Run Simulation")

if run_button:
    # Layout
    grid_container = st.container()
    chart_container = st.container()
    
    with grid_container:
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Student Movement")
            movement_grid = st.empty()
        with col2:
            st.write("### Fight Heatmap")
            fight_heatmap = st.empty()
    
    # Create placeholders for the charts once before the loop
    with chart_container:
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Fights per Step")
            fight_counter_placeholder = st.empty()
        with col2:
            st.write("### Mean Aggr. Over Time")
            average_aggressiveness_placeholder = st.empty()
            
    
    # Track data
    fight_history = []
    avg_aggressiveness_history = []
    fight_spots_grid = np.zeros((grid_size, grid_size))  # Initialize fight spots grid
    

    for step_num in range(steps):
        
        # Run one step of the simulation
        fight_counter = step(students, grid, layout_grid, aggress_threshold)

        # Create a masked array: True where bars are
        mask = np.array([[layout_grid[x][y] == "X" for y in range(grid_size)] for x in range(grid_size)])

        # Display grid
        grid_display = np.full((grid_size, grid_size), np.nan)
        
        for student in students:
            x, y = student.position
            grid_display[x, y] = student.aggressiveness
            

        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = colors.LinearSegmentedColormap.from_list("yellow_red", ["yellow", "orange", "red"])
        norm = colors.Normalize(vmin=0, vmax=1)
        ax.imshow(grid_display, cmap=cmap, norm=norm)
        
        # Overlay bars as gray cells
        ax.imshow(mask, cmap=colors.ListedColormap(["none", "gray"]), alpha=0.6)

        # Draw blue police officers
        for officer in police_officers:
            x, y = officer.position
            ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color='blue'))
            
        # Hide the axes    
        ax.set_xticks([])
        ax.set_yticks([])

        # Update the grid visualization placeholder
        movement_grid.pyplot(fig)
        
        # Define custom white-to-red colormap
        white_red_cmap = colors.LinearSegmentedColormap.from_list('white_red', ['white', 'red'])

        # Create the heatmap for fight spots
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        heatmap = ax2.imshow(fight_spots_grid, cmap=white_red_cmap, interpolation='nearest', vmin=0)
        ax2.set_xticks([])
        ax2.set_yticks([])
        fight_heatmap.pyplot(fig2)
        
        # Update metrics
        fight_history.append(fight_counter)
        avg_aggressiveness = np.mean([s.aggressiveness for s in students])
        avg_aggressiveness_history.append(avg_aggressiveness)

        # Update the line charts
        fight_counter_placeholder.line_chart(fight_history)
        average_aggressiveness_placeholder.line_chart(avg_aggressiveness_history)

        time.sleep(0.3)

