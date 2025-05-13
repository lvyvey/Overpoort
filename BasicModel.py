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

from Functions import initialize_agents, step, calculate_distance_to_bars


# Streamlit app layout
st.title('Nightlife Simulation')
st.sidebar.header("Simulation Parameters")

# User Inputs (using Streamlit widgets)
num_agents = st.sidebar.slider("Number of Students", 1, 250, 250)
num_police = st.sidebar.slider("Number of Police Officers", 0, 30, 20)
steps = st.sidebar.slider("Simulation Steps", 1, 360, 360)
aggress_threshold = st.sidebar.slider("Aggressiveness Threshold", 0.0, 1.0, 0.9)
mode = st.sidebar.selectbox("Police Movement Mode", ["random", "strategic","distributed-strategic"])
discount = st.sidebar.checkbox("Discount Bar", value=False)
graph_type = st.sidebar.selectbox("Friendship Network Type", ["barabasi", "watts", "erdos"])

grid_height = 30  # vertical length
grid_width = 18   # 14 walkable + 2 bar columns

# Initialize agents and run the simulation
students, police_officers, grid, layout_grid, friend_network = initialize_agents(grid_height, grid_width, num_agents,num_police, graph_type)


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
            st.write("### Steps between Fights")
            fight_counter_placeholder = st.empty()
        with col2:
            st.write("### Mean Aggr. Over Time")
            average_aggressiveness_placeholder = st.empty()
            
    
    # Track data
    fight_history = []
    avg_aggressiveness_history = []
    fight_spots_grid = np.zeros((grid_height, grid_width))  # Initialize fight spots grid
    steps_between_fights_history = []
    steps_between_fights = 0
    
    

    for step_num in range(steps):
        
        bar_discount = False
        
        if step_num % 30 == 0 and discount:
            bar_discount = True
        
        # Run one step of the simulation
        fight_counter = step(students, police_officers, mode, grid, layout_grid, aggress_threshold, fight_spots_grid, bar_discount)
        
        if fight_counter > 0:
            steps_between_fights = 0
        else:
            steps_between_fights += 1

        # Create a masked array: True where bars are
        mask = np.array([[layout_grid[x][y] == "X" for y in range(grid_width)] for x in range(grid_height)])

        # Display grid
        grid_display = np.full((grid_height, grid_width), np.nan)
        
        for student in students:
            x, y = student.position
            grid_display[x, y] = student.aggressiveness
            

        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = colors.LinearSegmentedColormap.from_list("green_red", ["green", "yellow", "red"])
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
        steps_between_fights_history.append(steps_between_fights)

        # Update the line charts
        fight_counter_placeholder.line_chart(steps_between_fights_history)
        average_aggressiveness_placeholder.line_chart(avg_aggressiveness_history)

        time.sleep(0.2)
        
    for officer in police_officers:
        st.write(f"Officer {officer.unique_id} took {officer.steps_taken} steps.")
        
    distances = calculate_distance_to_bars(fight_spots_grid, grid_width)
    
    # Display the distances
    st.write("Distance to Bars for each fight spot:")
    for i, distance in enumerate(distances):
        st.write(f"Fight Spot {i}: {distance}")
    

