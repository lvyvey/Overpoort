import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap, BoundaryNorm
import time
import math


class Student:
    def __init__(self, unique_id):
        self.unique_id = unique_id
            
        self.aggressiveness = np.random.beta(a=2, b=5) # Distribution of aggressiveness follows a beta distribution with a=2, b=5
        self.initial_aggressiveness = self.aggressiveness  # Store initial aggressiveness for later use
        self.budget = np.random.lognormal(3.5, 0.5) # Log normal distribution for budget
        self.position = None  # Position will be assigned later
        
        self.fought_this_step = False # Track if the student fought this step
        
        self.friends = []  # List of Student objects

    def check_for_fight(self, other_agent, threshold):
        """Check if two agents are adjacent and fight if they meet aggressiveness condition."""
        if self.unique_id < other_agent.unique_id:  # 🔹 Only one direction allowed
            if other_agent in self.friends:
                return False # No fight if they are friends
            if self.aggressiveness > threshold and other_agent.aggressiveness > threshold:
                self.aggressiveness = min(self.aggressiveness + 0.1, 1.0)
                other_agent.aggressiveness = min(other_agent.aggressiveness + 0.1, 1.0)
                self.fought_this_step = True
                other_agent.fought_this_step = True
                return True  # Fight occurred
        return False
            
    def get_neighbors(self, grid_height, grid_width):
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
                if 0 <= nx < grid_height and 0 <= ny < grid_width:
                    neighbors.append((nx, ny))
                
        return neighbors
    
    def move(self, grid, move_in_group=False):
        grid_height, grid_width = grid.shape

        # Fallback movement
        neighbors = self.get_neighbors(grid_height, grid_width)
        available_positions = [pos for pos in neighbors if grid[pos[0]][pos[1]] is None]
        if available_positions:
            return random.choice(available_positions)

        return self.position # Stay in place if no available move
    
    def buy_drink(self, bar_discount=False):
        
        if bar_discount:
            cost = 0.5
        else:
            cost = 1.0
        
        if self.budget >= cost:
            self.budget -= cost
            
            # Nonlinear increase in aggressiveness based on the bell curve
            mu = 0.5      # peak aggression level for max increase
            sigma = 0.2   # how wide the bell curve is
            A = 0.15      # maximum increment possible per drink

            x = self.aggressiveness
            increment = A * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
            self.aggressiveness = min(1.0, self.aggressiveness + increment)

            