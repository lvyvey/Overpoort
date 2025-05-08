import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap, BoundaryNorm
import time


class PoliceOfficer:
    def __init__(self, unique_id):
        self.unique_id = unique_id
        self.position = None  # Position will be assigned later
        self.response_time = random.uniform(300, 600)  # Random response time (300 to 600 seconds ~ 5 to 10 minutes)
        self.steps_taken = 0  # Track the number of steps taken
    
    def get_neighbors(self, grid_height, grid_width):
        x, y = self.position
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_height and 0 <= ny < grid_width:
                    neighbors.append((nx, ny))
        return neighbors
    
    def move(self, grid, mode, fight_spots_grid, target_position=None):
        old_position = self.position  # Store the old position before moving
        
        # Track movement based on the mode
        if mode == "random":
            """ Move the police officer to a random neighboring empty cell. """
            grid_height, grid_width = len(grid), len(grid[0])

            neighbors = self.get_neighbors(grid_height, grid_width)  # Get adjacent positions
            available_positions = [pos for pos in neighbors if grid[pos[0]][pos[1]] is None]

            if available_positions:
                new_pos = random.choice(available_positions)  # Pick a random free neighbor
            else:
                new_pos = self.position  # Stay in place if no available move

        elif mode == "strategic":
            # Find the position of the fight hotspot with the highest value
            max_fight_spot = np.unravel_index(np.argmax(fight_spots_grid), fight_spots_grid.shape)
            max_x, max_y = max_fight_spot
            x, y = self.position
            # Move towards the hotspot
            if max_x > x:
                new_x = x + 1
            elif max_x < x:
                new_x = x - 1
            else:
                new_x = x
            if max_y > y:
                new_y = y + 1
            elif max_y < y:
                new_y = y - 1
            else:
                new_y = y
            # Check if the new position is within bounds and empty
            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid) and grid[new_x][new_y] is None:
                new_pos = (new_x, new_y)
            else:
                new_pos = self.position  # Stay in place if no available move
        
        elif mode == "distributed-strategic" and target_position is not None:
            # Move toward assigned target
            target_x, target_y = target_position
            x, y = self.position
            
            new_x = x + np.sign(target_x - x)
            new_y = y + np.sign(target_y - y)
            
            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid) and grid[new_x][new_y] is None:
                new_pos = (new_x, new_y)
            else:
                new_pos = self.position  # Stay in place if no available move

        else:
            new_pos = self.position  # If no valid mode, stay in place

        # If the officer actually moved to a new position, increment steps_taken
        if new_pos != old_position:
            self.steps_taken += 1

        return new_pos