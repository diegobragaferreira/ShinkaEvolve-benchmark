# EVOLVE-BLOCK-START
import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy.spatial import cKDTree
import time
from typing import Tuple, List

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Problem parameters
    n_circles = 32
    bounds = [(0, 1), (0, 1), (0, 0.5)]  # x, y, r bounds
    max_radius = 0.5
    
    # Initialize toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define individual and population creation
    def create_individual():
        individual = []
        for _ in range(n_circles):
            x = np.random.uniform(bounds[0][0], bounds[0][1])
            y = np.random.uniform(bounds[1][0], bounds[1][1])
            r = np.random.uniform(bounds[2][0], bounds[2][1])
            individual.extend([x, y, r])
        return creator.Individual(individual)
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        """Evaluate fitness of individual - maximize sum of radii with penalties for violations"""
        circles = np.array(individual).reshape(-1, 3)
        
        # Validate constraints
        total_radius = np.sum(circles[:, 2])
        
        # Penalty for out-of-bounds
        penalty_bounds = 0
        for i in range(n_circles):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty_bounds += 1000
        
        # Penalty for overlapping circles
        penalty_overlap = 0
        for i in range(n_circles):
            x1, y1, r1 = circles[i]
            for j in range(i+1, n_circles):
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < (r1 + r2):
                    # Calculate overlap penalty
                    overlap = (r1 + r2 - distance)
                    penalty_overlap += overlap * 1000
        
        # Combine penalties with objective
        fitness = total_radius - penalty_bounds - penalty_overlap
        
        return (fitness,)
    
    # Register genetic operators
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Set up parameters
    pop_size = 50
    n_generations = 100
    cx_prob = 0.8
    mut_prob = 0.2
    
    # Create initial population
    population = toolbox.population(n=pop_size)
    
    # Track best fitness
    best_fitness_history = []
    
    # Evolution loop
    for generation in range(n_generations):
        # Evaluate population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Statistics
        fits = [ind.fitness.values[0] for ind in population]
        best_fitness = max(fits)
        best_fitness_history.append(best_fitness)
        
        # Check for convergence
        if len(best_fitness_history) > 10:
            recent_avg = np.mean(best_fitness_history[-10:])
            prev_avg = np.mean(best_fitness_history[-20:-10])
            if abs(recent_avg - prev_avg) < 1e-4:
                # Convergence detected, reduce population size and increase mutation rate
                pop_size = max(20, int(pop_size * 0.8))
                mut_prob = min(0.5, mut_prob * 1.2)
                if generation > 30:
                    break
        
        # Select next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Replace population
        population[:] = offspring
    
    # Extract best individual
    best_ind = tools.selBest(population, 1)[0]
    circles = np.array(best_ind).reshape(-1, 3)
    
    # Local optimization with gradient descent
    circles = local_refinement(circles, 1000)
    
    return circles

def local_refinement(initial_circles: np.ndarray, max_iter: int) -> np.ndarray:
    """Apply local refinement to improve solution quality"""
    circles = initial_circles.copy()
    n_circles = circles.shape[0]
    
    # Construct KDTree for fast neighbor search
    tree = cKDTree(circles[:, :2]) 
    
    # Simple gradient ascent approach
    for _ in range(max_iter):
        # Compute gradients for each circle
        gradients = np.zeros_like(circles)
        
        for i in range(n_circles):
            x, y, r = circles[i]
            
            # Calculate influence from neighbors
            neighbors = tree.query_ball_point([x, y], 2*r)
            neighbors = [idx for idx in neighbors if idx != i]
            
            # Gradient due to collisions
            grad_x, grad_y, grad_r = 0.0, 0.0, 0.0
            
            for j in neighbors:
                x2, y2, r2 = circles[j]
                dx = x - x2
                dy = y - y2
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Only consider when circles are close enough to collide
                if dist < (r + r2):
                    # Compute repulsion force
                    force_magnitude = (r + r2 - dist) / (dist + 1e-8)
                    
                    # Gradients due to repulsion
                    grad_x += force_magnitude * dx / dist
                    grad_y += force_magnitude * dy / dist
                    grad_r -= force_magnitude  # decrease radius to avoid collision
            
            # Add gradient towards center to keep circles inside bounds
            bound_grad_x = 0.0
            bound_grad_y = 0.0
            if x < 0.05:
                bound_grad_x += (0.05 - x) * 100
            elif x > 0.95:
                bound_grad_x -= (x - 0.95) * 100
                
            if y < 0.05:
                bound_grad_y += (0.05 - y) * 100
            elif y > 0.95:
                bound_grad_y -= (y - 0.95) * 100
                
            gradients[i] = [bound_grad_x + grad_x, bound_grad_y + grad_y, grad_r]
        
        # Update positions
        step_size = 0.01
        for i in range(n_circles):
            # Update circle properties
            new_x = circles[i, 0] + step_size * gradients[i, 0]
            new_y = circles[i, 1] + step_size * gradients[i, 1]
            new_r = circles[i, 2] + step_size * gradients[i, 2]
            
            # Enforce bounds
            new_x = np.clip(new_x, 0 + new_r, 1 - new_r)
            new_y = np.clip(new_y, 0 + new_r, 1 - new_r)
            new_r = np.clip(new_r, 0, 0.5)
            
            circles[i] = [new_x, new_y, new_r]
            
            # Ensure circles don't go out of bounds
            if new_x - new_r < 0 or new_x + new_r > 1 or new_y - new_r < 0 or new_y + new_r > 1:
                # Revert to previous good state for now
                pass
    
    return circles

# EVOLVE-BLOCK-END
