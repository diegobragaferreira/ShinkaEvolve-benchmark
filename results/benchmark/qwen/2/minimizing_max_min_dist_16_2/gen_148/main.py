# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time
from typing import Tuple, List
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0

        # Calculate pairwise distances
        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)

        if dmax == 0:
            return 0

        return dmin / dmax
    
    def calculate_sphere_packing_fitness(points):
        """Calculate a fitness metric based on sphere packing density."""
        if len(points) < 2:
            return 0
        
        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        if dmax == 0:
            return 0
            
        # Fitness based on minimum distance relative to maximum distance
        # Higher values indicate better packing (larger minimum distances)
        return dmin / dmax
    
    def create_better_hexagonal_initialization():
        """Create a more sophisticated hexagonal-like arrangement of points."""
        points = np.zeros((16, 2))

        # Create a more regular hexagonal arrangement with better spacing
        row_positions = [0, 1, 2, 3]
        col_positions = [0, 1, 2, 3]
        spacing_x = 1.0 / 4.0
        spacing_y = spacing_x * np.sqrt(3) / 2.0

        idx = 0
        for i, row in enumerate(row_positions):
            for j, col in enumerate(col_positions):
                if idx < 16:
                    # Offset every other row for proper hexagonal packing
                    x = (col + 0.5 * (row % 2)) * spacing_x
                    y = row * spacing_y
                    points[idx] = [x, y]
                    idx += 1

        return points

    def create_concentric_ring_initialization():
        """Create a concentric ring-like arrangement."""
        points = np.zeros((16, 2))

        # Place points in concentric rings
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.8, 4)  # Four layers
        layer_points = [4, 4, 4, 4]  # 4 points per layer

        idx = 0
        for i, radius in enumerate(radii):
            num_points_in_layer = layer_points[i]
            layer_angles = np.linspace(0, 2*np.pi, num_points_in_layer, endpoint=False)
            for angle in layer_angles:
                if idx < 16:
                    x = 0.5 + radius * np.cos(angle)
                    y = 0.5 + radius * np.sin(angle)
                    points[idx] = [x, y]
                    idx += 1

        return points

    def create_fibonacci_sphere_like_initialization():
        """Create a Fibonacci-like arrangement for better point distribution."""
        points = np.zeros((16, 2))

        # Use Fibonacci-inspired pattern in 2D
        golden_ratio = (1 + np.sqrt(5)) / 2.0
        for i in range(16):
            theta = 2 * np.pi * i / golden_ratio
            r = np.sqrt(i / 15.0)  # Normalize to [0,1]
            x = 0.5 + r * np.cos(theta) * 0.8
            y = 0.5 + r * np.sin(theta) * 0.8
            points[i] = [x, y]

        return points

    def create_grid_initialization():
        """Create a regular grid initialization."""
        points = np.zeros((16, 2))
        idx = 0
        
        # Create 4x4 grid
        for i in range(4):
            for j in range(4):
                if idx < 16:
                    x = j / 3.0 if j > 0 else 0.0
                    y = i / 3.0 if i > 0 else 0.0
                    points[idx] = [x, y]
                    idx += 1
        
        return points

    def create_perturbed_initialization(base_points, perturbation_magnitude=0.015):
        """Create a perturbed version of base initialization."""
        perturbed = base_points.copy()
        # Add random perturbation
        perturbed += np.random.normal(0, perturbation_magnitude, base_points.shape)
        # Ensure points stay within bounds
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def create_adaptive_perturbed_initialization(base_points, initial_ratio, iteration=0):
        """Create an adaptive perturbed initialization based on current optimization state."""
        # Base perturbation magnitude
        base_perturbation = 0.015

        # Adaptive scaling based on initial quality and optimization iteration
        if initial_ratio < 0.1:
            # Poor initial configuration - use larger perturbations to explore more
            perturbation_magnitude = base_perturbation * (1.0 + (0.1 - initial_ratio) * 5.0)
        elif initial_ratio > 0.25:
            # Good initial configuration - use smaller perturbations to refine
            perturbation_magnitude = base_perturbation * max(0.1, 1.0 - (initial_ratio - 0.25) * 2.0)
        else:
            # Medium quality - use moderate perturbations
            perturbation_magnitude = base_perturbation

        # Additional adjustment based on iteration (decrease over time)
        if iteration > 0:
            perturbation_magnitude *= max(0.1, 1.0 - iteration * 0.02)

        perturbed = base_points.copy()
        perturbed += np.random.normal(0, perturbation_magnitude, base_points.shape)
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def optimize_with_local_refinement(initial_points, max_iter=500):
        """Perform local optimization refinement on initial configuration."""
        # Flatten for optimization
        initial_flat = initial_points.flatten()

        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(len(initial_flat))]

        # Optimize using L-BFGS-B method with stricter tolerances
        result = minimize(
            lambda flat_points: -calculate_min_max_ratio(flat_points.reshape(-1, 2)),
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12},
            callback=None
        )

        # Extract optimized points
        optimized_points = result.x.reshape(-1, 2)

        # Ensure all points are within bounds
        optimized_points = np.clip(optimized_points, 0, 1)

        return optimized_points

    def tournament_selection(population: List[np.ndarray], fitnesses: List[float], tournament_size: int = 3) -> np.ndarray:
        """Select an individual from population using tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index].copy()

    def crossover(parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover between two parents."""
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
            
        # Create offspring through crossover
        mask = np.random.random(parent1.shape) > 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        
        return child1, child2

    def mutate(individual: np.ndarray, mutation_strength: float = 0.02):
        """Apply Gaussian mutation to an individual."""
        mutated = individual.copy()
        noise = np.random.normal(0, mutation_strength, individual.shape)
        mutated += noise
        # Ensure bounds
        mutated = np.clip(mutated, 0, 1)
        return mutated

    def sphere_packing_evolution(max_generations: int = 100, population_size: int = 30) -> np.ndarray:
        """Evolutionary algorithm focused on sphere packing optimization."""
        # Create initial population with diverse strategies
        population = []
        fitness_scores = []
        
        # Generate multiple initial configurations
        strategies = [
            create_better_hexagonal_initialization(),
            create_concentric_ring_initialization(), 
            create_fibonacci_sphere_like_initialization(),
            create_grid_initialization()
        ]
        
        # Add variants of each strategy
        for strategy in strategies:
            # Original strategy
            population.append(strategy)
            # Perturbed variants
            for _ in range(3):
                perturbed = create_perturbed_initialization(strategy, 0.02)
                population.append(perturbed)
        
        # Evaluate initial population
        for individual in population:
            fitness = calculate_sphere_packing_fitness(individual)
            fitness_scores.append(fitness)
        
        # Evolution loop
        best_individual = population[np.argmax(fitness_scores)].copy()
        best_fitness = max(fitness_scores)
        
        for gen in range(max_generations):
            # Create new population through selection, crossover, and mutation
            new_population = []
            
            # Elitism: keep the best individuals
            elite_count = population_size // 4
            sorted_indices = np.argsort(fitness_scores)[::-1][:elite_count]
            for i in sorted_indices:
                new_population.append(population[i].copy())
            
            # Generate rest through evolution
            while len(new_population) < population_size:
                # Tournament selection to get two parents
                parent1 = tournament_selection(population, fitness_scores)
                parent2 = tournament_selection(population, fitness_scores)
                
                # Crossover
                child1, child2 = crossover(parent1, parent2)
                
                # Mutation
                mutation_strength = max(0.005, 0.02 * (1.0 - gen/max_generations))
                child1 = mutate(child1, mutation_strength)
                child2 = mutate(child2, mutation_strength)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            new_population = new_population[:population_size]
            
            # Evaluate new population
            new_fitness_scores = []
            for individual in new_population:
                fitness = calculate_sphere_packing_fitness(individual)
                new_fitness_scores.append(fitness)
            
            # Update best solution
            current_best_idx = np.argmax(new_fitness_scores)
            if new_fitness_scores[current_best_idx] > best_fitness:
                best_fitness = new_fitness_scores[current_best_idx]
                best_individual = new_population[current_best_idx].copy()
            
            # Replace old population
            population = new_population
            fitness_scores = new_fitness_scores
        
        return best_individual.copy()
    
    def multi_strategy_optimization():
        """Main optimization routine using sphere packing evolution."""
        np.random.seed(42)
        
        # Phase 1: Evolutionary optimization
        print("Starting evolutionary optimization...")
        evolved_solution = sphere_packing_evolution(max_generations=150, population_size=40)
        
        # Phase 2: Local refinement of the best evolution result
        print("Refining evolution result with local optimization...")
        refined_solution = optimize_with_local_refinement(evolved_solution, max_iter=500)
        
        # Phase 3: Compare with multiple baseline strategies
        strategies = [
            ("hex", create_better_hexagonal_initialization()),
            ("ring", create_concentric_ring_initialization()),
            ("fibonacci", create_fibonacci_sphere_like_initialization()),
            ("grid", create_grid_initialization())
        ]
        
        best_solution = refined_solution
        best_ratio = calculate_min_max_ratio(refined_solution)
        
        for name, strategy in strategies:
            # Perturb the base strategy slightly
            perturbed = create_perturbed_initialization(strategy, 0.01)
            local_optimized = optimize_with_local_refinement(perturbed, max_iter=300)
            ratio = calculate_min_max_ratio(local_optimized)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_solution = local_optimized
        
        return best_solution
    
    # Execute the main optimization
    final_points = multi_strategy_optimization()
    
    return final_points

# EVOLVE-BLOCK-END