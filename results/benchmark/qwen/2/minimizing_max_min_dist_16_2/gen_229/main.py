# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
from typing import Tuple, List, Optional
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    class VoronoiBasedOptimizer:
        def __init__(self, num_points: int = 16):
            self.num_points = num_points
            self.boundary = 1.0
            self.max_iterations = 500
            self.population_size = 30
            self.mutation_rate = 0.1
            
        def calculate_voronoi_entropy(self, points: np.ndarray) -> float:
            """Calculate entropy of Voronoi cell areas to measure distribution uniformity."""
            try:
                vor = Voronoi(points)
                if len(vor.points) < 2:
                    return 0.0
                    
                # Get Voronoi cell areas
                areas = []
                for i in range(len(points)):
                    region = vor.regions[vor.point_region[i]]
                    if -1 not in region and len(region) > 0:
                        # Calculate area using shoelace formula
                        vertices = [vor.vertices[j] for j in region if j >= 0]
                        if len(vertices) >= 3:
                            area = self._polygon_area(vertices)
                            areas.append(area)
                
                if len(areas) == 0:
                    return 0.0
                    
                # Normalize areas
                areas = np.array(areas)
                mean_area = np.mean(areas)
                if mean_area == 0:
                    return 0.0
                    
                # Calculate entropy (lower entropy = more uniform)
                normalized_areas = areas / mean_area
                entropy = -np.sum(normalized_areas * np.log(normalized_areas + 1e-10))
                return entropy
            except:
                return 0.0
        
        def _polygon_area(self, vertices: List[List[float]]) -> float:
            """Calculate polygon area using shoelace formula."""
            if len(vertices) < 3:
                return 0.0
            n = len(vertices)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += vertices[i][0] * vertices[j][1]
                area -= vertices[j][0] * vertices[i][1]
            return abs(area) / 2.0
        
        def calculate_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
            """Calculate min/max distance ratio with proper error handling."""
            if len(points) < 2:
                return 0.0, 0.0, 0.0
            
            distances = pdist(points)
            if len(distances) == 0:
                return 0.0, 0.0, 0.0
                
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist == 0:
                return 0.0, min_dist, max_dist
                
            ratio = min_dist / max_dist
            return ratio, min_dist, max_dist
        
        def fitness_function(self, points: np.ndarray) -> float:
            """Combined fitness function using ratio and Voronoi entropy."""
            ratio, min_dist, max_dist = self.calculate_ratio(points)
            
            # Add entropy component to encourage uniform distribution
            entropy = self.calculate_voronoi_entropy(points)
            entropy_factor = np.exp(-entropy / 10.0)  # Scale entropy effect
            
            # Combined fitness: higher ratio AND more uniform distribution
            if ratio > 0:
                fitness = ratio * entropy_factor
            else:
                fitness = 0.0
                
            return fitness
        
        def generate_initial_population(self) -> List[np.ndarray]:
            """Generate diverse initial population using geometric constructions."""
            population = []
            
            # Strategy 1: Regular grid with noise
            grid_points = []
            for i in range(4):
                for j in range(4):
                    x = 0.1 + (i * 0.8) / 3.0
                    y = 0.1 + (j * 0.8) / 3.0
                    grid_points.append([x, y])
            base_grid = np.array(grid_points[:self.num_points])
            
            # Add noise to create variance
            for _ in range(self.population_size // 3):
                noisy_points = base_grid + np.random.normal(0, 0.03, (self.num_points, 2))
                noisy_points = np.clip(noisy_points, 0.001, 0.999)
                population.append(noisy_points)
            
            # Strategy 2: Hexagonal packing approximation
            hex_points = np.array([
                [0.25, 0.25], [0.75, 0.25],
                [0.25, 0.75], [0.75, 0.75],
                [0.1, 0.1], [0.9, 0.1],
                [0.1, 0.9], [0.9, 0.9],
                [0.3, 0.5], [0.7, 0.5],
                [0.5, 0.3], [0.5, 0.7],
                [0.4, 0.4], [0.6, 0.6],
                [0.4, 0.6], [0.6, 0.4]
            ][:self.num_points])
            
            for _ in range(self.population_size // 3):
                noisy_hex = hex_points + np.random.normal(0, 0.02, (self.num_points, 2))
                noisy_hex = np.clip(noisy_hex, 0.001, 0.999)
                population.append(noisy_hex)
            
            # Strategy 3: Random with boundary constraints
            for _ in range(self.population_size // 3):
                random_points = np.random.uniform(0.05, 0.95, (self.num_points, 2))
                population.append(random_points)
            
            return population
        
        def mutate_individual(self, individual: np.ndarray) -> np.ndarray:
            """Mutate individual with geometric-aware perturbations."""
            mutated = individual.copy()
            
            # Apply mutations to a subset of points
            num_mutations = max(1, int(self.num_points * self.mutation_rate))
            mutation_indices = random.sample(range(self.num_points), num_mutations)
            
            for idx in mutation_indices:
                # Apply geometrically meaningful perturbation
                # More aggressive near boundaries, less in center
                if (individual[idx][0] < 0.1 or individual[idx][0] > 0.9 or 
                    individual[idx][1] < 0.1 or individual[idx][1] > 0.9):
                    # Larger perturbations near boundaries
                    delta = np.random.normal(0, 0.04, 2)
                else:
                    # Smaller perturbations in center
                    delta = np.random.normal(0, 0.015, 2)
                
                mutated[idx] += delta
                # Clamp to boundary
                mutated[idx] = np.clip(mutated[idx], 0.001, 0.999)
            
            return mutated
        
        def crossover_individuals(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
            """Perform crossover between two individuals."""
            # Uniform crossover for point positions
            child = parent1.copy()
            mask = np.random.random(self.num_points) < 0.5
            child[mask] = parent2[mask]
            return child
        
        def evolve_population(self, population: List[np.ndarray]) -> List[np.ndarray]:
            """Evolve population using genetic operators."""
            # Evaluate fitness of current population
            fitnesses = [self.fitness_function(ind) for ind in population]
            
            # Sort by fitness
            sorted_indices = np.argsort(fitnesses)[::-1]
            elite = [population[i] for i in sorted_indices[:self.population_size//3]]
            
            # Generate new population
            new_population = elite.copy()
            
            # Fill rest with crossover and mutation
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
                
                # Select second parent
                second_parent_idx = random.choice([i for i in range(len(population)) if i != winner_idx])
                
                # Crossover
                child = self.crossover_individuals(population[winner_idx], population[second_parent_idx])
                
                # Mutation
                if random.random() < 0.8:  # 80% chance of mutation
                    child = self.mutate_individual(child)
                
                new_population.append(child)
            
            return new_population
        
        def optimize(self) -> np.ndarray:
            """Main optimization loop using evolutionary approach."""
            # Generate initial population
            population = self.generate_initial_population()
            
            # Track best solution
            best_fitness = -np.inf
            best_solution = None
            
            # Evolutionary optimization
            for generation in range(100):  # Limited generations for time constraints
                # Evolve population
                population = self.evolve_population(population)
                
                # Evaluate best in population
                current_fitnesses = [self.fitness_function(ind) for ind in population]
                max_fitness = max(current_fitnesses)
                
                if max_fitness > best_fitness:
                    best_fitness = max_fitness
                    best_idx = np.argmax(current_fitnesses)
                    best_solution = population[best_idx].copy()
                
                # Early stopping if we're converging
                if generation > 20 and abs(max_fitness - best_fitness) < 1e-6:
                    break
            
            # Final local refinement with gradient-based optimization
            if best_solution is not None:
                # Convert to flat array for optimization
                initial_flat = best_solution.flatten()
                
                def objective(x_flat):
                    points = x_flat.reshape(-1, 2)
                    ratio, _, _ = self.calculate_ratio(points)
                    return -ratio  # Negative because we're minimizing
                
                # Use L-BFGS-B for final refinement
                bounds = [(0.001, 0.999) for _ in range(32)]
                try:
                    result = minimize(
                        objective,
                        initial_flat,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
                    )
                    
                    if result.success:
                        refined_points = result.x.reshape(-1, 2)
                        ratio, _, _ = self.calculate_ratio(refined_points)
                        
                        # Only update if improvement
                        if ratio > self.calculate_ratio(best_solution)[0]:
                            return refined_points
                except:
                    pass
            
            return best_solution if best_solution is not None else np.random.uniform(0, 1, (self.num_points, 2))
    
    # Run the Voronoi-based optimization
    optimizer = VoronoiBasedOptimizer(16)
    best_points = optimizer.optimize()
    
    return best_points

# EVOLVE-BLOCK-END