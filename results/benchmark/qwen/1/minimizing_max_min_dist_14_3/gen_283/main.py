# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
from scipy.spatial import ConvexHull
import time
from typing import Tuple, List
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    
    class SphericalEvolutionOptimizer:
        def __init__(self, n_points: int = 14, n_generations: int = 100, population_size: int = 50):
            self.n_points = n_points
            self.n_generations = n_generations
            self.population_size = population_size
            self.best_individual = None
            self.best_fitness = -np.inf
            
        def normalize_to_sphere(self, points: np.ndarray) -> np.ndarray:
            """Normalize points to lie on unit sphere"""
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return points / norms
            
        def calculate_distance_ratio(self, points: np.ndarray) -> float:
            """Calculate minimum-to-maximum distance ratio"""
            if len(points) < 2:
                return 0.0
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist == 0:
                return 0.0
            return min_dist / max_dist
            
        def calculate_voronoi_entropy(self, points: np.ndarray) -> float:
            """Calculate entropy of Voronoi regions on sphere"""
            try:
                sv = SphericalVoronoi(points, radius=1.0, center=np.zeros(3))
                cell_areas = sv.voronoi_regions_area()
                if len(cell_areas) == 0:
                    return 0.0
                # Calculate entropy of area distribution
                total_area = np.sum(cell_areas)
                if total_area == 0:
                    return 0.0
                probabilities = cell_areas / total_area
                # Avoid log(0) by adding small epsilon
                eps = 1e-10
                probabilities = np.clip(probabilities, eps, 1.0)
                entropy = -np.sum(probabilities * np.log(probabilities))
                # Convert to normalized entropy (0 to 1)
                max_entropy = np.log(len(cell_areas))
                if max_entropy == 0:
                    return 0.0
                return entropy / max_entropy
            except:
                return 0.0
                
        def calculate_fitness(self, points: np.ndarray) -> float:
            """Calculate combined fitness: distance ratio + Voronoi uniformity"""
            # Normalize to sphere
            normalized_points = self.normalize_to_sphere(points)
            
            # Get distance ratio
            distance_ratio = self.calculate_distance_ratio(normalized_points)
            
            # Get Voronoi uniformity (higher entropy means more uniform)
            voronoi_uniformity = self.calculate_voronoi_entropy(normalized_points)
            
            # Combine with weights - prioritize distance ratio but encourage uniformity
            fitness = 0.7 * distance_ratio + 0.3 * voronoi_uniformity
            
            return fitness
            
        def initialize_population(self) -> List[np.ndarray]:
            """Initialize population with diverse strategies"""
            population = []
            
            # Strategy 1: Fibonacci spiral
            fib_points = self.fibonacci_sphere(self.n_points)
            population.append(fib_points.copy())
            
            # Strategy 2: Sobol sequence
            sobol_points = self.sobol_sphere(self.n_points)
            population.append(sobol_points.copy())
            
            # Strategy 3: Random points
            random_points = np.random.randn(self.n_points, 3)
            random_points = self.normalize_to_sphere(random_points)
            population.append(random_points.copy())
            
            # Strategy 4: Icosahedron points
            ico_points = self.icosahedron_points()
            population.append(ico_points.copy())
            
            # Fill remaining population with variations
            while len(population) < self.population_size:
                # Mutate existing individuals with small random perturbations
                parent = random.choice(population[:4])  # Only mutate from top strategies
                mutated = self.mutate_individual(parent)
                population.append(mutated)
                
                if len(population) >= self.population_size:
                    break
                    
            return population
            
        def fibonacci_sphere(self, samples: int) -> np.ndarray:
            """Generate points on sphere using Fibonacci spiral method"""
            points = []
            phi = np.pi * (3. - np.sqrt(5.))  # golden angle

            for i in range(samples):
                y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
                radius = np.sqrt(1 - y * y)  # radius at y

                theta = phi * i  # golden angle increment

                x = np.cos(theta) * radius
                z = np.sin(theta) * radius

                points.append([x, y, z])

            return np.array(points)
            
        def sobol_sphere(self, samples: int) -> np.ndarray:
            """Generate points using Sobol sequence mapped to sphere"""
            # Simple approach: generate Sobol-like distribution then normalize
            points = np.random.rand(samples, 3) * 2 - 1  # [-1, 1]^3
            points = self.normalize_to_sphere(points)
            return points
            
        def icosahedron_points(self) -> np.ndarray:
            """Generate points from regular icosahedron"""
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio
            vertices = np.array([
                [-1,  phi,  0],
                [ 1,  phi,  0],
                [-1, -phi,  0],
                [ 1, -phi,  0],
                [ 0, -1,  phi],
                [ 0,  1,  phi],
                [ 0, -1, -phi],
                [ 0,  1, -phi],
                [ phi,  0, -1],
                [ phi,  0,  1],
                [-phi,  0, -1],
                [-phi,  0,  1]
            ])
            # Normalize to unit sphere
            norms = np.linalg.norm(vertices, axis=1, keepdims=True)
            return vertices / norms
            
        def mutate_individual(self, individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
            """Mutate an individual by adding small random perturbations"""
            mutated = individual.copy()
            # Apply mutation to a subset of points
            n_mutation = max(1, int(self.n_points * mutation_rate))
            mutation_indices = np.random.choice(self.n_points, n_mutation, replace=False)
            
            for idx in mutation_indices:
                # Add small random perturbation
                perturbation = np.random.normal(0, 0.01, 3)
                mutated[idx] += perturbation
                
            # Re-normalize to sphere
            mutated = self.normalize_to_sphere(mutated)
            return mutated
            
        def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
            """Perform crossover between two parents"""
            # Simple uniform crossover
            mask = np.random.rand(self.n_points, 3) > 0.5
            child = np.where(mask, parent1, parent2)
            
            # Normalize to sphere
            child = self.normalize_to_sphere(child)
            return child
            
        def evolve(self) -> np.ndarray:
            """Main evolution loop"""
            # Initialize population
            population = self.initialize_population()
            
            for generation in range(self.n_generations):
                # Evaluate fitness of all individuals
                fitness_scores = []
                for individual in population:
                    fitness = self.calculate_fitness(individual)
                    fitness_scores.append(fitness)
                    
                    # Update best individual
                    if fitness > self.best_fitness:
                        self.best_fitness = fitness
                        self.best_individual = individual.copy()
                
                # Sort by fitness (descending)
                sorted_indices = np.argsort(fitness_scores)[::-1]
                population = [population[i] for i in sorted_indices]
                
                # Keep top 50% as elites
                elites = population[:self.population_size // 2]
                
                # Create new population through selection and crossover
                new_population = elites.copy()
                
                # Fill rest with offspring
                while len(new_population) < self.population_size:
                    # Tournament selection
                    parent1 = self.tournament_selection(population, fitness_scores)
                    parent2 = self.tournament_selection(population, fitness_scores)
                    
                    # Crossover
                    child = self.crossover(parent1, parent2)
                    
                    # Mutation
                    if np.random.rand() < 0.3:  # 30% mutation rate
                        child = self.mutate_individual(child)
                        
                    new_population.append(child)
                    
                    if len(new_population) >= self.population_size:
                        break
                
                population = new_population[:self.population_size]
                
                # Early stopping criteria
                if generation > 20 and abs(self.best_fitness - self.previous_best) < 1e-8:
                    # No improvement for last 20 generations
                    break
                    
                self.previous_best = self.best_fitness
            
            return self.best_individual
            
        def tournament_selection(self, population: List[np.ndarray], fitness_scores: List[float], 
                               tournament_size: int = 3) -> np.ndarray:
            """Select individual using tournament selection"""
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            return population[winner_index]
    
    # Run the evolutionary optimization
    optimizer = SphericalEvolutionOptimizer(n_points=14, n_generations=50, population_size=30)
    
    # Time limit check
    start_time = time.time()
    
    # Run evolution
    best_solution = optimizer.evolve()
    
    # Post-process: refine with local optimization
    refined_solution = best_solution.copy()
    try:
        # Apply small local optimizations using L-BFGS-B
        from scipy.optimize import minimize
        
        def objective(x):
            points = x.reshape(-1, 3)
            normalized_points = optimizer.normalize_to_sphere(points)
            return -optimizer.calculate_distance_ratio(normalized_points)
            
        # Local refinement
        bounds = [(-1, 1) for _ in range(14 * 3)]
        
        result = minimize(
            objective,
            refined_solution.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        
        refined_solution = result.x.reshape(-1, 3)
        refined_solution = optimizer.normalize_to_sphere(refined_solution)
        
    except Exception:
        # If local optimization fails, keep original solution
        pass
    
    # Final validation
    final_ratio = optimizer.calculate_distance_ratio(refined_solution)
    
    # Return the best solution found
    return refined_solution

# EVOLVE-BLOCK-END