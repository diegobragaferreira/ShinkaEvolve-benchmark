# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import warnings
import time
from typing import Tuple, List, Optional

class SphericalEvolutionOptimizer:
    """A spherical evolution optimizer that uses genetic algorithms with spherical operators."""
    
    def __init__(self, num_points: int = 14, population_size: int = 50, generations: int = 100):
        self.n = num_points
        self.pop_size = population_size
        self.generations = generations
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        self.best_solution = None
        self.best_ratio = -np.inf
        self.population = []
        self.fitness_history = []

    def _normalize_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Normalize points to unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms

    def _calculate_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Calculate min/max distance ratio and average distance."""
        if len(points) < 2:
            return 0.0, 0.0, 0.0
            
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        avg_dist = np.mean(distances[distances != np.inf])
        
        if max_dist > 0:
            ratio = min_dist / max_dist
        else:
            ratio = 0.0
            
        return ratio, min_dist, max_dist

    def _fitness_function(self, points: np.ndarray, uniformity_weight: float = 0.1) -> float:
        """Enhanced fitness function considering ratio and distribution uniformity."""
        ratio, min_dist, max_dist = self._calculate_ratio(points)
        
        if max_dist <= 1e-12:
            return -np.inf
            
        # Add uniformity penalty based on distance variance
        distances = pdist(points)
        distances = distances[distances > 1e-12]
        if len(distances) > 0:
            distance_variance = np.var(distances)
            uniformity_penalty = uniformity_weight * distance_variance
        else:
            uniformity_penalty = 0.0
            
        # Return fitness (higher is better)
        return ratio - uniformity_penalty

    def _generate_initial_population(self) -> List[np.ndarray]:
        """Generate diverse initial population using multiple strategies."""
        population = []
        
        # Strategy 1: Enhanced Fibonacci spiral
        def fibonacci_spiral():
            points = []
            for i in range(self.n):
                if i == 0:
                    phi = 0
                    theta = 0
                elif i == self.n - 1:
                    phi = np.pi
                    theta = 0
                else:
                    phi = np.arccos(1 - 2 * i / (self.n - 1))
                    theta = 2 * np.pi * i / self.golden_ratio + 0.1 * np.sin(i * 0.5)
                    
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                points.append([x, y, z])
            return np.array(points)
        
        # Strategy 2: Icosahedral-based
        def icosahedral_points():
            phi = (1 + np.sqrt(5)) / 2
            vertices = np.array([
                [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
            ])
            vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
            
            # Add two more points to make 14
            additional = np.array([[0, 0, 1], [0, 0, -1]])
            return np.vstack([vertices, additional])
        
        # Strategy 3: Random distributed
        def random_points():
            points = np.random.rand(self.n, 3) * 2 - 1
            return points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Generate individuals using different strategies
        strategies = [
            ("fibonacci", fibonacci_spiral),
            ("icosahedral", icosahedral_points),
            ("random", random_points)
        ]
        
        # Generate initial population
        for i in range(self.pop_size):
            np.random.seed(i * 42 + 1)
            
            # Select strategy based on index to ensure diversity
            strategy_idx = i % len(strategies)
            strategy_name, strategy_func = strategies[strategy_idx]
            
            # Apply strategy
            individual = strategy_func()
            
            # Add small random perturbation to break symmetry
            if strategy_name != "random":
                noise = np.random.normal(0, 0.01, individual.shape)
                individual += noise
                individual = self._normalize_to_sphere(individual)
            
            population.append(individual)
            
        return population

    def _spherical_crossover(self, parent1: np.ndarray, parent2: np.ndarray, 
                           alpha: float = 0.5) -> np.ndarray:
        """Perform spherical crossover that maintains unit sphere constraint."""
        # Spherical interpolation using spherical linear interpolation (slerp)
        # Normalize parents to unit sphere
        p1 = parent1 / np.linalg.norm(parent1, axis=1, keepdims=True)
        p2 = parent2 / np.linalg.norm(parent2, axis=1, keepdims=True)
        
        # Compute dot product for angle calculation
        dot_product = np.sum(p1 * p2, axis=1)
        # Clamp to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Compute angle
        angle = np.arccos(dot_product)
        # Avoid division by zero
        safe_angle = np.where(angle == 0, 1e-10, angle)
        
        # Spherical linear interpolation
        sin_angle = np.sin(safe_angle)
        t1 = np.sin((1 - alpha) * safe_angle) / sin_angle
        t2 = np.sin(alpha * safe_angle) / sin_angle
        
        # Vectorized slerp
        offspring = t1[:, np.newaxis] * p1 + t2[:, np.newaxis] * p2
        # Normalize result
        offspring = offspring / np.linalg.norm(offspring, axis=1, keepdims=True)
        return offspring

    def _spherical_mutation(self, individual: np.ndarray, mutation_rate: float = 0.1,
                          strength: float = 0.05) -> np.ndarray:
        """Apply spherical mutation that maintains unit sphere constraint."""
        # Create copy
        mutated = individual.copy()
        
        # Determine which points to mutate
        mask = np.random.random(mutated.shape[0]) < mutation_rate
        
        if np.any(mask):
            # Generate random perturbations
            noise = np.random.normal(0, strength, (mutated.shape[0], mutated.shape[1]))
            
            # Only apply to selected points
            noise[mask] = noise[mask] + np.random.normal(0, strength * 2, (np.sum(mask), mutated.shape[1]))
            
            # Apply noise
            mutated[mask] = mutated[mask] + noise[mask]
            
            # Normalize to unit sphere
            norms = np.linalg.norm(mutated[mask], axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            mutated[mask] = mutated[mask] / norms
            
        return mutated

    def _evolve_generation(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """Evolve one generation of the population."""
        # Evaluate fitness for current population
        fitness_scores = []
        for individual in population:
            fitness = self._fitness_function(individual)
            fitness_scores.append(fitness)
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Keep best individuals
        elite_count = max(1, self.pop_size // 4)
        new_population = sorted_population[:elite_count].copy()
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.pop_size:
            # Tournament selection
            tournament_size = 3
            parents = []
            for _ in range(2):  # Select 2 parents
                tournament_indices = np.random.choice(len(sorted_population), tournament_size)
                tournament_fitness = [sorted_fitness[i] for i in tournament_indices]
                winner_index = tournament_indices[np.argmax(tournament_fitness)]
                parents.append(sorted_population[winner_index])
            
            # Crossover
            if len(parents) == 2:
                # Use different alpha values for variety
                alpha = np.random.random()
                child = self._spherical_crossover(parents[0], parents[1], alpha)
                
                # Mutation
                child = self._spherical_mutation(child, mutation_rate=0.1, strength=0.02)
                
                # Ensure it stays normalized
                child = self._normalize_to_sphere(child)
                new_population.append(child)
        
        # Trim to exact population size
        return new_population[:self.pop_size]

    def _hybrid_local_refinement(self, points: np.ndarray) -> np.ndarray:
        """Apply local refinement using gradient-based methods."""
        try:
            # Define constraint function for unit sphere
            def constraint_func(x):
                points_matrix = x.reshape(-1, 3)
                norms = np.linalg.norm(points_matrix, axis=1)
                return norms - 1.0
            
            # Objectives for optimization
            def objective(x):
                points_matrix = x.reshape(-1, 3)
                ratio, _, _ = self._calculate_ratio(points_matrix)
                # We want to maximize ratio, so minimize negative ratio
                return -ratio
            
            # Use L-BFGS-B with strict bounds
            bounds = [(-2, 2) for _ in range(self.n * 3)]
            cons = {'type': 'eq', 'fun': constraint_func}
            
            # Optimize
            result = minimize(
                objective,
                points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                constraints=cons,
                options={'ftol': 1e-14, 'gtol': 1e-14, 'maxiter': 500}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                refined_points = self._normalize_to_sphere(refined_points)
                return refined_points
        except Exception as e:
            warnings.warn(f"Hybrid refinement failed: {e}")
            return points
            
        return points

    def _adaptive_evolution(self) -> np.ndarray:
        """Execute adaptive evolutionary optimization with multiple phases."""
        
        # Phase 1: Initial population generation
        self.population = self._generate_initial_population()
        
        # Phase 2: Evolution with adaptive parameters
        best_overall_ratio = -np.inf
        best_overall_solution = None
        
        # Track diversity to adapt parameters
        diversity_history = []
        
        for gen in range(self.generations):
            # Evolve population
            self.population = self._evolve_generation(self.population)
            
            # Evaluate current generation
            current_fitness = [self._fitness_function(ind) for ind in self.population]
            current_best_fitness = np.max(current_fitness)
            current_best_individual = self.population[np.argmax(current_fitness)]
            
            # Update best solution
            if current_best_fitness > best_overall_ratio:
                best_overall_ratio = current_best_fitness
                best_overall_solution = current_best_individual.copy()
            
            # Store fitness history
            self.fitness_history.append(current_best_fitness)
            
            # Adapt parameters based on generation
            if gen > 10:
                # Monitor diversity (standard deviation of fitness)
                diversity = np.std(self.fitness_history[-10:]) if len(self.fitness_history) >= 10 else 0
                diversity_history.append(diversity)
                
                # If diversity is low, increase mutation rate
                if len(diversity_history) > 5 and np.mean(diversity_history[-5:]) < 0.001:
                    pass  # Would increase mutation in real implementation
            
            # Print progress every 20 generations
            if gen % 20 == 0:
                ratio, _, _ = self._calculate_ratio(best_overall_solution)
                print(f"Generation {gen}: Best ratio = {ratio:.6f}")
        
        # Phase 3: Final hybrid refinement
        if best_overall_solution is not None:
            final_solution = self._hybrid_local_refinement(best_overall_solution)
            return final_solution
        else:
            # Fall back to best from initial population
            fitness_scores = [self._fitness_function(ind) for ind in self.population]
            best_idx = np.argmax(fitness_scores)
            return self.population[best_idx]
    
    def _evaluate_and_update_best(self, points: np.ndarray) -> bool:
        """Evaluate solution and update best if better."""
        ratio, _, _ = self._calculate_ratio(points)
        if ratio > self.best_ratio:
            self.best_ratio = ratio
            self.best_solution = points.copy()
            return True
        return False

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Set fixed seed for reproducibility
    np.random.seed(42)
    
    # Create optimizer instance with adapted parameters for faster execution
    optimizer = SphericalEvolutionOptimizer(num_points=14, population_size=30, generations=50)
    
    # Execute optimization
    result = optimizer._adaptive_evolution()
    
    # Final validation and cleanup
    if result is not None:
        # Ensure it's properly normalized
        result = result / np.linalg.norm(result, axis=1, keepdims=True)
        # Validate the solution
        try:
            ratio, _, _ = optimizer._calculate_ratio(result)
            if ratio <= 0:
                # If invalid, generate a reasonable fallback
                fallback = np.random.rand(14, 3) * 2 - 1
                fallback = fallback / np.linalg.norm(fallback, axis=1, keepdims=True)
                return fallback
        except Exception:
            # Fallback to random points if everything fails
            fallback = np.random.rand(14, 3) * 2 - 1
            fallback = fallback / np.linalg.norm(fallback, axis=1, keepdims=True)
            return fallback
    else:
        # Fallback to random distribution
        fallback = np.random.rand(14, 3) * 2 - 1
        fallback = fallback / np.linalg.norm(fallback, axis=1, keepdims=True)
        return fallback

    return result

# EVOLVE-BLOCK-END