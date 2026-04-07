# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import time
import random
from typing import Tuple, List
import warnings

class VoronoiEvolutionOptimizer:
    """Optimizes point distribution using Voronoi-based evolutionary approach."""
    
    def __init__(self, n_points=16, seed=42, max_time_seconds=180):
        self.n_points = n_points
        self.seed = seed
        self.max_time_seconds = max_time_seconds
        np.random.seed(seed)
        random.seed(seed)
        
    def _compute_min_max_ratio(self, points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0
            
        # Use efficient distance computation
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        if distances.size == 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 1e-12:
            return 0.0
            
        return d_min / d_max
    
    def _generate_voronoi_based_initialization(self):
        """Create initial configuration using Voronoi-based approach."""
        # Start with regular hexagonal grid
        points = []
        sqrt3 = np.sqrt(3)
        spacing_x = 1.0 / 3.0
        spacing_y = sqrt3 / 4.0
        
        # Create 4x4 hexagonal grid
        for i in range(4):
            for j in range(4):
                if len(points) >= self.n_points:
                    break
                    
                x = j * spacing_x
                y = i * spacing_y
                
                # Offset odd rows for hexagonal pattern
                if i % 2 == 1:
                    x += spacing_x / 2
                
                points.append([x, y])
        
        points = np.array(points[:self.n_points])
        
        # Normalize to [0,1] range
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        
        if x_max > x_min and x_max != x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min and y_max != y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
            
        # Ensure proper bounds
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)
        
        # Add structured perturbations to break symmetry
        for i in range(len(points)):
            # Different noise for different points
            noise_level = 0.01 + (i % 5) * 0.002
            points[i, 0] += np.random.normal(0, noise_level * 0.5)
            points[i, 1] += np.random.normal(0, noise_level * 0.5)
            
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)
        
        return points
    
    def _create_voronoi_representation(self, points):
        """Create Voronoi representation for the points."""
        try:
            # Create Voronoi diagram
            vor = Voronoi(points)
            return vor
        except Exception as e:
            warnings.warn(f"Voronoi creation failed: {str(e)}")
            # Fallback to simple point representation
            return None
    
    def _evaluate_voronoi_fitness(self, points):
        """Evaluate fitness based on Voronoi cell properties."""
        # Calculate the min/max distance ratio
        ratio = self._compute_min_max_ratio(points)
        
        # Additional geometric considerations from Voronoi structure
        try:
            vor = self._create_voronoi_representation(points)
            if vor is not None:
                # Penalize highly irregular cells (very large or very small)
                areas = []
                for i, region in enumerate(vor.point_region):
                    if region < len(vor.regions) and vor.regions[region]:
                        # Calculate area of Voronoi cell
                        points_in_region = np.array(vor.vertices)[vor.regions[region]]
                        if len(points_in_region) > 2:
                            # Simplified area calculation
                            area = abs(np.cross(points_in_region[:-1] - points_in_region[0], 
                                              points_in_region[1:] - points_in_region[0]).sum() / 2)
                            areas.append(area)
                
                # Penalize extreme variations in cell sizes
                if len(areas) > 1:
                    area_std = np.std(areas)
                    area_mean = np.mean(areas)
                    if area_mean > 0:
                        area_penalty = area_std / area_mean
                        ratio *= (1.0 - 0.1 * min(area_penalty, 1.0))
        except:
            pass
            
        return ratio
    
    def _generate_voronoi_mutant(self, points, mutation_strength=0.02):
        """Generate mutant points based on Voronoi structure."""
        # Start with copy of current points
        mutant_points = points.copy()
        
        # Select subset of points to modify (more points for better diversity)
        num_modify = max(1, min(len(points) // 3, 5))
        indices = np.random.choice(len(points), num_modify, replace=False)
        
        for idx in indices:
            # Apply perturbation that considers Voronoi neighbors
            # Find nearest neighbors to understand local structure
            distances = cdist([mutant_points[idx]], mutant_points)[0]
            distances[idx] = np.inf  # Exclude self
            nearest_idx = np.argmin(distances)
            
            # Perturb towards or away from neighbors based on structure
            if np.random.random() < 0.5:
                # Move away from nearest neighbor (for dispersion)
                direction = mutant_points[idx] - mutant_points[nearest_idx]
                magnitude = np.linalg.norm(direction)
                if magnitude > 0:
                    direction /= magnitude
                    perturbation = direction * np.random.uniform(0, mutation_strength)
                else:
                    perturbation = np.random.uniform(-mutation_strength, mutation_strength, 2)
            else:
                # Move towards nearest neighbor (for clustering if beneficial)
                direction = mutant_points[nearest_idx] - mutant_points[idx]
                magnitude = np.linalg.norm(direction)
                if magnitude > 0:
                    direction /= magnitude
                    perturbation = direction * np.random.uniform(0, mutation_strength * 0.5)
                else:
                    perturbation = np.random.uniform(-mutation_strength, mutation_strength, 2)
            
            # Apply perturbation
            mutant_points[idx] += perturbation
            
            # Ensure bounds
            mutant_points[idx, 0] = np.clip(mutant_points[idx, 0], 0, 1)
            mutant_points[idx, 1] = np.clip(mutant_points[idx, 1], 0, 1)
            
        return mutant_points
    
    def _generate_voronoi_crossover(self, parent1, parent2):
        """Generate offspring from two Voronoi structures."""
        # Simple uniform crossover on point positions
        mask = np.random.random(len(parent1)) < 0.5
        child = np.where(mask, parent1, parent2).copy()
        
        # Add some mutation to ensure diversity
        mutation_mask = np.random.random(len(child)) < 0.1
        for i in range(len(child)):
            if mutation_mask[i]:
                child[i] += np.random.normal(0, 0.01, 2)
                child[i, 0] = np.clip(child[i, 0], 0, 1)
                child[i, 1] = np.clip(child[i, 1], 0, 1)
                
        return child
    
    def _initialize_population(self, pop_size=30):
        """Initialize population using Voronoi-based strategies."""
        population = []
        
        # Generate diverse initial configurations
        for i in range(pop_size):
            # Mix different initialization strategies
            if i < 10:  # Hexagonal grid
                points = self._generate_voronoi_based_initialization()
            elif i < 20:  # Random with structured elements
                points = np.random.rand(self.n_points, 2)
                # Add some structure
                for j in range(0, self.n_points, 3):
                    if j < self.n_points:
                        points[j] = np.random.rand(2) * 0.5 + 0.25  # Center cluster
            else:  # Random
                points = np.random.rand(self.n_points, 2)
            
            # Add controlled noise based on population index
            noise_factor = 0.01 * (i % 3 + 1) / 3
            points += np.random.normal(0, noise_factor, points.shape)
            points[:, 0] = np.clip(points[:, 0], 0, 1)
            points[:, 1] = np.clip(points[:, 1], 0, 1)
            
            population.append(points)
            
        return population
    
    def _evolve_population(self, population, fitness_scores):
        """Evolve the population using tournament selection and genetic operators."""
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Elitism: keep top 20%
        elite_count = max(1, len(population) // 5)
        new_population = sorted_population[:elite_count]
        
        # Generate offspring through crossover and mutation
        while len(new_population) < len(population):
            # Tournament selection for parents
            tournament_size = 3
            parent1_idx = sorted_indices[np.random.choice(tournament_size, replace=False)]
            parent2_idx = sorted_indices[np.random.choice(tournament_size, replace=False)]
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            child = self._generate_voronoi_crossover(parent1, parent2)
            
            # Mutation
            if np.random.random() < 0.7:  # 70% chance of mutation
                child = self._generate_voronoi_mutant(child, mutation_strength=0.03)
            
            new_population.append(child)
        
        return new_population[:len(population)]
    
    def _local_refinement(self, points, max_iterations=100):
        """Perform local refinement using gradient-based approach."""
        best_points = points.copy()
        best_ratio = self._compute_min_max_ratio(best_points)
        
        # Try several local search strategies
        for iter_num in range(max_iterations):
            current_points = best_points.copy()
            
            # Try moving each point optimally
            for i in range(len(current_points)):
                # Store original point
                original_point = current_points[i].copy()
                
                # Try small perturbations in directions of neighbors
                distances = cdist([original_point], current_points)[0]
                distances[i] = np.inf  # Exclude self
                
                if len(distances) > 0:
                    # Find closest neighbors
                    nearest_indices = np.argsort(distances)[:3]  # Top 3 nearest
                    
                    # Try moving away from neighbors to increase minimum distance
                    avg_neighbor_pos = np.mean(current_points[nearest_indices], axis=0)
                    direction = original_point - avg_neighbor_pos
                    distance = np.linalg.norm(direction)
                    
                    if distance > 0:
                        direction /= distance
                        
                        # Try perturbation in this direction
                        test_point = current_points[i] + direction * 0.01
                        test_point[0] = np.clip(test_point[0], 0, 1)
                        test_point[1] = np.clip(test_point[1], 0, 1)
                        
                        current_points[i] = test_point
                        new_ratio = self._compute_min_max_ratio(current_points)
                        
                        if new_ratio > best_ratio:
                            best_ratio = new_ratio
                            best_points = current_points.copy()
                        else:
                            current_points[i] = original_point  # Revert
                
                # Also try random small perturbations
                test_point = original_point + np.random.normal(0, 0.005, 2)
                test_point[0] = np.clip(test_point[0], 0, 1)
                test_point[1] = np.clip(test_point[1], 0, 1)
                
                current_points[i] = test_point
                new_ratio = self._compute_min_max_ratio(current_points)
                
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = current_points.copy()
                else:
                    current_points[i] = original_point  # Revert
        
        return best_points
    
    def optimize(self):
        """Main optimization routine using Voronoi evolutionary approach."""
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population(pop_size=30)
        best_points = None
        best_ratio = -np.inf
        
        # Evolutionary process
        generations = 50
        for generation in range(generations):
            if time.time() - start_time > self.max_time_seconds - 10:
                break
                
            # Evaluate fitness for current population
            fitness_scores = []
            for points in population:
                # Evaluate Voronoi-based fitness
                ratio = self._evaluate_voronoi_fitness(points)
                fitness_scores.append(ratio)
                
                # Track best solution
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()
            
            # Evolve population
            population = self._evolve_population(population, fitness_scores)
            
            # Periodic local refinement
            if generation % 5 == 0 and best_points is not None:
                refined_points = self._local_refinement(best_points.copy())
                refined_ratio = self._compute_min_max_ratio(refined_points)
                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = refined_points.copy()
        
        # Final local refinement
        if best_points is not None:
            final_points = self._local_refinement(best_points.copy())
            final_ratio = self._compute_min_max_ratio(final_points)
            if final_ratio > best_ratio:
                best_points = final_points
        
        # Fallback to hexagonal initialization if nothing worked
        if best_points is None:
            best_points = self._generate_voronoi_based_initialization()
        
        return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = VoronoiEvolutionOptimizer(n_points=16, seed=42, max_time_seconds=180)
    points = optimizer.optimize()
    return points

# EVOLVE-BLOCK-END