# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist
import time
from typing import Tuple, List
import heapq

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a Voronoi-based evolutionary optimization approach with geometric fitness functions.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    np.random.seed(42)

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max <= 0:
            return 0
        return d_min / d_max

    def compute_voronoi_fitness(points):
        """Compute fitness based on Voronoi diagram properties"""
        if len(points) < 3:
            return 0
            
        try:
            vor = Voronoi(points)
            # Calculate Voronoi cell areas
            areas = []
            for region in vor.regions:
                if len(region) > 0 and -1 not in region:
                    # Compute area of polygon (simplified - just count vertices)
                    if len(region) >= 3:
                        # Just return a measure related to cell complexity
                        areas.append(len(region))
            
            if len(areas) == 0:
                return 0
            
            # Fitness components
            avg_area = np.mean(areas) if areas else 0
            area_variance = np.var(areas) if len(areas) > 1 else 0
            
            # Prefer more uniform cell structures
            uniformity_score = 1.0 / (1.0 + area_variance)
            
            # Also consider min distance ratio
            min_ratio = compute_min_max_ratio(points)
            
            # Combined fitness
            return uniformity_score * 0.5 + min_ratio * 0.5
            
        except:
            return compute_min_max_ratio(points)

    def create_initial_configurations():
        """Create diverse initial configurations using geometric patterns"""
        configurations = []
        
        # Pattern 1: Regular hexagonal grid (optimal for packing)
        points = []
        rows = 4
        cols = 4
        spacing_x = 1.0
        spacing_y = np.sqrt(3) / 2.0
        
        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x * 0.5
                y = i * spacing_y
                points.append([x, y])
        
        points = np.array(points)
        # Normalize to [0,1] x [0,1]
        max_x = (cols - 1) + 0.5
        max_y = (rows - 1) * spacing_y
        points[:, 0] = points[:, 0] / max_x
        points[:, 1] = points[:, 1] / max_y
        configurations.append(points.copy())
        
        # Pattern 2: Perturbed hexagonal grid with moderate noise
        noise = np.random.normal(0, 0.015, points.shape)
        noisy_points = points + noise
        noisy_points = np.clip(noisy_points, 0, 1)
        configurations.append(noisy_points.copy())
        
        # Pattern 3: Random with boundary awareness
        random_points = np.random.rand(16, 2)
        for i in range(len(random_points)):
            if random_points[i, 0] < 0.01:
                random_points[i, 0] = 0.01 + np.random.rand() * 0.01
            elif random_points[i, 0] > 0.99:
                random_points[i, 0] = 0.99 - np.random.rand() * 0.01
            if random_points[i, 1] < 0.01:
                random_points[i, 1] = 0.01 + np.random.rand() * 0.01
            elif random_points[i, 1] > 0.99:
                random_points[i, 1] = 0.99 - np.random.rand() * 0.01
        configurations.append(random_points.copy())
        
        # Pattern 4: Grid with noise
        uniform_points = []
        for i in range(4):
            for j in range(4):
                uniform_points.append([i/3, j/3])
        uniform_points = np.array(uniform_points[:16])
        noise = np.random.normal(0, 0.02, (16, 2))
        grid_noisy = uniform_points + noise
        grid_noisy = np.clip(grid_noisy, 0, 1)
        configurations.append(grid_noisy.copy())
        
        # Pattern 5: Circle arrangement
        theta = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 0.4
        circle_points = np.column_stack([radius * np.cos(theta) + 0.5, radius * np.sin(theta) + 0.5])
        configurations.append(circle_points.copy())
        
        # Pattern 6: Another random seed variation
        np.random.seed(123)
        configurations.append(np.random.rand(16, 2).copy())
        
        # Pattern 7: Highly perturbed hexagonal
        noise = np.random.normal(0, 0.03, points.shape)
        highly_noisy = points + noise
        highly_noisy = np.clip(highly_noisy, 0, 1)
        configurations.append(highly_noisy.copy())
        
        # Pattern 8: Uniform grid
        uniform_grid = []
        for i in range(4):
            for j in range(4):
                uniform_grid.append([i/3, j/3])
        uniform_grid = np.array(uniform_grid[:16])
        configurations.append(uniform_grid.copy())
        
        return configurations

    def voronoi_evolutionary_optimization(initial_populations: List[np.ndarray], max_generations: int = 100) -> np.ndarray:
        """Evolutionary optimization using Voronoi geometry"""
        
        population_size = len(initial_populations)
        num_parents = population_size // 2
        best_individual = None
        best_fitness = -np.inf
        
        # Initialize population with fitness evaluations
        population = []
        fitness_scores = []
        
        for i, individual in enumerate(initial_populations):
            fitness = compute_voronoi_fitness(individual)
            population.append(individual.copy())
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
        
        # Evolution loop
        for generation in range(max_generations):
            # Sort by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            top_individuals = [population[i] for i in sorted_indices[:num_parents]]
            top_fitness = [fitness_scores[i] for i in sorted_indices[:num_parents]]
            
            # Create offspring
            new_population = top_individuals[:]
            
            # Elitism: keep best individual
            if len(top_individuals) > 0:
                new_population.append(top_individuals[0].copy())
            
            # Generate new individuals through crossover and mutation
            while len(new_population) < population_size:
                # Selection: tournament selection
                parent1_idx = tournament_selection(fitness_scores, top_individuals)
                parent2_idx = tournament_selection(fitness_scores, top_individuals)
                
                # Crossover: blend between parents
                child = blend_crossover(top_individuals[parent1_idx], top_individuals[parent2_idx])
                
                # Mutation: Voronoi-aware mutation
                child = voronoi_mutation(child, generation, max_generations)
                
                # Add to population
                new_population.append(child)
            
            # Trim to exact population size
            population = new_population[:population_size]
            
            # Re-evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = compute_voronoi_fitness(individual)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Adaptive termination check
            if generation > 20 and abs(best_fitness - np.mean(fitness_scores)) < 1e-5:
                break
        
        return best_individual
    
    def tournament_selection(fitness_scores: List[float], individuals: List[np.ndarray], tournament_size: int = 3) -> int:
        """Select individual via tournament selection"""
        tournament_indices = np.random.choice(len(fitness_scores), tournament_size, replace=False)
        best_idx = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
        return best_idx
    
    def blend_crossover(parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Blend crossover between two parents"""
        # Simple average crossover
        child = alpha * parent1 + (1 - alpha) * parent2
        # Add some noise for diversity
        noise_magnitude = 0.01
        noise = np.random.normal(0, noise_magnitude, child.shape)
        child += noise
        child = np.clip(child, 0, 1)
        return child
    
    def voronoi_mutation(points: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
        """Mutation with Voronoi-aware adjustments"""
        mutated = points.copy()
        
        # Adaptive mutation rate based on generation
        mutation_rate = 0.3 * (1.0 - generation / max_generations)
        
        # Mutate points that are in high-density regions
        for i in range(len(mutated)):
            if np.random.rand() < mutation_rate:
                # Estimate local density
                local_density = estimate_local_density(mutated, i, radius=0.15)
                
                # Adjust mutation strength based on density
                if local_density > 3:
                    # Dense regions: smaller mutations
                    strength = 0.005 * (1.0 - generation / max_generations) 
                elif local_density < 2:
                    # Sparse regions: larger mutations 
                    strength = 0.02 * (1.0 - generation / max_generations)
                else:
                    strength = 0.01 * (1.0 - generation / max_generations)
                
                # Apply mutation
                mutated[i, 0] += np.random.normal(0, strength)
                mutated[i, 1] += np.random.normal(0, strength)
                
                # Boundary handling with more aggressive corrections
                if mutated[i, 0] < 0.01:
                    mutated[i, 0] = 0.01 + np.random.rand() * 0.01
                elif mutated[i, 0] > 0.99:
                    mutated[i, 0] = 0.99 - np.random.rand() * 0.01
                if mutated[i, 1] < 0.01:
                    mutated[i, 1] = 0.01 + np.random.rand() * 0.01
                elif mutated[i, 1] > 0.99:
                    mutated[i, 1] = 0.99 - np.random.rand() * 0.01
        
        return mutated
    
    def estimate_local_density(points: np.ndarray, target_idx: int, radius: float = 0.15) -> int:
        """Estimate local density around a point"""
        density = 0
        target_point = points[target_idx]
        
        for i in range(len(points)):
            if i != target_idx:
                dist = np.sqrt(np.sum((target_point - points[i])**2))
                if dist <= radius:
                    density += 1
                    
        return density

    def local_improvement(points: np.ndarray, iterations: int = 1000) -> np.ndarray:
        """Fine-tune with local search"""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        for _ in range(iterations):
            # Try to improve by moving one point at a time
            idx = np.random.randint(len(current_points))
            
            # Save current state
            old_points = current_points.copy()
            old_ratio = current_ratio
            
            # Make small random perturbation
            current_points[idx, 0] += np.random.normal(0, 0.005)
            current_points[idx, 1] += np.random.normal(0, 0.005)
            
            # Enforce boundaries
            current_points[:, 0] = np.clip(current_points[:, 0], 0, 1)
            current_points[:, 1] = np.clip(current_points[:, 1], 0, 1)
            
            # Handle boundary issues
            if current_points[idx, 0] < 0.01:
                current_points[idx, 0] = 0.01 + np.random.rand() * 0.01
            elif current_points[idx, 0] > 0.99:
                current_points[idx, 0] = 0.99 - np.random.rand() * 0.01
            if current_points[idx, 1] < 0.01:
                current_points[idx, 1] = 0.01 + np.random.rand() * 0.01
            elif current_points[idx, 1] > 0.99:
                current_points[idx, 1] = 0.99 - np.random.rand() * 0.01
            
            # Evaluate
            new_ratio = compute_min_max_ratio(current_points)
            
            # Accept if better, or sometimes accept worse moves
            if new_ratio > old_ratio or np.random.random() < 0.3:
                current_ratio = new_ratio
            else:
                # Revert if not accepted
                current_points = old_points
                current_ratio = old_ratio
        
        return current_points

    # Main optimization process
    initial_configs = create_initial_configurations()
    
    # Run Voronoi-based evolutionary optimization
    evolved_points = voronoi_evolutionary_optimization(initial_configs, max_generations=80)
    
    # Apply local improvement
    final_points = local_improvement(evolved_points, iterations=500)
    
    # Final evaluation
    final_ratio = compute_min_max_ratio(final_points)
    
    return final_points

# EVOLVE-BLOCK-END