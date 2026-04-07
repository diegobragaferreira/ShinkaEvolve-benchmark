# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Voronoi
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0

        # Compute pairwise distances with enhanced numerical stability
        distance_matrix = squareform(pdist(points))
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Get all finite distances (excluding NaN and inf values)
        finite_distances = distance_matrix[np.isfinite(distance_matrix)]
        
        if len(finite_distances) == 0:
            return 0

        # Get min and max distances
        dmin = np.min(finite_distances)
        dmax = np.max(finite_distances)

        # Avoid division by zero
        if dmax == 0:
            return 0

        return dmin / dmax

    def compute_voronoi_metrics(points):
        """Compute Voronoi-based metrics for local optimization."""
        try:
            vor = Voronoi(points)
            # Calculate average Voronoi cell area (more uniform = better)
            areas = []
            for i in range(len(points)):
                region = vor.regions[vor.point_region[i]]
                if -1 not in region and len(region) > 0:
                    # Simple polygon area calculation
                    vertices = [vor.vertices[j] for j in region if j >= 0]
                    if len(vertices) >= 3:
                        # Simplified area calculation using shoelace formula
                        x = [v[0] for v in vertices]
                        y = [v[1] for v in vertices]
                        area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(len(x)-1)))
                        areas.append(area)
            if areas:
                return np.mean(areas)
            return 0
        except:
            return 0

    def generate_sphere_packing_initial():
        """Generate initial points using sphere packing principles."""
        # Use a modified hexagonal lattice with optimized spacing
        points = []
        
        # Create a 4x4 grid with hexagonal offset
        rows, cols = 4, 4
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = np.sqrt(3) / 2 / (rows - 1) if rows > 1 else 1.0
        
        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x / 2
                y = i * spacing_y
                points.append([x, y])
        
        # Convert to numpy array
        points = np.array(points[:16])  # Ensure exactly 16 points
        
        # Add slight perturbations to break perfect symmetry
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        points = np.clip(points, 0, 1)
        
        return points

    def generate_diverse_initial_population(n_individuals=10):
        """Generate diverse initial population using multiple strategies."""
        population = []
        
        for i in range(n_individuals):
            np.random.seed(i * 100 + 42)
            
            if i == 0:
                # Sphere packing initialization
                individual = generate_sphere_packing_initial()
            elif i == 1:
                # Random with boundary padding
                individual = np.random.rand(16, 2) * 0.9 + 0.05
            elif i == 2:
                # Grid with jitter
                individual = []
                for k in range(4):
                    for l in range(4):
                        x = (k + 0.5) / 4.0 + np.random.normal(0, 0.02)
                        y = (l + 0.5) / 4.0 + np.random.normal(0, 0.02)
                        individual.append([x, y])
                individual = np.clip(np.array(individual), 0, 1)
            else:
                # Random distribution
                individual = np.random.rand(16, 2)
            
            population.append(individual)
        
        return population

    def geometric_mutation(individual, mutation_rate=0.1):
        """Apply geometric-based mutation that respects spatial constraints."""
        mutated = individual.copy()
        
        # Select subset of points to mutate
        n_mutate = int(len(individual) * mutation_rate)
        indices = np.random.choice(len(individual), n_mutate, replace=False)
        
        for idx in indices:
            # Use a combination of local neighborhood and global influence
            # Add small Gaussian noise with adaptive scale
            scale = 0.01 + 0.02 * np.random.random()
            
            # Mutate in a way that encourages better distribution
            delta = np.random.normal(0, scale, 2)
            
            # Apply mutation with boundary checking
            mutated[idx] += delta
            mutated[idx] = np.clip(mutated[idx], 0, 1)
            
            # Additional constraint: try to maintain distance to neighbors
            if idx > 0 and idx < len(individual) - 1:
                # Simple neighbor avoidance
                neighbor_idx = np.random.choice([idx-1, idx+1])
                dist = np.linalg.norm(mutated[idx] - mutated[neighbor_idx])
                if dist < 0.05:  # Too close to neighbor
                    # Push away from neighbor
                    direction = mutated[idx] - mutated[neighbor_idx]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        mutated[idx] += direction * 0.01
            
        return mutated

    def selection(population, fitness_scores):
        """Tournament selection with diversity preservation."""
        selected = []
        tournament_size = 3
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        # Take top performers
        top_performers = sorted_indices[:len(population)//2]
        selected.extend([population[i] for i in top_performers])
        
        # Fill remaining with diversity-based selection
        remaining = len(population) - len(selected)
        if remaining > 0:
            # Select randomly from top performers with bias towards diversity
            for _ in range(remaining):
                # Pick random parent
                parent_idx = np.random.choice(top_performers)
                selected.append(population[parent_idx].copy())
        
        return selected

    def crossover(parent1, parent2):
        """Geometric crossover that preserves point relationships."""
        # Uniform crossover with geometric constraints
        child = parent1.copy()
        
        # Determine crossover points
        mask = np.random.rand(16) > 0.5
        child[mask] = parent2[mask]
        
        # Apply boundary correction
        child = np.clip(child, 0, 1)
        
        return child

    def adaptive_evolution():
        """Main evolutionary algorithm with sphere packing constraints."""
        np.random.seed(42)
        
        # Phase 1: Initialize population
        population = generate_diverse_initial_population(20)
        
        best_individual = None
        best_fitness = 0
        stagnation_counter = 0
        max_stagnation = 15
        
        # Evolution loop
        for generation in range(50):
            # Evaluate fitness of population
            fitness_scores = []
            for individual in population:
                fitness = compute_min_max_ratio(individual)
                fitness_scores.append(fitness)
            
            fitness_scores = np.array(fitness_scores)
            
            # Track best solution
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Early termination if no improvement
            if stagnation_counter > max_stagnation:
                break
            
            # Selection
            selected = selection(population, fitness_scores)
            
            # Create new population through crossover and mutation
            new_population = []
            
            # Elitism: keep best individuals
            elite_size = 3
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < len(population):
                # Tournament selection for parents
                parent1_idx = np.random.choice(len(selected))
                parent2_idx = np.random.choice(len(selected))
                
                parent1 = selected[parent1_idx]
                parent2 = selected[parent2_idx]
                
                # Crossover
                child = crossover(parent1, parent2)
                
                # Mutation with geometric constraints
                mutation_prob = 0.3 if generation < 20 else 0.1
                if np.random.random() < mutation_prob:
                    child = geometric_mutation(child)
                
                new_population.append(child)
            
            population = new_population[:len(population)]
        
        return best_individual if best_individual is not None else population[0]

    def local_geometry_improvement(points):
        """Refine solution using local Voronoi-based geometry optimization."""
        # Try several local refinements
        current_points = points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        
        # Try to improve by moving points to better positions
        for iter in range(20):  # Limited iterations
            improved = False
            new_points = current_points.copy()
            
            # For each point, try to improve its position
            for i in range(len(current_points)):
                # Save original position
                original_pos = current_points[i].copy()
                
                # Try small movements
                for _ in range(10):  # Multiple tries per point
                    # Small random perturbation
                    perturbation = np.random.normal(0, 0.005, 2)
                    new_pos = original_pos + perturbation
                    new_pos = np.clip(new_pos, 0, 1)
                    
                    # Test if improvement
                    test_points = new_points.copy()
                    test_points[i] = new_pos
                    
                    ratio = compute_min_max_ratio(test_points)
                    if ratio > best_ratio:
                        new_points[i] = new_pos
                        best_ratio = ratio
                        improved = True
                        
            if not improved:
                break
                
            current_points = new_points.copy()
            
        return current_points

    # Main execution
    # Phase 1: Evolutionary optimization
    evolved_solution = adaptive_evolution()
    
    # Phase 2: Local geometry improvement
    final_solution = local_geometry_improvement(evolved_solution)
    
    # Final validation and refinement
    final_ratio = compute_min_max_ratio(final_solution)
    
    # Try a few more variants to ensure quality
    np.random.seed(1000)
    for _ in range(3):
        # Generate a random variation with better spread
        variation = evolved_solution.copy()
        noise = np.random.normal(0, 0.005, variation.shape)
        variation += noise
        variation = np.clip(variation, 0, 1)
        
        ratio = compute_min_max_ratio(variation)
        if ratio > final_ratio:
            final_ratio = ratio
            final_solution = variation.copy()
    
    return final_solution

# EVOLVE-BLOCK-END