# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import random
from copy import deepcopy
from typing import Tuple, List
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    n_circles = 26
    max_generations = 1000
    population_size = 50

    class SpatialIndexer:
        """Efficient spatial indexing for overlap detection using grid-based approach."""
        def __init__(self, resolution=30):
            self.resolution = resolution
            self.grid = {}
            
        def clear(self):
            self.grid.clear()
            
        def add_circle(self, idx, x, y, r):
            """Add circle to spatial grid."""
            # Calculate bounds of circle in grid coordinates
            min_x = max(0, int((x - r) * self.resolution))
            max_x = min(self.resolution - 1, int((x + r) * self.resolution))
            min_y = max(0, int((y - r) * self.resolution))
            max_y = min(self.resolution - 1, int((y + r) * self.resolution))
            
            # Add to all relevant grid cells
            for gx in range(min_x, max_x + 1):
                for gy in range(min_y, max_y + 1):
                    if (gx, gy) not in self.grid:
                        self.grid[(gx, gy)] = []
                    self.grid[(gx, gy)].append((idx, x, y, r))
        
        def get_candidates(self, x, y, r):
            """Get candidate circles from neighboring grid cells."""
            candidates = []
            min_x = max(0, int((x - r) * self.resolution))
            max_x = min(self.resolution - 1, int((x + r) * self.resolution))
            min_y = max(0, int((y - r) * self.resolution))
            max_y = min(self.resolution - 1, int((y + r) * self.resolution))
            
            for gx in range(min_x, max_x + 1):
                for gy in range(min_y, max_y + 1):
                    if (gx, gy) in self.grid:
                        candidates.extend(self.grid[(gx, gy)])
            return candidates

    def generate_voronoi_seeds(n_points, n_circles):
        """Generate initial points using a Voronoi-like approach to ensure good distribution."""
        # Generate a set of points using a modified Delaunay triangulation approach
        points = []
        # Use a hexagonal grid pattern with some randomness 
        grid_size = int(np.ceil(np.sqrt(n_circles))) + 2
        spacing = 1.0 / (grid_size + 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) < n_points:
                    # Offset every other row for better coverage
                    x = (i + 1) * spacing + np.random.uniform(-spacing*0.2, spacing*0.2)
                    y = (j + 1) * spacing + np.random.uniform(-spacing*0.2, spacing*0.2)
                    # Ensure within bounds
                    x = max(spacing, min(1-spacing, x))
                    y = max(spacing, min(1-spacing, y))
                    points.append((x, y))
        
        # Trim to desired number of points
        points = points[:n_circles]
        
        # If we don't have enough, add random points
        while len(points) < n_circles:
            points.append((np.random.random(), np.random.random()))
            
        return np.array(points[:n_circles])

    def compute_optimal_radii(positions, n_circles):
        """Compute initial optimal radii for circles based on their spatial relationships."""
        radii = np.zeros(n_circles)
        
        # For each circle, compute the maximum radius that fits
        for i in range(n_circles):
            x, y = positions[i]
            
            # Boundaries
            boundary_radius = min(x, 1-x, y, 1-y)
            
            # Neighbors
            min_neighbor_dist = float('inf')
            for j in range(n_circles):
                if i != j:
                    dist = np.sqrt((x - positions[j][0])**2 + (y - positions[j][1])**2)
                    min_neighbor_dist = min(min_neighbor_dist, dist)
            
            # Compute radius as half the minimum distance to neighbors, 
            # but limited by boundary constraints
            if min_neighbor_dist < float('inf'):
                neighbor_radius = min_neighbor_dist / 2.0
                radii[i] = min(boundary_radius, neighbor_radius, 0.1)
            else:
                radii[i] = min(boundary_radius, 0.1)
                
            # Ensure minimum radius
            radii[i] = max(0.005, radii[i])
            
        return radii

    def initialize_population(pop_size, n_circles):
        """Initialize population with Voronoi-inspired distribution."""
        population = []
        
        # Generate Voronoi-like seeds
        seeds = generate_voronoi_seeds(2*n_circles, n_circles)
        
        for _ in range(pop_size):
            # Start with Voronoi seeds
            positions = seeds.copy()
            
            # Add randomness to positions
            for i in range(len(positions)):
                # Apply moderate perturbation
                positions[i][0] += np.random.normal(0, 0.03)
                positions[i][1] += np.random.normal(0, 0.03)
                
                # Clamp to valid range
                positions[i][0] = max(0.01, min(0.99, positions[i][0]))
                positions[i][1] = max(0.01, min(0.99, positions[i][1]))
            
            # Compute radii
            radii = compute_optimal_radii(positions, n_circles)
            
            # Create circles array
            circles = np.column_stack([positions, radii])
            
            # Ensure validity
            circles = ensure_validity(circles)
            
            population.append(circles)
            
        return population

    def ensure_validity(circles):
        """Ensure all circles are within bounds."""
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Clamp positions to maintain containment
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            circles[i] = [x, y, r]
        return circles

    def is_valid(circles):
        """Check if circles satisfy all constraints."""
        n = len(circles)
        
        # Check containment first
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
                
        # Check overlaps using spatial indexing
        indexer = SpatialIndexer()
        
        # Add all circles to grid
        for i in range(n):
            x, y, r = circles[i]
            indexer.add_circle(i, x, y, r)
            
        # Check overlaps
        for i in range(n):
            x1, y1, r1 = circles[i]
            candidates = indexer.get_candidates(x1, y1, r1)
            
            for _, x2, y2, r2 in candidates:
                if i != _:
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return False
                        
        return True

    def evaluate_fitness(circles, generation=0):
        """Evaluate fitness with progressive constraint weighting."""
        if not is_valid(circles):
            # Progressive weighting: more focus on constraints in early generations
            constraint_weight = 1.0 if generation > max_generations * 0.3 else 1.0 + (1.0 - generation/max_generations) * 3.0
            
            penalty = 0
            
            # Boundary violations
            for i in range(len(circles)):
                x, y, r = circles[i]
                penalty += max(0, r - x) * 1000 * constraint_weight  # Left boundary
                penalty += max(0, x + r - 1) * 1000 * constraint_weight  # Right boundary
                penalty += max(0, r - y) * 1000 * constraint_weight  # Bottom boundary
                penalty += max(0, y + r - 1) * 1000 * constraint_weight  # Top boundary
            
            # Overlap violations
            indexer = SpatialIndexer()
            for i in range(len(circles)):
                x, y, r = circles[i]
                indexer.add_circle(i, x, y, r)
                
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                candidates = indexer.get_candidates(x1, y1, r1)
                
                for _, x2, y2, r2 in candidates:
                    if i != _:
                        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        overlap = max(0, r1 + r2 - dist)
                        penalty += overlap * 20000 * constraint_weight
                        
            return -penalty
            
        # Valid configuration - maximize sum of radii
        return np.sum(circles[:, 2])

    def mutate(circles, generation, max_generations):
        """Enhanced mutation with position-radius coupling."""
        mutated = deepcopy(circles)
        # Adaptive mutation rate that decreases over time
        mutation_rate = 0.15 * (1.0 - generation / max_generations)
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Choose mutation type with bias towards position
                mutation_type = random.choices(
                    ['position', 'radius', 'both'], 
                    weights=[0.6, 0.3, 0.1]
                )[0]
                
                if mutation_type == 'position':
                    # Mutate position with adaptive step size
                    step_size = 0.03 * (1.0 - generation / max_generations * 0.8)
                    mutated[i, 0] += np.random.normal(0, step_size)
                    mutated[i, 1] += np.random.normal(0, step_size)
                    mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0]))
                    mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1]))
                    
                elif mutation_type == 'radius':
                    # Mutate radius
                    mutated[i, 2] += np.random.normal(0, 0.01)
                    mutated[i, 2] = max(0.001, mutated[i, 2])
                    
                elif mutation_type == 'both':
                    # Mutate position and radius together
                    step_size = 0.02 * (1.0 - generation / max_generations * 0.8)
                    mutated[i, 0] += np.random.normal(0, step_size)
                    mutated[i, 1] += np.random.normal(0, step_size)
                    mutated[i, 2] += np.random.normal(0, 0.008)
                    mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0]))
                    mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1]))
                    mutated[i, 2] = max(0.001, mutated[i, 2])
                    
        return mutated

    def crossover(parent1, parent2):
        """Enhanced crossover with partial matching."""
        child = deepcopy(parent1)
        n = len(parent1)
        
        # Uniform crossover with preference for preserving good features
        for i in range(n):
            if random.random() < 0.5:
                child[i] = parent2[i].copy()
                
        return child

    def tournament_selection(population, k=3):
        """Tournament selection with fitness evaluation."""
        selected = random.sample(population, k)
        # Return the best among selected
        return max(selected, key=lambda x: evaluate_fitness(x))

    # Initialize population
    population = initialize_population(population_size, n_circles)
    
    # Evolution loop
    start_time = time.time()
    best_fitness = float('-inf')
    best_solution = None
    
    for generation in range(max_generations):
        # Evaluate fitness for entire population
        fitness_scores = [evaluate_fitness(individual, generation) for individual in population]
        
        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_solution = deepcopy(population[max_fitness_idx])
        
        # Early termination
        if time.time() - start_time > 55:
            break
            
        # Elitism: keep top 15%
        elite_count = max(1, population_size // 7)
        sorted_indices = np.argsort(fitness_scores)[::-1][:elite_count]
        elite = [population[i] for i in sorted_indices]
        
        # Generate new population
        new_population = deepcopy(elite)
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            mutated_child = mutate(child, generation, max_generations)
            
            new_population.append(mutated_child)
            
        population = new_population[:population_size]
    
    # Return best solution found
    if best_solution is not None:
        return best_solution
    else:
        return population[0]

# EVOLVE-BLOCK-END