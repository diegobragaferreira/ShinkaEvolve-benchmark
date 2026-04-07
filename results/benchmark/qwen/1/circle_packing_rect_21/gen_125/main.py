# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import random
import time
from collections import defaultdict

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Rectangle dimensions (width + height = 2)
    rect_width = 1.0
    rect_height = 1.0

    # Number of circles
    n = 21

    def generate_initial_patterns(width, height, n):
        """Generate multiple initial patterns and return the best one"""
        patterns = []
        
        # Hexagonal packing
        hex_pattern = generate_hexagonal_pattern(width, height, n)
        patterns.append(hex_pattern)
        
        # Grid-based packing
        grid_pattern = generate_grid_pattern(width, height, n)
        patterns.append(grid_pattern)
        
        # Voronoi-inspired pattern
        voronoi_pattern = generate_voronoi_pattern(width, height, n)
        patterns.append(voronoi_pattern)
        
        # Random with constraints
        random_pattern = generate_random_constrained_pattern(width, height, n)
        patterns.append(random_pattern)
        
        # Evaluate all patterns and select best
        best_pattern = None
        best_fitness = -float('inf')
        
        for pattern in patterns:
            fitness = evaluate_fitness(pattern, width, height)
            if fitness > best_fitness:
                best_fitness = fitness
                best_pattern = pattern
                
        return best_pattern if best_pattern is not None else generate_hexagonal_pattern(width, height, n)

    def generate_hexagonal_pattern(width, height, n):
        """Generate initial hexagonal packing pattern"""
        circles = np.zeros((n, 3))

        # Determine grid parameters
        rows = int(np.sqrt(n))
        cols = int(np.ceil(n / rows))

        # Calculate spacing
        margin = 0.05
        max_radius = min(width, height) * 0.08

        # Create hexagonal grid
        x_spacing = max_radius * 2.5
        y_spacing = max_radius * 2.165  # sqrt(3)/2 * 2

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = margin + j * x_spacing
                y = margin + i * y_spacing

                if i % 2 == 1:
                    x += x_spacing / 2

                # Adjust for bounds
                x = max(max_radius, min(width - max_radius, x))
                y = max(max_radius, min(height - max_radius, y))

                circles[idx] = [x, y, max_radius]
                idx += 1

        return circles

    def generate_grid_pattern(width, height, n):
        """Generate initial grid pattern"""
        circles = np.zeros((n, 3))

        # Find grid dimensions
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        # Calculate spacing
        margin = 0.05
        cell_width = (width - 2 * margin) / cols
        cell_height = (height - 2 * margin) / rows
        max_radius = min(cell_width, cell_height) * 0.4

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = margin + j * cell_width + cell_width / 2
                y = margin + i * cell_height + cell_height / 2
                circles[idx] = [x, y, max_radius]
                idx += 1

        return circles

    def generate_voronoi_pattern(width, height, n):
        """Generate initial pattern inspired by Voronoi diagrams"""
        circles = np.zeros((n, 3))
        
        # Place some circles at key positions
        key_positions = [
            (0.2 * width, 0.2 * height),
            (0.8 * width, 0.2 * height),
            (0.2 * width, 0.8 * height),
            (0.8 * width, 0.8 * height),
            (width / 2, height / 2),
            (0.1 * width, height / 2),
            (0.9 * width, height / 2),
            (width / 2, 0.1 * height),
            (width / 2, 0.9 * height)
        ]
        
        placed = 0
        for x, y in key_positions:
            if placed >= n:
                break
            r = min(x, y, width - x, height - y) * 0.15
            circles[placed] = [x, y, r]
            placed += 1
            
        # Fill remaining with strategic random placement
        attempts = 0
        while placed < n and attempts < 1000:
            x = np.random.uniform(0.05 * width, 0.95 * width)
            y = np.random.uniform(0.05 * height, 0.95 * height)
            
            # Check minimum distance to existing circles
            min_dist = float('inf')
            for i in range(placed):
                existing_x, existing_y, existing_r = circles[i]
                dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                min_dist = min(min_dist, dist)
            
            if min_dist > 0.1 * min(width, height) and min_dist > 0:
                r = min(x, y, width - x, height - y) * 0.1
                circles[placed] = [x, y, r]
                placed += 1
                
            attempts += 1
            
        # Fill remaining with random circles
        for i in range(placed, n):
            x = np.random.uniform(0.05 * width, 0.95 * width)
            y = np.random.uniform(0.05 * height, 0.95 * height)
            r = np.random.uniform(0.005, min(x, y, width - x, height - y) * 0.15)
            circles[i] = [x, y, r]
            
        return circles

    def generate_random_constrained_pattern(width, height, n):
        """Generate random pattern with basic constraints"""
        circles = np.zeros((n, 3))
        max_radius = min(width, height) * 0.08
        
        # Precompute some random values to avoid repeated calls
        random_x = np.random.uniform(0, width, n)
        random_y = np.random.uniform(0, height, n)
        random_r = np.random.uniform(0.005, max_radius, n)
        
        for i in range(n):
            x = random_x[i]
            y = random_y[i]
            radius = random_r[i]
            
            # Check if this circle overlaps with existing ones
            valid = True
            for j in range(i):
                existing_x, existing_y, existing_r = circles[j]
                dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                if dist < (radius + existing_r):
                    valid = False
                    break

            if valid:
                circles[i] = [x, y, radius]
            else:
                # If failed, try again with different random values
                attempts = 0
                while not valid and attempts < 100:
                    x = np.random.uniform(max_radius, width - max_radius)
                    y = np.random.uniform(max_radius, height - max_radius)
                    radius = np.random.uniform(0.005, max_radius)
                    
                    valid = True
                    for j in range(i):
                        existing_x, existing_y, existing_r = circles[j]
                        dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                        if dist < (radius + existing_r):
                            valid = False
                            break
                    attempts += 1
                    
                if valid:
                    circles[i] = [x, y, radius]
                else:
                    # If still failing, just place at random valid position
                    circles[i] = [np.random.uniform(max_radius, width - max_radius), 
                                  np.random.uniform(max_radius, height - max_radius),
                                  np.random.uniform(0.005, max_radius)]

        return circles

    def evaluate_fitness(individual, width, height):
        """Evaluate fitness of an individual - sum of radii with penalty for violations"""
        circles = individual.copy()
        total_radius = np.sum(circles[:, 2])

        # Penalty for boundary violations
        penalty = 0
        
        # Vectorized boundary check
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]
        
        # Check if any circle violates bounds
        boundary_violations = (x_coords - radii < 0) | (x_coords + radii > width) | \
                              (y_coords - radii < 0) | (y_coords + radii > height)
        if np.any(boundary_violations):
            penalty -= 1000 * np.sum(boundary_violations)

        # Penalty for overlaps using spatial hashing for efficiency
        if len(circles) > 1:
            # Spatial hash grid approach for collision detection
            try:
                # Use spatial grid for O(n) collision detection instead of O(n^2)
                collisions = detect_collisions_spatial_hash(circles, width, height)
                if collisions > 0:
                    penalty -= 1000 * collisions
            except:
                # Fallback to classic O(n^2) method if spatial hash fails
                coords = circles[:, :2]
                radii = circles[:, 2]
                distances = cdist(coords, coords)
                # Create mask for upper triangle (avoid double counting)
                mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
                # Compute overlap penalties
                overlap_distances = distances[mask]
                overlap_radii = (radii[:, None] + radii[None, :])[mask]
                overlaps = overlap_distances < overlap_radii
                if np.any(overlaps):
                    overlap_penalty = -np.sum(overlap_radii[overlaps] - overlap_distances[overlaps]) * 100
                    penalty += overlap_penalty

        return total_radius + penalty

    def detect_collisions_spatial_hash(circles, width, height, grid_size=None):
        """Detect collisions using spatial hashing for O(n) complexity"""
        if grid_size is None:
            # Estimate grid size based on average radius
            avg_radius = np.mean(circles[:, 2])
            grid_size = max(avg_radius, 0.001) * 2
            
        # Create spatial hash grid
        grid = defaultdict(list)
        cell_size = grid_size
        
        # Hash each circle to its grid cell
        for i, (x, y, r) in enumerate(circles):
            # Only consider circles that are within bounds
            if x >= r and x <= width - r and y >= r and y <= height - r:
                cell_row = int(y // cell_size)
                cell_col = int(x // cell_size)
                grid[(cell_row, cell_col)].append(i)
                
                # Also add to neighboring cells to handle edge cases
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        neighbor_cell = (cell_row + dr, cell_col + dc)
                        if neighbor_cell != (cell_row, cell_col):
                            grid[neighbor_cell].append(i)
        
        # Check for collisions within each cell and neighbors
        collisions = 0
        checked_pairs = set()
        
        for cell, indices in grid.items():
            if len(indices) > 1:
                # Check all pairs in this cell
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        # Skip if already checked
                        if (idx1, idx2) in checked_pairs or (idx2, idx1) in checked_pairs:
                            continue
                            
                        # Check actual distance
                        x1, y1, r1 = circles[idx1]
                        x2, y2, r2 = circles[idx2]
                        
                        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if dist < (r1 + r2):
                            collisions += 1
                            checked_pairs.add((idx1, idx2))
                            
        return collisions

    def get_voronoi_criticality(individual):
        """Calculate criticality based on minimum distance to neighbors"""
        circles = individual.copy()
        n = len(circles)
        
        if n <= 1:
            return np.ones(n) * 0.01
            
        # Vectorized computation of distances to all others
        coords = circles[:, :2]
        distances = cdist(coords, coords)
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)
        
        # Minimum distances for each circle
        min_distances = np.min(distances, axis=1)
        
        # Criticality is inverse of minimum distance
        # Add small epsilon to avoid division by zero
        criticality_scores = 1.0 / (min_distances + 1e-8)
        
        # Also consider boundary constraints
        for i in range(n):
            x, y, r = circles[i]
            # Distance to nearest boundary
            min_boundary_dist = min(x, y, 1-x, 1-y)
            # If very close to boundary, increase criticality
            if min_boundary_dist < 0.05:
                criticality_scores[i] *= (1 + 5 * (0.05 - min_boundary_dist))
        
        # Normalize
        if np.max(criticality_scores) > 0:
            criticality_scores = criticality_scores / np.max(criticality_scores)
            
        # Ensure minimum values
        criticality_scores = np.maximum(criticality_scores, 0.01)
        
        return criticality_scores

    def mut_radius(individual, indpb=0.25):
        """Mutation operator that modifies only the radius of selected circles with adaptive strength"""
        mutated_individual = individual.copy()
        n = len(mutated_individual)

        # Get criticality scores
        criticality = get_voronoi_criticality(mutated_individual)

        # Sort by criticality (most critical first)
        sorted_indices = np.argsort(-criticality)  # Descending order

        # Mutate top 40% of critical circles (focus on the most constrained)
        num_mutations = int(n * 0.4)
        mutation_indices = sorted_indices[:num_mutations]

        for i in range(num_mutations):
            idx = mutation_indices[i]
            if random.random() < indpb:
                old_radius = mutated_individual[idx, 2]

                # Adaptive mutation strength based on criticality
                # High criticality (constrained) = small mutation
                # Low criticality (loosely constrained) = large mutation
                adaptive_strength = 0.003 * (1.0 / (criticality[idx] + 0.001))
                adaptive_strength = min(adaptive_strength, 0.02)  # Cap maximum mutation

                # Small random change to radius
                delta = np.random.normal(0, adaptive_strength)
                new_radius = max(0.001, old_radius + delta)
                mutated_individual[idx, 2] = new_radius

        return mutated_individual,

    def crossover(parent1, parent2):
        """Crossover operator that exchanges radii of most critical circles"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Get criticality scores for both parents
        crit1 = get_voronoi_criticality(parent1)
        crit2 = get_voronoi_criticality(parent2)

        # Exchange radii of circles with highest criticality
        combined_criticality = np.maximum(crit1, crit2)
        sorted_indices = np.argsort(-combined_criticality)

        # Exchange radii for top 40% of circles (larger percentage than before)
        num_exchanges = int(len(parent1) * 0.4)
        for i in range(num_exchanges):
            idx = sorted_indices[i]
            child1[idx, 2], child2[idx, 2] = child2[idx, 2], child1[idx, 2]

        return child1, child2

    def is_valid_solution(circles, width, height):
        """Check if solution is valid - faster version using spatial hash"""
        # Check boundary constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                return False

        # Check overlap constraints using spatial hash
        if len(circles) > 1:
            collisions = detect_collisions_spatial_hash(circles, width, height)
            return collisions == 0

        return True

    def optimize_boundaries(circles, width, height):
        """Focus on edge-constrained circles for boundary refinement"""
        # Identify circles near boundaries
        edge_threshold = 0.05
        edge_circles = []
        
        for i in range(len(circles)):
            x, y, r = circles[i]
            if (x < edge_threshold + r or x > width - edge_threshold - r or 
                y < edge_threshold + r or y > height - edge_threshold - r):
                edge_circles.append(i)
        
        if not edge_circles:
            return circles
            
        # Refine edge circles with more aggressive boundary adjustments
        refined = circles.copy()
        
        for idx in edge_circles:
            x, y, r = refined[idx]
            
            # Adjust to keep within bounds
            if x - r < 0:
                x = r
            elif x + r > width:
                x = width - r
                
            if y - r < 0:
                y = r
            elif y + r > height:
                y = height - r
                
            refined[idx] = [x, y, r]
            
        return refined

    def local_refinement(circles, width, height):
        """Apply local refinement to improve packing quality"""
        refined = circles.copy()
        
        # Try to increase radii of unconstrained circles
        for iteration in range(50):
            improved = False
            # Focus on circles with low criticality (more room to grow)
            criticality = get_voronoi_criticality(refined)
            sorted_indices = np.argsort(criticality)  # Ascending order (low criticality first)
            
            # Try to expand top 10 least constrained circles
            for i in range(min(10, len(refined))):
                idx = sorted_indices[i]
                old_x, old_y, old_r = refined[idx]
                
                max_radius = min(old_x, width - old_x, old_y, height - old_y) - 0.001
                
                # Check neighbor collisions
                for j in range(len(refined)):
                    if i != j:
                        ox, oy, oradius = refined[j]
                        dist = np.sqrt((old_x - ox)**2 + (old_y - oy)**2)
                        if dist < (old_r + oradius):
                            # Compute max possible radius to avoid collision
                            collision_radius = dist - oradius - 0.001
                            if collision_radius > 0:
                                max_radius = min(max_radius, collision_radius)
                
                # Try to increase radius
                if max_radius > old_r and max_radius > 0.001:
                    # Adaptive expansion based on criticality
                    expansion_factor = 0.5 + 0.5 * (1.0 - criticality[idx])  # More expansion for less critical
                    new_radius = min(max_radius, old_r + (max_radius - old_r) * expansion_factor * 0.1)
                    if new_radius > old_r:
                        refined[idx] = [old_x, old_y, new_radius]
                        improved = True
            
            if not improved:
                break
                
        return refined

    # Main algorithm - improved version
    start_time = time.time()

    # Phase 1: Initialize with best pattern
    best_individual = generate_initial_patterns(rect_width, rect_height, n)

    # Phase 2: Evolutionary optimization with improved parameters
    population_size = 25
    generations = 200
    elite_size = 5
    tournament_size = 4
    
    # Generate initial population
    population = [best_individual]
    for _ in range(population_size - 1):
        # Add variation to the best solution
        variant = best_individual.copy()
        # Small random perturbation
        for i in range(len(variant)):
            if np.random.random() < 0.1:
                variant[i, 0] += np.random.normal(0, 0.01)
                variant[i, 1] += np.random.normal(0, 0.01)
                variant[i, 2] += np.random.normal(0, 0.005)
        population.append(variant)

    # Evolutionary loop
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_fitness(ind, rect_width, rect_height) for ind in population]
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Keep elite
        elite = population[:elite_size]
        
        # Generate new population
        new_population = elite[:]
        
        # Create offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = sorted_indices[np.random.choice(min(tournament_size, len(sorted_indices)))]
            parent2_idx = sorted_indices[np.random.choice(min(tournament_size, len(sorted_indices)))]
            
            parent1 = population[parent1_idx].copy()
            parent2 = population[parent2_idx].copy()
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutate
            child1 = mut_radius(child1, 0.25)[0]
            child2 = mut_radius(child2, 0.25)[0]
            
            # Repair
            child1 = optimize_boundaries(child1, rect_width, rect_height)
            child2 = optimize_boundaries(child2, rect_width, rect_height)
            
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
        
        # Early stopping if no improvement
        if gen > 0 and abs(fitness_scores[0] - previous_fitness) < 1e-6:
            stagnation_count += 1
            if stagnation_count > 20:
                break
        else:
            stagnation_count = 0
        previous_fitness = fitness_scores[0]

    # Phase 3: Local optimization on best solution
    best_individual = population[0].copy()
    
    # Multiple rounds of refinement
    for _ in range(3):
        # Local refinement
        best_individual = local_refinement(best_individual, rect_width, rect_height)
        
        # Boundary-focused refinement
        best_individual = optimize_boundaries(best_individual, rect_width, rect_height)
        
        # Final local search
        for _ in range(100):
            test_individual = best_individual.copy()
            
            # Focus on most critical circles
            criticality = get_voronoi_criticality(test_individual)
            sorted_indices = np.argsort(-criticality)
            
            # Perturb top 10 circles
            for i in range(min(10, len(test_individual))):
                idx = sorted_indices[i]
                old_x, old_y, old_r = test_individual[idx]
                
                # Make small adjustments
                new_x = max(0.005, min(0.995, old_x + np.random.normal(0, 0.003)))
                new_y = max(0.005, min(0.995, old_y + np.random.normal(0, 0.003)))
                new_r = max(0.001, old_r + np.random.normal(0, 0.001))
                
                test_individual[idx] = [new_x, new_y, new_r]
            
            # Validate and accept improvement
            if evaluate_fitness(test_individual, rect_width, rect_height) > evaluate_fitness(best_individual, rect_width, rect_height):
                best_individual = test_individual.copy()

    # Final validation
    if not is_valid_solution(best_individual, rect_width, rect_height):
        # Fallback to structured pattern
        best_individual = generate_hexagonal_pattern(rect_width, rect_height, n)

    return best_individual

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")