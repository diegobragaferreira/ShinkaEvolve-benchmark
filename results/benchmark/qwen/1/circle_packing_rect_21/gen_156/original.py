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
        
        # Hexagonal packing with refined parameters
        hex_pattern = generate_hexagonal_pattern(width, height, n)
        patterns.append(hex_pattern)
        
        # Grid-based packing with balanced aspect ratio
        grid_pattern = generate_grid_pattern(width, height, n)
        patterns.append(grid_pattern)
        
        # Spiral pattern with optimal spacing
        spiral_pattern = generate_spiral_pattern(width, height, n)
        patterns.append(spiral_pattern)
        
        # Random with better overlap handling
        random_pattern = generate_random_constrained_pattern(width, height, n)
        patterns.append(random_pattern)
        
        # Boundary-focused pattern with strategic placements
        boundary_pattern = generate_boundary_focused_pattern(width, height, n)
        patterns.append(boundary_pattern)
        
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
        """Generate initial hexagonal packing pattern with precise spacing"""
        circles = np.zeros((n, 3))

        # Optimize grid parameters for hexagonal packing efficiency
        rows = int(np.sqrt(n * 1.1))
        cols = int(np.ceil(n / rows))

        # Precise margin and radius calculation
        margin = 0.04
        max_radius = min(width, height) * 0.07

        # Better hexagonal grid spacing (closer to ideal packing)
        x_spacing = max_radius * 2.1
        y_spacing = max_radius * 1.866  # sqrt(3) * max_radius

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = margin + j * x_spacing
                y = margin + i * y_spacing

                if i % 2 == 1:
                    x += x_spacing / 2

                # Strict bounds enforcement
                x = max(max_radius, min(width - max_radius, x))
                y = max(max_radius, min(height - max_radius, y))

                circles[idx] = [x, y, max_radius]
                idx += 1

        return circles

    def generate_grid_pattern(width, height, n):
        """Generate initial grid pattern with optimized cell sizing"""
        circles = np.zeros((n, 3))

        # Balanced aspect ratio grid
        cols = int(np.ceil(np.sqrt(n * 1.05)))
        rows = int(np.ceil(n / cols))

        # Precise spacing calculations
        margin = 0.04
        cell_width = (width - 2 * margin) / cols
        cell_height = (height - 2 * margin) / rows
        max_radius = min(cell_width, cell_height) * 0.45

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

    def generate_spiral_pattern(width, height, n):
        """Generate initial spiral pattern with improved distribution"""
        circles = np.zeros((n, 3))
        center_x, center_y = width / 2, height / 2
        max_radius = min(width, height) * 0.085
        angle_step = 2 * np.pi / 4.2
        radius_step = 0.04

        for i in range(n):
            angle = i * angle_step
            radius = i * radius_step
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)

            # Strict boundary enforcement
            x = max(max_radius, min(width - max_radius, x))
            y = max(max_radius, min(height - max_radius, y))

            circles[i] = [x, y, max_radius]

        return circles

    def generate_random_constrained_pattern(width, height, n):
        """Generate random pattern with enhanced constraint resolution"""
        circles = np.zeros((n, 3))
        max_radius = min(width, height) * 0.075
        
        # Optimized random sampling
        random_x = np.random.uniform(0, width, n)
        random_y = np.random.uniform(0, height, n)
        random_r = np.random.uniform(0.005, max_radius, n)
        
        for i in range(n):
            x = random_x[i]
            y = random_y[i]
            radius = random_r[i]
            
            # Check for overlaps efficiently
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
                # Retry with improved sampling strategy - reduce overlap likelihood
                attempts = 0
                while not valid and attempts < 200:
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
                    # Last resort: place at random valid position
                    circles[i] = [np.random.uniform(max_radius, width - max_radius), 
                                  np.random.uniform(max_radius, height - max_radius),
                                  np.random.uniform(0.005, max_radius)]

        return circles

    def generate_boundary_focused_pattern(width, height, n):
        """Generate pattern focused on strategic boundary placements"""
        circles = np.zeros((n, 3))
        
        # Place circles at key positions
        # Corner positions
        corners = [(0.1*width, 0.1*height), (0.9*width, 0.1*height), 
                   (0.1*width, 0.9*height), (0.9*width, 0.9*height)]
        
        # Edge centers
        edges = [(width/2, 0.1*height), (width/2, 0.9*height),
                 (0.1*width, height/2), (0.9*width, height/2)]
        
        # Center
        center = (width/2, height/2)
        
        # Place circles in sequence
        positions = [center] + corners + edges
        placed = 0
        
        for x, y in positions:
            if placed >= n:
                break
            # Calculate max allowable radius to stay within bounds
            max_radius = min(x, y, width-x, height-y) * 0.25
            radius = max_radius * 0.6
            circles[placed] = [x, y, radius]
            placed += 1
            
        # Fill remaining with random but constraint-aware placement
        for i in range(placed, n):
            # Sample from a distribution that favors less-constrained areas
            attempts = 0
            valid = False
            while not valid and attempts < 100:
                x = np.random.uniform(0.05*width, 0.95*width)
                y = np.random.uniform(0.05*height, 0.95*height)
                max_radius = min(x, y, width-x, height-y) * 0.2
                radius = np.random.uniform(0.005, max_radius)
                
                # Check overlap with all previously placed circles
                valid = True
                for j in range(i):
                    px, py, pr = circles[j]
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    if dist < (radius + pr):
                        valid = False
                        break
                        
                attempts += 1
                
            if valid:
                circles[i] = [x, y, radius]
            else:
                # Fallback to simple random placement with checks
                circles[i] = [np.random.uniform(0.05*width, 0.95*width), 
                              np.random.uniform(0.05*height, 0.95*height),
                              np.random.uniform(0.005, min(0.1*width, 0.1*height))]
            
        return circles

    def evaluate_fitness(individual, width, height):
        """Enhanced fitness evaluation with weighted penalties"""
        circles = individual.copy()
        total_radius = np.sum(circles[:, 2])

        # Heavy penalty for boundary violations
        penalty = 0
        
        # Vectorized boundary check
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]
        
        # Check if any circle violates bounds
        boundary_violations = (x_coords - radii < 0) | (x_coords + radii > width) | \
                              (y_coords - radii < 0) | (y_coords + radii > height)
        if np.any(boundary_violations):
            penalty -= 5000 * np.sum(boundary_violations)

        # Overlap penalty with improved calculation
        if len(circles) > 1:
            # Use spatial hash for efficiency
            try:
                collisions = detect_collisions_spatial_hash(circles, width, height)
                if collisions > 0:
                    penalty -= 3000 * collisions
            except:
                # Fallback to direct calculation
                coords = circles[:, :2]
                radii = circles[:, 2]
                distances = cdist(coords, coords)
                mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
                overlap_distances = distances[mask]
                overlap_radii = (radii[:, None] + radii[None, :])[mask]
                overlaps = overlap_distances < overlap_radii
                if np.any(overlaps):
                    overlap_penalty = -np.sum(overlap_radii[overlaps] - overlap_distances[overlaps]) * 50
                    penalty += overlap_penalty

        return total_radius + penalty

    def detect_collisions_spatial_hash(circles, width, height, grid_size=None):
        """Efficient collision detection using spatial hashing with optimal grid size"""
        if grid_size is None:
            avg_radius = np.mean(circles[:, 2])
            grid_size = max(avg_radius * 2.5, 0.001)
            
        grid = defaultdict(list)
        cell_size = grid_size
        
        # Hash circles into grid cells
        for i, (x, y, r) in enumerate(circles):
            if x >= r and x <= width - r and y >= r and y <= height - r:
                cell_row = int(y // cell_size)
                cell_col = int(x // cell_size)
                grid[(cell_row, cell_col)].append(i)
                
                # Include adjacent cells to catch boundary collisions
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        neighbor_cell = (cell_row + dr, cell_col + dc)
                        if neighbor_cell != (cell_row, cell_col):
                            grid[neighbor_cell].append(i)
        
        # Check for collisions
        collisions = 0
        checked_pairs = set()
        
        for cell, indices in grid.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        if (idx1, idx2) in checked_pairs or (idx2, idx1) in checked_pairs:
                            continue
                            
                        x1, y1, r1 = circles[idx1]
                        x2, y2, r2 = circles[idx2]
                        
                        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if dist < (r1 + r2):
                            collisions += 1
                            checked_pairs.add((idx1, idx2))
                            
        return collisions

    def get_voronoi_adaptive_criticality(individual):
        """Calculate criticality based on Voronoi cell sizes for adaptive mutation"""
        circles = individual.copy()
        n = len(circles)
        
        if n <= 1:
            return np.ones(n) * 0.01
            
        # Generate Voronoi diagram for better understanding of spatial relationships
        try:
            points = circles[:, :2]
            vor = Voronoi(points)
            
            # Calculate Voronoi cell areas for each circle
            voronoi_areas = []
            for i in range(n):
                # Get vertices of Voronoi cell for circle i
                region_indices = np.where(vor.point_region == i)[0]
                if len(region_indices) > 0:
                    # Get the region vertices
                    region_vertices = vor.vertices[vor.regions[region_indices[0]]]
                    if len(region_vertices) > 0:
                        # Calculate area using shoelace formula
                        area = 0.5 * abs(sum(
                            region_vertices[j][0] * region_vertices[(j+1)%len(region_vertices)][1] - 
                            region_vertices[(j+1)%len(region_vertices)][0] * region_vertices[j][1]
                            for j in range(len(region_vertices))
                        ))
                        voronoi_areas.append(area)
                    else:
                        voronoi_areas.append(float('inf'))
                else:
                    voronoi_areas.append(float('inf'))
                    
            # Convert to criticality (lower area = higher criticality)
            voronoi_areas = np.array(voronoi_areas)
            voronoi_areas[voronoi_areas == 0] = 1e-10  # Avoid division by zero
            voronoi_areas[voronoi_areas == float('inf')] = 1e10  # Handle invalid areas
            
            # Invert areas to get criticality (smaller areas = higher criticality)
            criticality_scores = 1.0 / voronoi_areas
            
        except:
            # Fallback to distance-based criticality if Voronoi fails
            coords = circles[:, :2]
            distances = cdist(coords, coords)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            criticality_scores = 1.0 / (min_distances + 1e-8)
            
        # Incorporate boundary effects
        for i in range(n):
            x, y, r = circles[i]
            # Distance to nearest boundary
            min_boundary_dist = min(x, y, 1-x, 1-y)
            # If very close to boundary, increase criticality
            if min_boundary_dist < 0.05:
                criticality_scores[i] *= (1 + 8 * (0.05 - min_boundary_dist))
            elif min_boundary_dist < 0.1:
                criticality_scores[i] *= (1 + 3 * (0.1 - min_boundary_dist))
        
        # Normalize and ensure minimum values
        if np.max(criticality_scores) > 0:
            criticality_scores = criticality_scores / np.max(criticality_scores)
        criticality_scores = np.maximum(criticality_scores, 0.01)
        
        return criticality_scores

    def mut_radius_adaptive(individual, indpb=0.2):
        """Adaptive mutation operator using Voronoi-based criticality"""
        mutated_individual = individual.copy()
        n = len(mutated_individual)

        # Get Voronoi-based criticality scores
        criticality = get_voronoi_adaptive_criticality(mutated_individual)

        # Sort by criticality (most critical first)
        sorted_indices = np.argsort(-criticality)

        # Mutate top 55% of critical circles (focus on the most constrained)
        num_mutations = int(n * 0.55)
        mutation_indices = sorted_indices[:num_mutations]

        for i in range(num_mutations):
            idx = mutation_indices[i]
            if random.random() < indpb:
                old_radius = mutated_individual[idx, 2]

                # Adaptive mutation strength inversely proportional to criticality
                # Circles in high-density regions (small Voronoi cells) get smaller mutations
                adaptive_strength = 0.004 * (1.0 / (criticality[idx] + 0.001))
                adaptive_strength = min(adaptive_strength, 0.02)  # Cap maximum mutation

                # Apply Gaussian noise with adaptive standard deviation
                delta = np.random.normal(0, adaptive_strength)
                new_radius = max(0.001, old_radius + delta)
                mutated_individual[idx, 2] = new_radius

        return mutated_individual,

    def crossover_adaptive(parent1, parent2):
        """Adaptive crossover focusing on critical circles"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Get criticality scores for both parents
        crit1 = get_voronoi_adaptive_criticality(parent1)
        crit2 = get_voronoi_adaptive_criticality(parent2)

        # Exchange radii of circles with highest combined criticality
        combined_criticality = np.maximum(crit1, crit2)
        sorted_indices = np.argsort(-combined_criticality)

        # Exchange radii for top 45% of circles
        num_exchanges = int(len(parent1) * 0.45)
        for i in range(num_exchanges):
            idx = sorted_indices[i]
            child1[idx, 2], child2[idx, 2] = child2[idx, 2], child1[idx, 2]

        return child1, child2

    def is_valid_solution(circles, width, height):
        """Check if solution is valid with optimized collision detection"""
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

    def adaptive_local_search(individual, width, height, max_iterations=300):
        """Local search with adaptive perturbation based on criticality"""
        best_individual = individual.copy()
        best_fitness = evaluate_fitness(best_individual, width, height)
        
        for iteration in range(max_iterations):
            test_individual = best_individual.copy()
            
            # Select circles based on criticality - focus more on critical ones
            criticality = get_voronoi_adaptive_criticality(best_individual)
            
            # Select 6-10 circles based on criticality (higher criticality = more likely to be selected)
            probabilities = criticality / np.sum(criticality)
            num_selected = max(6, min(10, int(np.random.exponential(5))))
            selected_indices = np.random.choice(len(test_individual), 
                                              num_selected, 
                                              replace=False, 
                                              p=probabilities)
            
            for idx in selected_indices:
                old_x, old_y, old_r = test_individual[idx]
                
                # Adaptive perturbation based on criticality
                # Critical circles have smaller perturbations
                adaptation_factor = 1.0 / (criticality[idx] + 0.001)
                adaptation_factor = min(adaptation_factor, 5.0)
                
                # Apply perturbations with adaptive magnitude
                new_x = max(0.005, min(0.995, old_x + np.random.normal(0, 0.005 * adaptation_factor)))
                new_y = max(0.005, min(0.995, old_y + np.random.normal(0, 0.005 * adaptation_factor)))
                new_r = max(0.001, old_r + np.random.normal(0, 0.002 * adaptation_factor))

                # Boundary check before detailed validation
                if new_x - new_r < 0 or new_x + new_r > width or \
                   new_y - new_r < 0 or new_y + new_r > height:
                    continue

                # Detailed overlap validation
                valid = True
                for other_idx in range(len(test_individual)):
                    if other_idx != idx:
                        ox, oy, oradius = test_individual[other_idx]
                        dist = np.sqrt((new_x - ox)**2 + (new_y - oy)**2)
                        if dist < (new_r + oradius):
                            valid = False
                            break

                if valid:
                    test_individual[idx] = [new_x, new_y, new_r]

            # Evaluate and accept improvement
            new_fitness = evaluate_fitness(test_individual, width, height)
            if new_fitness > best_fitness:
                best_individual = test_individual.copy()
                best_fitness = new_fitness

        return best_individual

    # Main algorithm - improved version
    start_time = time.time()

    # Stage 1: Multiple initialization strategies
    best_individual = generate_initial_patterns(rect_width, rect_height, n)

    # Stage 2: Global adaptive search with enhanced mutation
    for iteration in range(600):  # Increased iterations
        test_individual = best_individual.copy()

        # Select circles with probability weighted by criticality
        criticality = get_voronoi_adaptive_criticality(best_individual)
        
        # Use exponential distribution to select variable number of circles
        num_selected = max(5, min(12, int(np.random.exponential(6))))
        probabilities = criticality / np.sum(criticality)
        selected_indices = np.random.choice(len(test_individual), 
                                          num_selected, 
                                          replace=False, 
                                          p=probabilities)

        for idx in selected_indices:
            old_x, old_y, old_r = test_individual[idx]
            
            # Adaptive perturbation based on current criticality
            adaptation_factor = 1.0 / (criticality[idx] + 0.001)
            adaptation_factor = min(adaptation_factor, 5.0)
            
            # Apply perturbations
            new_x = max(0.005, min(0.995, old_x + np.random.normal(0, 0.008 * adaptation_factor)))
            new_y = max(0.005, min(0.995, old_y + np.random.normal(0, 0.008 * adaptation_factor)))
            new_r = max(0.001, old_r + np.random.normal(0, 0.004 * adaptation_factor))

            # Quick boundary check
            if new_x - new_r < 0 or new_x + new_r > rect_width or \
               new_y - new_r < 0 or new_y + new_r > rect_height:
                continue

            # Overlap check
            valid = True
            for other_idx in range(len(test_individual)):
                if other_idx != idx:
                    ox, oy, oradius = test_individual[other_idx]
                    dist = np.sqrt((new_x - ox)**2 + (new_y - oy)**2)
                    if dist < (new_r + oradius):
                        valid = False
                        break

            if valid:
                test_individual[idx] = [new_x, new_y, new_r]

        # Accept improvement
        old_fitness = evaluate_fitness(best_individual, rect_width, rect_height)
        new_fitness = evaluate_fitness(test_individual, rect_width, rect_height)

        if new_fitness > old_fitness:
            best_individual = test_individual.copy()

    # Early termination check
    if time.time() - start_time > 55:
        return best_individual

    # Stage 3: Adaptive local search
    best_individual = adaptive_local_search(best_individual, rect_width, rect_height, 400)

    # Stage 4: Final refinement with criticality-aware approach
    criticality = get_voronoi_adaptive_criticality(best_individual)
    sorted_indices = np.argsort(-criticality)
    
    # Refine the top 15 most critical circles
    for i in range(min(15, len(best_individual))):
        idx = sorted_indices[i]
        test_individual = best_individual.copy()
        
        # Fine-grained perturbation around the most critical circle
        old_x, old_y, old_r = test_individual[idx]
        new_x = max(0.005, min(0.995, old_x + np.random.normal(0, 0.003)))
        new_y = max(0.005, min(0.995, old_y + np.random.normal(0, 0.003)))
        new_r = max(0.001, old_r + np.random.normal(0, 0.0015))

        test_individual[idx] = [new_x, new_y, new_r]
        
        # Validate and accept improvement
        if evaluate_fitness(test_individual, rect_width, rect_height) > evaluate_fitness(best_individual, rect_width, rect_height):
            best_individual = test_individual.copy()

    # Final validation
    if not is_valid_solution(best_individual, rect_width, rect_height):
        # Fallback to best structured pattern
        best_individual = generate_hexagonal_pattern(rect_width, rect_height, n)

    return best_individual

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")