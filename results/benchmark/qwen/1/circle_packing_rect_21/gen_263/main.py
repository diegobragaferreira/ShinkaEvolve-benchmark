# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Set seeds for determinism
random.seed(42)
np.random.seed(42)

def is_valid_solution(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
    """
    Fast validity check using grid-based spatial indexing for collision detection.
    """
    n = len(circles)

    # Check boundary constraints first
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
            return False

    if n <= 1:
        return True

    # Use grid-based spatial indexing for efficient overlap detection
    coords = circles[:, :2]
    radii = circles[:, 2]

    try:
        # Build grid index
        avg_radius = np.mean(radii) if n > 0 else 0.1
        cell_size = avg_radius * 1.5  # Slightly larger than average radius

        # Calculate grid dimensions
        cols = max(1, int(rect_width / cell_size) + 1)
        rows = max(1, int(rect_height / cell_size) + 1)

        # Create cell dictionary mapping (row, col) to circle indices
        cell_dict = {}

        # Assign circles to grid cells
        for i in range(n):
            x, y, r = circles[i]
            # Find grid cell coordinates for circle center
            col = int(x / cell_size)
            row = int(y / cell_size)

            # Clamp to grid bounds
            col = max(0, min(col, cols - 1))
            row = max(0, min(row, rows - 1))

            # Store in dictionary
            key = (row, col)
            if key not in cell_dict:
                cell_dict[key] = []
            cell_dict[key].append(i)

        # Check for collisions
        for i in range(n):
            x1, y1, r1 = circles[i]

            # Find the grid cell that contains this circle
            col = int(x1 / cell_size)
            row = int(y1 / cell_size)
            col = max(0, min(col, cols - 1))
            row = max(0, min(row, rows - 1))

            # Check neighboring cells (including current cell)
            # Check 3x3 neighborhood around current cell
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    neighbor_row = row + dr
                    neighbor_col = col + dc

                    # Check bounds
                    if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                        key = (neighbor_row, neighbor_col)
                        if key in cell_dict:
                            # Check all circles in this cell against current circle
                            for j in cell_dict[key]:
                                if i != j:  # Don't compare with self
                                    x2, y2, r2 = circles[j]

                                    # Fast distance check using squared distances
                                    dx = x1 - x2
                                    dy = y1 - y2
                                    distance_sq = dx*dx + dy*dy
                                    min_distance_sq = (r1 + r2) * (r1 + r2)

                                    if distance_sq < min_distance_sq:
                                        return False

        return True
    except:
        # Fallback to brute force if grid indexing fails
        for i in range(n):
            for j in range(i+1, n):
                distance = np.linalg.norm(coords[i] - coords[j])
                min_distance = radii[i] + radii[j]

                if distance < min_distance:
                    return False
        return True

def compute_voronoi_constraints(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Compute refined constraint density for each circle based on actual Voronoi cell areas and geometric properties.
    """
    n = len(circles)
    if n < 2:
        return np.zeros(n)

    # Get circle centers
    centers = circles[:, :2]

    # Compute Voronoi-based constraint density using actual Voronoi diagrams
    constraint_density = np.zeros(n)

    try:
        # For each circle, compute Voronoi cell area and constraint density
        for i in range(n):
            # Extract the current circle's center
            current_center = centers[i]

            # For Voronoi computation, we need to include boundary constraints
            # Create extended set of points to simulate bounded Voronoi
            # We'll add artificial points around the rectangle to simulate boundaries
            extended_centers = centers.copy()

            # Add boundary points if not already included
            # These represent the rectangle boundaries
            boundary_points = []

            # Add some boundary points to help compute proper Voronoi regions
            # Only add if we don't already have points near boundaries
            boundary_margin = 0.1  # Margin to avoid placing points too close to real boundaries

            # Add corner points
            corners = [
                [boundary_margin, boundary_margin],
                [rect_width - boundary_margin, boundary_margin],
                [boundary_margin, rect_height - boundary_margin],
                [rect_width - boundary_margin, rect_height - boundary_margin]
            ]

            # Add edge points
            edges = [
                [rect_width/2, boundary_margin],
                [rect_width/2, rect_height - boundary_margin],
                [boundary_margin, rect_height/2],
                [rect_width - boundary_margin, rect_height/2]
            ]

            # Only add boundary points that aren't too close to existing circles
            for point in corners + edges:
                # Check if this point is too close to any existing circle
                is_far_enough = True
                for j in range(n):
                    dist = np.linalg.norm(np.array(point) - centers[j])
                    if dist < circles[j, 2] * 2:  # Too close to existing circle
                        is_far_enough = False
                        break

                if is_far_enough:
                    boundary_points.append(point)

            # Only proceed with Voronoi if we can form meaningful regions
            if len(extended_centers) + len(boundary_points) >= 3:
                # Try to compute Voronoi but with a fallback logic
                try:
                    # Create extended points list
                    all_points = np.vstack([extended_centers, np.array(boundary_points)])

                    # Compute Voronoi (may fail in some edge cases)
                    vor = Voronoi(all_points)

                    # Find the Voronoi region for the current point
                    region_index = None
                    for idx, region in enumerate(vor.point_region):
                        if region < len(vor.regions) and len(vor.regions[region]) > 0:
                            # Check if this region corresponds to our point
                            if idx == i:
                                region_index = region
                                break

                    # If we found a valid region, compute the area
                    if region_index is not None and region_index < len(vor.regions):
                        region = vor.regions[region_index]
                        if len(region) > 0 and -1 not in region:
                            # Extract the vertices of the Voronoi cell
                            vertices = [vor.vertices[k] for k in region if k < len(vor.vertices)]
                            if len(vertices) >= 3:
                                # Calculate the area of the polygon (Voronoi cell)
                                # Using shoelace formula for polygon area
                                vertices = np.array(vertices)
                                # Close the polygon
                                vertices = np.vstack([vertices, vertices[0]])
                                area = 0.5 * np.abs(np.dot(vertices[:-1, 0], vertices[1:, 1]) -
                                                   np.dot(vertices[:-1, 1], vertices[1:, 0]))
                                # Convert area to constraint measure
                                # Area inversely relates to constraint density
                                # Large areas = more space = less constraint
                                # Small areas = less space = more constraint
                                if area > 1e-8:
                                    constraint_density[i] = 1.0 / area
                                else:
                                    # Fall back to neighbor-based approach if area is tiny
                                    constraint_density[i] = 1.0
                            else:
                                # Not enough vertices, fall back to neighbor-based
                                constraint_density[i] = 1.0
                        else:
                            # Invalid region, fall back to neighbor-based approach
                            constraint_density[i] = 1.0
                    else:
                        # Could not locate region, fall back to neighbor-based
                        constraint_density[i] = 1.0

                except:
                    # Voronoi computation failed, fall back to simple approach
                    constraint_density[i] = 1.0
            else:
                # Not enough points for meaningful Voronoi, use neighbor-based approach
                constraint_density[i] = 1.0

    except Exception as e:
        # If Voronoi computation fails entirely for any reason, fall back to neighbor counting
        constraint_density.fill(1.0)  # Default to uniform constraint density

    # Fallback: always include neighbor-based density for robustness
    try:
        # Compute neighbor density as backup
        tree = cKDTree(centers)

        # For each circle, compute neighbor-based constraint density
        neighbor_constraint_density = np.zeros(n)

        for i in range(n):
            center_i = centers[i]
            # Use 3x the maximum radius as threshold for neighbors
            max_radius = np.max(circles[:, 2])
            threshold = max_radius * 3.0

            # Query neighbors efficiently
            neighbors = tree.query_ball_point(center_i, threshold)
            neighbor_count = len(neighbors) - 1  # Exclude self

            # Normalize neighbor count
            normalized_neighbors = neighbor_count / max(1, n - 1)

            # Convert to constraint density (higher neighbor count = higher constraint)
            neighbor_constraint_density[i] = normalized_neighbors

        # Combine both approaches with weighted average
        # Use Voronoi-based approach as primary, neighbor-based as backup
        combined_density = np.zeros(n)
        for i in range(n):
            # If Voronoi failed, use neighbor-based
            if constraint_density[i] <= 1e-8:  # Invalid Voronoi density
                combined_density[i] = neighbor_constraint_density[i]
            else:
                # Blend both approaches with 70% Voronoi, 30% neighbor
                combined_density[i] = 0.7 * constraint_density[i] + 0.3 * neighbor_constraint_density[i]

        constraint_density = combined_density

    except Exception as e:
        # If everything fails, return simple neighbor density
        pass

    # Normalize and ensure reasonable values
    if np.any(constraint_density > 0):
        # Normalize using max value (but cap at 10.0 to avoid extreme values)
        max_density = np.max(constraint_density)
        if max_density > 0:
            constraint_density = np.minimum(constraint_density / max_density, 10.0)

    # Ensure minimum constraint density to avoid zero values that could cause issues
    constraint_density = np.maximum(constraint_density, 0.01)

    return constraint_density

def initialize_hexagonal_lattice(n_circles: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Initialize circle positions using a hexagonal lattice pattern.
    """
    # Use hexagonal packing approach for initial placement
    # Estimate radius based on area
    total_area = rect_width * rect_height
    circle_area = total_area / n_circles * 0.9  # Leave some margin
    estimated_radius = np.sqrt(circle_area / np.pi)

    # Hexagon parameters
    side_length = 2 * estimated_radius

    # Determine grid dimensions
    cols = max(1, int(rect_width / side_length) + 1)
    rows = max(1, int(rect_height / (side_length * np.sqrt(3) / 2)) + 1)

    points = []
    for i in range(rows):
        for j in range(cols):
            x = (j + (i % 2) * 0.5) * side_length
            y = i * side_length * np.sqrt(3) / 2

            # Only include points that fit within the rectangle
            if x >= estimated_radius and x <= rect_width - estimated_radius and \
               y >= estimated_radius and y <= rect_height - estimated_radius:
                points.append([x, y])

    # If we have too few points, add more by expanding
    while len(points) < n_circles:
        # Add points at random locations within bounds
        x = random.uniform(estimated_radius, rect_width - estimated_radius)
        y = random.uniform(estimated_radius, rect_height - estimated_radius)
        points.append([x, y])

    # Trim to exact number needed
    points = points[:n_circles]

    # Create initial circles with estimated radii
    circles = np.zeros((n_circles, 3))
    for i, (x, y) in enumerate(points):
        circles[i] = [x, y, estimated_radius * 0.8]

    return circles

def calculate_fitness(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> Tuple[float, float]:
    """
    Calculate fitness of circle configuration with penalty for constraint violations.
    """
    n = len(circles)

    # Check boundary constraints
    penalty = 0.0
    for i in range(n):
        x, y, r = circles[i]
        # Circle must be fully contained within rectangle
        if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
            # Apply penalty based on how much it violates boundaries
            overlap = 0.0
            if x - r < 0:
                overlap += abs(x - r)
            if x + r > rect_width:
                overlap += abs(x + r - rect_width)
            if y - r < 0:
                overlap += abs(y - r)
            if y + r > rect_height:
                overlap += abs(y + r - rect_height)
            penalty += overlap * 1000

    # Check overlap constraints using efficient spatial indexing
    overlap_penalty = 0.0
    if n > 1:
        # Use spatial indexing with KDTree for efficient overlap detection
        coords = circles[:, :2]
        radii = circles[:, 2]

        try:
            tree = cKDTree(coords)
            max_radius = np.max(radii)

            # For each circle, find neighbors and check overlaps
            for i in range(n):
                # Query neighbors within 2 * max_radius distance
                neighbors = tree.query_ball_point(coords[i], 2 * max_radius)

                # Check overlaps with neighbors
                for j in neighbors:
                    if i != j:
                        distance = np.linalg.norm(coords[i] - coords[j])
                        min_distance = radii[i] + radii[j]

                        if distance < min_distance:
                            # Overlap exists
                            overlap_amount = min_distance - distance
                            overlap_penalty += overlap_amount * 1000  # Heavy penalty for overlaps
        except:
            # Fallback to brute force if spatial indexing fails
            for i in range(n):
                for j in range(i+1, n):
                    distance = np.linalg.norm(coords[i] - coords[j])
                    min_distance = radii[i] + radii[j]

                    if distance < min_distance:
                        # Overlap exists
                        overlap_amount = min_distance - distance
                        overlap_penalty += overlap_amount * 1000  # Heavy penalty for overlaps

    # Fitness is sum of radii minus penalties
    total_radius = np.sum(circles[:, 2])
    fitness = total_radius - penalty - overlap_penalty

    return fitness, overlap_penalty

def mutate_circles_adaptive(circles: np.ndarray,
                          constraint_densities: np.ndarray,
                          rect_width: float = 1.0,
                          rect_height: float = 1.0,
                          max_radius: float = 0.5) -> np.ndarray:
    """
    Mutate circle positions and radii with adaptive weights based on Voronoi cell areas.
    """
    mutated = circles.copy()
    n = len(mutated)

    # Compute Voronoi cell areas for each circle (more precise than density)
    voronoi_areas = np.zeros(n)
    try:
        # Use a simpler and more reliable approach to compute local constraint information
        # Instead of complex Voronoi computation, compute approximate constraint from neighbors
        centers = circles[:, :2]
        radii = circles[:, 2]

        # For each circle, compute average distance to neighbors (inverse of constraint density)
        tree = cKDTree(centers)

        for i in range(n):
            center_i = centers[i]
            # Find neighbors within 3x the maximum radius
            neighbors = tree.query_ball_point(center_i, 3 * max_radius)

            # Calculate average distance to neighbors (excluding self)
            if len(neighbors) > 1:
                neighbor_distances = []
                for j in neighbors:
                    if i != j:
                        dist = np.linalg.norm(center_i - centers[j])
                        neighbor_distances.append(dist)

                if neighbor_distances:
                    avg_dist = np.mean(neighbor_distances)
                    # Convert distance to area-like constraint (smaller distances = smaller areas = higher constraint)
                    voronoi_areas[i] = 1.0 / (avg_dist + 1e-8)
                else:
                    voronoi_areas[i] = 1.0
            else:
                voronoi_areas[i] = 1.0

    except:
        # Fallback to simple constraint density
        voronoi_areas = constraint_densities

    # Normalize Voronoi areas to reasonable range
    if np.max(voronoi_areas) > 0:
        voronoi_areas = voronoi_areas / np.max(voronoi_areas)

    # Mutation parameters adapted based on local constraint (Voronoi area-based)
    for i in range(n):
        x, y, r = mutated[i]

        # Voronoi area-based constraint: smaller areas = higher constraint = smaller steps
        # Scale mutation strength inversely to Voronoi area (smaller area = more constrained)
        area_factor = max(0.1, voronoi_areas[i])
        # Base mutation strengths
        base_pos_strength = 0.02
        base_rad_strength = 0.01

        # Adjust mutation strengths based on local constraint
        pos_strength = base_pos_strength * (0.5 + 0.5 * area_factor)
        rad_strength = base_rad_strength * (0.5 + 0.5 * area_factor)

        # Mutate position
        x += np.random.normal(0, pos_strength)
        y += np.random.normal(0, pos_strength)

        # Ensure position stays within bounds
        x = np.clip(x, r, rect_width - r)
        y = np.clip(y, r, rect_height - r)

        # Mutate radius
        r += np.random.normal(0, rad_strength)
        # Ensure radius remains positive and reasonable
        r = np.clip(r, 0.001, max_radius * 0.9)

        mutated[i] = [x, y, r]

    return mutated

def crossover_circles(parent1: np.ndarray, parent2: np.ndarray,
                     crossover_rate: float = 0.8) -> np.ndarray:
    """
    Perform uniform crossover between two circle configurations.
    """
    if random.random() > crossover_rate:
        return parent1.copy()  # Return first parent if no crossover

    offspring = parent1.copy()
    n = len(parent1)

    # Uniform crossover
    for i in range(n):
        if random.random() < 0.5:
            offspring[i] = parent2[i].copy()

    return offspring

def refine_solution_fast(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0,
                        iterations: int = 100) -> np.ndarray:
    """
    Fast local refinement using gradient-like approach and constraint-aware moves.
    """
    refined = circles.copy()
    n = len(refined)

    if n <= 1:
        return refined

    # Precompute constraint information
    constraint_densities = compute_voronoi_constraints(refined, rect_width, rect_height)

    # Iterative refinement
    for iter_num in range(iterations):
        # Work on one circle at a time
        for i in range(n):
            # Save current state
            old_x, old_y, old_r = refined[i]

            # Adaptive mutation based on constraint density
            density_weight = 1.0 + constraint_densities[i] * 2.0
            pos_strength = 0.01 / density_weight
            rad_strength = 0.005 / density_weight

            # Try small random moves
            new_x = old_x + np.random.normal(0, pos_strength)
            new_y = old_y + np.random.normal(0, pos_strength)
            new_r = old_r + np.random.normal(0, rad_strength)

            # Clip to bounds
            new_x = np.clip(new_x, new_r, rect_width - new_r)
            new_y = np.clip(new_y, new_r, rect_height - new_r)
            new_r = np.clip(new_r, 0.001, rect_width / 2)

            # Test if this change improves fitness
            test_config = refined.copy()
            test_config[i] = [new_x, new_y, new_r]

            # Quick constraint check before full fitness evaluation
            if not is_valid_solution(test_config, rect_width, rect_height):
                continue

            # Check if this move improves fitness
            current_fitness, _ = calculate_fitness(refined, rect_width, rect_height)
            test_fitness, _ = calculate_fitness(test_config, rect_width, rect_height)

            if test_fitness > current_fitness:
                refined = test_config

            # Occasionally do larger moves to escape local optima
            elif random.random() < 0.05 and iter_num > iterations // 2:
                # Do bigger perturbation
                new_x = old_x + np.random.normal(0, pos_strength * 3)
                new_y = old_y + np.random.normal(0, pos_strength * 3)
                new_r = old_r + np.random.normal(0, rad_strength * 3)

                # Clip to bounds
                new_x = np.clip(new_x, new_r, rect_width - new_r)
                new_y = np.clip(new_y, new_r, rect_height - new_r)
                new_r = np.clip(new_r, 0.001, rect_width / 2)

                if is_valid_solution(test_config, rect_width, rect_height):
                    test_config[i] = [new_x, new_y, new_r]
                    test_fitness, _ = calculate_fitness(test_config, rect_width, rect_height)
                    if test_fitness > current_fitness:
                        refined = test_config

    return refined

def optimize_with_voronoi_evolution(n_circles: int = 21,
                                  rect_width: float = 1.0,
                                  rect_height: float = 1.0,
                                  population_size: int = 100,
                                  generations: int = 100) -> np.ndarray:
    """
    Optimize circle packing using Voronoi-enhanced evolutionary algorithm.
    """
    # Initialize population
    population = []
    for _ in range(population_size):
        circles = initialize_hexagonal_lattice(n_circles, rect_width, rect_height)
        # Add some randomness to initial positions
        for i in range(n_circles):
            circles[i][0] += random.uniform(-0.05, 0.05)
            circles[i][1] += random.uniform(-0.05, 0.05)
            circles[i][0] = np.clip(circles[i][0], circles[i][2], rect_width - circles[i][2])
            circles[i][1] = np.clip(circles[i][1], circles[i][2], rect_height - circles[i][2])
        population.append(circles)

    # Evolutionary loop
    best_fitness_history = []

    for gen in range(generations):
        # Evaluate fitness of population
        fitness_scores = []
        for circles in population:
            fitness, _ = calculate_fitness(circles, rect_width, rect_height)
            fitness_scores.append(fitness)

        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
        population = [population[i] for i in sorted_indices]
        fitness_scores.sort(reverse=True)

        best_fitness_history.append(fitness_scores[0])

        # Print progress
        if gen % 20 == 0:
            print(f"Generation {gen}, Best fitness: {fitness_scores[0]:.6f}")

        # Create new generation
        new_population = [population[0]]  # Elitism - keep best individual

        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 5
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]

            # Select parent
            parent1 = population[winner_index]

            # Select second parent
            tournament_indices.remove(winner_index)
            tournament_fitness.remove(max(tournament_fitness))
            winner_index2 = tournament_indices[np.argmax(tournament_fitness)]
            parent2 = population[winner_index2]

            # Crossover
            offspring = crossover_circles(parent1, parent2)

            # Compute constraint densities and mutate adaptively
            constraint_densities = compute_voronoi_constraints(offspring, rect_width, rect_height)
            offspring = mutate_circles_adaptive(offspring, constraint_densities,
                                              rect_width, rect_height,
                                              max_radius=min(rect_width, rect_height) / 2)

            new_population.append(offspring)

        population = new_population[:population_size]

        # Early stopping if no improvement
        if len(best_fitness_history) >= 5:
            recent_improvement = best_fitness_history[-1] - best_fitness_history[-5]
            if recent_improvement < 1e-6 and gen > 30:
                break

    # Return best solution
    best_index = np.argmax([calculate_fitness(ind, rect_width, rect_height)[0] for ind in population])
    return population[best_index]

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: perimeter = 4 => width + height = 2
    # Optimize rectangle aspect ratio for better packing
    rect_width = 1.2
    rect_height = 0.8

    # Run Voronoi-enhanced optimization
    best_solution = optimize_with_voronoi_evolution(
        n_circles=21,
        rect_width=rect_width,
        rect_height=rect_height,
        population_size=100,
        generations=100
    )

    # Apply fast refinement
    refined_solution = refine_solution_fast(best_solution, rect_width, rect_height)

    return refined_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")