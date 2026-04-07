# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List, Dict
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    width, height = 1.0, 1.0

    def is_valid_position(x: float, y: float, r: float) -> bool:
        """Check if circle is within bounds"""
        return (r <= x <= width - r) and (r <= y <= height - r)

    def evaluate_fitness(circles_array: np.ndarray) -> Tuple[float, float]:
        """Evaluate fitness: sum of radii and penalty for overlaps/bounds"""
        total_radius = np.sum(circles_array[:, 2])

        # Check bounds
        valid = True
        for x, y, r in circles_array:
            if not is_valid_position(x, y, r):
                valid = False
                break

        if not valid:
            return -1e6, total_radius

        # Check overlaps using efficient distance matrix
        if len(circles_array) > 1:
            positions = circles_array[:, :2]
            radii = circles_array[:, 2]
            distances = cdist(positions, positions)
            
            for i in range(len(circles_array)):
                for j in range(i+1, len(circles_array)):
                    distance = distances[i, j]
                    overlap_distance = radii[i] + radii[j]
                    if distance < overlap_distance:
                        return -1e6, total_radius  # Overlap penalty

        return total_radius, total_radius

    def compute_multi_scale_voronoi_constraints(circles_list: List[Tuple[float, float, float]]) -> Dict[int, Dict[str, float]]:
        """Compute multi-scale Voronoi constraints for each circle"""
        if len(circles_list) < 2:
            return {i: {'density': 1.0, 'area': 1.0, 'neighborhood': 1.0} for i in range(len(circles_list))}
        
        try:
            # Build Voronoi diagram
            points = np.array([[x, y] for x, y, r in circles_list])
            vor = Voronoi(points)
            
            # Compute constraint metrics for each circle
            constraints = {}
            for i, (x, y, r) in enumerate(circles_list):
                # Basic local density based on neighbor count and distances
                distances = np.sqrt(np.sum((points - points[i])**2, axis=1))
                distances[i] = np.inf  # Remove self-distance
                nearest_distances = np.sort(distances)[:min(5, len(distances))]
                
                # Density metric (inverse of average nearest neighbor distance)
                if len(nearest_distances) > 0:
                    avg_distance = np.mean(nearest_distances)
                    density = 1.0 / max(avg_distance, 1e-6)
                else:
                    density = 1.0
                
                # Voronoi cell area (simplified estimation)
                area = 1.0  # Default value
                try:
                    # Find Voronoi cell for this point
                    cell_vertices = []
                    for j, region in enumerate(vor.regions):
                        if len(region) > 0 and -1 not in region:
                            # Check if this region corresponds to our point
                            if np.allclose(vor.points[j], [x, y], atol=1e-10):
                                vertices = [vor.vertices[k] for k in region if k < len(vor.vertices)]
                                if len(vertices) >= 3:
                                    # Compute area of polygon
                                    vertices = np.array(vertices)
                                    # Use shoelace formula
                                    x_coords = vertices[:, 0]
                                    y_coords = vertices[:, 1]
                                    area = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1)))
                                break
                except:
                    pass  # Fall back to default area
                
                # Neighborhood constraint (how many neighbors are close)
                neighbor_threshold = r * 2.0
                neighbor_count = np.sum(distances < neighbor_threshold)
                neighborhood = neighbor_count / max(1.0, len(circles_list) - 1)
                
                constraints[i] = {
                    'density': density,
                    'area': area,
                    'neighborhood': neighborhood
                }
            
            return constraints
        except:
            # Fallback to simple distance-based constraints if Voronoi fails
            constraints = {}
            for i in range(len(circles_list)):
                constraints[i] = {'density': 1.0, 'area': 1.0, 'neighborhood': 1.0}
            return constraints

    def initialize_hexagonal_arrangement(width: float, height: float, n_circles: int) -> np.ndarray:
        """Initialize circles in a hexagonal pattern"""
        circles = np.zeros((n_circles, 3))
        base_radius = min(width, height) * 0.05

        # Try to place in hexagonal grid
        rows = int(np.ceil(np.sqrt(n_circles)))
        cols = int(np.ceil(n_circles / rows))

        spacing_x = width / (cols + 1)
        spacing_y = height / (rows + 1)

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Adjust for hexagonal pattern
                if i % 2 == 1:
                    x += spacing_x / 2
                # Bound checks and adjust radius
                r = min(base_radius,
                       min(x, width-x), min(y, height-y))
                if r > 0.001:
                    circles[idx] = [x, y, r]
                    idx += 1
        return circles

    def adaptive_mutation_with_voronoi_direction(circles: np.ndarray, 
                                               voronoi_constraints: Dict[int, Dict[str, float]], 
                                               generation: int) -> np.ndarray:
        """Mutate circles with direction guided by Voronoi constraints"""
        mutated = circles.copy()
        mutation_rate = 0.2 + 0.1 * np.exp(-generation/50)  # Decrease over generations
        
        # Compute Voronoi-based weight for each circle
        weights = np.array([voronoi_constraints[i]['density'] for i in range(len(circles))])
        # Normalize weights
        weights = weights / (np.sum(weights) + 1e-8)
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Get constraint info for this circle
                constraint_info = voronoi_constraints.get(i, {'density': 1.0, 'area': 1.0, 'neighborhood': 1.0})
                density = constraint_info['density']
                
                # Adaptive step size based on constraint density
                # More constrained areas get smaller steps
                base_step = 0.02
                # Step size inversely related to density but capped
                step_size = base_step / (1.0 + density * 0.5) * (0.5 + 0.5 * np.exp(-generation/100))
                
                # Determine mutation type: position or radius
                mutation_type = np.random.choice(['position', 'radius'], p=[0.7, 0.3])
                
                if mutation_type == 'position':
                    # Mutation direction informed by Voronoi constraints
                    # Prefer moving toward less constrained regions
                    direction_bias = 0.0
                    
                    # Get spatial tree for neighbor queries
                    points = circles[:, :2]
                    if len(points) > 1:
                        tree = cKDTree(points)
                        # Find neighbors
                        neighbors = tree.query_ball_point(points[i], 0.1)
                        # Bias movement towards regions where neighbors are far
                        if len(neighbors) > 1:
                            # Calculate average distance to neighbors
                            neighbor_points = points[neighbors]
                            avg_dist = np.mean(np.sqrt(np.sum((neighbor_points - points[i])**2, axis=1)))
                            # If neighbors are far, move more boldly; otherwise, be careful
                            direction_bias = max(0, 0.1 - avg_dist) * 0.5
                    
                    # Apply mutation with directional bias
                    mutated[i, 0] += np.random.normal(0, step_size) * (1.0 + direction_bias)
                    mutated[i, 1] += np.random.normal(0, step_size) * (1.0 + direction_bias)
                    
                    # Ensure bounds
                    mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], width - mutated[i, 2])
                    mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], height - mutated[i, 2])
                    
                else:  # radius mutation
                    # Modify radius with adaptive scaling
                    if density > 2.0:  # High constraint area - be conservative
                        radius_step = np.random.normal(0, step_size * 0.3)
                    elif density < 1.0:  # Low constraint area - be exploratory
                        radius_step = np.random.normal(0, step_size * 2.0)
                    else:  # Medium constraint area
                        radius_step = np.random.normal(0, step_size * 1.0)
                        
                    mutated[i, 2] = max(0.001, mutated[i, 2] + radius_step)
                    
                    # Ensure radius fits in its position
                    max_radius_x = min(mutated[i, 0], width - mutated[i, 0])
                    max_radius_y = min(mutated[i, 1], height - mutated[i, 1])
                    mutated[i, 2] = min(mutated[i, 2], max_radius_x, max_radius_y)
        
        return mutated

    def local_optimization(circles: np.ndarray, max_iterations: int = 30) -> np.ndarray:
        """Perform local optimization focusing on improving radii"""
        current = circles.copy()
        best_fitness = evaluate_fitness(current)[0]

        for iteration in range(max_iterations):
            improved = False
            # Try small adjustments to each circle
            for i in range(len(current)):
                x, y, r = current[i]
                best_r = r
                best_x, best_y = x, y
                best_fitness_local = best_fitness

                # Sample positions more intelligently based on constraint information
                # Get constraint info for this circle
                constraint_density = 1.0  # default if not computed
                
                # Try positions in a more structured grid
                grid_size = max(0.005, r * 0.2)  # Dynamic grid size based on circle size
                sample_points = []
                for dx in np.arange(-0.02, 0.021, grid_size):
                    for dy in np.arange(-0.02, 0.021, grid_size):
                        sample_points.append((dx, dy))
                
                # Add some random perturbations to explore
                for _ in range(10):
                    sample_points.append((np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02)))
                
                for dx, dy in sample_points[:20]:  # Limit samples for efficiency
                    new_x, new_y = x + dx, y + dy
                    new_r = r

                    # Check if new position is valid
                    if is_valid_position(new_x, new_y, new_r):
                        # Check collision with others
                        valid = True
                        for j in range(len(current)):
                            if i != j:
                                ox, oy, oradius = current[j]
                                distance = np.sqrt((new_x - ox)**2 + (new_y - oy)**2)
                                if distance < (new_r + oradius):
                                    valid = False
                                    break

                        if valid:
                            # Test if we can increase radius
                            test_r = min(
                                new_r,
                                new_x, width - new_x,
                                new_y, height - new_y
                            )

                            test_circles = current.copy()
                            test_circles[i] = [new_x, new_y, test_r]

                            new_fitness, _ = evaluate_fitness(test_circles)

                            if new_fitness > best_fitness_local:
                                best_fitness_local = new_fitness
                                best_r = test_r
                                best_x, best_y = new_x, new_y
                                improved = True

                current[i] = [best_x, best_y, best_r]

            # Update best fitness
            new_fitness, _ = evaluate_fitness(current)
            if new_fitness > best_fitness:
                best_fitness = new_fitness
            elif not improved:
                break  # No improvement, stop early

        return current

    # Main algorithm
    
    # Initialize with hexagonal packing
    circles = initialize_hexagonal_arrangement(width, height, 21)
    
    # Evolution parameters
    generations = 100
    population_size = 30
    elite_size = 4
    
    # Initialize population with better diversity
    population = [circles.copy()]
    
    # Add some variation to initial population
    for _ in range(population_size - 1):
        mutated = circles.copy()
        for i in range(len(mutated)):
            # Apply more diverse mutations for initial population
            mutated[i, 0] += np.random.uniform(-0.03, 0.03)
            mutated[i, 1] += np.random.uniform(-0.03, 0.03)
            mutated[i, 2] += np.random.uniform(-0.015, 0.015)
            mutated[i, 2] = max(0.001, mutated[i, 2])
        population.append(mutated)
    
    best_solution = None
    best_fitness = -1e10
    
    # Evolution loop
    for gen in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            score, _ = evaluate_fitness(individual)
            fitness_scores.append(score)
            
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Keep elite
        elite = sorted_population[:elite_size]
        
        # Update best solution
        if sorted_fitness[0] > best_fitness:
            best_fitness = sorted_fitness[0]
            best_solution = sorted_population[0].copy()
            
        # Create new population
        new_population = elite.copy()
        
        # Generate offspring through mutation with adaptation
        while len(new_population) < population_size:
            # Select parent from top half
            parent_idx = random.randint(0, population_size // 2 - 1)
            parent = sorted_population[parent_idx]
            
            # Add constraint density information
            constraint_info = compute_multi_scale_voronoi_constraints([(x, y, r) for x, y, r in parent])
            
            # Mutate parent with Voronoi-guided direction
            child = adaptive_mutation_with_voronoi_direction(parent, constraint_info, gen)
            
            # Local optimization on child
            child = local_optimization(child, max_iterations=3)
            
            new_population.append(child)
            
        population = new_population[:population_size]
        
        # Early stopping if no improvement for several generations
        if gen > 20 and abs(sorted_fitness[0] - best_fitness) < 1e-6:
            break
    
    # Final local optimization
    if best_solution is not None:
        final_solution = local_optimization(best_solution, max_iterations=30)
        return final_solution
    
    # Fallback to hexagonal arrangement if nothing worked
    return initialize_hexagonal_arrangement(width, height, 21)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")