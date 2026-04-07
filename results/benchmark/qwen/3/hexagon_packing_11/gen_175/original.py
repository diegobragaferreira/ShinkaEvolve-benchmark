# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import time
from numba import jit, prange
from joblib import Parallel, delayed
import random
import math

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Constants
UNIT_HEX_RADIUS = 1.0  # radius of unit hexagon circumcircle
UNIT_HEX_SIDE = 1.0    # side length of unit hexagon
PI = np.pi

@jit(nopython=True)
def hexagon_vertices_jit(center_x, center_y, angle_rad, side_length):
    """Compute hexagon vertices efficiently using numba"""
    vertices = np.zeros((6, 2))
    for i in range(6):
        angle = angle_rad + i * PI / 3
        vertices[i, 0] = center_x + side_length * np.cos(angle)
        vertices[i, 1] = center_y + side_length * np.sin(angle)
    return vertices

@jit(nopython=True)
def distance_point_to_segment(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment"""
    # Vector from (x1,y1) to (x2,y2)
    dx = x2 - x1
    dy = y2 - y1

    # Length squared of segment
    len_sq = dx*dx + dy*dy

    # Avoid division by zero
    if len_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    # Project point onto line
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = max(0, min(1, t))  # Clamp t to [0,1]

    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Distance squared
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def point_in_hexagon_fast(point_x, point_y, hex_center_x, hex_center_y, hex_angle_rad, hex_side_length):
    """Fast point-in-hexagon check using distance to edges"""
    # Get vertices
    vertices = hexagon_vertices_jit(hex_center_x, hex_center_y, hex_angle_rad, hex_side_length)

    # Calculate distance to each edge
    min_dist = np.inf
    for i in range(6):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i+1)%6]
        dist = distance_point_to_segment(point_x, point_y, x1, y1, x2, y2)
        min_dist = min(min_dist, dist)

    # For a regular hexagon with side length s, inradius = s * sqrt(3)/2
    inradius = hex_side_length * np.sqrt(3) / 2
    return min_dist >= inradius

def hexagon_vertices(center_x, center_y, angle_rad, side_length):
    """Compute hexagon vertices"""
    return hexagon_vertices_jit(center_x, center_y, angle_rad, side_length)

def check_overlap_hexagons(h1_center_x, h1_center_y, h1_angle, h1_side,
                          h2_center_x, h2_center_y, h2_angle, h2_side):
    """Check if two hexagons overlap using vertices inclusion test"""
    vertices1 = hexagon_vertices(h1_center_x, h1_center_y, np.radians(h1_angle), h1_side)
    vertices2 = hexagon_vertices(h2_center_x, h2_center_y, np.radians(h2_angle), h2_side)

    # Create shapely polygons
    poly1 = Polygon(vertices1)
    poly2 = Polygon(vertices2)

    # Check if they intersect
    return poly1.intersects(poly2)

def check_all_overlaps(inner_hex_data):
    """Check all pairs of hexagons for overlaps"""
    n = len(inner_hex_data)
    # Early return if too few hexagons
    if n < 2:
        return False

    # Check only unique pairs
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, angle1 = inner_hex_data[i]
            cx2, cy2, angle2 = inner_hex_data[j]

            if check_overlap_hexagons(cx1, cy1, angle1, UNIT_HEX_SIDE,
                                    cx2, cy2, angle2, UNIT_HEX_SIDE):
                return True
    return False

def check_containment(inner_hex_data, outer_center=(0,0), outer_side=10):
    """Check if all inner hexagons are contained in outer hexagon"""
    outer_vertices = hexagon_vertices(outer_center[0], outer_center[1], 0, outer_side)
    outer_polygon = Polygon(outer_vertices)

    for i in range(len(inner_hex_data)):
        cx, cy, angle = inner_hex_data[i]
        vertices = hexagon_vertices(cx, cy, np.radians(angle), UNIT_HEX_SIDE)

        # Create hexagon polygon
        inner_polygon = Polygon(vertices)

        # Check if it's contained
        if not outer_polygon.contains(inner_polygon):
            return False

    return True

def compute_outer_hexagon_radius(inner_hex_data, outer_hex_center=(0,0)):
    """Estimate minimum outer hexagon radius needed to contain all inner hexagons"""
    max_distance = 0
    for i in range(len(inner_hex_data)):
        cx, cy, angle = inner_hex_data[i]
        # Get all vertices of this hexagon
        vertices = hexagon_vertices(cx, cy, np.radians(angle), UNIT_HEX_SIDE)
        # Find maximum distance from center
        for vx, vy in vertices:
            dist = np.sqrt((vx - outer_hex_center[0])**2 + (vy - outer_hex_center[1])**2)
            max_distance = max(max_distance, dist)

    # Add safety margin for numerical precision
    return max_distance * 1.05

def evaluate_layout(inner_hex_data, outer_side_estimate=None):
    """Evaluate the quality of a given hexagon layout"""
    # Check overlaps first (early rejection)
    if check_all_overlaps(inner_hex_data):
        return 1e10  # Large penalty for overlaps

    # Estimate outer hexagon size
    if outer_side_estimate is None:
        estimated_outer_radius = compute_outer_hexagon_radius(inner_hex_data)
        outer_side = estimated_outer_radius * 2  # rough estimate
    else:
        outer_side = outer_side_estimate

    # Check containment
    if not check_containment(inner_hex_data, (0,0), outer_side):
        return 1e10  # Large penalty for containment violations

    # Return inverse of outer hexagon side length (we want to maximize 1/outer_side)
    return 1.0 / outer_side

def parallel_evaluate_population(population):
    """Evaluate multiple configurations in parallel"""
    results = Parallel(n_jobs=-1)(delayed(evaluate_layout)(config) for config in population)
    return results

def generate_initial_config():
    """
    Generate initial configuration using mathematically informed pattern.
    This uses a more systematic approach based on hexagonal packing theory with
    optimized spacing to improve convergence.
    """
    # Create a systematic arrangement inspired by mathematical hexagonal close packing
    # This pattern aims to maximize density while minimizing overlaps
    
    config = []
    
    # Central hexagon
    config.append([0.0, 0.0, 0.0])
    
    # First shell - 6 hexagons arranged in a tight ring
    # Spacing calculated to minimize overhangs while maximizing density
    shell_radius = 1.732  # Approximately sqrt(3) which provides good spacing
    
    for i in range(6):
        angle = i * 60  # 60 degree increments
        rad = np.radians(angle)
        x = shell_radius * np.cos(rad)
        y = shell_radius * np.sin(rad)
        config.append([x, y, 0.0])
    
    # Additional strategic placements to fill gaps and improve arrangement
    # These are placed to balance the packing density and avoid clustering issues
    
    # Additional hexagons in the gaps of the first shell
    extra_positions = [
        (0.0, 3.464, 0.0),   # Top
        (0.0, -3.464, 0.0),  # Bottom
        (-3.0, 1.732, 0.0),  # Left-top
        (3.0, 1.732, 0.0),   # Right-top
        (-3.0, -1.732, 0.0), # Left-bottom
        (3.0, -1.732, 0.0),  # Right-bottom
    ]
    
    for pos in extra_positions:
        config.append(list(pos))
    
    # Trim to exactly 11 hexagons and apply small symmetric jitter to break degeneracy
    config = np.array(config[:11])
    
    # Add small jitter to break symmetry
    noise_scale = 0.01
    np.random.seed(42)  # For reproducibility
    config[:, 0] += np.random.normal(0, noise_scale, config.shape[0])
    config[:, 1] += np.random.normal(0, noise_scale, config.shape[0])
    
    return config

def adaptive_evolution_step(population, fitness_scores, generation, max_generations):
    """Evolution step with adaptive parameters based on convergence"""
    # Adaptive mutation rate that decreases with generations
    progress = generation / max_generations
    mutation_rate = 0.2 * (1 - progress * 0.7)  # Start high, decrease gradually
    
    # Select top performers
    sorted_indices = np.argsort(fitness_scores)[::-1][:len(population)//2]
    selected = [population[i] for i in sorted_indices]
    
    # Create offspring with adaptive crossover and mutation
    new_population = selected.copy()
    
    # Generate offspring through crossover and mutation
    while len(new_population) < len(population):
        parent1 = selected[np.random.randint(len(selected))]
        parent2 = selected[np.random.randint(len(selected))]
        
        # Crossover - blend positions, rotate angles
        child = parent1.copy()
        for i in range(len(child)):
            if np.random.random() < 0.5:
                child[i] = parent2[i].copy()
        
        # Mutation with adaptive rate
        for i in range(len(child)):
            if np.random.random() < mutation_rate:
                # Position mutation with adaptive magnitude
                mutation_magnitude = 0.2 * (1 - progress)
                child[i][0] += np.random.normal(0, mutation_magnitude)
                child[i][1] += np.random.normal(0, mutation_magnitude)
                # Angle mutation  
                child[i][2] += np.random.normal(0, 15)
                # Keep angle in [0, 360)
                child[i][2] = child[i][2] % 360
        
        new_population.append(child)
    
    return new_population

def local_optimization_step(individual, outer_side, max_iter=50):
    """Improved local optimization with multiple restarts and better convergence"""
    def objective_func(params):
        # Reshape parameters back into hexagon data
        hex_data = params.reshape(-1, 3)
        return evaluate_layout(hex_data, outer_side)

    # Flatten current configuration for optimization
    flat_params = individual.flatten()
    
    best_result = None
    best_fitness = 1e10
    
    # Multiple restart strategies for better convergence
    for restart in range(3):
        # Add small random perturbation to starting point
        perturbed_params = flat_params.copy()
        for i in range(len(perturbed_params)):
            if i % 3 < 2:  # Only perturb x,y, not angle
                perturbed_params[i] += np.random.uniform(-0.1, 0.1)
        
        try:
            # Try Nelder-Mead method for better handling of discrete-like constraints
            result = minimize(objective_func, perturbed_params, method='Nelder-Mead',
                             options={'maxiter': max_iter // 3, 'adaptive': True, 'disp': False})
            
            if result.success:
                fitness = result.fun
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_result = result.x
                    
        except:
            continue
    
    # Return best result if found
    if best_result is not None:
        refined_config = best_result.reshape(-1, 3)
        return refined_config
    else:
        return individual

def adaptive_local_search(initial_config, max_iterations=100):
    """Multi-phase local search with adaptive strategies"""
    
    # Phase 1: Coarse search with larger steps
    current_config = initial_config.copy()
    phase1_iterations = max_iterations // 2
    
    for iter_num in range(phase1_iterations):
        # Perturb positions more aggressively in early phases
        perturbed = current_config.copy()
        for i in range(len(perturbed)):
            if np.random.random() < 0.3:  # 30% chance to move each hexagon
                perturbed[i][0] += np.random.normal(0, 0.3)  # Larger moves
                perturbed[i][1] += np.random.normal(0, 0.3)
                perturbed[i][2] += np.random.normal(0, 20)  # Larger angle changes
                perturbed[i][2] = perturbed[i][2] % 360
            
        # Evaluate
        current_fitness = evaluate_layout(current_config)
        candidate_fitness = evaluate_layout(perturbed)
        
        # Accept better solutions
        if candidate_fitness < current_fitness:
            current_config = perturbed
    
    # Phase 2: Fine search with smaller steps
    phase2_iterations = max_iterations - phase1_iterations
    for iter_num in range(phase2_iterations):
        # Perturb with smaller steps
        perturbed = current_config.copy()
        for i in range(len(perturbed)):
            if np.random.random() < 0.5:
                perturbed[i][0] += np.random.normal(0, 0.05)
                perturbed[i][1] += np.random.normal(0, 0.05)
                perturbed[i][2] += np.random.normal(0, 5)
                perturbed[i][2] = perturbed[i][2] % 360
                
        # Evaluate and accept
        current_fitness = evaluate_layout(current_config)
        candidate_fitness = evaluate_layout(perturbed)
        
        if candidate_fitness < current_fitness:
            current_config = perturbed
    
    return current_config

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    try:
        # Generate initial configuration with better heuristic
        current_config = generate_initial_config()
        
        best_eval = evaluate_layout(current_config)
        best_config = current_config.copy()
        
        # Multi-stage optimization with adaptive strategies
        max_generations = 100
        best_improvement_count = 0
        max_no_improvement = 15
        
        for generation in range(max_generations):
            # Evaluate current population
            population = [current_config]
            # Generate diverse offspring for this generation
            for _ in range(15):  # Increase offspring generation
                # Create variation with some geometric constraints
                offspring = current_config.copy()
                
                # Add some structured changes based on hexagonal relationship
                for i in range(len(offspring)):
                    if np.random.random() < 0.4:  # 40% chance per hexagon to change
                        # More structured mutations
                        offspring[i][0] += np.random.normal(0, 0.15)
                        offspring[i][1] += np.random.normal(0, 0.15)
                        offspring[i][2] += np.random.normal(0, 10)
                        offspring[i][2] = offspring[i][2] % 360
                
                population.append(offspring)
            
            # Evaluate all candidates in parallel
            evaluations = parallel_evaluate_population(population)
            
            # Select best from this generation
            min_idx = np.argmin(evaluations)
            if evaluations[min_idx] < best_eval:
                best_eval = evaluations[min_idx]
                best_config = population[min_idx].copy()
                best_improvement_count = 0  # Reset counter
            else:
                best_improvement_count += 1
                
            # Early stopping if no improvement for too long
            if best_improvement_count > max_no_improvement:
                break
                
            # Update current config to the best in this generation
            current_config = population[min_idx]
            
            # Periodic local refinement
            if generation % 10 == 0:
                # Find the optimal radius for current config
                estimated_radius = compute_outer_hexagon_radius(current_config)
                refined_config = local_optimization_step(current_config, estimated_radius * 2.0, 30)
                refined_fitness = evaluate_layout(refined_config)
                if refined_fitness < best_eval:
                    best_eval = refined_fitness
                    best_config = refined_config.copy()
                    best_improvement_count = 0  # Reset counter

        # Final adaptive local search for maximum refinement
        final_config = adaptive_local_search(best_config, max_iterations=100)
        final_fitness = evaluate_layout(final_config)
        
        if final_fitness < best_eval:
            best_eval = final_fitness
            best_config = final_config

        # Ensure we have a valid solution
        if best_eval >= 1e9:
            raise ValueError("No valid solution found")
            
        # Extract the best configuration found
        inner_hex_data = best_config

        # Compute actual outer hexagon size
        estimated_outer_radius = compute_outer_hexagon_radius(inner_hex_data)
        outer_hex_side_length = estimated_outer_radius * 2.0

        # Set outer hexagon at center with zero rotation
        outer_hex_data = np.array([0, 0, 0])

    except Exception as e:
        print(f"Error during optimization: {e}")
        # Fallback to baseline approach
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ])
        outer_hex_side_length = 8.0
        outer_hex_data = np.array([0, 0, 0])

    elapsed = time.time() - start_time
    print(f"Eval time: {elapsed:.4f}s")

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END