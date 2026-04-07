# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import math
from joblib import Parallel, delayed
import time

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_SIDE = 1.0
PI = np.pi

def create_regular_hexagon(center_x, center_y, side_length=1, rotation_deg=0):
    """Create a regular hexagon as a Shapely polygon"""
    rotation_rad = math.radians(rotation_deg)
    points = []
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        points.append((x, y))
    return Polygon(points)

def hexagon_vertices(center_x, center_y, angle_rad, side_length):
    """Compute hexagon vertices"""
    vertices = []
    for i in range(6):
        angle = angle_rad + i * PI / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def compute_outer_hexagon_radius(inner_hex_data, outer_hex_center=(0,0)):
    """Estimate minimum outer hexagon radius needed to contain all inner hexagons"""
    max_distance = 0
    for i in range(len(inner_hex_data)):
        cx, cy, angle = inner_hex_data[i]
        # Get all vertices of this hexagon
        vertices = hexagon_vertices(cx, cy, math.radians(angle), UNIT_HEX_SIDE)
        # Find maximum distance from center
        for vx, vy in vertices:
            dist = math.sqrt((vx - outer_hex_center[0])**2 + (vy - outer_hex_center[1])**2)
            max_distance = max(max_distance, dist)

    # Add safety margin for numerical precision
    return max_distance * 1.02  # Reduced margin for better packing

def check_all_overlaps(inner_hex_data):
    """Check all pairs of hexagons for overlaps with early termination and optimized geometry"""
    n = len(inner_hex_data)
    if n < 2:
        return False

    # Pre-compute hexagons for faster access
    hexagons = []
    for i in range(n):
        cx, cy, angle = inner_hex_data[i]
        hexagons.append(create_regular_hexagon(cx, cy, UNIT_HEX_SIDE, angle))

    # Check only unique pairs
    for i in range(n):
        for j in range(i+1, n):
            if hexagons[i].intersects(hexagons[j]):
                return True
    return False

def check_containment(inner_hex_data, outer_center=(0,0), outer_side=10):
    """Check if all inner hexagons are contained in outer hexagon"""
    outer_vertices = hexagon_vertices(outer_center[0], outer_center[1], 0, outer_side)
    outer_polygon = Polygon(outer_vertices)

    for i in range(len(inner_hex_data)):
        cx, cy, angle = inner_hex_data[i]
        vertices = hexagon_vertices(cx, cy, math.radians(angle), UNIT_HEX_SIDE)
        inner_polygon = Polygon(vertices)

        if not outer_polygon.contains(inner_polygon):
            return False

    return True

def adaptive_binary_search_radius(inner_hex_data, initial_guess, max_iterations=30):
    """Adaptive binary search with improved bounds and convergence"""
    # Start with a reasonable range based on known properties
    low = initial_guess * 0.99
    high = initial_guess * 1.2

    # Set initial tolerance - start more coarse and get finer
    tolerance = 1e-4

    for iteration in range(max_iterations):
        if high - low < tolerance:
            break

        # Adjust precision based on iteration
        if iteration > 15:
            tolerance = 1e-6
        elif iteration > 10:
            tolerance = 1e-5

        mid = (low + high) / 2

        # Check containment with safety margin
        if check_containment(inner_hex_data, (0,0), mid * 1.005):  # Increased safety margin
            high = mid
        else:
            low = mid

    return (low + high) / 2

def evaluate_layout(inner_hex_data, outer_side_estimate=None):
    """Evaluate the quality of a given hexagon layout"""
    # Check overlaps first (early rejection)
    if check_all_overlaps(inner_hex_data):
        return 1e10  # Large penalty for overlaps

    # Estimate outer hexagon size
    if outer_side_estimate is None:
        estimated_outer_radius = compute_outer_hexagon_radius(inner_hex_data)
        outer_side = estimated_outer_radius * 1.5  # More conservative estimate
    else:
        outer_side = outer_side_estimate

    # Check containment
    if not check_containment(inner_hex_data, (0,0), outer_side):
        return 1e10  # Large penalty for containment violations

    # Refine outer radius with binary search for tighter bound
    try:
        refined_radius = adaptive_binary_search_radius(inner_hex_data, outer_side)
        # Return inverse of outer hexagon side length (we want to maximize 1/outer_side)
        return 1.0 / refined_radius
    except:
        # Fallback to simple calculation
        return 1.0 / outer_side

def generate_smart_initial_config():
    """Generate a smart initial configuration using hexagonal close packing with better spatial distribution"""
    # Based on hexagonal close packing principles with optimized spacing
    # Arrange in three rings: center, first ring (6 hexagons), second ring (4 hexagons)
    config = np.array([
        [0.0, 0.0, 0.0],           # Center hexagon
        [-2.0, 0.0, 0.0],          # Left hexagon
        [2.0, 0.0, 0.0],           # Right hexagon
        [0.0, 3.464, 0.0],         # Top hexagon
        [0.0, -3.464, 0.0],        # Bottom hexagon
        [-1.0, 1.732, 0.0],        # Top-left hexagon
        [1.0, 1.732, 0.0],         # Top-right hexagon
        [-1.0, -1.732, 0.0],       # Bottom-left hexagon
        [1.0, -1.732, 0.0],        # Bottom-right hexagon
        [-2.5, 2.17, 0.0],         # Far top-left
        [2.5, 2.17, 0.0],          # Far top-right
    ])

    # Add small random jitter to avoid degenerate cases
    np.random.seed(42)
    noise_scale = 0.01  # Reduced noise to maintain structure
    config[:, 0] += np.random.normal(0, noise_scale, config.shape[0])
    config[:, 1] += np.random.normal(0, noise_scale, config.shape[0])

    return config

def adaptive_optimization_step(current_config, iteration, max_iter, base_mutation_rate=0.3):
    """Perform one step of adaptive optimization with dynamic parameters"""
    # Adaptive mutation rate based on iteration progress
    progress = iteration / max_iter
    adaptive_mutation_rate = base_mutation_rate * (1 - progress * 0.7)

    perturbed = current_config.copy()

    # Perturb hexagons with adaptive strategy
    n_to_perturb = max(1, int(len(perturbed) * 0.4))
    indices_to_perturb = np.random.choice(len(perturbed), n_to_perturb, replace=False)

    for idx in indices_to_perturb:
        # Apply adaptive perturbations
        perturbed[idx][0] += np.random.normal(0, 0.15 * adaptive_mutation_rate)
        perturbed[idx][1] += np.random.normal(0, 0.15 * adaptive_mutation_rate)
        perturbed[idx][2] += np.random.normal(0, 10 * adaptive_mutation_rate) % 360

    return perturbed

def parallel_evaluate_population(population):
    """Evaluate multiple configurations in parallel"""
    results = Parallel(n_jobs=-1, verbose=0)(delayed(evaluate_layout)(config) for config in population)
    return results

def multi_stage_optimization():
    """Multi-stage optimization approach with enhanced parameters"""
    # Stage 1: Generate good initial guess
    current_config = generate_smart_initial_config()

    best_eval = evaluate_layout(current_config)
    best_config = current_config.copy()

    # Stage 2: Evolutionary search with adaptive parameters
    max_iterations = 80  # Increased iterations
    stagnation_count = 0
    max_stagnation = 15  # Increased tolerance
    population_size = 25  # Larger population

    for epoch in range(max_iterations):
        # Create offspring through adaptive mutation
        offspring = []
        for _ in range(population_size):
            mutated = adaptive_optimization_step(current_config, epoch, max_iterations)
            offspring.append(mutated)

        # Evaluate all offspring in parallel
        evaluations = parallel_evaluate_population(offspring)

        # Select best offspring
        min_idx = np.argmin(evaluations)
        if evaluations[min_idx] < best_eval:
            best_eval = evaluations[min_idx]
            best_config = offspring[min_idx].copy()
            stagnation_count = 0  # Reset stagnation counter
        else:
            stagnation_count += 1

        # Early stopping if no improvement for too long
        if stagnation_count > max_stagnation:
            break

        # Update current config
        current_config = offspring[min_idx]

    return best_config, best_eval

def optimize_with_local_refinement(initial_config):
    """Use multi-start local optimization to refine the solution"""
    def objective_func(params):
        # Reshape parameters back into hexagon data
        hex_data = params.reshape(-1, 3)
        return evaluate_layout(hex_data)

    best_result = initial_config
    best_score = evaluate_layout(initial_config)

    # Multi-start approach with different initializations
    for start in range(3):  # Try 3 different starting points
        # Add small random perturbation to the current config
        perturbed = initial_config.copy()
        np.random.seed(start)
        noise = np.random.normal(0, 0.05, perturbed.shape)
        perturbed += noise

        # Flatten current configuration for optimization
        flat_params = perturbed.flatten()

        # Local optimization using L-BFGS-B
        try:
            result = minimize(objective_func, flat_params, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8})

            # Reshape back to 2D array
            refined_config = result.x.reshape(-1, 3)
            score = evaluate_layout(refined_config)

            if score > best_score:
                best_score = score
                best_result = refined_config

        except:
            # If local optimization fails, continue with current best
            continue

    return best_result

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
        # Multi-stage optimization with adaptive parameters
        initial_config, initial_eval = multi_stage_optimization()

        # Local refinement with multi-start approach
        refined_config = optimize_with_local_refinement(initial_config)

        # Final evaluation
        final_eval = evaluate_layout(refined_config)

        # Ensure we have a valid solution
        if final_eval >= 1e9:
            # Fall back to original method if optimization fails
            print("Optimization failed, falling back to basic configuration")
            inner_hex_data = np.array([
                [0, 0, 0],          # center
                [-2.5, 0, 0],       # left
                [2.5, 0, 0],        # right
                [-1.25, 2.17, 0],   # top-left
                [1.25, 2.17, 0],    # top-right
                [-1.25, -2.17, 0],  # bottom-left
                [1.25, -2.17, 0],   # bottom-right
                [-3.75, 2.17, 0],   # far top-left
                [3.75, 2.17, 0],    # far top-right
                [-3.75, -2.17, 0],  # far bottom-left
                [3.75, -2.17, 0],   # far bottom-right
            ])
            outer_hex_side_length = 8.0
        else:
            # Extract the best configuration found
            inner_hex_data = refined_config

            # Compute actual outer hexagon size
            estimated_outer_radius = compute_outer_hexagon_radius(inner_hex_data)
            outer_hex_side_length = adaptive_binary_search_radius(inner_hex_data, estimated_outer_radius)

        # Set outer hexagon at center with zero rotation
        outer_hex_data = np.array([0, 0, 0])

        # Validate solution with comprehensive checks
        if (check_all_overlaps(inner_hex_data) or
            not check_containment(inner_hex_data, (0,0), outer_hex_side_length)):
            # If validation fails, fall back to a known good configuration
            print("Validation failed, using fallback configuration")
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

    except Exception as e:
        print(f"Exception in optimization: {e}")
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