# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import time
import random

# Constants for unit hexagon geometry
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3) / 2
UNIT_HEX_VERTEX_ANGLE = np.pi / 3
SQRT_3 = np.sqrt(3)

def create_unit_hexagon_vertices(center=(0,0), rotation=0):
    """Create vertices of a unit regular hexagon centered at center with given rotation."""
    vertices = []
    for i in range(6):
        angle = rotation + i * UNIT_HEX_VERTEX_ANGLE
        x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
        y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def compute_outer_hexagon_vertices(center=(0,0), side_length=1.0, rotation=0):
    """Create vertices of the outer hexagon."""
    vertices = []
    for i in range(6):
        angle = rotation + i * UNIT_HEX_VERTEX_ANGLE
        x = center[0] + side_length * np.cos(angle)
        y = center[1] + side_length * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def check_hexagon_containment(inner_vertices, outer_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon using vectorized operations."""
    inner_polygon = Polygon(inner_vertices)
    outer_polygon = Polygon(outer_vertices)
    return outer_polygon.contains(inner_polygon)

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def evaluate_configuration_fast(config):
    """Fast evaluation of configuration with early rejection of invalid states."""
    # Config: 36 position/angle values + 1 outer side length = 37 total
    n = 12
    positions = config[:3*n].reshape(n, 3)
    outer_side_length = config[3*n]

    # Very small outer hexagon is invalid
    if outer_side_length < 1.0:
        return False, 0.0

    # Compute outer hexagon vertices once
    outer_hex_vertices = compute_outer_hexagon_vertices((0,0), outer_side_length)

    # Check containment and overlap in single pass
    for i in range(n):
        center_x, center_y, angle = positions[i]
        hex_vertices = create_unit_hexagon_vertices((center_x, center_y), np.radians(angle))

        # Check containment first (early rejection)
        if not check_hexagon_containment(hex_vertices, outer_hex_vertices):
            return False, 0.0

        # Check overlap with all previous hexagons (symmetry-aware)
        for j in range(i):
            prev_center_x, prev_center_y, prev_angle = positions[j]
            prev_hex_vertices = create_unit_hexagon_vertices((prev_center_x, prev_center_y), np.radians(prev_angle))

            if check_hexagon_overlap(hex_vertices, prev_hex_vertices):
                return False, 0.0

    # If we get here, all checks passed
    return True, outer_side_length

def objective_function(config):
    """Objective function to minimize (negative inverse of outer hexagon side length)."""
    valid, outer_side_length = evaluate_configuration_fast(config)

    if not valid:
        # Heavy penalty for invalid configurations
        return 1e10

    # Return negative inverse (since we want to maximize 1/R)
    return -1.0 / outer_side_length

def generate_initial_symmetric_config():
    """Generate a highly symmetric initial configuration based on hexagonal lattice."""
    config = []

    # Central hexagon
    config.extend([0.0, 0.0, 0.0])

    # First ring: 6 hexagons around center
    for i in range(6):
        angle = i * UNIT_HEX_VERTEX_ANGLE
        x = SQRT_3 * np.cos(angle)  # Circumradius for adjacent hexagons
        y = SQRT_3 * np.sin(angle)
        config.extend([x, y, 0.0])

    # Second ring: 6 hexagons forming a larger ring
    for i in range(6):
        angle = i * UNIT_HEX_VERTEX_ANGLE + UNIT_HEX_VERTEX_ANGLE / 2
        x = 2 * SQRT_3 * np.cos(angle)
        y = 2 * SQRT_3 * np.sin(angle)
        config.extend([x, y, 0.0])

    # Add one extra hexagon to make 12 total
    config.extend([0.0, 2 * SQRT_3, 0.0])

    # Add outer side length parameter (start with larger value to allow optimization)
    config.append(6.0)

    return np.array(config)

def generate_random_config():
    """Generate a random but plausible initial configuration."""
    config = []

    # Generate random positions within a reasonable range
    for i in range(12):
        # Position with some clustering around center
        angle = random.uniform(0, 2*np.pi)
        radius = random.uniform(0, 3.0)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        angle_deg = random.uniform(0, 360)
        config.extend([x, y, angle_deg])

    # Add outer side length parameter
    config.append(random.uniform(5.0, 10.0))

    return np.array(config)

def generate_fallback_config():
    """Generate a fallback configuration when optimization fails."""
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
        [0, -4, 0],         # far bottom-center
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 8  # large enough to contain all inner hexagons

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# Adaptive mutation class with decreasing strength
class AdaptiveMutation:
    def __init__(self, initial_strength=1.0, decay_rate=0.95):
        self.initial_strength = initial_strength
        self.decay_rate = decay_rate
        self.generation = 0

    def get_mutation_strength(self):
        """Return current mutation strength with decay"""
        return self.initial_strength * (self.decay_rate ** self.generation)

    def mutate_config(self, config, max_dist_factor=0.5):
        """Mutate configuration with adaptive strength"""
        mutated = config.copy()
        mutation_strength = self.get_mutation_strength()
        max_dist = max_dist_factor * mutation_strength

        # Apply mutation to all parameters
        for i in range(len(mutated)):
            if random.random() < 0.3:  # 30% chance to mutate each parameter
                if i < 36:  # Position/angle parameters
                    # Apply normal mutation with adaptive strength
                    delta = np.random.normal(0, max_dist)
                    mutated[i] += delta

                    # Keep angles within [0, 360)
                    if i % 3 == 2:  # angle parameter
                        mutated[i] = mutated[i] % 360

                else:  # outer side length parameter
                    delta = np.random.normal(0, max_dist)
                    mutated[i] = max(1.0, mutated[i] + delta)

        # Increment generation counter
        self.generation += 1

        return mutated

def optimize_with_adaptive_grid_refinement():
    """Perform adaptive grid refinement optimization to improve convergence."""
    best_result = None
    best_objective = float('inf')
    max_time = 170  # Leave some buffer for final processing

    # Grid refinement levels: coarse, medium, fine
    refinement_levels = [
        {"grid_spacing": 1.0, "maxiter": 100, "ftol": 1e-5, "gtol": 1e-5},
        {"grid_spacing": 0.5, "maxiter": 150, "ftol": 1e-6, "gtol": 1e-6},
        {"grid_spacing": 0.2, "maxiter": 200, "ftol": 1e-7, "gtol": 1e-7}
    ]

    # Try multiple starting configurations
    configs = []

    # Add symmetric configurations
    for _ in range(3):
        configs.append(generate_initial_symmetric_config())

    # Add random configurations
    for _ in range(2):
        configs.append(generate_random_config())

    # Perform optimization from each start with multi-resolution approach
    for start_idx, initial_config in enumerate(configs):
        print(f"Starting optimization from configuration {start_idx + 1}")

        # Track the best result found so far for this start
        start_best_result = None
        start_best_objective = float('inf')

        # Apply adaptive grid refinement across multiple levels
        for level_idx, level_config in enumerate(refinement_levels):
            if time.time() - start_time > max_time:
                break

            print(f"  Refinement level {level_idx + 1} with spacing {level_config['grid_spacing']}")

            # Create grid-refined version of the current configuration
            if level_idx == 0:
                # Start with original configuration
                current_config = initial_config.copy()
            else:
                # Improve on previous best for this start
                current_config = start_best_result.copy() if start_best_result is not None else initial_config.copy()

            # Apply grid-based mutation to explore neighborhood
            mutated_config = current_config.copy()

            # For grid refinement, we want more structured exploration
            # Adjust parameters based on grid spacing
            grid_mutation_strength = level_config['grid_spacing'] * 0.3

            # Mutate positions to sample around current configuration
            for i in range(12):  # Only mutate position parameters
                if i < 12:  # Position parameters
                    # Apply grid-based perturbations
                    if random.random() < 0.5:  # 50% chance to perturb
                        mutated_config[i*3] += random.uniform(-grid_mutation_strength, grid_mutation_strength)
                        mutated_config[i*3 + 1] += random.uniform(-grid_mutation_strength, grid_mutation_strength)

                        # Ensure we don't go too far from reasonable bounds
                        mutated_config[i*3] = np.clip(mutated_config[i*3], -10, 10)
                        mutated_config[i*3 + 1] = np.clip(mutated_config[i*3 + 1], -10, 10)

            # Stage 1: Position-only optimization with fixed rotations
            config_stage1 = mutated_config.copy()
            for i in range(12):
                config_stage1[i*3 + 2] = 0  # Fix rotations initially

            bounds_pos_only = [(None, None)] * 36
            bounds_pos_only.extend([(1.0, 15.0)])  # Outer side length bound

            try:
                result1 = minimize(
                    lambda x: objective_function(np.concatenate([x, [x[-1]]])),
                    config_stage1[:-1],  # Only positions
                    method='L-BFGSB',
                    bounds=bounds_pos_only[:-1],
                    options={'maxiter': level_config['maxiter'], 'ftol': level_config['ftol'], 'gtol': level_config['gtol']},
                    tol=level_config['ftol']
                )
                stage1_result = result1.x if result1.success else config_stage1[:-1]
            except Exception:
                stage1_result = config_stage1[:-1]

            # Stage 2: Limited rotation adjustment with refined position
            config_stage2 = np.concatenate([stage1_result, [config_stage1[-1]]])

            # Allow some rotation adjustments for hexagons
            for i in range(0, 12, 2):  # Every second hexagon gets rotation adjustment
                config_stage2[i*3 + 2] = random.uniform(0, 30)

            bounds_stage2 = [(None, None)] * 36
            bounds_stage2.extend([(1.0, 15.0)])

            try:
                result2 = minimize(
                    objective_function,
                    config_stage2,
                    method='L-BFGSB',
                    bounds=bounds_stage2,
                    options={'maxiter': level_config['maxiter'], 'ftol': level_config['ftol'], 'gtol': level_config['gtol']},
                    tol=level_config['ftol']
                )
                stage2_result = result2.x if result2.success else config_stage2
            except Exception:
                stage2_result = config_stage2

            # Stage 3: Fine-tuned optimization with full freedom
            try:
                bounds_final = [(None, None)] * 36
                bounds_final.extend([(1.0, 15.0)])

                result_final = minimize(
                    objective_function,
                    stage2_result,
                    method='L-BFGSB',
                    bounds=bounds_final,
                    options={'maxiter': level_config['maxiter'], 'ftol': level_config['ftol'], 'gtol': level_config['gtol']},
                    tol=level_config['ftol']
                )

                final_result = result_final.x if result_final.success else stage2_result

            except Exception as e:
                final_result = stage2_result

            # Evaluate final result at this refinement level
            objective_value = objective_function(final_result)
            if objective_value < start_best_objective:
                start_best_objective = objective_value
                start_best_result = final_result.copy()

            # Early termination if we're approaching the benchmark
            if objective_value < -0.24:  # Near the target
                break

        # Update global best if this start was better
        if start_best_objective < best_objective:
            best_objective = start_best_objective
            best_result = start_best_result.copy()

    return best_result

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    global start_time  # Make start_time accessible globally
    start_time = time.time()

    try:
        final_config = optimize_with_adaptive_grid_refinement()

        # Extract the final configuration
        final_positions = final_config[:-1].reshape(12, 3)
        final_side_length = final_config[-1]

        # Return in the required format
        inner_hex_data = final_positions.copy()
        outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered

        # Validate final result
        valid, _ = evaluate_configuration_fast(final_config)
        if not valid:
            print("Warning: Final configuration may be invalid")
            # Fallback to a better-known configuration
            inner_hex_data, outer_hex_data, final_side_length = generate_fallback_config()

    except Exception as e:
        # Fallback to old method if anything goes wrong
        print(f"Optimization error: {e}")
        inner_hex_data, outer_hex_data, final_side_length = generate_fallback_config()

    end_time = time.time()

    # Calculate performance metrics
    inv_outer_hex_side_length = 1.0 / final_side_length if final_side_length > 0 else 0.0
    benchmark_ratio = inv_outer_hex_side_length / 0.2537

    print(f"Optimized result: inverse_side_length={inv_outer_hex_side_length:.6f}, "
          f"benchmark_ratio={benchmark_ratio:.6f}, eval_time={(end_time-start_time):.3f}s")

    return inner_hex_data, outer_hex_data, final_side_length

# EVOLVE-BLOCK-END