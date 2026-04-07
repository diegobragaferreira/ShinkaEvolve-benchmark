# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
import time

# Constants for hexagons
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_WIDTH = 2.0  # Distance between parallel sides
UNIT_HEX_SIDE_LENGTH = 1.0  # Side length of unit hexagon
MAX_EVAL_TIME = 180.0  # seconds
TARGET_RATIO = 0.2537  # Benchmark ratio we're trying to beat

class SymmetryAwareMutation:
    """Custom mutation operator that respects hexagonal symmetries"""

    @staticmethod
    def mutate_symmetrically(individual, mutation_strength=0.5, stage=1, temperature=1.0, convergence_progress=0.0):
        """
        Apply mutation that maintains certain symmetry properties
        stage: 1=initial exploration, 2=refinement, 3=final polishing
        temperature: controls randomness in early stages
        convergence_progress: 0.0 to 1.0 indicating how much optimization has progressed
        """
        # Determine mutation strength based on stage, temperature, and convergence progress
        if stage == 1:
            # More aggressive for initial exploration with adaptive strength based on convergence
            current_strength = mutation_strength * 1.5 * temperature * (1.0 - convergence_progress * 0.5)
        elif stage == 2:
            # Moderate for refinement with convergence-based adjustment
            current_strength = mutation_strength * 0.8 * temperature * (1.0 - convergence_progress * 0.3)
        else:  # stage == 3
            # Conservative for final polishing with strong convergence influence
            current_strength = mutation_strength * 0.3 * temperature * (1.0 - convergence_progress * 0.8)

        # Apply mutations to positions (first 24 values of 36 total)
        mutated = individual.copy()
        for i in range(12):  # 12 hexagons
            # Mutate positions (x, y) with some symmetry consideration
            for j in range(2):  # x and y coordinates
                # Apply Gaussian mutation with adaptive strength
                mutated[i*3 + j] += np.random.normal(0, current_strength)

            # Mutate rotation
            mutated[i*3 + 2] += np.random.normal(0, current_strength * 0.5)
            # Keep rotation within [0, 360) range
            mutated[i*3 + 2] = mutated[i*3 + 2] % 360

        return mutated

    @staticmethod
    def generate_symmetric_configurations():
        """Generate mathematically-derived symmetric configurations"""
        configs = []

        # Configuration 1: Kagome lattice pattern (triangular packing)
        config1 = np.array([
            [0.0, 0.0, 0.0],       # Center
            [0.0, 2.0, 0.0],       # Top
            [1.732, 1.0, 0.0],     # Top right
            [1.732, -1.0, 0.0],    # Bottom right
            [0.0, -2.0, 0.0],      # Bottom
            [-1.732, -1.0, 0.0],   # Bottom left
            [-1.732, 1.0, 0.0],    # Top left
            [3.464, 0.0, 0.0],     # Far right
            [3.464, 2.0, 0.0],     # Far top right
            [3.464, -2.0, 0.0],    # Far bottom right
            [-3.464, 0.0, 0.0],    # Far left
            [-3.464, 2.0, 0.0],    # Far top left
        ])
        configs.append(config1)

        # Configuration 2: Hexagonal Close Packed (HCP) structure
        config2 = np.array([
            [0.0, 0.0, 0.0],       # Center
            [0.0, 2.0, 0.0],       # Top
            [1.732, 1.0, 0.0],     # Top right
            [1.732, -1.0, 0.0],    # Bottom right
            [0.0, -2.0, 0.0],      # Bottom
            [-1.732, -1.0, 0.0],   # Bottom left
            [-1.732, 1.0, 0.0],    # Top left
            [3.464, 0.0, 0.0],     # Far right
            [-3.464, 0.0, 0.0],    # Far left
            [0.0, 3.464, 0.0],     # Very top
            [0.0, -3.464, 0.0],    # Very bottom
            [1.732, 2.0, 0.0],     # Corner adjustment
        ])
        configs.append(config2)

        # Configuration 3: Dihedral symmetry pattern (D6)
        config3 = np.array([
            [0.0, 0.0, 0.0],       # Center
            [0.0, 2.0, 0.0],       # Top
            [1.732, 1.0, 0.0],     # Top right
            [1.732, -1.0, 0.0],    # Bottom right
            [0.0, -2.0, 0.0],      # Bottom
            [-1.732, -1.0, 0.0],   # Bottom left
            [-1.732, 1.0, 0.0],    # Top left
            [3.464, 0.0, 0.0],     # Far right
            [3.464, 2.0, 0.0],     # Far top right
            [3.464, -2.0, 0.0],    # Far bottom right
            [-3.464, 0.0, 0.0],    # Far left
            [-3.464, 2.0, 0.0],    # Far top left
        ])
        configs.append(config3)

        return configs

def get_hexagon_vertices(x, y, angle_deg, radius=1.0):
    """Get vertices of a hexagon given center, angle, and radius"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

def hexagon_contains_point(x, y, point_x, point_y, radius=1.0):
    """Check if a point is inside a hexagon centered at (x,y)"""
    # Convert point to hexagon coordinate system
    vertices = get_hexagon_vertices(x, y, 0, radius)
    # Simple check using polygon containment
    hex_poly = Polygon(vertices)
    pt = Point(point_x, point_y)
    return hex_poly.contains(pt)

def compute_outer_hexagon_radius(inner_hex_data):
    """Compute minimum outer hexagon radius that contains all inner hexagons"""
    if len(inner_hex_data) == 0:
        return 0.0

    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)
        all_vertices.extend(vertices)

    if len(all_vertices) == 0:
        return 0.0

    # Compute centroid
    centroid_x = np.mean([v[0] for v in all_vertices])
    centroid_y = np.mean([v[1] for v in all_vertices])

    # Find maximum distance from centroid to any vertex
    max_distance = 0.0
    for x, y in all_vertices:
        distance = math.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
        max_distance = max(max_distance, distance)

    return max_distance + UNIT_HEX_RADIUS

def check_hexagon_overlap(h1_center_x, h1_center_y, h1_angle, h2_center_x, h2_center_y, h2_angle):
    """Check if two hexagons overlap using their vertices"""
    vertices1 = get_hexagon_vertices(h1_center_x, h1_center_y, h1_angle)
    vertices2 = get_hexagon_vertices(h2_center_x, h2_center_y, h2_angle)
    poly1 = Polygon(vertices1)
    poly2 = Polygon(vertices2)
    return poly1.intersects(poly2) and not poly1.touches(poly2)

def validate_solution(inner_hex_data):
    """Validate that solution meets all constraints with buffer for numerical precision"""
    if len(inner_hex_data) != 12:
        return False, "Wrong number of hexagons"

    # Check for overlaps between any pair of hexagons using buffer for numerical precision
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            x1, y1, angle1 = inner_hex_data[i]
            x2, y2, angle2 = inner_hex_data[j]
            if check_hexagon_overlap(x1, y1, angle1, x2, y2, angle2):
                return False, f"Overlapping hexagons {i} and {j}"

    # Compute outer hexagon size needed to contain all
    outer_radius = compute_outer_hexagon_radius(inner_hex_data)

    # Check containment within a hexagon of this size with buffer
    outer_vertices = get_hexagon_vertices(0, 0, 0, outer_radius)
    outer_poly = Polygon(outer_vertices).buffer(1e-10)  # Add small buffer for numerical issues

    # Check if all inner hexagon vertices are inside outer hexagon
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        inner_vertices = get_hexagon_vertices(x, y, angle)
        for vx, vy in inner_vertices:
            inner_point = Point(vx, vy)
            if not outer_poly.contains(inner_point):
                return False, f"Hexagon {i} vertex ({vx}, {vy}) outside outer hexagon"

    return True, "Valid solution"

def optimize_with_local_search(initial_config, max_iter=100, temperature=1.0, stage=1):
    """Refine configuration using local search with symmetry awareness and multi-level approach"""
    # Flatten the initial configuration
    params = initial_config.flatten()

    # Objective function to minimize
    def objective(x):
        # Reshape back to hexagon configuration
        hex_config = x.reshape(12, 3)

        # Validate and penalize invalid solutions
        valid, msg = validate_solution(hex_config)
        if not valid:
            return 1e10  # High penalty for invalid solutions

        # Minimize negative of 1/outer_radius (i.e., maximize 1/outer_radius)
        outer_radius = compute_outer_hexagon_radius(hex_config)
        if outer_radius <= 0:
            return 1e10
        return -1.0 / outer_radius

    # Multi-level optimization approach
    best_result = None
    best_fitness = -float('inf')

    # Level 1: Coarse optimization
    try:
        result = minimize(
            objective,
            params,
            method='L-BFGS-B',
            bounds=[(-10.0, 10.0)] * 36,
            options={'maxiter': max_iter // 3, 'ftol': 1e-6}
        )
        if result.success:
            level1_config = result.x.reshape(12, 3)
            level1_fitness = 1.0 / compute_outer_hexagon_radius(level1_config) if compute_outer_hexagon_radius(level1_config) > 0 else 0.0
            if level1_fitness > best_fitness:
                best_fitness = level1_fitness
                best_result = level1_config
    except Exception as e:
        pass

    # Level 2: Fine optimization
    if best_result is not None:
        try:
            result = minimize(
                objective,
                best_result.flatten(),
                method='L-BFGS-B',
                bounds=[(-10.0, 10.0)] * 36,
                options={'maxiter': max_iter // 2, 'ftol': 1e-8}
            )
            if result.success:
                level2_config = result.x.reshape(12, 3)
                level2_fitness = 1.0 / compute_outer_hexagon_radius(level2_config) if compute_outer_hexagon_radius(level2_config) > 0 else 0.0
                if level2_fitness > best_fitness:
                    best_fitness = level2_fitness
                    best_result = level2_config
        except Exception as e:
            pass

    # Level 3: Very fine optimization
    if best_result is not None:
        try:
            result = minimize(
                objective,
                best_result.flatten(),
                method='L-BFGS-B',
                bounds=[(-10.0, 10.0)] * 36,
                options={'maxiter': max_iter // 2, 'ftol': 1e-10}
            )
            if result.success:
                level3_config = result.x.reshape(12, 3)
                level3_fitness = 1.0 / compute_outer_hexagon_radius(level3_config) if compute_outer_hexagon_radius(level3_config) > 0 else 0.0
                if level3_fitness > best_fitness:
                    best_fitness = level3_fitness
                    best_result = level3_config
        except Exception as e:
            pass

    # Return best result found, fallback to original if nothing worked
    if best_result is not None:
        return best_result
    else:
        return initial_config

def simulate_annealing(initial_config, max_iterations=500, initial_temp=1.0, cooling_rate=0.95):
    """Apply simulated annealing to escape local optima"""
    current_config = initial_config.copy()
    current_fitness = 1.0 / compute_outer_hexagon_radius(current_config) if compute_outer_hexagon_radius(current_config) > 0 else 0.0
    best_config = current_config.copy()
    best_fitness = current_fitness
    temperature = initial_temp

    # Track convergence progress
    stagnation_count = 0
    stagnation_threshold = 20
    prev_best_fitness = best_fitness

    for iteration in range(max_iterations):
        # Generate neighbor solution with temperature-based mutation
        mutated_config = SymmetryAwareMutation.mutate_symmetrically(
            current_config.flatten(),
            mutation_strength=0.3,
            stage=1,
            temperature=temperature,
            convergence_progress=iteration / max_iterations
        ).reshape(12, 3)

        # Evaluate neighbor
        valid, _ = validate_solution(mutated_config)
        if valid:
            mutated_fitness = 1.0 / compute_outer_hexagon_radius(mutated_config) if compute_outer_hexagon_radius(mutated_config) > 0 else 0.0
        else:
            mutated_fitness = 0  # Invalid solutions get low fitness

        # Accept or reject based on Metropolis criterion
        if mutated_fitness > current_fitness or random.random() < math.exp((mutated_fitness - current_fitness) / temperature):
            current_config = mutated_config.copy()
            current_fitness = mutated_fitness

            if mutated_fitness > best_fitness:
                best_config = mutated_config.copy()
                best_fitness = mutated_fitness
                stagnation_count = 0  # Reset stagnation counter on improvement
            else:
                stagnation_count += 1
        else:
            stagnation_count += 1

        # Adaptive cooling based on convergence
        if stagnation_count > stagnation_threshold:
            # Increase cooling rate when stagnating to speed up convergence
            cooling_rate = min(cooling_rate * 1.05, 0.99)
            stagnation_count = 0  # Reset stagnation counter

        # Cool down
        temperature *= cooling_rate

        # Further adaptive cooling when we're close to target
        if best_fitness >= TARGET_RATIO * 0.95:
            temperature *= 0.98  # Aggressive cooling near target

        # Early stopping if we've reached a good solution
        if best_fitness >= TARGET_RATIO:
            break

        # Reduce temperature more aggressively after significant improvements
        if best_fitness > prev_best_fitness * 1.01:  # If improvement is significant
            temperature *= 0.92
            prev_best_fitness = best_fitness

    return best_config

def optimize_hexagon_arrangement():
    """Use a multi-stage optimization with symmetry awareness and simulated annealing"""
    # Get mathematically-derived symmetric configurations
    initial_configs = SymmetryAwareMutation.generate_symmetric_configurations()

    # Add some stochastic perturbations to increase diversity
    for i in range(len(initial_configs)):
        # Add random perturbations to each configuration
        perturbed = initial_configs[i].copy()
        for j in range(12):
            # Perturb positions slightly
            perturbed[j][0] += np.random.normal(0, 0.1)
            perturbed[j][1] += np.random.normal(0, 0.1)
        initial_configs.append(perturbed)

    best_config = None
    best_fitness = -float('inf')
    best_outer_radius = float('inf')

    for i, initial_config in enumerate(initial_configs):
        # Stage 1: Simulated annealing for global exploration
        sa_config = simulate_annealing(initial_config, max_iterations=200)

        # Stage 2: Local optimization for refinement
        refined_config = optimize_with_local_search(sa_config, max_iter=150, stage=2)

        # Stage 3: Final polishing with lower temperature
        final_config = optimize_with_local_search(refined_config, max_iter=100, temperature=0.1, stage=3)

        # Evaluate the final configuration
        valid, msg = validate_solution(final_config)
        if valid:
            outer_radius = compute_outer_hexagon_radius(final_config)
            fitness = 1.0 / outer_radius if outer_radius > 0 else 0.0
            if fitness > best_fitness:
                best_fitness = fitness
                best_config = final_config.copy()
                best_outer_radius = outer_radius

    # If none of the initial configs worked, return a default configuration
    if best_config is None:
        # Default configuration that's known to work reasonably well
        best_config = np.array([
            [0.0, 0.0, 0.0],       # Center
            [0.0, 2.0, 0.0],       # Top
            [0.0, -2.0, 0.0],      # Bottom
            [1.732, 1.0, 0.0],     # Top right
            [-1.732, 1.0, 0.0],    # Top left
            [1.732, -1.0, 0.0],    # Bottom right
            [-1.732, -1.0, 0.0],   # Bottom left
            [3.464, 0.0, 0.0],     # Far right
            [-3.464, 0.0, 0.0],    # Far left
            [0.0, 3.464, 0.0],     # Very top
            [0.0, -3.464, 0.0],    # Very bottom
            [1.732, 2.0, 0.0],     # Additional corner
        ])

    return best_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Start with a symmetry-aware optimized configuration
    inner_hex_data = optimize_hexagon_arrangement()

    # Compute the outer hexagon radius required
    outer_radius = compute_outer_hexagon_radius(inner_hex_data)

    # Convert to side length (for regular hexagon, radius = side length)
    outer_hex_side_length = outer_radius

    # Outer hexagon centered at origin, no rotation
    outer_hex_data = np.array([0, 0, 0])

    # Validate final solution
    valid, message = validate_solution(inner_hex_data)
    if not valid:
        # Fallback to a simple but safe configuration
        inner_hex_data = np.array([
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right,
            [0, -4, 0],  # far bottom-center
        ])
        outer_hex_side_length = 8

    end_time = time.time()
    eval_time = end_time - start_time

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END