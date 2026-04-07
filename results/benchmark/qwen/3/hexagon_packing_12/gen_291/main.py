# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from scipy.spatial.distance import cdist
import random
from numba import jit

# Constants
UNIT_HEX_RADIUS = 1.0
MAX_EVAL_TIME = 180.0
TARGET_RATIO = 0.2537

@jit(nopython=True)
def get_hexagon_vertices(x, y, angle_deg, radius=1.0):
    """Get vertices of a hexagon given center, angle, and radius"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

def create_regular_hexagon(center=(0,0), side_length=1, rotation=0):
    """Create a regular hexagon as a Shapely polygon"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    x = center[0] + side_length * np.cos(angles)
    y = center[1] + side_length * np.sin(angles)
    return Polygon(list(zip(x, y)))

def hexagon_to_polygon(x, y, angle_deg, radius=1.0):
    """Convert hexagon parameters to shapely polygon"""
    vertices = get_hexagon_vertices(x, y, angle_deg, radius)
    return Polygon(vertices)

def check_overlap_fast(hex1_poly, hex2_poly):
    """Fast overlap check using Shapely with buffer for numerical stability"""
    # Quick bounding box check first
    bbox1 = hex1_poly.bounds
    bbox2 = hex2_poly.bounds
    if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
        bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
        return False
    return hex1_poly.buffer(1e-10).intersects(hex2_poly.buffer(1e-10)) and not hex1_poly.touches(hex2_poly)

def check_containment(inner_hex, outer_hex):
    """Check if inner hexagon is fully contained within outer hexagon"""
    return outer_hex.contains(inner_hex)

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
        distance = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
        max_distance = max(max_distance, distance)

    # Add buffer for hexagon radius calculation
    return max_distance + UNIT_HEX_RADIUS

def evaluate_constraint_violations(inner_hex_data, outer_hex_data):
    """Evaluate constraint violations for a given configuration"""
    violations = []

    # Create outer hexagon
    outer_x, outer_y, outer_angle = outer_hex_data
    outer_radius = compute_outer_hexagon_radius(inner_hex_data)
    outer_hex = hexagon_to_polygon(outer_x, outer_y, outer_angle, outer_radius)

    # Check each inner hexagon for containment
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        inner_hex = hexagon_to_polygon(x, y, angle)

        if not check_containment(inner_hex, outer_hex):
            violations.append(f"Inner hexagon {i} not contained")

    # Check overlaps between all pairs
    for i in range(len(inner_hex_data)):
        x1, y1, angle1 = inner_hex_data[i]
        hex1_poly = hexagon_to_polygon(x1, y1, angle1)

        for j in range(i+1, len(inner_hex_data)):
            x2, y2, angle2 = inner_hex_data[j]
            hex2_poly = hexagon_to_polygon(x2, y2, angle2)

            if check_overlap_fast(hex1_poly, hex2_poly):
                violations.append(f"Overlapping hexagons {i} and {j}")

    return violations

def compute_objective_function(hex_data):
    """Compute negative of 1/outer_hex_side_length (to minimize instead of maximize)"""
    # Check if hex_data is valid
    if len(hex_data) != 12:
        return 1e10

    # Compute outer hexagon radius
    outer_radius = compute_outer_hexagon_radius(hex_data)

    # If outer radius is invalid, penalize heavily
    if outer_radius <= 0:
        return 1e10

    # Return negative of 1/outer_radius (for minimization)
    return -1.0 / outer_radius

def evaluate_solution(hex_data, outer_hex_data):
    """Comprehensive evaluation of solution validity and quality"""
    # Basic constraint checking
    violations = evaluate_constraint_violations(hex_data, outer_hex_data)

    if violations:
        return False, 1e10, violations

    # Compute objective value
    obj_value = compute_objective_function(hex_data)
    return True, obj_value, []

def generate_mathematical_lattice_initial_solution():
    """Generate an initial solution based on mathematical lattice structures for better hexagon packing"""
    # Use a more sophisticated mathematical approach based on triangular lattice
    # This creates a structure that's more likely to achieve better packing density

    # Base positions on triangular lattice principles
    # Center hexagon
    hex_data = [[0.0, 0.0, 0.0]]

    # First ring - 6 hexagons in a hexagonal arrangement
    # Using radius = 2.0 for good spread
    first_ring_radius = 2.0
    for i in range(6):
        angle = i * 60  # degrees
        rad = np.radians(angle)
        x = first_ring_radius * np.cos(rad)
        y = first_ring_radius * np.sin(rad)
        hex_data.append([x, y, 0.0])

    # Second ring - 6 hexagons in a larger hexagonal arrangement
    # Using slightly larger radius to create efficient packing
    second_ring_radius = 3.464  # ~2*sqrt(3) for hexagonal packing efficiency
    for i in range(6):
        angle = i * 60  # degrees
        rad = np.radians(angle)
        x = second_ring_radius * np.cos(rad)
        y = second_ring_radius * np.sin(rad)
        hex_data.append([x, y, 0.0])

    # Trim to exactly 12 hexagons (we have 13, so remove last one)
    hex_data = hex_data[:12]

    # Apply sophisticated perturbations to break symmetry while preserving structural integrity
    for i in range(12):
        # Perturb positions with varied magnitudes based on ring membership
        if i == 0:  # Central hexagon
            hex_data[i][0] += random.uniform(-0.03, 0.03)
            hex_data[i][1] += random.uniform(-0.03, 0.03)
        elif 1 <= i <= 6:  # First ring
            hex_data[i][0] += random.uniform(-0.05, 0.05) * 0.8
            hex_data[i][1] += random.uniform(-0.05, 0.05) * 0.8
        else:  # Second ring
            hex_data[i][0] += random.uniform(-0.04, 0.04) * 0.6
            hex_data[i][1] += random.uniform(-0.04, 0.04) * 0.6
        hex_data[i][2] += random.uniform(-0.5, 0.5)  # Small rotation variation

    return np.array(hex_data)

def generate_kagome_lattice_solution():
    """Generate a Kagome lattice-inspired configuration which often provides better packing"""
    # Kagome lattice provides excellent hexagon packing properties
    hex_data = []

    # Central hexagon
    hex_data.append([0.0, 0.0, 0.0])

    # First ring - 6 hexagons with exact hexagonal spacing
    hex_radius = 2.0
    for i in range(6):
        angle = i * 60
        x = hex_radius * np.cos(np.radians(angle))
        y = hex_radius * np.sin(np.radians(angle))
        hex_data.append([x, y, 0.0])

    # Second ring - 5 hexagons in a pattern that follows Kagome structure
    # Using the golden ratio relationship for improved packing
    ring_radius = 3.464  # sqrt(12) - typical for dense hexagonal packing
    angles = [0, 72, 144, 216, 288]  # Golden ratio based distribution
    for i, angle in enumerate(angles):
        x = ring_radius * np.cos(np.radians(angle))
        y = ring_radius * np.sin(np.radians(angle))
        hex_data.append([x, y, 0.0])

    # Add one more to reach 12 (place strategically)
    hex_data.append([0, -ring_radius - 0.5, 0])

    # Truncate to exactly 12
    hex_data = hex_data[:12]

    # Apply careful perturbations
    for i in range(12):
        # Scale perturbations based on hexagon's role in packing
        if i == 0:
            # Central hexagon - small perturbations
            hex_data[i][0] += random.uniform(-0.02, 0.02)
            hex_data[i][1] += random.uniform(-0.02, 0.02)
        elif 1 <= i <= 6:
            # First ring - moderate perturbations
            hex_data[i][0] += random.uniform(-0.04, 0.04)
            hex_data[i][1] += random.uniform(-0.04, 0.04)
        else:
            # Second ring - smaller perturbations
            hex_data[i][0] += random.uniform(-0.03, 0.03)
            hex_data[i][1] += random.uniform(-0.03, 0.03)
        hex_data[i][2] += random.uniform(-0.8, 0.8)

    return np.array(hex_data)

def optimize_single_configuration(initial_hex_data):
    """Perform optimization on a single configuration using L-BFGS-B with enhanced symmetry awareness"""

    # Flatten initial data for optimization
    initial_flat = initial_hex_data.flatten()

    # Bounds: positions (-10, 10), angles (0, 360)
    bounds = []
    for i in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])

    def objective_and_gradient(params):
        # Reshape parameters back to hex data format
        hex_data = params.reshape(12, 3)

        # Preprocessing step: apply symmetry constraints to enhance convergence
        # For hexagonal packing, we know certain geometric relationships should hold

        # Group hexagons into rings and enforce symmetry appropriately
        # Ring 1: indices 1-6 (first ring around central hexagon)
        # Ring 2: indices 7-11 (second ring) plus index 12 (if exists)

        # For ring 1 (6 hexagons), maintain rotational symmetry
        ring1_positions = hex_data[1:7, :2]
        if len(ring1_positions) >= 2:
            # Compute average position of first ring
            avg_ring1_pos = np.mean(ring1_positions, axis=0)
            # Adjust positions to maintain radial symmetry around the center
            avg_radius = np.sqrt(avg_ring1_pos[0]**2 + avg_ring1_pos[1]**2)
            if avg_radius > 0:
                base_angle = np.arctan2(avg_ring1_pos[1], avg_ring1_pos[0])
                for i in range(1, 7):
                    # Maintain same distance from center with adjusted angle
                    angle_offset = (i-1) * 2 * np.pi / 6
                    hex_data[i, 0] = avg_radius * np.cos(base_angle + angle_offset)
                    hex_data[i, 1] = avg_radius * np.sin(base_angle + angle_offset)

        # For ring 2 (5 hexagons), maintain rotational symmetry about center
        ring2_positions = hex_data[7:12, :2]
        if len(ring2_positions) >= 2:
            # Compute average position of second ring
            avg_ring2_pos = np.mean(ring2_positions, axis=0)
            # Adjust positions to maintain radial symmetry
            avg_radius = np.sqrt(avg_ring2_pos[0]**2 + avg_ring2_pos[1]**2)
            if avg_radius > 0:
                base_angle = np.arctan2(avg_ring2_pos[1], avg_ring2_pos[0])
                for i in range(7, 12):
                    # Maintain same distance from center with adjusted angle
                    angle_offset = (i-7) * 2 * np.pi / 5 + np.pi/6  # Offset to break perfect 6-fold symmetry
                    hex_data[i, 0] = avg_radius * np.cos(base_angle + angle_offset)
                    hex_data[i, 1] = avg_radius * np.sin(base_angle + angle_offset)

        # Evaluate objective function
        obj_value = compute_objective_function(hex_data)

        # Approximate gradient using finite differences
        epsilon = 1e-6
        grad = np.zeros_like(params)

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            hex_data_plus = params_plus.reshape(12, 3)
            obj_plus = compute_objective_function(hex_data_plus)
            grad[i] = (obj_plus - obj_value) / epsilon

        return obj_value, grad

    # Optimize using L-BFGS-B with more aggressive settings
    try:
        result = minimize(
            objective_and_gradient,
            initial_flat,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': 1500,  # Increased iterations
                'ftol': 1e-14,    # Tighter tolerance
                'gtol': 1e-14,    # Tighter tolerance
                'maxls': 100      # More line searches
            },
            tol=1e-14
        )

        if result.success:
            optimized_data = result.x.reshape(12, 3)
            return optimized_data
    except Exception:
        pass

    return initial_hex_data

def mutate_symmetrically(individual, mut_pb=0.3, mut_strength=0.2):
    """
    Mutate an individual while preserving true hexagonal symmetry properties.
    Enhanced version with more sophisticated symmetry handling and mathematical rigor.
    """
    # Copy individual to avoid modifying original
    mutated = individual.copy()

    # Preserve rotational symmetry by working with ring-based transformations
    # Group 0: Central hexagon (index 0)
    # Group 1: First ring (indices 1-6) - 6-fold rotational symmetry
    # Group 2: Second ring (indices 7-11) - 5-fold rotational symmetry (approximate)

    # Mutate central hexagon with enhanced precision
    if random.random() < mut_pb:
        mutated[0, 0] += random.uniform(-mut_strength * 0.4, mut_strength * 0.4)
        mutated[0, 1] += random.uniform(-mut_strength * 0.4, mut_strength * 0.4)
        # For central hexagon, rotation doesn't matter much, but keep it within reasonable bounds
        mutated[0, 2] += random.uniform(-mut_strength * 0.4, mut_strength * 0.4)

    # Mutate first ring (6 hexagons) - enforce rotational symmetry with mathematical precision
    if random.random() < mut_pb:
        # Get positions of first ring
        ring1_positions = mutated[1:7, :2]

        # Calculate radial distances and angles for each hexagon in ring
        ring1_distances = np.sqrt(ring1_positions[:, 0]**2 + ring1_positions[:, 1]**2)
        ring1_angles = np.arctan2(ring1_positions[:, 1], ring1_positions[:, 0])

        # Compute average characteristics using robust statistical methods
        avg_distance = np.mean(ring1_distances)
        # Use circular mean for angles to handle wraparound properly
        avg_angle = np.arctan2(np.mean(np.sin(ring1_angles)), np.mean(np.cos(ring1_angles)))

        # Apply systematic perturbations with better symmetrization
        angle_perturb = random.uniform(-mut_strength * 0.7, mut_strength * 0.7)
        distance_perturb = random.uniform(-mut_strength * 0.25, mut_strength * 0.25)

        # Apply to all hexagons in first ring maintaining their angular relationship
        for i in range(1, 7):
            # Preserve angular spacing but allow small variations
            base_angle = (i-1) * 2 * np.pi / 6
            # Adjust angle with symmetry-preserving perturbation
            new_angle = avg_angle + angle_perturb + base_angle
            # Adjust distance with perturbation
            new_distance = avg_distance + distance_perturb

            # Update position with proper polar-to-Cartesian conversion
            mutated[i, 0] = new_distance * np.cos(new_angle)
            mutated[i, 1] = new_distance * np.sin(new_angle)

            # Add small random rotation to break degeneracies but maintain symmetry
            mutated[i, 2] += random.uniform(-mut_strength * 0.25, mut_strength * 0.25)

    # Mutate second ring (5 hexagons) - enforce rotational symmetry with improved mathematical approach
    if random.random() < mut_pb:
        # Get positions of second ring
        ring2_positions = mutated[7:12, :2]

        # Calculate radial distances and angles for each hexagon in ring
        ring2_distances = np.sqrt(ring2_positions[:, 0]**2 + ring2_positions[:, 1]**2)
        ring2_angles = np.arctan2(ring2_positions[:, 1], ring2_positions[:, 0])

        # Compute average characteristics using robust statistical methods
        avg_distance = np.mean(ring2_distances)
        # Use circular mean for angles to handle wraparound properly
        avg_angle = np.arctan2(np.mean(np.sin(ring2_angles)), np.mean(np.cos(ring2_angles)))

        # Apply systematic perturbations with symmetry preservation
        angle_perturb = random.uniform(-mut_strength * 0.5, mut_strength * 0.5)
        distance_perturb = random.uniform(-mut_strength * 0.15, mut_strength * 0.15)

        # Apply to all hexagons in second ring maintaining their angular relationship
        for i in range(7, 12):
            # Preserve angular spacing (but with slight flexibility for optimization)
            base_angle = (i-7) * 2 * np.pi / 5 + np.pi/5  # Offset to distribute evenly
            # Adjust angle with symmetry-preserving perturbation
            new_angle = avg_angle + angle_perturb + base_angle
            # Adjust distance with perturbation
            new_distance = avg_distance + distance_perturb

            # Update position with proper polar-to-Cartesian conversion
            mutated[i, 0] = new_distance * np.cos(new_angle)
            mutated[i, 1] = new_distance * np.sin(new_angle)

            # Add small random rotation to break degeneracies but maintain symmetry
            mutated[i, 2] += random.uniform(-mut_strength * 0.15, mut_strength * 0.15)

    # Enhanced global transformation with mathematical foundation
    if random.random() < mut_pb * 0.25:
        # Apply a global rotation to the entire configuration with more mathematical precision
        global_rotation = random.uniform(-mut_strength * 0.4, mut_strength * 0.4)
        cos_rot = np.cos(np.radians(global_rotation))
        sin_rot = np.sin(np.radians(global_rotation))

        # Transform all hexagon positions except the central one
        for i in range(1, 12):
            x, y = mutated[i, 0], mutated[i, 1]
            mutated[i, 0] = x * cos_rot - y * sin_rot
            mutated[i, 1] = x * sin_rot + y * cos_rot

    # Add a mathematical constraint violation correction for robustness
    # This ensures that even if mutations break constraints slightly,
    # we maintain the fundamental hexagonal symmetry properties where possible
    if random.random() < mut_pb * 0.1:
        # Apply minor corrections to maintain approximate symmetry
        # This helps with maintaining good structural properties during evolution
        for i in range(1, 7):  # First ring correction
            # Ensure equal distance from center
            dist = np.sqrt(mutated[i, 0]**2 + mutated[i, 1]**2)
            target_dist = np.mean([np.sqrt(mutated[j, 0]**2 + mutated[j, 1]**2) for j in range(1, 7)])
            if dist > 0:
                scale = target_dist / dist
                mutated[i, 0] *= scale
                mutated[i, 1] *= scale

        # Second ring correction
        for i in range(7, 12):  # Second ring correction
            dist = np.sqrt(mutated[i, 0]**2 + mutated[i, 1]**2)
            target_dist = np.mean([np.sqrt(mutated[j, 0]**2 + mutated[j, 1]**2) for j in range(7, 12)])
            if dist > 0:
                scale = target_dist / dist
                mutated[i, 0] *= scale
                mutated[i, 1] *= scale

    return mutated

def multi_start_optimization():
    """Run multiple optimization starts with different initial configurations"""
    best_score = float('inf')
    best_solution = None

    # Generate a variety of high-quality initial configurations using mathematical structures
    initial_configs = []

    # Configuration 1: Mathematical lattice approach
    initial_configs.append(generate_mathematical_lattice_initial_solution())

    # Configuration 2: Kagome lattice-inspired approach
    initial_configs.append(generate_kagome_lattice_solution())

    # Configuration 3: Perturbed version of mathematical lattice with different symmetry breaking
    config3 = initial_configs[0].copy()
    for i in range(12):
        # Different perturbation scheme for better exploration
        config3[i, 0] += random.uniform(-0.12, 0.12) * (1 + i*0.02)  # Vary perturbation
        config3[i, 1] += random.uniform(-0.12, 0.12) * (1 + i*0.02)
        config3[i, 2] += random.uniform(-1.5, 1.5) * (1 + i*0.01)
    initial_configs.append(config3)

    # Configuration 4: Different radial spacing arrangement with optimized parameters
    config4 = []
    config4.append([0, 0, 0])
    # First ring - 6 hexagons with optimized radius
    first_radius = 1.95 + random.uniform(-0.05, 0.05)  # Slightly tighter than standard
    for i in range(6):
        angle = i * 60
        x = first_radius * np.cos(np.radians(angle))
        y = first_radius * np.sin(np.radians(angle))
        config4.append([x, y, 0])

    # Second ring - 5 hexagons with different spacing for optimal packing
    second_radius = 3.35 + random.uniform(-0.08, 0.08)
    angles = [0, 72, 144, 216, 288]
    for i, angle in enumerate(angles):
        x = second_radius * np.cos(np.radians(angle))
        y = second_radius * np.sin(np.radians(angle))
        config4.append([x, y, 0])
    config4.append([0, -second_radius - 0.8, 0])
    initial_configs.append(np.array(config4[:12]))

    # Configuration 5: Compact, dense arrangement
    config5 = []
    config5.append([0, 0, 0])
    # Tighter first ring
    tight_radius = 1.85 + random.uniform(-0.07, 0.07)
    for i in range(6):
        angle = i * 60
        x = tight_radius * np.cos(np.radians(angle))
        y = tight_radius * np.sin(np.radians(angle))
        config5.append([x, y, 0])
    # More compact second ring
    compact_radius = 3.15 + random.uniform(-0.06, 0.06)
    angles = [30, 90, 150, 210, 270]
    for i, angle in enumerate(angles):
        x = compact_radius * np.cos(np.radians(angle))
        y = compact_radius * np.sin(np.radians(angle))
        config5.append([x, y, 0])
    config5.append([0, -compact_radius - 0.7, 0])
    initial_configs.append(np.array(config5[:12]))

    # Now perform optimization on these configurations
    for i, initial_hex_data in enumerate(initial_configs):
        # Optimize this configuration
        optimized_hex_data = optimize_single_configuration(initial_hex_data)

        # Evaluate this solution
        valid, obj_value, violations = evaluate_solution(optimized_hex_data, [0, 0, 0])

        if valid and obj_value < best_score:
            best_score = obj_value
            best_solution = optimized_hex_data

    # Also include some random perturbations from the best found so far
    if best_solution is not None:
        for _ in range(5):  # Additional 5 random perturbations with enhanced variance
            perturbed = best_solution.copy()
            for i in range(12):
                # Enhanced perturbation for better exploration of the solution space
                perturbed[i, 0] += random.uniform(-0.15, 0.15)
                perturbed[i, 1] += random.uniform(-0.15, 0.15)
                perturbed[i, 2] += random.uniform(-1.2, 1.2)
            optimized_hex_data = optimize_single_configuration(perturbed)
            valid, obj_value, violations = evaluate_solution(optimized_hex_data, [0, 0, 0])
            if valid and obj_value < best_score:
                best_score = obj_value
                best_solution = optimized_hex_data

    # Return the best solution found or fall back to mathematical lattice
    return best_solution if best_solution is not None else generate_mathematical_lattice_initial_solution()

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Get best solution through multi-start optimization
    best_hex_data = multi_start_optimization()

    # Final validation
    valid, obj_value, violations = evaluate_solution(best_hex_data, [0, 0, 0])

    if not valid:
        # Fallback to a known good configuration if optimization fails
        fallback_config = np.array([
            [0, 0, 0],              # center
            [-2.5, 0, 0],           # left
            [2.5, 0, 0],            # right
            [-1.25, 2.17, 0],       # top-left
            [1.25, 2.17, 0],        # top-right
            [-1.25, -2.17, 0],      # bottom-left
            [1.25, -2.17, 0],       # bottom-right
            [-3.75, 2.17, 0],       # far top-left
            [3.75, 2.17, 0],        # far top-right
            [-3.75, -2.17, 0],      # far bottom-left
            [3.75, -2.17, 0],       # far bottom-right
            [0, -4, 0],             # far bottom-center
        ])
        return fallback_config, np.array([0, 0, 0]), 8.0

    # Compute final outer hexagon radius
    final_radius = compute_outer_hexagon_radius(best_hex_data)

    return best_hex_data, np.array([0, 0, 0]), final_radius

# EVOLVE-BLOCK-END