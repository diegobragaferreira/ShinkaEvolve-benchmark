# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.spatial.transform import Rotation as R
import random

# Constants
UNIT_HEXAGON_RADIUS = 1.0  # Circumradius of unit hexagon
UNIT_HEXAGON_APOGEE = np.sqrt(3)/2  # Apothem of unit hexagon
UNIT_HEXAGON_VERTEX_ANGLE = np.pi/3  # Angle between adjacent vertices
PI_3 = np.pi/3
SQRT_3 = np.sqrt(3)

def create_unit_hexagon_vertices(center=(0,0), rotation=0):
    """Create vertices of a unit regular hexagon centered at center with given rotation."""
    vertices = []
    for i in range(6):
        angle = rotation + i * UNIT_HEXAGON_VERTEX_ANGLE
        x = center[0] + UNIT_HEXAGON_RADIUS * np.cos(angle)
        y = center[1] + UNIT_HEXAGON_RADIUS * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def check_hexagon_containment(inner_hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon."""
    inner_polygon = Polygon(inner_hex_vertices)
    outer_polygon = Polygon(outer_hex_vertices)

    # Check if inner polygon is completely contained within outer polygon
    return outer_polygon.contains(inner_polygon)

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)

    # Return True if they overlap (intersection area > 0)
    return poly1.intersects(poly2)

def compute_outer_hexagon_vertices(center=(0,0), side_length=1.0, rotation=0):
    """Create vertices of the outer hexagon."""
    vertices = []
    for i in range(6):
        angle = rotation + i * UNIT_HEXAGON_VERTEX_ANGLE
        x = center[0] + side_length * np.cos(angle)
        y = center[1] + side_length * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def compute_inner_hex_positions(config, outer_side_length):
    """Compute actual hexagon positions from configuration."""
    # Config should be 12*(x,y,angle) = 36 values
    positions = config.reshape(12, 3)

    # Ensure hexagons don't exceed outer boundary
    hex_vertices_list = []
    for i, (x, y, angle) in enumerate(positions):
        # Create hexagon vertices with given position and rotation
        hex_v = create_unit_hexagon_vertices((x, y), np.radians(angle))
        hex_vertices_list.append(hex_v)

    return hex_vertices_list

def evaluate_configuration(config, outer_side_length):
    """Evaluate a configuration of hexagon positions."""
    # Config should be flattened array of 12*(x,y,angle) = 36 values
    hex_vertices_list = compute_inner_hex_positions(config, outer_side_length)

    # Test containment
    outer_hex_vertices = compute_outer_hexagon_vertices((0,0), outer_side_length)

    # Check if all inner hexgons are contained
    for hex_v in hex_vertices_list:
        if not check_hexagon_containment(hex_v, outer_hex_vertices):
            return False

    # Check for overlaps
    n = len(hex_vertices_list)
    for i in range(n):
        for j in range(i+1, n):
            if check_hexagon_overlap(hex_vertices_list[i], hex_vertices_list[j]):
                return False

    return True

def objective_function(config):
    """Objective function to minimize (negative inverse of outer hexagon side length)."""
    # Extract outer side length (last value in config)
    outer_side_length = config[-1]

    # If outer side length is too small, penalize heavily
    if outer_side_length < 1.0:
        return 1e10

    # Check validity of configuration
    if not evaluate_configuration(config[:-1], outer_side_length):
        return 1e10

    # Return negative inverse (since we want to maximize 1/R)
    return -1.0 / outer_side_length

def generate_initial_symmetric_config():
    """Generate a symmetric initial configuration for 12 hexagons using hexagonal lattice principles."""
    # This configuration is based on hexagonal close packing principles
    # with mathematical relationships that enhance packing density

    # Initialize with known high-density patterns
    config = []

    # Central hexagon
    config.extend([0.0, 0.0, 0.0])

    # First ring - 6 hexagons forming a tight hexagonal arrangement
    # Using spacing of 2*unit_radius = 2 for optimal packing in hexagonal lattice
    for i in range(6):
        angle = i * PI_3
        # Spacing of 2.0 ensures minimal overlap while maintaining good packing
        x = 2.0 * UNIT_HEXAGON_RADIUS * np.cos(angle)
        y = 2.0 * UNIT_HEXAGON_RADIUS * np.sin(angle)
        config.extend([x, y, 0.0])

    # Second ring - 5 hexagons arranged in a larger hexagon pattern
    # Placed at distance of ~3.464 (sqrt(12)) to allow for good packing
    angles = [0, 72, 144, 216, 288]  # 5 evenly distributed angles
    for angle_deg in angles:
        angle_rad = np.radians(angle_deg)
        # Distance of sqrt(12) approximately 3.464 to maintain hexagonal symmetry
        x = 3.464 * UNIT_HEXAGON_RADIUS * np.cos(angle_rad)
        y = 3.464 * UNIT_HEXAGON_RADIUS * np.sin(angle_rad)
        config.extend([x, y, 0.0])

    # Final hexagon to complete the 12-count
    config.extend([0.0, -3.464 * UNIT_HEXAGON_RADIUS, 0.0])

    # Add outer side length parameter (this will be optimized)
    # Initial estimate based on maximum expected distance plus buffer
    config.append(6.0)

    return np.array(config)

def create_hexagonal_lattice_pattern():
    """Create a hexagonal lattice pattern that respects mathematical hexagonal symmetry."""
    # This creates a configuration that is mathematically derived from
    # hexagonal close packing principles to achieve high packing density

    positions = []

    # Central position
    positions.append([0.0, 0.0, 0.0])

    # First shell - 6 hexagons arranged in a perfect hexagon
    shell1_radius = 2.0  # Unit hexagon radii spacing
    for i in range(6):
        angle = i * 60  # 60 degree increments for hexagonal symmetry
        rad = np.radians(angle)
        x = shell1_radius * np.cos(rad)
        y = shell1_radius * np.sin(rad)
        positions.append([x, y, 0.0])

    # Second shell - 6 hexagons in a larger hexagon
    shell2_radius = 3.464  # Approximately sqrt(12) for optimal hexagonal packing
    for i in range(6):
        angle = i * 60  # Maintain hexagonal symmetry
        rad = np.radians(angle)
        x = shell2_radius * np.cos(rad)
        y = shell2_radius * np.sin(rad)
        positions.append([x, y, 0.0])

    # Trim to exactly 12 hexagons with careful selection
    while len(positions) < 12:
        positions.append([0.0, 0.0, 0.0])

    return np.array(positions[:12])

class AdvancedSymmetryAwareMutation:
    """Enhanced mutation operator with deeper symmetry understanding."""

    def __init__(self):
        self.initial_mut_rate = 0.1
        self.initial_max_dist = 0.5
        self.initial_angle_std = 10.0
        self.mutation_decay_factor = 0.95
        self.symmetry_preservation_weight = 0.7  # Weight for symmetry preservation

    def _apply_hexagonal_symmetry_constraints(self, positions, indices):
        """Apply hexagonal symmetry constraints to maintain structural integrity."""
        # For hexagonal symmetry, positions should respect rotational invariance
        # This maintains the mathematical relationships between hexagon locations

        # If working with symmetric clusters, propagate changes according to symmetry
        if len(indices) >= 6:
            # For rings of 6 hexagons, maintain rotational symmetry
            base_positions = positions[indices[:6]]
            base_x = np.mean(base_positions[:, 0])
            base_y = np.mean(base_positions[:, 1])

            # Apply average displacement to all positions in this cluster
            avg_dx = np.mean(positions[indices][:, 0] - base_x)
            avg_dy = np.mean(positions[indices][:, 1] - base_y)

            for idx in indices:
                positions[idx, 0] = base_x + avg_dx
                positions[idx, 1] = base_y + avg_dy

    def mutate_symmetrically(self, individual, generation=0, max_generations=100):
        """Apply advanced symmetry-aware mutation with mathematical constraints."""
        mutated = individual.copy()

        # Adaptively scale mutation parameters based on generation
        current_mut_rate = self.initial_mut_rate * (self.mutation_decay_factor ** generation)
        current_max_dist = self.initial_max_dist * (self.mutation_decay_factor ** generation)
        current_angle_std = self.initial_angle_std * (self.mutation_decay_factor ** generation)

        # Parse positions and angles
        positions = mutated[:-1].reshape(12, 3)

        # Apply symmetry-preserving mutations
        for i in range(12):
            if random.random() < current_mut_rate:
                # Get current position and angle
                x, y, angle = positions[i, 0], positions[i, 1], positions[i, 2]

                # Apply small random displacements
                dx = np.random.normal(0, current_max_dist * 0.3)
                dy = np.random.normal(0, current_max_dist * 0.3)
                d_angle = np.random.normal(0, current_angle_std * 0.5)

                # Apply transformations with attention to symmetry
                positions[i, 0] += dx
                positions[i, 1] += dy
                positions[i, 2] += d_angle

                # Apply symmetry constraints based on hexagonal lattice relationships
                # Hexagonal clusters should maintain their relative positioning
                if i < 7:  # First ring positions
                    # Maintain relative distances to center
                    center_dist = np.sqrt(x**2 + y**2)
                    if center_dist > 0:
                        factor = 1.0  # Keep relative scaling
                        positions[i, 0] = factor * x / center_dist * center_dist
                        positions[i, 1] = factor * y / center_dist * center_dist

                # Clip to reasonable bounds
                positions[i, 0] = np.clip(positions[i, 0], -10, 10)
                positions[i, 1] = np.clip(positions[i, 1], -10, 10)
                positions[i, 2] = positions[i, 2] % 360

        # Reshape back to flattened format
        mutated[:-1] = positions.flatten()
        return mutated

class SymmetryAwareMutation:
    """Custom mutation class that respects hexagon symmetry relationships."""

    def __init__(self):
        self.initial_mut_rate = 0.1
        self.initial_max_dist = 0.5
        self.initial_angle_std = 10.0
        self.mutation_decay_factor = 0.9

    def mutate_symmetrically(self, individual, generation=0, max_generations=100):
        """Apply mutation that preserves symmetry relationships with adaptive scaling."""
        mutated = individual.copy()

        # Adaptively scale mutation parameters based on generation
        current_mut_rate = self.initial_mut_rate * (self.mutation_decay_factor ** generation)
        current_max_dist = self.initial_max_dist * (self.mutation_decay_factor ** generation)
        current_angle_std = self.initial_angle_std * (self.mutation_decay_factor ** generation)

        # Apply mutations to 12 hexagons
        for i in range(12):
            if random.random() < current_mut_rate:
                # Get position and angle for this hexagon
                pos_idx = i * 3
                x, y, angle = mutated[pos_idx], mutated[pos_idx+1], mutated[pos_idx+2]

                # Mutate x and y coordinates with scaled random displacements
                dx = np.random.normal(0, current_max_dist * 0.5)
                dy = np.random.normal(0, current_max_dist * 0.5)

                # For rotational symmetry preservation, apply similar changes to related hexagons
                # Keep some rotational consistency
                d_angle = np.random.normal(0, current_angle_std)  # Small angle change with adaptive std

                mutated[pos_idx] += dx
                mutated[pos_idx+1] += dy
                mutated[pos_idx+2] += d_angle

                # Clip to reasonable bounds
                mutated[pos_idx] = np.clip(mutated[pos_idx], -10, 10)
                mutated[pos_idx+1] = np.clip(mutated[pos_idx+1], -10, 10)
                mutated[pos_idx+2] = mutated[pos_idx+2] % 360

        return mutated

def optimize_with_stages():
    """Perform multi-stage optimization for better convergence with hybrid approach."""
    # Initialize mutation object
    mutation_operator = SymmetryAwareMutation()

    # Stage 1: Position-only optimization with fixed rotations
    print("Stage 1: Position-only optimization...")
    initial_config = generate_initial_symmetric_config()

    # Fix rotations to 0 for this stage
    config_stage1 = initial_config.copy()
    for i in range(12):
        config_stage1[i*3 + 2] = 0

    # Optimize only positions (first 36 values)
    bounds_pos_only = [(None, None)] * 36
    bounds_pos_only.extend([(1.0, 15.0)])  # Outer side length bound

    # Use L-BFGS-B since we're optimizing position variables
    try:
        result1 = minimize(
            lambda x: objective_function(np.concatenate([x, [x[-1]]])),  # Pass entire x including last element
            config_stage1[:-1],  # Only positions
            method='L-BFGSB',
            bounds=bounds_pos_only[:-1],
            options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6},
            tol=1e-6
        )
        if result1.success:
            stage1_result = np.concatenate([result1.x, [config_stage1[-1]]])
        else:
            stage1_result = config_stage1
    except:
        stage1_result = config_stage1

    # Stage 2: Limited rotation adjustment with refined position
    print("Stage 2: Limited rotation adjustment...")
    config_stage2 = stage1_result.copy()

    # Allow small rotation adjustments for some hexagons
    for i in range(12):
        if i % 3 == 0:  # Every third hexagon gets a rotation adjustment
            config_stage2[i*3 + 2] = random.uniform(0, 30)  # Small rotation

    # Optimize both positions and rotations with tight tolerance
    bounds_stage2 = [(None, None)] * 36
    bounds_stage2.extend([(1.0, 15.0)])

    try:
        result2 = minimize(
            objective_function,
            config_stage2,
            method='L-BFGSB',
            bounds=bounds_stage2,
            options={'maxiter': 500, 'ftol': 1e-7, 'gtol': 1e-7},
            tol=1e-7
        )
        if result2.success:
            stage2_result = result2.x
        else:
            stage2_result = config_stage2
    except:
        stage2_result = config_stage2

    # Stage 3: Hybrid evolutionary-local search with simulated annealing
    print("Stage 3: Hybrid evolutionary-local search...")

    # Parameters for hybrid approach
    max_generations = 50
    population_size = 20
    elite_count = 2
    temperature_start = 0.8
    temperature_end = 0.01
    cooling_rate = (temperature_end / temperature_start) ** (1.0 / max_generations)

    # Generate initial population around best solution
    population = [stage2_result.copy()]
    for _ in range(population_size - 1):
        individual = stage2_result.copy()
        # Add small random noise to create diversity
        for i in range(len(individual)):
            if i < len(individual) - 1:  # Not the side length
                individual[i] += np.random.normal(0, 0.1)
        population.append(individual)

    best_individual = stage2_result.copy()
    best_fitness = objective_function(best_individual)

    # Evolutionary loop with simulated annealing
    for generation in range(max_generations):
        # Calculate current temperature
        current_temp = temperature_start * (cooling_rate ** generation)

        # Evaluate fitness of entire population
        fitness_scores = []
        for individual in population:
            fitness = objective_function(individual)
            fitness_scores.append(fitness)

        # Sort population by fitness (lower is better)
        sorted_indices = np.argsort(fitness_scores)
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Update best individual
        if fitness_scores[0] < best_fitness:
            best_individual = population[0].copy()
            best_fitness = fitness_scores[0]

        # Create new population
        new_population = population[:elite_count]  # Elitism

        # Generate offspring through mutation
        for _ in range(population_size - elite_count):
            # Select parent (tournament selection)
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmin(tournament_fitness)]

            # Mutate selected parent
            child = mutation_operator.mutate_symmetrically(population[winner_index], generation)

            # Accept with probability based on temperature and fitness difference
            current_fitness = objective_function(child)
            if current_fitness < fitness_scores[winner_index]:
                # Always accept better solutions
                new_population.append(child)
            else:
                # Accept worse solutions with probability based on temperature
                fitness_diff = current_fitness - fitness_scores[winner_index]
                acceptance_prob = np.exp(-fitness_diff / max(current_temp, 1e-10))
                if random.random() < acceptance_prob:
                    new_population.append(child)

        population = new_population

        # Debug info every few generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.8f}")

    # Final optimization of best individual found
    final_result = best_individual.copy()
    try:
        bounds_final = [(None, None)] * 36  # All positions and angles
        bounds_final.extend([(1.0, 15.0)])  # Outer side length

        result_final = minimize(
            objective_function,
            final_result,
            method='L-BFGSB',
            bounds=bounds_final,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )

        if result_final.success:
            final_result = result_final.x
        else:
            # If minimization fails, keep the best found
            pass

    except Exception as e:
        print(f"Final optimization error: {e}")
        pass  # Keep the best result found from evolution

    return final_result

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

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Time the optimization
    start_time = time.time()

    try:
        final_config = optimize_with_stages()
        # Extract the final configuration
        final_positions = final_config[:-1].reshape(12, 3)
        final_side_length = final_config[-1]

        # Return in the required format
        inner_hex_data = final_positions.copy()
        outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered

        # Validate final result
        if not evaluate_configuration(final_config[:-1], final_side_length):
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