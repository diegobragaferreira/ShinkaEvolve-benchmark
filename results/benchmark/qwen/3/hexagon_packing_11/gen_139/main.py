# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
from typing import Tuple, List
import time
from joblib import Parallel, delayed

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def create_regular_hexagon(center_x: float, center_y: float, side_length: float = 1.0, rotation_deg: float = 0.0) -> Polygon:
    """Create a regular hexagon as a Shapely polygon."""
    angle_rad = np.radians(rotation_deg)
    points = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        points.append((x, y))
    return Polygon(points)

def check_hexagon_containment(hexagon: Polygon, outer_hex: Polygon) -> bool:
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex.contains(hexagon)

def check_hexagon_collision(hex1: Polygon, hex2: Polygon) -> bool:
    """Check if two hexagons collide (overlap)."""
    return hex1.intersects(hex2)

def compute_min_outer_radius(inner_hex_data: np.ndarray) -> float:
    """Compute the minimum outer hexagon radius required to contain all inner hexagons."""
    max_dist = 0.0
    for i in range(len(inner_hex_data)):
        center_x, center_y, _ = inner_hex_data[i]
        # Distance from center to hexagon center plus the hexagon's circumradius
        dist = np.sqrt(center_x**2 + center_y**2) + 1.0  # 1.0 is the circumradius of unit hexagon
        max_dist = max(max_dist, dist)
    return max_dist * 1.05  # Add safety margin

def binary_search_outer_radius(inner_hex_data: np.ndarray, min_radius: float,
                              max_radius: float, tolerance: float = 0.001) -> float:
    """Binary search to find the minimum valid outer radius."""
    while max_radius - min_radius > tolerance:
        mid_radius = (min_radius + max_radius) / 2.0
        penalty, is_valid, _ = evaluate_packing(inner_hex_data, mid_radius)
        if is_valid:
            max_radius = mid_radius
        else:
            min_radius = mid_radius
    return max_radius

def evaluate_packing(inner_hex_data: np.ndarray, outer_hex_side_length: float) -> Tuple[float, bool, str]:
    """
    Evaluate a packing configuration.

    Returns:
        tuple: (penalty_score, is_valid, message)
    """
    # Precompute outer hexagon (centered at origin) for reuse
    outer_hex = create_regular_hexagon(0, 0, outer_hex_side_length)

    # Check containment and collisions for all inner hexagons
    inner_hexagons = []
    total_penalty = 0.0

    for i in range(len(inner_hex_data)):
        center_x, center_y, rotation = inner_hex_data[i]
        inner_hex = create_regular_hexagon(center_x, center_y, 1.0, rotation)

        # Check containment
        if not check_hexagon_containment(inner_hex, outer_hex):
            total_penalty += 1000.0  # Large penalty for containment violation

        inner_hexagons.append(inner_hex)

    # Check pairwise collisions in parallel
    def check_collision_pair(args):
        i, j, hex1, hex2 = args
        return check_hexagon_collision(hex1, hex2)

    # Create list of pairs to check
    pairs_to_check = [(i, j, inner_hexagons[i], inner_hexagons[j])
                      for i in range(len(inner_hexagons))
                      for j in range(i+1, len(inner_hexagons))]

    # Parallel collision checking
    collision_results = Parallel(n_jobs=-1)(delayed(check_collision_pair)(pair) for pair in pairs_to_check)

    # Count collisions
    num_collisions = sum(collision_results)
    total_penalty += 100.0 * num_collisions  # Penalty for collision

    # Calculate number of hexagons that fit (should be 11)
    num_fits = len(inner_hexagons)

    if num_fits != 11:
        total_penalty += 10000.0  # Very high penalty for wrong count

    # Return penalty score (lower is better) and validity flag
    is_valid = (total_penalty == 0.0)
    return total_penalty, is_valid, f"Penalty: {total_penalty}"

def generate_structured_initial_config() -> np.ndarray:
    """Generate a structured initial configuration based on efficient hexagonal packing principles."""
    # Create a configuration that starts with a central hexagon and builds upon it
    # This is inspired by known efficient packings of hexagons
    config = np.zeros((11, 3))

    # Central hexagon
    config[0] = [0.0, 0.0, 0.0]

    # First ring (6 hexagons around center)
    ring1_angles = [i * 60 for i in range(6)]
    ring1_distance = 2.0  # Distance between centers (approximately 2 unit hex radii)

    for i, angle in enumerate(ring1_angles):
        rad = np.radians(angle)
        x = ring1_distance * np.cos(rad)
        y = ring1_distance * np.sin(rad)
        config[i+1] = [x, y, 0.0]

    # Second ring (4 hexagons)
    ring2_angles = [30, 90, 150, 210]  # Specific angles for second ring
    ring2_distance = 3.5  # Distance for second ring

    for i, angle in enumerate(ring2_angles):
        rad = np.radians(angle)
        x = ring2_distance * np.cos(rad)
        y = ring2_distance * np.sin(rad)
        config[i+7] = [x, y, 0.0]

    return config

def generate_initial_population(pop_size: int, max_outer_radius: float = 15.0) -> List[np.ndarray]:
    """Generate initial population of hexagon configurations."""
    population = []

    # Start with structured configuration
    structured_config = generate_structured_initial_config()

    for _ in range(pop_size):
        # Start with structured configuration and apply small random perturbations
        individual = structured_config.copy()

        # Apply small random noise to each hexagon's position and rotation
        for i in range(11):
            individual[i, 0] += np.random.normal(0, 0.2)  # Position perturbation
            individual[i, 1] += np.random.normal(0, 0.2)
            individual[i, 2] += np.random.normal(0, 10)  # Rotation perturbation
            individual[i, 2] %= 360.0  # Keep within [0, 360)

        population.append(individual)

    return population

def mutate_individual(individual: np.ndarray, mutation_rate: float = 0.1,
                     max_displacement: float = 0.5, max_rotation: float = 30.0) -> np.ndarray:
    """Apply mutation to an individual."""
    mutated = individual.copy()

    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[i, 0] += np.random.normal(0, max_displacement)
        if np.random.random() < mutation_rate:
            mutated[i, 1] += np.random.normal(0, max_displacement)
        if np.random.random() < mutation_rate:
            mutated[i, 2] += np.random.normal(0, max_rotation)
            mutated[i, 2] %= 360.0  # Keep within [0, 360)

    return mutated

def crossover_parents(parent1: np.ndarray, parent2: np.ndarray,
                      crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents."""
    if np.random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    # Single point crossover on the 11 hexagons
    crossover_point = np.random.randint(1, 11)

    child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])

    return child1, child2

def optimize_hexagon_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Evolve an optimal arrangement of 11 unit regular hexagons.

    Returns:
        tuple: (inner_hex_data, outer_hex_data, outer_hex_side_length)
    """
    # Parameters
    pop_size = 50
    generations = 200
    elite_size = 5
    mutation_rate = 0.2
    min_outer_radius = 2.0
    max_outer_radius = 15.0

    # Time limit enforcement
    start_time = time.time()
    timeout_seconds = 175  # Leave some buffer

    # Initialize population
    population = generate_initial_population(pop_size, max_outer_radius)

    best_individual = None
    best_penalty = float('inf')
    best_outer_radius = max_outer_radius

    # Evolution loop
    for generation in range(generations):
        if time.time() - start_time > timeout_seconds:
            break

        # Evaluate fitness in parallel
        def evaluate_individual(individual):
            penalty, is_valid, _ = evaluate_packing(individual, max_outer_radius)
            return penalty, individual, is_valid

        # Process all individuals in parallel
        results = Parallel(n_jobs=-1)(delayed(evaluate_individual)(individual) for individual in population)

        fitness_scores = results
        valid_individuals = [individual for penalty, individual, is_valid in results if is_valid]

        # Sort by fitness (lower penalty better)
        fitness_scores.sort(key=lambda x: x[0])

        # Track best solution so far
        if fitness_scores and fitness_scores[0][0] < best_penalty:
            best_penalty = fitness_scores[0][0]
            best_individual = fitness_scores[0][1].copy()
            best_outer_radius = max_outer_radius

        # If we have valid individuals, try smaller outer radius with binary search
        if len(valid_individuals) > 0 and generation > 50:
            # Use binary search to find minimal outer radius for best solution
            estimated_min_radius = compute_min_outer_radius(best_individual)
            min_test_radius = max(min_outer_radius, estimated_min_radius)

            test_radius = binary_search_outer_radius(best_individual, min_test_radius, max_outer_radius)

            # Verify the result
            final_penalty, final_valid, _ = evaluate_packing(best_individual, test_radius)
            if final_valid and final_penalty < best_penalty:
                best_penalty = final_penalty
                best_outer_radius = test_radius

        # Early stopping if very good solution found
        if best_penalty < 1.0:
            break

        # Select elite (top performers)
        elite = [ind for _, ind, valid in fitness_scores[:elite_size]]

        # Generate next generation
        new_population = elite.copy()

        while len(new_population) < pop_size:
            # Tournament selection with better probability of selecting good individuals
            parent1 = tournament_selection(fitness_scores, tournament_size=3)
            parent2 = tournament_selection(fitness_scores, tournament_size=3)

            # Crossover with higher probability
            child1, child2 = crossover_parents(parent1, parent2, crossover_rate=0.9)

            # Mutate with adjusted rates
            child1 = mutate_individual(child1, mutation_rate=0.15, max_displacement=0.3, max_rotation=20.0)
            child2 = mutate_individual(child2, mutation_rate=0.15, max_displacement=0.3, max_rotation=20.0)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:pop_size]

    # Final refinement using binary search instead of linear search
    if best_individual is not None:
        estimated_min_radius = compute_min_outer_radius(best_individual)
        min_test_radius = max(min_outer_radius, estimated_min_radius)

        final_radius = binary_search_outer_radius(best_individual, min_test_radius, best_outer_radius)

        # Double-check the final result
        final_penalty, final_valid, _ = evaluate_packing(best_individual, final_radius)
        if final_valid and final_penalty < best_penalty:
            best_penalty = final_penalty
            best_outer_radius = final_radius

    # Construct final return values
    inner_hex_data = best_individual if best_individual is not None else np.zeros((11, 3))
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin

    return inner_hex_data, outer_hex_data, best_outer_radius

def tournament_selection(fitness_scores: List[Tuple[float, np.ndarray, bool]],
                        tournament_size: int = 3) -> np.ndarray:
    """Select an individual using tournament selection."""
    # Use a more sophisticated tournament selection that considers both fitness and diversity
    participants = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
    # Select best among participants
    return min(participants, key=lambda x: x[0])[1]

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing()

    # Ensure we return at least the minimum possible result
    if outer_hex_side_length <= 0:
        outer_hex_side_length = 10.0

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END