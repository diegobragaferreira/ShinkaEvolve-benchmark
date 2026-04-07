# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
import random
from typing import Tuple, List
import time
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution, minimize
import warnings

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
                              max_radius: float, tolerance: float = 0.0001) -> float:
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
    Evaluate a packing configuration with improved geometric validation.

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

    # Spatial index for efficient collision checking
    try:
        hex_tree = STRtree(inner_hexagons)
        
        # Check pairwise collisions efficiently using spatial indexing
        for i in range(len(inner_hexagons)):
            hex1 = inner_hexagons[i]
            # Get potential candidates using spatial index
            candidates = hex_tree.query(hex1)
            for j in candidates:
                if i < j:  # Avoid checking pair twice
                    hex2 = inner_hexagons[j]
                    if check_hexagon_collision(hex1, hex2):
                        total_penalty += 100.0  # Penalty for collision
    except:
        # Fallback to brute force in case of spatial tree issues
        for i in range(len(inner_hexagons)):
            for j in range(i+1, len(inner_hexagons)):
                if check_hexagon_collision(inner_hexagons[i], inner_hexagons[j]):
                    total_penalty += 100.0  # Penalty for collision

    # Calculate number of hexagons that fit (should be 11)
    num_fits = len(inner_hexagons)
    
    if num_fits != 11:
        total_penalty += 10000.0  # Very high penalty for wrong count

    # Return penalty score (lower is better) and validity flag
    is_valid = (total_penalty == 0.0)
    return total_penalty, is_valid, f"Penalty: {total_penalty}"

def generate_hierarchical_initial_config() -> np.ndarray:
    """Generate a highly structured initial configuration inspired by hexagonal close packing."""
    # Start with a mathematically optimized arrangement
    # Place hexagons in a way that mimics dense hexagonal packing patterns
    config = np.zeros((11, 3))

    # Central hexagon
    config[0] = [0.0, 0.0, 0.0]

    # First ring (6 hexagons in hexagonal pattern)
    # Using the pattern where each hexagon is at distance 2 from center
    ring1_angles = [i * 60 for i in range(6)]
    ring1_distance = 2.0

    for i, angle in enumerate(ring1_angles):
        rad = np.radians(angle)
        x = ring1_distance * np.cos(rad)
        y = ring1_distance * np.sin(rad)
        config[i+1] = [x, y, 0.0]

    # Second ring (4 hexagons, but place them strategically)
    # These are positioned to fill gaps in the hexagonal structure
    # This placement is inspired by the most efficient packing arrangements
    ring2_positions = [
        [-1.0, -1.732, 0.0],   # Bottom left
        [1.0, -1.732, 0.0],    # Bottom right  
        [-1.0, 1.732, 0.0],    # Top left
        [1.0, 1.732, 0.0],     # Top right
    ]
    
    for i, (x, y, rot) in enumerate(ring2_positions):
        config[i+7] = [x, y, rot]

    return config

def generate_initial_population(pop_size: int, max_outer_radius: float = 15.0) -> List[np.ndarray]:
    """Generate initial population with enhanced diversity and structure."""
    population = []

    # Start with hierarchical configuration
    structured_config = generate_hierarchical_initial_config()

    # Add multiple variants with different perturbation strategies
    for _ in range(pop_size):
        individual = structured_config.copy()
        
        # Apply more sophisticated perturbations
        for i in range(11):
            # Position perturbation with varying magnitudes
            pos_noise_x = np.random.normal(0, 0.15)  # Reduced for stability
            pos_noise_y = np.random.normal(0, 0.15)
            individual[i, 0] += pos_noise_x
            individual[i, 1] += pos_noise_y
            
            # Rotation with smaller perturbations
            rot_noise = np.random.normal(0, 5.0)
            individual[i, 2] += rot_noise
            individual[i, 2] %= 360.0  # Keep within [0, 360)

        population.append(individual)

    return population

def adaptive_local_optimization(individual: np.ndarray, outer_radius: float, 
                               max_iter: int = 100) -> Tuple[np.ndarray, float]:
    """
    Enhanced local optimization using multiple strategies combined intelligently.
    """
    # Strategy 1: Differential evolution for global search
    def objective_de(x_flat):
        temp_individual = individual.copy()
        temp_individual[:, 0] = x_flat[::3]
        temp_individual[:, 1] = x_flat[1::3]
        temp_individual[:, 2] = x_flat[2::3]
        penalty, _, _ = evaluate_packing(temp_individual, outer_radius)
        return penalty
    
    # Flatten individual for optimization
    flat_individual = np.concatenate([
        individual[:, 0], individual[:, 1], individual[:, 2]
    ])
    
    bounds = []
    for i in range(len(flat_individual)):
        if i % 3 == 0:  # x coordinates
            bounds.append((-10, 10))
        elif i % 3 == 1:  # y coordinates
            bounds.append((-10, 10))
        else:  # rotations
            bounds.append((0, 360))
    
    # Try differential evolution first (good for global search)
    try:
        de_result = differential_evolution(
            objective_de, bounds, maxiter=max_iter//2, popsize=10, 
            strategy='best1bin', tol=1e-8, disp=False, updating='immediate'
        )
        
        if de_result.success:
            refined_individual = individual.copy()
            refined_individual[:, 0] = de_result.x[::3]
            refined_individual[:, 1] = de_result.x[1::3]
            refined_individual[:, 2] = de_result.x[2::3]
            
            penalty, _, _ = evaluate_packing(refined_individual, outer_radius)
            return refined_individual, penalty
    except:
        pass
    
    # Strategy 2: Local gradient-based optimization if DE doesn't work
    try:
        # Use scipy minimize with L-BFGS-B
        def objective_lbfgs(x_flat):
            temp_individual = individual.copy()
            temp_individual[:, 0] = x_flat[::3]
            temp_individual[:, 1] = x_flat[1::3]
            temp_individual[:, 2] = x_flat[2::3]
            penalty, _, _ = evaluate_packing(temp_individual, outer_radius)
            return penalty
            
        result = minimize(
            objective_lbfgs, flat_individual, method='L-BFGS-B',
            bounds=bounds, options={'maxiter': max_iter//2, 'ftol': 1e-8}
        )
        
        if result.success:
            refined_individual = individual.copy()
            refined_individual[:, 0] = result.x[::3]
            refined_individual[:, 1] = result.x[1::3]
            refined_individual[:, 2] = result.x[2::3]
            
            penalty, _, _ = evaluate_packing(refined_individual, outer_radius)
            return refined_individual, penalty
    except:
        pass

    # Strategy 3: Simple gradient-free method for robustness
    try:
        # Try a basic coordinate descent approach with successively smaller steps
        current_params = flat_individual.copy()
        current_penalty, _, _ = evaluate_packing(individual, outer_radius)
        
        step_sizes = [0.5, 0.2, 0.1, 0.05]  # Decreasing step sizes
        
        for step_size in step_sizes:
            improved = True
            while improved and len(current_params) > 0:
                improved = False
                for i in range(len(current_params)):
                    # Try positive and negative step
                    old_val = current_params[i]
                    
                    # Test positive step
                    current_params[i] = old_val + step_size
                    temp_individual = individual.copy()
                    temp_individual[:, 0] = current_params[::3]
                    temp_individual[:, 1] = current_params[1::3]
                    temp_individual[:, 2] = current_params[2::3]
                    new_penalty, _, _ = evaluate_packing(temp_individual, outer_radius)
                    
                    if new_penalty < current_penalty:
                        current_penalty = new_penalty
                        improved = True
                    else:
                        current_params[i] = old_val  # Revert
                        
                    # Test negative step
                    current_params[i] = old_val - step_size
                    temp_individual = individual.copy()
                    temp_individual[:, 0] = current_params[::3]
                    temp_individual[:, 1] = current_params[1::3]
                    temp_individual[:, 2] = current_params[2::3]
                    new_penalty, _, _ = evaluate_packing(temp_individual, outer_radius)
                    
                    if new_penalty < current_penalty:
                        current_penalty = new_penalty
                        improved = True
                    else:
                        current_params[i] = old_val  # Revert
                        
        # Convert back to individual
        refined_individual = individual.copy()
        refined_individual[:, 0] = current_params[::3]
        refined_individual[:, 1] = current_params[1::3]
        refined_individual[:, 2] = current_params[2::3]
        
        return refined_individual, current_penalty
    except:
        pass

    # Return original if all strategies fail
    penalty, _, _ = evaluate_packing(individual, outer_radius)
    return individual, penalty

def hierarchical_evolutionary_search(max_generations: int = 100, 
                                   population_size: int = 40,
                                   max_runtime_seconds: int = 175) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Execute hierarchical evolutionary search with multi-scale optimization.
    """
    start_time = time.time()
    
    # Phase 1: Coarse optimization with large steps
    # Initialize population with enhanced structure
    population = generate_initial_population(population_size)
    
    best_individual = None
    best_penalty = float('inf')
    best_outer_radius = 15.0
    
    # Phase 1: Global search with larger steps
    for generation in range(max_generations // 2):
        if time.time() - start_time > max_runtime_seconds:
            break
            
        # Evaluate fitness in parallel
        def evaluate_individual(individual):
            penalty, is_valid, _ = evaluate_packing(individual, 15.0)  # Use large radius for initial search
            return penalty, individual, is_valid

        results = Parallel(n_jobs=-1)(delayed(evaluate_individual)(individual) for individual in population)

        # Sort by fitness
        fitness_scores = results
        fitness_scores.sort(key=lambda x: x[0])

        # Track best solution
        if fitness_scores and fitness_scores[0][0] < best_penalty:
            best_penalty = fitness_scores[0][0]
            best_individual = fitness_scores[0][1].copy()
            best_outer_radius = 15.0

        # Select best 20% for breeding
        elite_size = max(2, population_size // 5)
        elite = [ind for _, ind, valid in fitness_scores[:elite_size]]

        # Generate new population with more diverse mutations
        new_population = elite.copy()
        
        # Generate offspring with stronger mutations in early generations
        mutation_intensity = max(0.1, 0.3 * (1.0 - generation / (max_generations // 2)))
        
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament_selection(fitness_scores, tournament_size=3)
            parent2 = tournament_selection(fitness_scores, tournament_size=3)
            
            # Crossover with some uniform aspects
            child1 = parent1.copy()
            child2 = parent2.copy()
            
            # Mix elements with probability
            for i in range(11):
                if np.random.random() < 0.5:
                    child1[i] = parent2[i].copy()
                if np.random.random() < 0.5:
                    child2[i] = parent1[i].copy()
            
            # Mutate with adaptive intensity
            child1 = mutate_individual(child1, mutation_rate=mutation_intensity)
            child2 = mutate_individual(child2, mutation_rate=mutation_intensity)
            
            new_population.extend([child1, child2])

        population = new_population[:population_size]
    
    # Phase 2: Fine optimization 
    if best_individual is not None:
        # Refine the best solution with binary search to find tightest outer radius
        estimated_min_radius = compute_min_outer_radius(best_individual)
        min_test_radius = max(2.0, estimated_min_radius * 0.95)
        final_radius = binary_search_outer_radius(best_individual, min_test_radius, 15.0, tolerance=0.0001)
        
        # Final validation and refinement
        final_penalty, _, _ = evaluate_packing(best_individual, final_radius)
        
        if final_penalty < best_penalty:
            best_penalty = final_penalty
            best_outer_radius = final_radius

        # Use adaptive local optimization for final refinement
        if time.time() - start_time < max_runtime_seconds - 10:
            refined_individual, refined_penalty = adaptive_local_optimization(best_individual, best_outer_radius, 50)
            if refined_penalty < best_penalty:
                best_penalty = refined_penalty
                best_individual = refined_individual
                best_outer_radius = best_outer_radius

    # Final binary search to ensure optimal outer radius
    if best_individual is not None:
        try:
            final_radius = binary_search_outer_radius(best_individual, 
                                                    compute_min_outer_radius(best_individual) * 0.99, 
                                                    best_outer_radius, 
                                                    tolerance=0.0001)
            final_penalty, _, _ = evaluate_packing(best_individual, final_radius)
            if final_penalty < best_penalty:
                best_penalty = final_penalty
                best_outer_radius = final_radius
        except:
            pass

    return best_individual, best_outer_radius

def mutate_individual(individual: np.ndarray, mutation_rate: float = 0.1,
                     max_displacement: float = 0.5, max_rotation: float = 30.0) -> np.ndarray:
    """Enhanced mutation operator with adaptive parameters."""
    mutated = individual.copy()

    # Apply mutations with different intensities per parameter
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Position mutation with lower intensity for stability
            mutated[i, 0] += np.random.normal(0, max_displacement * 0.5)
        if np.random.random() < mutation_rate:
            mutated[i, 1] += np.random.normal(0, max_displacement * 0.5)
        if np.random.random() < mutation_rate:
            mutated[i, 2] += np.random.normal(0, max_rotation * 0.3)
            mutated[i, 2] %= 360.0  # Keep within [0, 360)

    return mutated

def tournament_selection(fitness_scores: List[Tuple[float, np.ndarray, bool]],
                        tournament_size: int = 3) -> np.ndarray:
    """Enhanced tournament selection with diversity considerations."""
    # Use a more sophisticated variant that considers both fitness and diversity
    participants = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
    # Select best among participants
    return min(participants, key=lambda x: x[0])[1]

def optimize_hexagon_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Evolve an optimal arrangement of 11 unit regular hexagons with hierarchical approach.

    Returns:
        tuple: (inner_hex_data, outer_hex_data, outer_hex_side_length)
    """
    try:
        # Execute hierarchical evolutionary search
        best_individual, best_radius = hierarchical_evolutionary_search(
            max_generations=80,
            population_size=40, 
            max_runtime_seconds=175
        )
        
        # Construct final return values
        inner_hex_data = best_individual if best_individual is not None else np.zeros((11, 3))
        outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
        
        return inner_hex_data, outer_hex_data, best_radius
    
    except Exception as e:
        # Fallback to a good heuristic solution
        warnings.warn(f"Fallback mode activated due to error: {e}")
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
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Run the optimized search
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing()

    # Ensure we return at least the minimum possible result
    if outer_hex_side_length <= 0:
        outer_hex_side_length = 10.0

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END