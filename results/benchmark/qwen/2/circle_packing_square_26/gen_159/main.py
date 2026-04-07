# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
from collections import defaultdict

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def validate_circles(circles: np.ndarray) -> bool:
    """
    Validates that all circles are within bounds and don't overlap.
    Uses efficient spatial indexing for overlap checking.
    """
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False

    # Check overlap constraints using KDTree for efficiency
    points = circles[:, :2]  # Get (x, y) coordinates
    tree = cKDTree(points)

    # For each circle, check overlap with others
    for i in range(n):
        x1, y1, r1 = circles[i]
        # Find nearby circles (within 2*(r1+r2) distance)
        nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

        # Check overlap with each nearby circle
        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2

                if distance_sq < min_distance_sq:
                    return False

    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii"""
    return np.sum(circles[:, 2])

def create_voronoi_initialization(n_circles: int) -> np.ndarray:
    """Create initial configuration using a physics-inspired Voronoi-like approach"""
    circles = np.zeros((n_circles, 3))
    
    # Start with a few strategic points
    initial_points = []
    
    # Corners first
    initial_points.extend([(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)])
    
    # Add some edge points
    for i in range(3):
        initial_points.append((0.1 + 0.3*i, 0.1))
        initial_points.append((0.1 + 0.3*i, 0.9))
        initial_points.append((0.1, 0.1 + 0.3*i))
        initial_points.append((0.9, 0.1 + 0.3*i))
    
    # Add center point
    initial_points.append((0.5, 0.5))
    
    # If we have more circles than initial points, fill with random points
    if n_circles > len(initial_points):
        additional_points = []
        for _ in range(n_circles - len(initial_points)):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            additional_points.append((x, y))
        initial_points.extend(additional_points)
    
    # Limit to exactly n_circles
    initial_points = initial_points[:n_circles]
    
    # Use Delaunay triangulation for initial spatial distribution
    points_array = np.array(initial_points)
    
    # Assign initial radii based on distance to nearest neighbors
    if len(points_array) >= 2:
        distances = cdist(points_array, points_array)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distance
        min_distances = np.min(distances, axis=1)
        
        # Use minimum distances to determine initial radii
        # Scale according to how "dense" each region is
        for i, (x, y) in enumerate(initial_points):
            # Distance to nearest neighbor
            dist_to_nearest = min_distances[i] 
            # Base radius on density (closer points = smaller initial radii)
            base_radius = min(0.08, dist_to_nearest * 0.3)
            
            # Adjust based on distance from edges
            min_edge_dist = min(x, 1-x, y, 1-y)
            adjusted_radius = min(base_radius, min_edge_dist * 0.8)
            
            circles[i] = [x, y, max(0.01, adjusted_radius)]
    else:
        # Fallback for small cases
        for i, (x, y) in enumerate(initial_points):
            min_edge_dist = min(x, 1-x, y, 1-y)
            r = min(0.05, min_edge_dist * 0.7)
            circles[i] = [x, y, max(0.01, r)]
    
    return circles

def create_initial_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Create initial population using Voronoi-inspired initialization"""
    population = []
    
    # Most individuals use Voronoi-based initialization
    for i in range(pop_size):
        if i < pop_size * 0.7:  # 70% use Voronoi method
            circles = create_voronoi_initialization(n_circles)
        else:  # 30% use grid-based method for diversity
            circles = np.zeros((n_circles, 3))
            grid_size = int(np.ceil(np.sqrt(n_circles)))
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)
            
            idx = 0
            for i_grid in range(grid_size):
                for j_grid in range(grid_size):
                    if idx >= n_circles:
                        break
                    x = (j_grid + 1) * spacing_x + np.random.uniform(-spacing_x/6, spacing_x/6)
                    y = (i_grid + 1) * spacing_y + np.random.uniform(-spacing_y/6, spacing_y/6)
                    x = np.clip(x, 0.01, 0.99)
                    y = np.clip(y, 0.01, 0.99)
                    min_dist_to_edge = min(x, 1-x, y, 1-y)
                    r = min(0.08, min_dist_to_edge * np.random.uniform(0.6, 0.9))
                    circles[idx] = [x, y, r]
                    idx += 1
                if idx >= n_circles:
                    break
            
            # Fill remaining with random
            for i_fill in range(idx, n_circles):
                x = np.random.uniform(0.01, 0.99)
                y = np.random.uniform(0.01, 0.99)
                min_dist_to_edge = min(x, 1-x, y, 1-y)
                r = min(0.08, min_dist_to_edge * np.random.uniform(0.5, 0.8))
                circles[i_fill] = [x, y, r]
        
        # Apply a bit of noise to introduce diversity
        for i in range(n_circles):
            if np.random.rand() < 0.3:  # 30% chance to modify
                circles[i, 0] += np.random.uniform(-0.005, 0.005)
                circles[i, 1] += np.random.uniform(-0.005, 0.005)
                circles[i, 2] += np.random.uniform(-0.003, 0.003)
                # Clamp to bounds
                circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
                circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
                circles[i, 2] = max(0.001, circles[i, 2])
        
        population.append(circles)
    
    return population

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int = 5) -> np.ndarray:
    """Select parent using tournament selection with adaptive size"""
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def compute_overlap_severity(circles: np.ndarray) -> int:
    """Compute total number of overlap violations"""
    n = len(circles)
    tree = cKDTree(circles[:, :2])
    violations = 0
    
    for i in range(n):
        x1, y1, r1 = circles[i]
        nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))
        
        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2
                if distance_sq < min_distance_sq:
                    violations += 1
    
    return violations

def constraint_aware_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Crossover that considers overlap severity to preserve valid configurations"""
    child = parent1.copy()
    
    # Calculate overlap severity for parents
    parent1_violations = compute_overlap_severity(parent1)
    parent2_violations = compute_overlap_severity(parent2)
    
    # Weighted crossover based on parent quality
    crossover_weight = 0.5
    if parent1_violations == 0 and parent2_violations > 0:
        crossover_weight = 0.8  # Prefer valid parent
    elif parent2_violations == 0 and parent1_violations > 0:
        crossover_weight = 0.2  # Prefer valid parent
    elif parent1_violations == 0 and parent2_violations == 0:
        crossover_weight = 0.7  # Both valid, more mixing
    
    # Apply crossover with weighted probabilities
    for i in range(len(child)):
        if np.random.rand() < crossover_weight:
            child[i] = parent2[i].copy()
    
    return child

def mutate(circles: np.ndarray, generation: int, max_generations: int,
           base_mutation_rate: float = 0.15) -> np.ndarray:
    """Apply mutation with adaptive rates and constraint awareness"""
    # Adaptive mutation rate
    mutation_rate = base_mutation_rate * (0.1**(generation/max_generations))
    
    mutated = circles.copy()
    
    # Calculate overlap severity to adapt mutation intensity
    overlap_severity = compute_overlap_severity(mutated)
    if overlap_severity > 10:  # High overlap - more aggressive mutation
        mutation_rate *= 1.5
    elif overlap_severity < 3:  # Low overlap - more conservative
        mutation_rate *= 0.8
    
    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            # Mutate position or radius with different intensities
            if np.random.rand() < 0.5:  # Mutate position
                # Larger mutations early, smaller later
                scale = 0.03 * (1 - generation/max_generations) + 0.005
                mutated[i, 0] += np.random.normal(0, scale)
                mutated[i, 1] += np.random.normal(0, scale)
                
                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
            else:  # Mutate radius
                # Use smaller scale for radius mutations
                scale = 0.015 * (1 - generation/max_generations) + 0.001
                mutated[i, 2] += np.random.normal(0, scale)
                mutated[i, 2] = max(0.001, mutated[i, 2])
    
    return mutated

def repair_circles(circles: np.ndarray) -> np.ndarray:
    """Improved repair with more sophisticated overlap resolution"""
    repaired = circles.copy()

    # First ensure all circles are within bounds
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        # Keep within bounds
        repaired[i, 0] = np.clip(x, r, 1 - r)
        repaired[i, 1] = np.clip(y, r, 1 - r)
        repaired[i, 2] = max(0.001, repaired[i, 2])  # Ensure positive radius

    # Then resolve overlaps using iterative repulsion with early termination
    points = repaired[:, :2]
    tree = cKDTree(points)
    
    # Try to resolve overlaps iteratively with early stopping
    for _ in range(30):  # Increased iterations
        any_changes = False
        for i in range(len(repaired)):
            x1, y1, r1 = repaired[i]

            # Find nearby circles more efficiently
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = repaired[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2

                    if distance < min_distance:
                        # Repel circles apart with smarter movement
                        if distance > 0.001:  # Avoid division by zero
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance

                            # Move them apart proportionally to the overlap with force scaling
                            move_amount = (min_distance - distance) * 0.6
                            repaired[i, 0] += dx * move_amount
                            repaired[i, 1] += dy * move_amount
                            repaired[j, 0] -= dx * move_amount
                            repaired[j, 1] -= dy * move_amount

                            # Clamp to bounds
                            repaired[i, 0] = np.clip(repaired[i, 0], r1, 1 - r1)
                            repaired[i, 1] = np.clip(repaired[i, 1], r1, 1 - r1)
                            repaired[j, 0] = np.clip(repaired[j, 0], r2, 1 - r2)
                            repaired[j, 1] = np.clip(repaired[j, 1], r2, 1 - r2)
                        
                        any_changes = True
                        
        if not any_changes:
            break

    return repaired

def multi_level_local_refinement(circles: np.ndarray) -> np.ndarray:
    """Apply multi-level refinement based on overlap severity"""
    refined = circles.copy()
    n_circles = len(refined)
    
    # Classify overlap severity
    overlap_severity = compute_overlap_severity(refined)
    
    if overlap_severity == 0:
        # No overlaps, do focused refinement
        max_iterations = 100
        strategy = "light"
    elif overlap_severity <= 5:
        # Few overlaps, moderate refinement
        max_iterations = 150
        strategy = "moderate"
    else:
        # Many overlaps, intensive refinement
        max_iterations = 200
        strategy = "intensive"
    
    # Apply strategy-specific refinement
    if strategy == "light":
        # Light-weight optimization
        improved = True
        iteration = 0
        while improved and iteration < 50:
            improved = False
            iteration += 1
            for i in range(n_circles):
                original_r = refined[i, 2]
                x, y, _ = refined[i]
                max_possible_r = min(x, 1-x, y, 1-y)
                
                # Try small radius increase
                test_r = min(original_r + 0.001, max_possible_r * 0.98)
                if test_r > original_r + 1e-6:
                    temp_circles = refined.copy()
                    temp_circles[i, 2] = test_r
                    
                    if validate_circles(temp_circles):
                        refined = temp_circles
                        improved = True
    elif strategy == "moderate":
        # Moderate refinement with position adjustment
        for _ in range(50):
            improved = False
            for i in range(n_circles):
                x, y, r = refined[i]
                
                # Try several small position adjustments
                best_x, best_y, best_r = x, y, r
                best_radius = r
                
                adjustments = [(0, 0, 0), (0.002, 0, 0), (-0.002, 0, 0),
                              (0, 0.002, 0), (0, -0.002, 0)]
                
                for dx, dy, dr in adjustments:
                    test_x = max(0.001, min(0.999, x + dx))
                    test_y = max(0.001, min(0.999, y + dy))
                    test_r = max(0.001, min(r + dr,
                                          min(test_x, test_y, 1-test_x, 1-test_y) * 0.99))
                    
                    temp_circles = refined.copy()
                    temp_circles[i] = [test_x, test_y, test_r]
                    
                    if validate_circles(temp_circles) and test_r > best_radius:
                        best_x, best_y, best_r = test_x, test_y, test_r
                        best_radius = test_r
                
                if best_radius > r:
                    refined[i] = [best_x, best_y, best_r]
                    improved = True
                    
            if not improved:
                break
    else:  # intensive
        # Intensive refinement with comprehensive search
        improved = True
        iteration = 0
        while improved and iteration < 100:
            improved = False
            iteration += 1
            
            # Try to expand each circle's radius
            for i in range(n_circles):
                original_r = refined[i, 2]
                x, y, _ = refined[i]
                max_possible_r = min(x, 1-x, y, 1-y)
                
                # Try larger steps first
                steps = [0.008, 0.005, 0.003, 0.001, 0.0005]
                for step in steps:
                    test_r = min(original_r + step, max_possible_r * 0.95)
                    if test_r > original_r + 1e-6:
                        temp_circles = refined.copy()
                        temp_circles[i, 2] = test_r
                        
                        if validate_circles(temp_circles):
                            refined = temp_circles
                            improved = True
                            break
    
    # Final repair
    final_repaired = repair_circles(refined)
    return final_repaired if validate_circles(final_repaired) else refined

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Algorithm parameters
    pop_size = 80  # Increased population for better exploration
    n_generations = 150  # More generations for better convergence
    elite_size = 10  # More elites to preserve good solutions
    tournament_size = 7  # Larger tournaments for better selection pressure
    
    # Create initial population
    population = create_initial_population(pop_size, 26)

    # Evolution loop
    best_fitness = -np.inf
    best_individual = None

    for generation in range(n_generations):
        # Calculate fitness for all individuals
        fitnesses = []
        valid_individuals = []

        for circles in population:
            if validate_circles(circles):
                fitness = calculate_sum_radii(circles)
                fitnesses.append(fitness)
                valid_individuals.append(circles)
            else:
                # Repair invalid individuals
                repaired = repair_circles(circles)
                if validate_circles(repaired):
                    fitness = calculate_sum_radii(repaired)
                    fitnesses.append(fitness)
                    valid_individuals.append(repaired)
                else:
                    # If still invalid, penalize heavily
                    fitnesses.append(-np.inf)
                    valid_individuals.append(circles)

        # Track best individual
        if valid_individuals:
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_fitness:
                best_fitness = fitnesses[max_idx]
                best_individual = valid_individuals[max_idx].copy()

        # Elitism: keep top individuals
        elite_indices = np.argsort(fitnesses)[-elite_size:]
        elites = [valid_individuals[i] for i in elite_indices if fitnesses[i] > -np.inf]

        # Generate new population
        new_population = elites[:]

        # Fill remaining slots with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(valid_individuals, fitnesses, tournament_size)
            parent2 = tournament_selection(valid_individuals, fitnesses, tournament_size)

            # Constraint-aware crossover
            child = constraint_aware_crossover(parent1, parent2)

            # Mutation with adaptive rate
            child = mutate(child, generation, n_generations)

            # Repair
            child = repair_circles(child)

            new_population.append(child)

        population = new_population[:pop_size]

    # Return the best solution found with final refinement
    if best_individual is not None:
        # Apply multi-level local refinement to the best solution
        refined_solution = multi_level_local_refinement(best_individual)
        if validate_circles(refined_solution):
            return refined_solution
        else:
            return best_individual
    else:
        # If no valid solution found, return the best from final population
        fitnesses = [calculate_sum_radii(circles) for circles in population if validate_circles(circles)]
        if fitnesses:
            best_idx = np.argmax(fitnesses)
            # Apply local refinement to the best candidate from final population
            refined_solution = multi_level_local_refinement(population[best_idx])
            if validate_circles(refined_solution):
                return refined_solution
            else:
                return population[best_idx]
        else:
            # Fallback: return a valid random solution
            circles = np.zeros((26, 3))
            for i in range(26):
                circles[i] = [0.5, 0.5, 0.01]
            return circles

# EVOLVE-BLOCK-END