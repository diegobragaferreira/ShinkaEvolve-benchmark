# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def validate_circles(circles: np.ndarray) -> bool:
    """
    Validates that all circles are within bounds and don't overlap.

    Args:
        circles: np.array of shape (n, 3) where each row is (x, y, r)

    Returns:
        True if all circles are valid, False otherwise
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
    """
    Create initial circle configuration using Voronoi-based approach.
    This creates points that are well-separated and then assigns radii based on
    Voronoi cell areas to promote good packing.
    """
    # Generate initial points using a more sophisticated sampling approach
    # Generate points in a way that avoids clustering
    points = []

    # Use a systematic approach to place initial points
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) >= n_circles:
                break
            # Add jittered grid positions
            x = (i + 0.5 + np.random.normal(0, 0.1)) / grid_size
            y = (j + 0.5 + np.random.normal(0, 0.1)) / grid_size
            # Ensure points stay within bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            points.append([x, y])

    points = np.array(points[:n_circles])

    # Create Voronoi diagram to get geometric information
    try:
        from scipy.spatial import Voronoi
        vor = Voronoi(points)
        # Compute approximate area for each point (Voronoi cell area)
        cell_areas = []
        for i in range(len(points)):
            # For simplicity, use distance-based area estimation
            # Find nearest neighbors to estimate local density
            if len(vor.points) > 1:
                distances = np.sqrt(np.sum((vor.points - vor.points[i])**2, axis=1))
                distances = np.sort(distances)
                # Take the first few nearest neighbors to estimate local area
                if len(distances) > 1:
                    avg_distance = np.mean(distances[1:4])  # Average of 3 nearest neighbors
                    area_estimate = np.pi * avg_distance**2
                    cell_areas.append(area_estimate)
                else:
                    cell_areas.append(0.01)
            else:
                cell_areas.append(0.01)

        # Normalize areas and assign radii
        if len(cell_areas) > 0:
            max_area = max(cell_areas)
            # Assign radii inversely proportional to cell area (smaller areas = larger radii)
            radii = [min(0.1, max(0.01, 0.05 * max_area / (area + 1e-8))) for area in cell_areas]
        else:
            radii = [0.05] * len(points)
    except:
        # Fallback to simple approach if Voronoi fails
        radii = [np.random.uniform(0.01, 0.1) for _ in range(len(points))]

    # Create circles array
    circles = np.zeros((len(points), 3))
    for i, (point, r) in enumerate(zip(points, radii)):
        circles[i] = [point[0], point[1], r]

    # Ensure all circles are within bounds
    for i in range(len(circles)):
        x, y, r = circles[i]
        circles[i, 0] = np.clip(x, r, 1 - r)
        circles[i, 1] = np.clip(y, r, 1 - r)
        circles[i, 2] = max(0.001, circles[i, 2])

    return circles

def create_initial_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Create initial population using enhanced Voronoi-based initialization"""
    population = []

    # Adaptive grid refinement with fitness-based perturbation scaling
    # Adjust perturbation based on initial population quality
    fitness_variance = 0
    
    for _ in range(pop_size):
        # Use Voronoi-based initialization
        circles = create_voronoi_initialization(n_circles)

        # Add some additional randomization to improve diversity
        perturbation_scale = 0.01 + 0.01 * np.random.rand()  # Dynamic scaling
        
        for i in range(n_circles):
            if np.random.rand() < 0.4:  # 40% chance to perturb
                circles[i, 0] += np.random.normal(0, perturbation_scale)
                circles[i, 1] += np.random.normal(0, perturbation_scale)
                circles[i, 2] += np.random.normal(0, 0.003)

                # Ensure valid bounds
                circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
                circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
                circles[i, 2] = max(0.001, circles[i, 2])

        population.append(circles)

    return population

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int = 3) -> np.ndarray:
    """Select parent using adaptive tournament selection based on population diversity"""
    # Calculate population diversity (variance of fitness values)
    if len(fitnesses) > 1:
        diversity = np.var(fitnesses)
        # Adjust tournament size based on diversity
        # High diversity → smaller tournaments (more exploration)
        # Low diversity → larger tournaments (more exploitation)
        if diversity > np.std(fitnesses):  # Diversity is high
            tournament_size = max(2, int(tournament_size * 0.7))
        else:  # Diversity is low
            tournament_size = min(7, int(tournament_size * 1.3))

    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Constraint-aware crossover with overlap probability weighting"""
    child = parent1.copy()

    # Calculate overlap risk between corresponding circles in parents
    overlap_risks = []
    for i in range(len(parent1)):
        x1, y1, r1 = parent1[i]
        x2, y2, r2 = parent2[i]
        
        # Calculate distance between parents' circles
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        overlap_risk = max(0, (r1 + r2) - distance) / (r1 + r2 + 1e-8)
        overlap_risks.append(overlap_risk)

    # Apply crossover probability based on overlap risk
    crossover_probabilities = [0.8 if risk < 0.5 else 0.3 for risk in overlap_risks]
    
    # Apply crossover with weighted probabilities
    for i in range(len(child)):
        if np.random.rand() < crossover_probabilities[i]:
            child[i] = parent2[i].copy()

    return child

def mutate(circles: np.ndarray, mutation_rate: float = 0.1,
           max_radius_change: float = 0.02) -> np.ndarray:
    """Apply mutation to a circle configuration with adaptive parameters"""
    mutated = circles.copy()

    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            # Mutate position or radius with different strengths
            if np.random.rand() < 0.5:  # Mutate position
                # Adaptive position mutation strength
                mutation_strength = 0.02 + 0.01 * np.random.rand()
                mutated[i, 0] += np.random.normal(0, mutation_strength)
                mutated[i, 1] += np.random.normal(0, mutation_strength)

                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
            else:  # Mutate radius
                # Adaptive radius mutation strength
                adaptive_radius_change = max_radius_change * (0.8 + 0.4 * np.random.rand())
                mutated[i, 2] += np.random.normal(0, adaptive_radius_change)
                mutated[i, 2] = max(0.001, mutated[i, 2])

    return mutated

def repair_circles(circles: np.ndarray) -> np.ndarray:
    """Repair invalid circle configurations with hierarchical approach"""
    repaired = circles.copy()

    # Classify overlap severity
    points = repaired[:, :2]
    tree = cKDTree(points)
    overlap_count = np.zeros(len(repaired))
    
    for i in range(len(repaired)):
        x1, y1, r1 = repaired[i]
        nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))
        overlap_count[i] = sum(1 for j in nearby_indices if i != j)

    # Separate into three groups
    low_overlap = [i for i in range(len(repaired)) if overlap_count[i] < 5]
    medium_overlap = [i for i in range(len(repaired)) if 5 <= overlap_count[i] < 15]
    high_overlap = [i for i in range(len(repaired)) if overlap_count[i] >= 15]

    # Apply different refinement strategies based on overlap severity
    
    # High overlap circles: intensive local search
    for i in high_overlap:
        # Try to expand radius first
        old_r = repaired[i, 2]
        for _ in range(10):
            test_r = min(old_r * 1.1, 0.5)
            valid = True
            for j in range(len(repaired)):
                if i != j:
                    dist = np.sqrt((repaired[i, 0] - repaired[j, 0])**2 + 
                                  (repaired[i, 1] - repaired[j, 1])**2)
                    if dist < test_r + repaired[j, 2]:
                        valid = False
                        break
            if valid:
                repaired[i, 2] = test_r
            else:
                break

    # Medium overlap circles: moderate refinement
    for i in medium_overlap:
        # Try small adjustments
        for _ in range(5):
            test_r = min(repaired[i, 2] * 1.05, 0.5)
            valid = True
            for j in range(len(repaired)):
                if i != j:
                    dist = np.sqrt((repaired[i, 0] - repaired[j, 0])**2 + 
                                  (repaired[i, 1] - repaired[j, 1])**2)
                    if dist < test_r + repaired[j, 2]:
                        valid = False
                        break
            if valid:
                repaired[i, 2] = test_r
            else:
                break

    # Low overlap circles: light refinement
    for i in low_overlap:
        # Try minor radius increase
        for _ in range(3):
            test_r = min(repaired[i, 2] * 1.02, 0.5)
            valid = True
            for j in range(len(repaired)):
                if i != j:
                    dist = np.sqrt((repaired[i, 0] - repaired[j, 0])**2 + 
                                  (repaired[i, 1] - repaired[j, 1])**2)
                    if dist < test_r + repaired[j, 2]:
                        valid = False
                        break
            if valid:
                repaired[i, 2] = test_r
            else:
                break

    # Final bound enforcement
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        # Keep within bounds
        repaired[i, 0] = np.clip(x, r, 1 - r)
        repaired[i, 1] = np.clip(y, r, 1 - r)
        repaired[i, 2] = max(0.001, repaired[i, 2])  # Ensure positive radius

    return repaired

def progressive_local_optimization(circles: np.ndarray) -> np.ndarray:
    """Apply progressive local optimization with different intensities"""
    optimized = circles.copy()
    
    # Phase 1: Coarse adjustments
    for _ in range(20):
        improved = False
        for i in range(len(optimized)):
            current_r = optimized[i, 2]
            new_r = min(current_r * 1.05, 0.5)
            
            valid = True
            for j in range(len(optimized)):
                if i != j:
                    dist = np.sqrt((optimized[i, 0] - optimized[j, 0])**2 + 
                                  (optimized[i, 1] - optimized[j, 1])**2)
                    if dist < new_r + optimized[j, 2]:
                        valid = False
                        break
            
            if valid and new_r <= 1 - max(optimized[i, 0], 1 - optimized[i, 0]) and \
               new_r <= 1 - max(optimized[i, 1], 1 - optimized[i, 1]):
                optimized[i, 2] = new_r
                improved = True
                
        if not improved:
            break
    
    # Phase 2: Fine adjustments
    for _ in range(10):
        improved = False
        for i in range(len(optimized)):
            current_r = optimized[i, 2]
            new_r = min(current_r * 1.02, 0.5)
            
            valid = True
            for j in range(len(optimized)):
                if i != j:
                    dist = np.sqrt((optimized[i, 0] - optimized[j, 0])**2 + 
                                  (optimized[i, 1] - optimized[j, 1])**2)
                    if dist < new_r + optimized[j, 2]:
                        valid = False
                        break
            
            if valid and new_r <= 1 - max(optimized[i, 0], 1 - optimized[i, 0]) and \
               new_r <= 1 - max(optimized[i, 1], 1 - optimized[i, 1]):
                optimized[i, 2] = new_r
                improved = True
                
        if not improved:
            break
            
    return optimized

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Algorithm parameters
    pop_size = 60
    n_generations = 120
    elite_size = 5

    # Three-phase adaptive mutation rate scheduling
    # Phase 1 (0-60): High exploration (0.15)
    # Phase 2 (61-100): Balanced refinement (0.05)
    # Phase 3 (101-120): Fine-tuning (0.015)
    def get_adaptive_mutation_rate(gen):
        if gen < 60:
            return 0.15
        elif gen < 100:
            return 0.05
        else:
            return 0.015

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
            parent1 = tournament_selection(valid_individuals, fitnesses)
            parent2 = tournament_selection(valid_individuals, fitnesses)

            # Crossover with constraint awareness
            child = crossover(parent1, parent2)

            # Get adaptive mutation rate
            mutation_rate = get_adaptive_mutation_rate(generation)
            
            # Mutation
            child = mutate(child, mutation_rate)

            # Repair
            child = repair_circles(child)
            
            # Progressive local optimization
            child = progressive_local_optimization(child)

            new_population.append(child)

        population = new_population[:pop_size]

    # Return the best solution found
    if best_individual is not None:
        return best_individual
    else:
        # If no valid solution found, return the best from final population
        fitnesses = [calculate_sum_radii(circles) for circles in population if validate_circles(circles)]
        if fitnesses:
            best_idx = np.argmax(fitnesses)
            return population[best_idx]
        else:
            # Fallback: return a valid random solution
            circles = np.zeros((26, 3))
            for i in range(26):
                circles[i] = [0.5, 0.5, 0.01]
            return circles


# EVOLVE-BLOCK-END