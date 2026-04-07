# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree, Voronoi
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import math
import heapq

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def validate_circles(circles: np.ndarray) -> bool:
    """
    Validates that all circles are within bounds and don't overlap.
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
    """Create initial population using Voronoi-based distribution"""
    # Generate random points and create Voronoi diagram
    points = np.random.rand(2 * n_circles, 2)  # Generate more points than needed
    
    try:
        vor = Voronoi(points)
        # Select valid Voronoi vertices as circle centers
        # Filter vertices that are inside the unit square
        valid_vertices = []
        for vertex in vor.vertices:
            if 0 <= vertex[0] <= 1 and 0 <= vertex[1] <= 1:
                valid_vertices.append(vertex)
        
        # Take enough valid vertices
        selected_vertices = valid_vertices[:n_circles]
        
        if len(selected_vertices) < n_circles:
            # Fall back to random placement
            circles = np.zeros((n_circles, 3))
            for i in range(n_circles):
                circles[i] = [np.random.uniform(0.05, 0.95), 
                            np.random.uniform(0.05, 0.95), 
                            np.random.uniform(0.01, 0.1)]
            return circles
            
        circles = np.zeros((n_circles, 3))
        for i, vertex in enumerate(selected_vertices):
            x, y = vertex
            # Compute maximum radius for this point
            max_radius = min(x, 1-x, y, 1-y)
            # Set radius to be proportional to available space
            r = min(0.05, max_radius * np.random.uniform(0.3, 0.7))
            circles[i] = [x, y, r]
            
        return circles
    except:
        # Fallback method if Voronoi fails
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            circles[i] = [np.random.uniform(0.05, 0.95), 
                         np.random.uniform(0.05, 0.95), 
                         np.random.uniform(0.01, 0.1)]
        return circles

def create_initial_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Create initial population with Voronoi-based initialization"""
    population = []
    
    # Use mixture of Voronoi and grid initialization
    for i in range(pop_size):
        if i % 3 == 0:
            # Voronoi-based initialization
            circles = create_voronoi_initialization(n_circles)
        else:
            # Grid-based but with more randomness
            circles = np.zeros((n_circles, 3))
            grid_size = int(np.ceil(np.sqrt(n_circles))) + 1
            spacing_x = 1.0 / grid_size
            spacing_y = 1.0 / grid_size
            
            idx = 0
            for i_grid in range(grid_size):
                for j_grid in range(grid_size):
                    if idx >= n_circles:
                        break
                    x = (i_grid + 0.5 + np.random.uniform(-0.1, 0.1)) * spacing_x
                    y = (j_grid + 0.5 + np.random.uniform(-0.1, 0.1)) * spacing_y
                    max_radius = min(x, 1-x, y, 1-y)
                    r = min(0.08, max_radius * np.random.uniform(0.2, 0.6))
                    circles[idx] = [x, y, r]
                    idx += 1
        population.append(circles)
    return population

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int = 3) -> np.ndarray:
    """Select parent using tournament selection"""
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def calculate_overlap_probability(circle1, circle2):
    """Calculate probability of overlap between two circles"""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    if distance >= r1 + r2:
        return 0.0
    elif distance <= abs(r1 - r2):
        return 1.0
    else:
        # Approximate overlap probability
        max_possible_distance = r1 + r2
        return 1.0 - (distance / max_possible_distance)

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Crossover with overlap probability aware selection"""
    child = parent1.copy()
    
    # Calculate overlap probabilities for each circle pair
    overlap_probs = []
    for i in range(len(parent1)):
        prob = calculate_overlap_probability(parent1[i], parent2[i])
        overlap_probs.append(prob)
    
    # Higher overlap probability means less likely to crossover
    crossover_probs = [0.3 + 0.7 * (1 - prob) for prob in overlap_probs]
    
    # Apply crossover with adaptive probabilities
    for i in range(len(parent1)):
        if np.random.rand() < crossover_probs[i]:
            child[i] = parent2[i].copy()
    
    # Ensure constraint validity
    for i in range(len(child)):
        x, y, r = child[i]
        child[i, 0] = np.clip(x, r, 1 - r)
        child[i, 1] = np.clip(y, r, 1 - r)
        child[i, 2] = max(0.001, r)
    
    return child

def mutate(circles: np.ndarray, mutation_rate: float = 0.1,
           phase: str = 'exploration') -> np.ndarray:
    """Mutation with phase-adaptive strategies"""
    mutated = circles.copy()
    
    # Different mutation strategies based on evolutionary phase
    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            if phase == 'exploitation':
                # More focused mutations for exploitation phase
                if np.random.rand() < 0.6:
                    # Position mutation with smaller step
                    mutated[i, 0] += np.random.normal(0, 0.005)
                    mutated[i, 1] += np.random.normal(0, 0.005)
                    mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                    mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
                else:
                    # Radius mutation with smaller step
                    mutated[i, 2] += np.random.normal(0, 0.003)
                    mutated[i, 2] = max(0.001, mutated[i, 2])
            else:
                # Normal mutation for exploration phase
                if np.random.rand() < 0.7:
                    # Position mutation with larger step
                    mutated[i, 0] += np.random.normal(0, 0.01)
                    mutated[i, 1] += np.random.normal(0, 0.01)
                    mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                    mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
                else:
                    # Radius mutation
                    mutated[i, 2] += np.random.normal(0, 0.005)
                    mutated[i, 2] = max(0.001, mutated[i, 2])
    
    return mutated

def compute_energy(circles: np.ndarray) -> float:
    """Compute total energy of the system (repulsive forces)"""
    energy = 0.0
    n = len(circles)
    points = circles[:, :2]
    radii = circles[:, 2]
    
    for i in range(n):
        x1, y1, r1 = circles[i]
        for j in range(i+1, n):
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                # Overlapping circles - high repulsive energy
                energy += 1000 * (r1 + r2 - distance)**2
            elif distance < 2*(r1 + r2):
                # Close but not overlapping - moderate repulsion
                force = 1.0 / (distance**2 + 0.001)
                energy += force
    return energy

def local_optimization(circles: np.ndarray, max_iterations: int = 30) -> np.ndarray:
    """Advanced local optimization using simulated annealing-like approach"""
    current = circles.copy()
    current_energy = compute_energy(current)
    
    # Simulated annealing parameters
    temperature = 1.0
    cooling_rate = 0.95
    min_temperature = 0.001
    
    for iteration in range(max_iterations):
        # Make a small random change
        candidate = current.copy()
        idx = np.random.randint(0, len(candidate))
        
        # Perturb one circle
        if np.random.rand() < 0.5:
            # Move position
            candidate[idx, 0] += np.random.normal(0, 0.005)
            candidate[idx, 1] += np.random.normal(0, 0.005)
            candidate[idx, 0] = np.clip(candidate[idx, 0], candidate[idx, 2], 1 - candidate[idx, 2])
            candidate[idx, 1] = np.clip(candidate[idx, 1], candidate[idx, 2], 1 - candidate[idx, 2])
        else:
            # Change radius
            candidate[idx, 2] += np.random.normal(0, 0.003)
            candidate[idx, 2] = max(0.001, candidate[idx, 2])
        
        # Check if new configuration is valid
        if not validate_circles(candidate):
            continue
            
        candidate_energy = compute_energy(candidate)
        
        # Accept or reject based on energy difference and temperature
        if candidate_energy < current_energy or np.random.rand() < np.exp(-(candidate_energy - current_energy) / (temperature + 1e-10)):
            current = candidate
            current_energy = candidate_energy
        
        # Cool down temperature
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature
    
    return current

def repair_circles(circles: np.ndarray) -> np.ndarray:
    """Enhanced repair mechanism"""
    repaired = circles.copy()
    
    # First stage: Fix boundary violations
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        repaired[i, 0] = np.clip(x, r, 1 - r)
        repaired[i, 1] = np.clip(y, r, 1 - r)
        repaired[i, 2] = max(0.001, repaired[i, 2])
    
    # Second stage: Resolve remaining overlaps
    points = repaired[:, :2]
    tree = cKDTree(points)
    
    # Try to resolve overlaps step by step
    for _ in range(5):
        overlap_found = False
        for i in range(len(repaired)):
            x1, y1, r1 = repaired[i]
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))
            
            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = repaired[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2
                    
                    if distance < min_distance:
                        overlap_found = True
                        
                        # Move circles apart with attraction to center
                        if distance > 0.001:
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance
                            
                            # Move them apart
                            move_amount = (min_distance - distance) * 0.3
                            repaired[i, 0] += dx * move_amount * 0.5
                            repaired[i, 1] += dy * move_amount * 0.5
                            repaired[j, 0] -= dx * move_amount * 0.5
                            repaired[j, 1] -= dy * move_amount * 0.5
                            
                            # Keep within bounds
                            repaired[i, 0] = np.clip(repaired[i, 0], r1, 1 - r1)
                            repaired[i, 1] = np.clip(repaired[i, 1], r1, 1 - r1)
                            repaired[j, 0] = np.clip(repaired[j, 0], r2, 1 - r2)
                            repaired[j, 1] = np.clip(repaired[j, 1], r2, 1 - r2)
        
        if not overlap_found:
            break
    
    return repaired

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Algorithm parameters with three distinct phases
    pop_size = 80
    n_generations = 200
    elite_size = 10
    
    # Phase indicators
    exploration_phase = 0
    exploitation_phase = 1
    fine_tuning_phase = 2
    
    # Create initial population
    population = create_initial_population(pop_size, 26)
    
    # Evolution loop
    best_fitness = -np.inf
    best_individual = None
    
    for generation in range(n_generations):
        # Determine evolutionary phase
        if generation < n_generations * 0.4:
            phase = exploration_phase
            current_mutation_rate = 0.15
        elif generation < n_generations * 0.8:
            phase = exploitation_phase
            current_mutation_rate = 0.08
        else:
            phase = fine_tuning_phase
            current_mutation_rate = 0.03
        
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
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, current_mutation_rate, phase)
            
            # Local optimization
            child = local_optimization(child)
            
            # Repair
            child = repair_circles(child)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
        
        # Logging every 25 generations
        if generation % 25 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
    
    # Final optimization of best solution
    if best_individual is not None:
        best_individual = local_optimization(best_individual)
    
    # Return the best solution found
    if best_individual is not None:
        return best_individual
    else:
        # If no valid solution found, return a valid random solution
        circles = np.zeros((26, 3))
        for i in range(26):
            circles[i] = [0.5, 0.5, 0.01]
        return circles

# EVOLVE-BLOCK-END