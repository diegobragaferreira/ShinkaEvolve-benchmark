# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import time

# Global constant for rectangle dimensions
RECT_WIDTH = 1.3
RECT_HEIGHT = 0.7

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Container dimensions - rectangle with perimeter 4, so width + height = 2
    # Using optimized 1.3:0.7 ratio based on prior analysis
    container_width = RECT_WIDTH
    container_height = RECT_HEIGHT

    # Two-stage optimization approach for better results
    # Stage 1: Coarse optimization with adaptive grid refinement
    coarse_solution = two_stage_optimization(container_width, container_height, 21, stage='coarse')

    # Stage 2: Fine optimization with evolutionary approach
    fine_solution = two_stage_optimization(container_width, container_height, 21, stage='fine',
                                          initial_solution=coarse_solution)

    return fine_solution

def two_stage_optimization(width: float, height: float, n_circles: int, stage: str,
                          initial_solution: np.ndarray = None) -> np.ndarray:
    """Two-stage optimization approach: coarse for exploration, fine for exploitation"""

    if stage == 'coarse':
        # Coarse stage: simpler, faster optimization focusing on good initial layout
        circles = generate_adaptive_grid_solution(width, height, n_circles)

        # Quick refinement to resolve overlaps
        circles = quick_refinement(circles, width, height)

        return circles

    elif stage == 'fine':
        # Fine stage: more sophisticated evolutionary optimization
        if initial_solution is None:
            circles = generate_adaptive_grid_solution(width, height, n_circles)
        else:
            circles = initial_solution.copy()

        # Parameters for fine-tuning
        max_iterations = 2000
        population_size = 120
        elite_size = 12
        mutation_rate = 0.12
        crossover_rate = 0.75

        # Initialize population
        population = []
        for _ in range(population_size):
            if initial_solution is not None and random.random() < 0.3:  # 30% chance to use initial
                population.append(initial_solution.copy())
            else:
                circles = generate_adaptive_grid_solution(width, height, n_circles)
                population.append(circles)

        best_solution = None
        best_sum = 0

        # Evolutionary loop
        for generation in range(max_iterations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = evaluate_fitness(individual, width, height)
                fitness_scores.append(fitness)

            # Update best solution
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_sum:
                best_sum = fitness_scores[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()

            # Selection (tournament)
            selected = tournament_selection(population, fitness_scores, population_size)

            # Crossover and mutation
            new_population = []
            for i in range(0, population_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % population_size]

                if random.random() < crossover_rate:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation with adaptive scaling based on generation
                adaptive_mutation_rate = mutation_rate * (1.0 - generation/max_iterations)
                mutate(child1, width, height, adaptive_mutation_rate)
                mutate(child2, width, height, adaptive_mutation_rate)

                new_population.extend([child1, child2])

            population = new_population[:population_size]

        # Return best solution found
        if best_solution is not None:
            return best_solution
        else:
            # Fallback to initial solution
            return generate_adaptive_grid_solution(width, height, n_circles)

def generate_adaptive_grid_solution(width: float, height: float, n_circles: int) -> np.ndarray:
    """Generate initial solution using adaptive grid approach considering rectangle proportions."""
    circles = np.zeros((n_circles, 3))

    # Calculate optimal grid spacing using proven formula for uniform coverage
    # This formula ensures better distribution across the rectangle
    spacing = np.sqrt((width * height) / n_circles)

    # Determine grid dimensions based on spacing
    cols = max(1, int(np.ceil(width / spacing)))
    rows = max(1, int(np.ceil(height / spacing)))

    # Adjust to ensure enough circles
    if cols * rows < n_circles:
        cols = max(cols, int(np.ceil(n_circles / rows)))

    # Calculate actual cell dimensions
    cell_width = width / cols
    cell_height = height / rows

    # Place circles in a grid pattern with better randomness and positioning
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_circles:
                break

            # Position with better spatial distribution
            # Alternate row offset for honeycomb-like packing
            offset = (i % 2) * (cell_width / 2) if cols > 1 else 0

            x = (j + 0.5) * cell_width + offset + (random.random() - 0.5) * cell_width * 0.15
            y = (i + 0.5) * cell_height + (random.random() - 0.5) * cell_height * 0.15

            # Ensure valid bounds with margin
            x = max(0.01, min(width - 0.01, x))
            y = max(0.01, min(height - 0.01, y))

            # Initial radius based on available space with better factors
            min_radius = min(x, width - x, y, height - y) * 0.35

            # Use more aggressive initial radius for better exploration
            radius = min_radius * random.uniform(0.4, 0.8)

            circles[idx] = [x, y, radius]
            idx += 1

    # Refine using improved local optimization
    circles = refine_positions_improved(circles, width, height)

    return circles

def quick_refinement(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Quick refinement to fix major issues"""
    # Basic constraint enforcement
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Ensure within bounds
        circles[i, 0] = max(r, min(width - r, x))
        circles[i, 1] = max(r, min(height - r, y))

    # Simple overlap resolution
    for _ in range(50):
        improved = False
        for i in range(len(circles)):
            x, y, r = circles[i]

            # Look for nearby circles and adjust position to avoid overlap
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x2 - x)**2 + (y2 - y)**2)

                    if distance < (r + r2) and distance > 0:
                        # Move circle away from overlapping neighbor
                        dx = x2 - x
                        dy = y2 - y
                        separation = (r + r2) - distance

                        # Normalize direction vector
                        norm = np.sqrt(dx*dx + dy*dy)
                        if norm > 0:
                            move_x = (dx / norm) * separation * 0.1
                            move_y = (dy / norm) * separation * 0.1

                            circles[i, 0] += move_x
                            circles[i, 1] += move_y
                            improved = True

        if not improved:
            break

    return circles

def evaluate_fitness(circles: np.ndarray, width: float, height: float) -> float:
    """Evaluate fitness based on sum of radii, penalizing constraint violations with more sophisticated penalty."""
    sum_radii = np.sum(circles[:, 2])

    # Check constraints and apply penalties using spatial indexing for efficiency
    penalty = 0

    # Boundary violations (more severe penalty)
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
            penalty += 10000  # Heavier penalty for boundary violations

    # Overlap violations using spatial indexing
    try:
        points = circles[:, :2]
        tree = cKDTree(points)

        # Find overlapping pairs efficiently
        max_radius = np.max(circles[:, 2])
        pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

        for i, j in pairs:
            if i < j:  # Avoid double counting
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]

                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if distance < (r1 + r2):
                    # Weight overlap penalty by the amount of overlap
                    overlap = (r1 + r2) - distance
                    penalty += overlap * 3000  # Reduced penalty to allow some overlap exploration

    except Exception:
        # Fallback to brute force if spatial indexing fails
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]

                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if distance < (r1 + r2):
                    overlap = (r1 + r2) - distance
                    penalty += overlap * 3000

    # Add penalty for very small radii to encourage meaningful circle sizes
    for i in range(len(circles)):
        if circles[i, 2] < 0.005:
            penalty += 1000

    return sum_radii - penalty

def tournament_selection(population: List[np.ndarray], fitness_scores: List[float], k: int) -> List[np.ndarray]:
    """Perform tournament selection with variable tournament size."""
    selected = []
    for _ in range(k):
        # Use variable tournament size to balance exploration and exploitation
        tournament_size = random.choice([3, 4, 5])
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_index])
    return selected

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform uniform crossover with bias towards better parent."""
    child1 = parent1.copy()
    child2 = parent2.copy()

    # For each circle, choose from parent based on fitness
    parent1_fitness = np.sum(parent1[:, 2])
    parent2_fitness = np.sum(parent2[:, 2])

    # Bias towards better parent based on fitness difference
    fitness_diff = abs(parent1_fitness - parent2_fitness)
    bias = 0.6 + 0.2 * (parent1_fitness / (parent1_fitness + parent2_fitness + 1e-8))

    for i in range(len(parent1)):
        if random.random() < bias:
            child1[i] = parent1[i].copy()
            child2[i] = parent2[i].copy()
        else:
            child1[i] = parent2[i].copy()
            child2[i] = parent1[i].copy()

    return child1, child2

def mutate(individual: np.ndarray, width: float, height: float, rate: float) -> None:
    """Mutate an individual with adaptive mutation strategy."""
    for i in range(len(individual)):
        if random.random() < rate:
            # Mutate position or radius with adaptive probabilities
            if random.random() < 0.65:  # 65% chance to mutate position for better stability
                # Mutate position with adaptive magnitude
                magnitude_x = random.uniform(-0.015, 0.015) * width
                magnitude_y = random.uniform(-0.015, 0.015) * height

                individual[i][0] += magnitude_x
                individual[i][1] += magnitude_y

                # Keep within bounds
                individual[i][0] = max(0.01, min(width - 0.01, individual[i][0]))
                individual[i][1] = max(0.01, min(height - 0.01, individual[i][1]))
            else:  # 35% chance to mutate radius
                # Mutate radius with adaptive scaling based on current radius
                scale_factor = random.uniform(0.85, 1.15)
                individual[i][2] *= scale_factor

                # Ensure positive radius with minimum threshold
                individual[i][2] = max(0.001, individual[i][2])

def refine_positions_improved(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Improved refinement positions to ensure no overlaps and respect boundaries."""
    # Use a more sophisticated local optimization approach
    for _ in range(150):  # Fewer iterations but more targeted
        updated = False

        # Try to increase radii while maintaining no overlaps
        for i in range(len(circles)):
            x, y, r = circles[i]

            # Calculate maximum possible radius at this location
            max_r = min(x, width - x, y, height - y)

            # Check for overlap with other circles
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x2 - x)**2 + (y2 - y)**2)

                    # Can't get closer than sum of radii
                    if distance < (r + r2):
                        max_r = min(max_r, distance - r2)

            # Increase radius if beneficial and safe
            if max_r > r and max_r > 0.001:
                # Use a more conservative increase to prevent instability
                increment = min((max_r - r) * 0.2, 0.03)
                circles[i][2] = min(max_r, r + increment)
                updated = True

        if not updated:
            break

    # Final boundary correction
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Correct if out of bounds
        if x - r < 0:
            circles[i][0] = r
        elif x + r > width:
            circles[i][0] = width - r
        if y - r < 0:
            circles[i][1] = r
        elif y + r > height:
            circles[i][1] = height - r

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")