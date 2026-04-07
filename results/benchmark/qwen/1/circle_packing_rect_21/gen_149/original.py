# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import random
from typing import Tuple, List, Optional
import warnings

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class ConstraintChecker:
    """Handles constraint validation for circle packing solutions."""

    @staticmethod
    def check_bounds(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
        """Check if all circles are within rectangle bounds."""
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                return False
        return True

    @staticmethod
    def check_overlaps(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
        """Check if any circles overlap using spatial indexing for efficiency."""
        n = len(circles)
        if n <= 1:
            return True

        # Use spatial grid for efficient overlap detection
        grid_size = 0.2
        grid_width = int(np.ceil(rect_width / grid_size))
        grid_height = int(np.ceil(rect_height / grid_size))
        grid = {}

        # Place circles in grid cells
        for i in range(n):
            x, y, r = circles[i]
            grid_x = int(x / grid_size)
            grid_y = int(y / grid_size)
            if (grid_x, grid_y) not in grid:
                grid[(grid_x, grid_y)] = []
            grid[(grid_x, grid_y)].append(i)

        # Check for overlaps using grid-based approach
        for i in range(n):
            x1, y1, r1 = circles[i]
            grid_x = int(x1 / grid_size)
            grid_y = int(y1 / grid_size)

            # Check this cell and adjacent cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_cell = (grid_x + dx, grid_y + dy)
                    if neighbor_cell in grid:
                        for j in grid[neighbor_cell]:
                            if i != j:
                                x2, y2, r2 = circles[j]
                                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                                if distance < (r1 + r2):
                                    return False
        return True

    @staticmethod
    def validate(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
        """Validate that all constraints are satisfied."""
        return ConstraintChecker.check_bounds(circles, rect_width, rect_height) and \
               ConstraintChecker.check_overlaps(circles, rect_width, rect_height)

class FitnessEvaluator:
    """Evaluates fitness of circle solutions."""

    @staticmethod
    def evaluate(circles: np.ndarray) -> float:
        """Evaluate fitness as the sum of radii with constraint validation."""
        if not ConstraintChecker.validate(circles):
            return -np.inf
        return np.sum(circles[:, 2])

class InitializerFactory:
    """Factory for creating different initial solution strategies."""

    @staticmethod
    def create_hexagonal_solution(rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
        """Create initial solution using hexagonal lattice pattern."""
        circles = np.zeros((21, 3))

        # Hexagonal grid parameters
        rows = int(np.ceil(np.sqrt(21)))
        cols = int(np.ceil(21 / rows))

        # Adjust for rectangular container
        if rect_width >= rect_height:
            grid_width = rect_width * 0.9
            grid_height = rect_height * 0.9
        else:
            grid_width = rect_width * 0.9
            grid_height = rect_height * 0.9

        # Calculate spacing
        cell_width = grid_width / cols
        cell_height = grid_height / rows
        min_cell_dim = min(cell_width, cell_height)
        hex_radius = min_cell_dim * 0.4

        # Arrange in hexagonal pattern
        placed = 0
        for row in range(rows):
            if placed >= 21:
                break
            for col in range(cols):
                if placed >= 21:
                    break

                # Offset every other row for hexagonal pattern
                offset = (row % 2) * (cell_width / 2)
                x = offset + col * cell_width + cell_width / 2
                y = row * cell_height + cell_height / 2

                # Ensure within bounds
                x = np.clip(x, hex_radius, rect_width - hex_radius)
                y = np.clip(y, hex_radius, rect_height - hex_radius)

                # Adjust radius to prevent boundary issues
                max_radius = min(x, y, rect_width - x, rect_height - y)
                r = min(hex_radius, max_radius * 0.8)

                circles[placed] = [x, y, r]
                placed += 1

        # Fill remaining positions with small random circles
        for i in range(placed, 21):
            x = np.random.uniform(hex_radius, rect_width - hex_radius)
            y = np.random.uniform(hex_radius, rect_height - hex_radius)
            r = np.random.uniform(0.005, hex_radius * 0.5)
            circles[i] = [x, y, r]

        return circles

    @staticmethod
    def create_voronoi_solution(rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
        """Create initial solution inspired by Voronoi diagrams."""
        circles = np.zeros((21, 3))

        # Corner placements
        corner_positions = [
            (0.2 * rect_width, 0.2 * rect_height),
            (0.8 * rect_width, 0.2 * rect_height),
            (0.2 * rect_width, 0.8 * rect_height),
            (0.8 * rect_width, 0.8 * rect_height),
            (rect_width/2, rect_height/2)
        ]

        placed = 0
        for x, y in corner_positions:
            if placed >= 21:
                break
            r = min(x, y, rect_width - x, rect_height - y) * 0.15
            circles[placed] = [x, y, r]
            placed += 1

        # Fill remaining positions with greedy approach
        max_attempts = 10000
        for attempt in range(max_attempts):
            if placed >= 21:
                break

            x = np.random.uniform(0.05 * rect_width, 0.95 * rect_width)
            y = np.random.uniform(0.05 * rect_height, 0.95 * rect_height)

            # Find closest existing circle
            min_dist = float('inf')
            for i in range(placed):
                existing_x, existing_y, existing_r = circles[i]
                distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                min_dist = min(min_dist, distance)

            if min_dist > 0.1 * min(rect_width, rect_height):
                r = np.random.uniform(0.01, min(x, y, rect_width - x, rect_height - y) * 0.2)
                circles[placed] = [x, y, r]
                placed += 1

        # Fill remaining positions with small random circles
        for i in range(placed, 21):
            x = np.random.uniform(0.05 * rect_width, 0.95 * rect_width)
            y = np.random.uniform(0.05 * rect_height, 0.95 * rect_height)
            r = np.random.uniform(0.005, 0.05)
            circles[i] = [x, y, r]

        return circles

    @staticmethod
    def create_strategic_solution(rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
        """Create initial solution with strategic placement near boundaries and center."""
        circles = np.zeros((21, 3))

        # Place circles in key locations
        circles[0] = [rect_width / 2, rect_height / 2, min(rect_width, rect_height) * 0.1]

        # Corners
        corners = [
            (rect_width * 0.1, rect_height * 0.1),
            (rect_width * 0.9, rect_height * 0.1),
            (rect_width * 0.1, rect_height * 0.9),
            (rect_width * 0.9, rect_height * 0.9)
        ]

        placed = 1
        for x, y in corners:
            if placed >= 21:
                break
            r = min(x, y, rect_width - x, rect_height - y) * 0.1
            circles[placed] = [x, y, r]
            placed += 1

        # Along edges (not corners)
        edges = [
            (rect_width * 0.5, rect_height * 0.1),  # Top edge
            (rect_width * 0.5, rect_height * 0.9),  # Bottom edge
            (rect_width * 0.1, rect_height * 0.5),  # Left edge
            (rect_width * 0.9, rect_height * 0.5),  # Right edge
        ]

        for x, y in edges:
            if placed >= 21:
                break
            r = min(x, y, rect_width - x, rect_height - y) * 0.08
            circles[placed] = [x, y, r]
            placed += 1

        # Fill remaining with uniform random placement
        for i in range(placed, 21):
            x = np.random.uniform(0.05 * rect_width, 0.95 * rect_width)
            y = np.random.uniform(0.05 * rect_height, 0.95 * rect_height)
            r = np.random.uniform(0.005, min(x, y, rect_width - x, rect_height - y) * 0.15)
            circles[i] = [x, y, r]

        return circles

    @staticmethod
    def create_best_initial_solution(rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
        """Create the best initial solution by trying multiple strategies."""
        strategies = [
            InitializerFactory.create_hexagonal_solution,
            InitializerFactory.create_voronoi_solution,
            InitializerFactory.create_strategic_solution
        ]

        best_solution = None
        best_fitness = -np.inf

        for strategy in strategies:
            solution = strategy(rect_width, rect_height)
            fitness = FitnessEvaluator.evaluate(solution)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution.copy()

        return best_solution if best_solution is not None else InitializerFactory.create_hexagonal_solution(rect_width, rect_height)

class MutationOperator:
    """Handles mutation operations with adaptive parameters."""

    @staticmethod
    def compute_voronoi_criticality(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
        """Compute criticality scores for each circle based on Voronoi diagram."""
        if len(circles) < 3:
            return np.ones(len(circles)) * 0.5

        try:
            # Get circle centers
            points = circles[:, :2]

            # Shift points to avoid boundary issues
            shifted_points = points.copy()
            shifted_points[:, 0] = np.clip(shifted_points[:, 0], 0.01, rect_width - 0.01)
            shifted_points[:, 1] = np.clip(shifted_points[:, 1], 0.01, rect_height - 0.01)

            # Compute Voronoi diagram
            vor = Voronoi(shifted_points)

            # Compute area of Voronoi cells
            areas = []
            for i in range(len(shifted_points)):
                # Get vertices of Voronoi cell for point i
                region = vor.regions[vor.point_region[i]]
                if -1 in region:
                    # Infinite region, skip
                    areas.append(1000000)  # Large area for infinite regions
                else:
                    # Compute polygon area
                    vertices = [vor.vertices[j] for j in region if j >= 0]
                    if len(vertices) < 3:
                        areas.append(1000000)
                    else:
                        # Simplified area calculation using cross product
                        vertices_array = np.array(vertices)
                        x = vertices_array[:, 0]
                        y = vertices_array[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        areas.append(area)

            # Convert to criticality score (lower area = higher criticality)
            areas = np.array(areas)
            # Normalize to [0,1] where 0 = most critical
            normalized_areas = (areas - areas.min()) / (areas.max() - areas.min() + 1e-8)
            return 1.0 - normalized_areas  # Higher criticality = closer to 1

        except:
            # Fallback to uniform criticality if Voronoi fails
            return np.ones(len(circles)) * 0.5

    @staticmethod
    def mutate(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
        """Improved mutation operator with Voronoi-based criticality weighting and adaptive step sizes."""
        mutated = circles.copy()

        # Compute criticality scores
        criticality_scores = MutationOperator.compute_voronoi_criticality(circles, rect_width, rect_height)

        for i in range(21):
            if np.random.random() < 0.25:  # 25% mutation rate
                # Choose mutation type with bias towards position (70%)
                mutation_type = np.random.choice(['position', 'radius'], p=[0.7, 0.3])

                # Get criticality score for this circle
                crit_score = criticality_scores[i]

                if mutation_type == 'position':
                    # Mutate position with adaptive step size based on criticality
                    base_step = 0.03 + 0.02 * crit_score  # Smaller steps in critical regions
                    step_size = base_step + np.random.random() * base_step * 0.5

                    mutated[i, 0] += np.random.normal(0, step_size)
                    mutated[i, 1] += np.random.normal(0, step_size)

                    # Boundary awareness: penalize movements that go out of bounds
                    bound_penalty_x = max(0, mutated[i, 0] - (rect_width - mutated[i, 2]))
                    bound_penalty_y = max(0, mutated[i, 1] - (rect_height - mutated[i, 2]))
                    if bound_penalty_x > 0 or bound_penalty_y > 0:
                        mutated[i, 0] -= bound_penalty_x * 0.5
                        mutated[i, 1] -= bound_penalty_y * 0.5

                else:
                    # Mutate radius with log-normal distribution
                    # Scale based on criticality (less change in high-criticality regions)
                    scale_factor = np.exp(np.random.normal(0, 0.15 * (1 - crit_score)))
                    mutated[i, 2] *= scale_factor
                    mutated[i, 2] = max(0.001, mutated[i, 2])

                    # Additional boundary awareness for radius
                    max_possible_radius = min(mutated[i, 0], mutated[i, 1],
                                            rect_width - mutated[i, 0], rect_height - mutated[i, 1])
                    mutated[i, 2] = min(mutated[i, 2], max_possible_radius * 0.9)

        return mutated

class CrossoverOperator:
    """Handles crossover operations for genetic algorithm."""

    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Improved crossover operator with better recombination and symmetry preservation."""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Uniform crossover with blended inheritance
        for i in range(21):
            if np.random.random() < 0.5:
                # Swap entire circle data
                child1[i], child2[i] = child2[i], child1[i]
            else:
                # Blend positions and radii
                alpha_pos = np.random.random() * 0.6 + 0.2  # Blend factor for position [0.2, 0.8]
                alpha_rad = np.random.random() * 0.6 + 0.2  # Blend factor for radius [0.2, 0.8]

                # Position blending
                child1[i, :2] = parent1[i, :2] * alpha_pos + parent2[i, :2] * (1 - alpha_pos)
                child2[i, :2] = parent1[i, :2] * (1 - alpha_pos) + parent2[i, :2] * alpha_pos

                # Radius blending
                child1[i, 2] = parent1[i, 2] * alpha_rad + parent2[i, 2] * (1 - alpha_rad)
                child2[i, 2] = parent1[i, 2] * (1 - alpha_rad) + parent2[i, 2] * alpha_rad

                # Ensure positive radii
                child1[i, 2] = max(0.001, child1[i, 2])
                child2[i, 2] = max(0.001, child2[i, 2])

        return child1, child2

class RepairMechanism:
    """Manages repairing constraint violations in solutions."""

    @staticmethod
    def repair(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
        """Enhanced repair mechanism for fixing constraint violations with boundary-aware approach."""
        repaired = circles.copy()

        # Ensure positive radii
        repaired[:, 2] = np.maximum(repaired[:, 2], 0.001)

        # Enforce bounds and fix boundary violations first
        for i in range(len(repaired)):
            x, y, r = repaired[i]

            # Push back to bounds if necessary
            if x - r < 0:
                x = r
            elif x + r > rect_width:
                x = rect_width - r

            if y - r < 0:
                y = r
            elif y + r > rect_height:
                y = rect_height - r

            repaired[i] = [x, y, r]

        # Resolve overlaps iteratively with spatial indexing for improved efficiency
        for iteration in range(100):
            # Use spatial indexing to detect conflicts efficiently
            grid_size = 0.2
            grid_width = int(np.ceil(rect_width / grid_size))
            grid_height = int(np.ceil(rect_height / grid_size))

            # Initialize grid
            grid = {}

            # Place circles in grid cells
            for i in range(len(repaired)):
                x, y, r = repaired[i]
                # Get grid coordinates for this circle
                grid_x = int(x / grid_size)
                grid_y = int(y / grid_size)

                if (grid_x, grid_y) not in grid:
                    grid[(grid_x, grid_y)] = []
                grid[(grid_x, grid_y)].append(i)

            # Find conflicts using grid-based approach
            conflicts = []

            for i in range(len(repaired)):
                x1, y1, r1 = repaired[i]
                # Get grid coordinates for this circle
                grid_x = int(x1 / grid_size)
                grid_y = int(y1 / grid_size)

                # Check this cell and adjacent cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor_cell = (grid_x + dx, grid_y + dy)
                        if neighbor_cell in grid:
                            for j in grid[neighbor_cell]:
                                if i != j:  # Don't compare with self
                                    x2, y2, r2 = repaired[j]
                                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                                    if distance < (r1 + r2):
                                        conflicts.append((i, j))

            if not conflicts:
                break

            # Move conflicting pairs apart with more intelligent algorithm
            for i, j in conflicts:
                x1, y1, r1 = repaired[i]
                x2, y2, r2 = repaired[j]

                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx*dx + dy*dy)

                if distance > 0:
                    # Move circles away from each other
                    overlap = (r1 + r2) - distance
                    move_distance = overlap / 2

                    dx_norm = dx / distance
                    dy_norm = dy / distance

                    # Apply movement with biased correction
                    # Prioritize moving from boundary if circle is close to boundary
                    boundary_penalty1 = 0
                    boundary_penalty2 = 0

                    # Check proximity to boundaries
                    if x1 <= r1 * 1.5 or x1 >= rect_width - r1 * 1.5 or y1 <= r1 * 1.5 or y1 >= rect_height - r1 * 1.5:
                        boundary_penalty1 = 1.0

                    if x2 <= r2 * 1.5 or x2 >= rect_width - r2 * 1.5 or y2 <= r2 * 1.5 or y2 >= rect_height - r2 * 1.5:
                        boundary_penalty2 = 1.0

                    # Weighted movement
                    total_penalty = boundary_penalty1 + boundary_penalty2
                    weight1 = (1 + boundary_penalty1) / (total_penalty + 1e-8)
                    weight2 = (1 + boundary_penalty2) / (total_penalty + 1e-8)

                    move1 = move_distance * weight2 / (weight1 + weight2 + 1e-8)
                    move2 = move_distance * weight1 / (weight1 + weight2 + 1e-8)

                    repaired[i, 0] -= dx_norm * move1 * 0.5
                    repaired[i, 1] -= dy_norm * move1 * 0.5
                    repaired[j, 0] += dx_norm * move2 * 0.5
                    repaired[j, 1] += dy_norm * move2 * 0.5

                    # Keep within bounds
                    repaired[i, 0] = np.clip(repaired[i, 0], r1, rect_width - r1)
                    repaired[i, 1] = np.clip(repaired[i, 1], r1, rect_height - r1)
                    repaired[j, 0] = np.clip(repaired[j, 0], r2, rect_width - r2)
                    repaired[j, 1] = np.clip(repaired[j, 1], r2, rect_height - r2)

        return repaired

class CirclePacker:
    """Main orchestrator for the circle packing evolutionary algorithm."""

    def __init__(self):
        self.population_size = 30
        self.generations = 150
        self.elite_size = 6
        self.tournament_size = 5
        self.max_stagnation = 25

    def optimize_rectangle_dimensions(self, circles: np.ndarray) -> Tuple[float, float]:
        """Heuristic to determine optimal rectangle dimensions."""
        # Estimate minimum width and height needed based on circle radii
        total_area = np.sum(circles[:, 2]**2) * np.pi
        # Assume 60% packing efficiency for circles
        estimated_width = np.sqrt(total_area / 0.6)
        estimated_height = estimated_width

        # Use a reasonable range around the estimate
        if estimated_width + estimated_height > 2.0:
            # Normalize to perimeter 4
            scale = 2.0 / (estimated_width + estimated_height)
            estimated_width *= scale
            estimated_height *= scale

        # Prefer slightly wider rectangle (more common in practice)
        optimized_width = min(1.8, max(0.2, estimated_width))
        optimized_height = 2.0 - optimized_width

        return optimized_width, optimized_height

    def run_evolution(self) -> np.ndarray:
        """Run the evolutionary algorithm to find optimal circle packing."""
        # Rectangle dimensions (perimeter = 4, so width + height = 2)
        rect_width = 1.0
        rect_height = 1.0

        # Track best fitness for convergence detection
        previous_best = -np.inf
        stagnation_count = 0

        # Initialize population with diverse strategies
        population = []

        # Mix of initialization strategies
        for _ in range(self.population_size // 3):
            solution = InitializerFactory.create_hexagonal_solution(rect_width, rect_height)
            population.append(solution)

        for _ in range(self.population_size // 3):
            solution = InitializerFactory.create_voronoi_solution(rect_width, rect_height)
            population.append(solution)

        for _ in range(self.population_size // 3):
            solution = InitializerFactory.create_strategic_solution(rect_width, rect_height)
            population.append(solution)

        # Track best overall solution
        best_overall_fitness = -np.inf
        best_overall_solution = None

        # Evolutionary algorithm
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [FitnessEvaluator.evaluate(individual) for individual in population]

            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]

            # Update best overall solution
            current_best_fitness = max(fitness_scores)
            if current_best_fitness > best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_overall_solution = population[0].copy()

            # Keep elite
            elite = population[:self.elite_size]

            # Generate new population
            new_population = elite[:]

            # Create offspring using tournament selection and crossover
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1_idx = sorted_indices[np.random.choice(min(self.tournament_size, len(sorted_indices)))]
                parent2_idx = sorted_indices[np.random.choice(min(self.tournament_size, len(sorted_indices)))]

                parent1 = population[parent1_idx].copy()
                parent2 = population[parent2_idx].copy()

                # Crossover
                child1, child2 = CrossoverOperator.crossover(parent1, parent2)

                # Mutate
                child1 = MutationOperator.mutate(child1, rect_width, rect_height)
                child2 = MutationOperator.mutate(child2, rect_width, rect_height)

                # Repair
                child1 = RepairMechanism.repair(child1, rect_width, rect_height)
                child2 = RepairMechanism.repair(child2, rect_width, rect_height)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

            # Convergence detection with gradient monitoring
            current_best = max(fitness_scores)
            if abs(current_best - previous_best) < 1e-6:
                stagnation_count += 1
            else:
                stagnation_count = 0
            previous_best = current_best

            # Early stopping if stagnated too long
            if stagnation_count > self.max_stagnation:
                print(f"Early stopping at generation {generation} due to convergence")
                break

            # Periodically optimize rectangle dimensions
            if generation % 20 == 0 and generation > 0:
                # Use the best solution so far to get rough estimate
                if best_overall_solution is not None:
                    optimized_width, optimized_height = self.optimize_rectangle_dimensions(best_overall_solution)
                    rect_width = optimized_width
                    rect_height = optimized_height
                    print(f"Optimized rectangle to width={rect_width:.3f}, height={rect_height:.3f}")

            # Print progress
            if generation % 30 == 0:
                print(f"Generation {generation}: Best fitness = {current_best:.6f}")

        # Return the best solution found
        fitness_scores = [FitnessEvaluator.evaluate(individual) for individual in population]
        best_idx = np.argmax(fitness_scores)
        best_solution = population[best_idx]

        # Final validation
        final_fitness = FitnessEvaluator.evaluate(best_solution)
        if final_fitness == -np.inf:
            print("Warning: Final solution violated constraints. Returning best found.")
            return best_overall_solution if best_overall_solution is not None else InitializerFactory.create_best_initial_solution(rect_width, rect_height)

        return best_solution

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    packer = CirclePacker()
    return packer.run_evolution()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")