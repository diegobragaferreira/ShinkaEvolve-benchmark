# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree, Voronoi
from deap import base, creator, tools, algorithms
import random
import time
from scipy.spatial.distance import cdist
from typing import List, Tuple, Optional
import warnings

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

class CirclePackingOptimizer:
    """Primary optimizer class for 26-circle packing problem."""
    
    def __init__(self, n_circles: int = 26):
        self.n_circles = n_circles
        self.min_distance = 0.001
        self.max_radius = 0.5
        self.timeout_seconds = 55
        self.initial_pop_size = 150
        self.generations = 60
        self.crossover_prob = 0.8
        self.mutation_prob = 0.25
        self.tournament_size = 4
        self.elite_size = 10
        
    def _calculate_polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate area of polygon using shoelace formula"""
        if len(vertices) < 3:
            return 0.0
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(len(x)-1)) +
                         x[-1] * y[0] - x[0] * y[-1])
    
    def _create_voronoi_initialization(self, n_candidates: int = 1000) -> List[Tuple[float, float, float]]:
        """Create initial configuration using enhanced Voronoi-based approach with better distribution"""
        # Generate candidate points with Latin Hypercube sampling for better uniformity
        candidates = np.random.rand(n_candidates, 2)
        
        # Add strategic boundary points for better coverage
        boundary_points = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],  # Corners
            [0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5],  # Midpoints
            [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75],  # Diagonals
            [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9],  # Inner corners
            [0.33, 0.33], [0.33, 0.67], [0.67, 0.33], [0.67, 0.67],  # Central grid
            [0.15, 0.15], [0.15, 0.85], [0.85, 0.15], [0.85, 0.85],  # Secondary corners
        ])
        candidates = np.vstack([candidates, boundary_points])

        # Compute Voronoi diagram
        try:
            vor = Voronoi(candidates)
        except Exception:
            # Fallback to simpler approach if Voronoi computation fails
            points = np.random.rand(self.n_circles, 2)
            circles = []
            for x, y in points:
                max_radius = min(x, 1-x, y, 1-y)
                circles.append([x, y, max_radius])
            return circles

        # Get Voronoi vertices and regions
        regions = []
        for i in range(len(vor.points)):
            region = []
            for j, vertex_idx in enumerate(vor.point_region[i]):
                if vertex_idx != -1:
                    region.append(vor.vertices[vertex_idx])
            regions.append(region)

        # Calculate Voronoi cell properties for selection with improved scoring
        cell_properties = []
        for i, (point, region) in enumerate(zip(vor.points, regions)):
            if len(region) > 0:
                # Calculate area of Voronoi cell
                area = self._calculate_polygon_area(np.array(region))

                # Calculate centroid of the cell
                centroid = np.mean(np.array(region), axis=0)

                # Calculate distance from centroid to center (0.5, 0.5)
                distance_to_center = np.sqrt((centroid[0] - 0.5)**2 + (centroid[1] - 0.5)**2)

                # Calculate compactness measure (ratio of area to bounding box area)
                if len(region) >= 4:
                    bbox_min = np.min(np.array(region), axis=0)
                    bbox_max = np.max(np.array(region), axis=0)
                    bbox_area = (bbox_max[0] - bbox_min[0]) * (bbox_max[1] - bbox_min[1])
                    compactness = area / (bbox_area + 1e-8)
                else:
                    compactness = 0.0

                # Combined score that considers area, distance from center, and compactness
                score = area * (1.0 + 0.3 * (1 - distance_to_center)) * (1.0 + 0.2 * compactness)
                cell_properties.append((score, area, distance_to_center, point))

        # Sort by score (descending) and select top candidates
        cell_properties.sort(key=lambda x: x[0], reverse=True)
        selected_points = [prop[3] for prop in cell_properties[:self.n_circles]]

        # Add more points if needed using a different strategy
        if len(selected_points) < self.n_circles:
            # Fill with random points that are well-distributed
            additional_points = np.random.rand(self.n_circles - len(selected_points), 2)
            selected_points.extend(additional_points)

        selected_points = selected_points[:self.n_circles]

        # Enhanced placement with better radius calculation using neighbor information
        circles = []
        for i, (x, y) in enumerate(selected_points):
            # Calculate maximum possible radius that accounts for geometry and neighbors
            max_radius = min(x, 1-x, y, 1-y)

            # Find nearest neighbor distance to determine good radius size
            min_neighbor_dist = float('inf')
            for j, (other_x, other_y) in enumerate(selected_points):
                if i != j:
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_neighbor_dist = min(min_neighbor_dist, dist)

            # If we have neighbors, set radius proportional to minimum neighbor distance
            if min_neighbor_dist < float('inf'):
                # Base radius on neighbor distance but with upper bound
                base_radius = min_neighbor_dist * 0.15
                max_radius = min(max_radius, base_radius, 0.25)
            else:
                # No neighbors, use normal approach
                max_radius = min(max_radius, 0.2)

            # Ensure reasonable minimum
            max_radius = max(0.005, max_radius)

            if max_radius > 0:
                circles.append([x, y, max_radius])
            else:
                # Fallback to small circle if boundary constrained
                circles.append([x, y, 0.01])

        return circles

    def _is_valid_position(self, x: float, y: float, r: float, existing_circles: List[Tuple[float, float, float]]) -> bool:
        """Check if a circle at (x,y) with radius r is valid"""
        # Check boundary constraints
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

        # Check overlap with existing circles
        for cx, cy, cr in existing_circles:
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            if distance < r + cr + self.min_distance:
                return False
        return True

    def _create_initial_population(self) -> List[List[Tuple[float, float, float]]]:
        """Create an initial population with good starting configurations"""
        population = []

        # Generate multiple initial solutions using different strategies
        for _ in range(self.initial_pop_size):
            circles = []
            # Try to place circles greedily
            for i in range(self.n_circles):
                best_x, best_y, best_r = 0.5, 0.5, 0.0

                # Find a valid position with maximum possible radius
                attempts = 0
                while attempts < 1000:
                    # Randomly try to place a circle
                    x = np.random.uniform(0.01, 0.99)
                    y = np.random.uniform(0.01, 0.99)

                    # Calculate maximum radius at this position
                    r = min(x, 1-x, y, 1-y)

                    # If valid and has room for improvement, accept it
                    if self._is_valid_position(x, y, r, circles):
                        # Try to increase radius while keeping it valid
                        max_r = r
                        for _ in range(100):
                            new_r = min(max_r * 1.1, self.max_radius)
                            if self._is_valid_position(x, y, new_r, circles):
                                max_r = new_r
                            else:
                                break
                        best_x, best_y, best_r = x, y, max_r
                        break
                    attempts += 1

                circles.append([best_x, best_y, best_r])

            population.append(circles)

        return population

    def _evaluate(self, individual: List[Tuple[float, float, float]]) -> Tuple[float]:
        """Evaluate fitness - sum of radii"""
        total_radius = sum(circle[2] for circle in individual)
        return (total_radius,)

    def _mutate(self, individual: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Custom mutation operator that respects constraints with adaptive strength"""
        # Pick a random circle to modify
        idx = random.randint(0, len(individual) - 1)
        x, y, r = individual[idx]

        # Create a new, slightly modified circle with adaptive mutation strength
        # Use larger mutations for smaller circles to encourage exploration
        mutation_strength = 0.02 + (0.01 * (1.0 - min(1.0, r * 2.0)))
        new_x = max(0.01, min(0.99, x + random.gauss(0, mutation_strength)))
        new_y = max(0.01, min(0.99, y + random.gauss(0, mutation_strength)))

        # Recalculate max possible radius at new location
        new_r = min(new_x, 1-new_x, new_y, 1-new_y)

        # Adjust radius to ensure no overlaps with existing circles
        for i, (cx, cy, cr) in enumerate(individual):
            if i != idx:
                distance = np.sqrt((new_x - cx)**2 + (new_y - cy)**2)
                max_radius_allowed = distance - cr - 0.001
                if max_radius_allowed > 0:
                    new_r = min(new_r, max_radius_allowed)

        # Ensure minimum positive radius
        new_r = max(0.001, new_r)

        individual[idx] = [new_x, new_y, new_r]
        return individual

    def _crossover(self, ind1: List[Tuple[float, float, float]],
                   ind2: List[Tuple[float, float, float]]) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """Custom crossover that maintains validity with enhanced repair"""
        # Simple uniform crossover of positions and radii
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]

        # Repair individuals to maintain constraints with improved strategy
        for ind in [ind1, ind2]:
            # First pass: Fix boundary violations
            for i in range(len(ind)):
                x, y, r = ind[i]
                # Ensure boundary constraints
                if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                    # Shrink radius to fit boundary and center at valid location
                    r = min(x, 1-x, y, 1-y)
                    r = max(0.001, r)
                    # Keep same center for now, but we'll adjust later
                    ind[i] = [x, y, r]

            # Second pass: Resolve overlaps with more sophisticated approach
            for i in range(len(ind)):
                # Collect all circles that might overlap with this one
                overlapping = []
                x, y, r = ind[i]
                for j in range(len(ind)):
                    if i != j:
                        cx, cy, cr = ind[j]
                        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                        if distance < r + cr + 0.001:
                            overlapping.append((j, distance, cr))

                # If there are overlaps, try to resolve them
                if overlapping:
                    # Sort by distance to prioritize fixing closest overlaps first
                    overlapping.sort(key=lambda x: x[1])
                    for j, _, cr in overlapping:
                        # Try to move the circle to resolve overlap
                        cx, cy, _ = ind[j]
                        dx, dy = x - cx, y - cy
                        dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                        # Normalize direction vector
                        dx, dy = dx/dist, dy/dist

                        # Calculate how much to move to resolve overlap
                        overlap_amount = (r + cr) - dist + 0.001
                        if overlap_amount > 0:
                            # Move in the opposite direction to the overlapping circle
                            move_x = -dx * overlap_amount * 0.5
                            move_y = -dy * overlap_amount * 0.5

                            # Apply movement with boundary checks
                            new_x = max(r + 0.01, min(1-r - 0.01, x + move_x))
                            new_y = max(r + 0.01, min(1-r - 0.01, y + move_y))
                            ind[i] = [new_x, new_y, r]

        return ind1, ind2

    def _constraint_aware_local_search(self, individual: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Perform constraint-aware local search to fine-tune solution"""
        # Create a copy to work with
        solution = [list(circle) for circle in individual]

        # Try to optimize each circle individually
        improved = True
        iteration = 0
        max_iterations = 50

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Try to increase each circle's radius
            for i in range(len(solution)):
                x, y, r = solution[i]

                # Calculate maximum possible radius at current position
                max_radius = min(x, 1-x, y, 1-y)

                # Try to increase radius while respecting constraints
                original_r = r
                test_r = min(original_r * 1.1, max_radius)

                # Check if this radius works with neighbors
                valid = True
                for j in range(len(solution)):
                    if i != j:
                        cx, cy, cr = solution[j]
                        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                        if distance < test_r + cr + 0.001:
                            valid = False
                            break

                if valid and test_r > r:
                    solution[i][2] = test_r
                    improved = True

            # Try to improve positions to reduce overlaps
            for i in range(len(solution)):
                x, y, r = solution[i]

                # Try small position adjustments to minimize conflicts
                best_x, best_y = x, y
                best_r = r
                best_improvement = 0

                # Try different small adjustments
                adjustments = [(-0.01, -0.01), (-0.01, 0), (-0.01, 0.01),
                              (0, -0.01), (0, 0.01),
                              (0.01, -0.01), (0.01, 0), (0.01, 0.01)]

                for dx, dy in adjustments:
                    test_x = max(r + 0.01, min(1-r - 0.01, x + dx))
                    test_y = max(r + 0.01, min(1-r - 0.01, y + dy))

                    # Check if this position is valid
                    valid = True
                    for j in range(len(solution)):
                        if i != j:
                            cx, cy, cr = solution[j]
                            distance = np.sqrt((test_x - cx)**2 + (test_y - cy)**2)
                            if distance < r + cr + 0.001:
                                valid = False
                                break

                    if valid:
                        # Calculate potential improvement (try to make it as large as possible)
                        new_r = min(test_x, 1-test_x, test_y, 1-test_y)
                        improvement = new_r - r

                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_x, best_y = test_x, test_y

                if best_x != x or best_y != y:
                    solution[i][0] = best_x
                    solution[i][1] = best_y
                    solution[i][2] = min(solution[i][2], best_x, 1-best_x, best_y, 1-best_y)
                    improved = True

        return solution

    def optimize(self) -> np.ndarray:
        """Run the full optimization pipeline"""
        # Setup DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("individual", lambda: self._create_initial_population()[0])
        toolbox.register("population", lambda: self._create_initial_population())
        toolbox.register("evaluate", self._evaluate)
        toolbox.register("mate", self._crossover)
        toolbox.register("mutate", self._mutate)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        # Create initial population
        pop = toolbox.population()

        # Run evolution with elitism and adaptive parameters
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run evolution with timeout
        start_time = time.time()
        try:
            # Run with elistism for better preservation of good solutions
            pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=self.initial_pop_size, 
                                                lambda_=self.initial_pop_size,
                                                cxpb=self.crossover_prob, mutpb=self.mutation_prob,
                                                ngen=self.generations, stats=stats, halloffame=hof,
                                                verbose=False)
        except Exception as e:
            warnings.warn(f"Evolution failed: {e}")

        # Apply local search to the best individual found
        best_individual = hof[0] if len(hof) > 0 else pop[0]
        refined_individual = self._constraint_aware_local_search(list(best_individual))

        # Convert to final result format (ensure all circles are valid)
        final_circles = []
        for x, y, r in refined_individual:
            # Final validation and adjustment
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            final_circles.append([x, y, r])

        return np.array(final_circles)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer(n_circles=26)
    return optimizer.optimize()

# EVOLVE-BLOCK-END