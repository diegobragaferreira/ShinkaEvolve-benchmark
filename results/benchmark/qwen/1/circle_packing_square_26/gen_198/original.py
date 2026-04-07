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
        self.initial_pop_size = 20
        self.generations = 50
        self.crossover_prob = 0.7
        self.mutation_prob = 0.3
        self.tournament_size = 3

    def _calculate_polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate area of polygon using shoelace formula"""
        if len(vertices) < 3:
            return 0.0
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(len(x)-1)) +
                         x[-1] * y[0] - x[0] * y[-1])

    def _create_voronoi_initialization(self, n_candidates: int = 1000) -> List[Tuple[float, float, float]]:
        """Create initial configuration using enhanced Voronoi-based approach"""
        # Generate random candidate points with better distribution
        candidates = np.random.rand(n_candidates, 2)

        # Add strategic boundary points for better coverage
        boundary_points = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],  # Corners
            [0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5],  # Midpoints
            [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]  # Diagonals
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

        # Calculate Voronoi cell properties for selection
        cell_properties = []
        for i, (point, region) in enumerate(zip(vor.points, regions)):
            if len(region) > 0:
                # Calculate area of Voronoi cell
                area = self._calculate_polygon_area(np.array(region))

                # Calculate centroid of the cell
                centroid = np.mean(np.array(region), axis=0)

                # Calculate distance from centroid to center (0.5, 0.5)
                distance_to_center = np.sqrt((centroid[0] - 0.5)**2 + (centroid[1] - 0.5)**2)

                # Store properties: (area, distance_from_center, point)
                cell_properties.append((area, distance_to_center, point))

        # Sort by area (descending) and select top candidates
        cell_properties.sort(key=lambda x: x[0], reverse=True)
        selected_points = [prop[2] for prop in cell_properties[:self.n_circles]]

        # Add more points if needed
        if len(selected_points) < self.n_circles:
            # Fill with random points
            additional_points = np.random.rand(self.n_circles - len(selected_points), 2)
            selected_points.extend(additional_points)

        selected_points = selected_points[:self.n_circles]

        # Enhanced placement with better radius calculation
        circles = []
        for i, (x, y) in enumerate(selected_points):
            # Calculate maximum possible radius that accounts for geometry
            max_radius = min(x, 1-x, y, 1-y)

            # If in the center region, allow larger radii
            center_distance = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            if center_distance < 0.3:
                # Allow larger radius in center
                max_radius = min(max_radius, 0.25)
            else:
                # Reduce radius for edges to allow better packing
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
        """Custom mutation operator that respects constraints"""
        # Pick a random circle to modify
        idx = random.randint(0, len(individual) - 1)
        x, y, r = individual[idx]

        # Create a new, slightly modified circle
        new_x = max(0.01, min(0.99, x + random.gauss(0, 0.02)))
        new_y = max(0.01, min(0.99, y + random.gauss(0, 0.02)))

        # Recalculate max possible radius at new location
        new_r = min(new_x, 1-new_x, new_y, 1-new_y)

        # Adjust radius to ensure no overlaps
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
        """Custom crossover that maintains validity"""
        # Simple uniform crossover of positions and radii
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]

        # Repair individuals to maintain constraints
        for ind in [ind1, ind2]:
            # Repeatedly repair until no more repairs needed
            changed = True
            while changed:
                changed = False
                for i in range(len(ind)):
                    x, y, r = ind[i]
                    # Ensure boundary constraints
                    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                        # Try to shrink radius and reposition
                        r = min(x, 1-x, y, 1-y)
                        r = max(0.001, r)
                        changed = True

                    # Check for overlaps
                    for j in range(len(ind)):
                        if i != j:
                            cx, cy, cr = ind[j]
                            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                            if distance < r + cr + 0.001:
                                # Reduce radius to prevent overlap
                                r = max(0.001, distance - cr - 0.001)
                                changed = True

                    ind[i] = [x, y, r]

        return ind1, ind2

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

        # Run evolution
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run evolution with timeout
        start_time = time.time()
        try:
            pop, log = algorithms.eaSimple(pop, toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob,
                                          ngen=self.generations, stats=stats, halloffame=hof,
                                          verbose=False)
        except Exception as e:
            warnings.warn(f"Evolution failed: {e}")

        # Return the best individual found
        best_individual = hof[0] if len(hof) > 0 else pop[0]

        # Convert to final result format (ensure all circles are valid)
        final_circles = []
        for x, y, r in best_individual:
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