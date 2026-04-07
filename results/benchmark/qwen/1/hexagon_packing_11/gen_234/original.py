# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
from numba import jit
import warnings
import random
from collections import defaultdict
from typing import Tuple, List, Optional, Any
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Calculate vertices of a hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vx = x + side_length * np.cos(theta)
        vy = y + side_length * np.sin(theta)
        vertices.append((vx, vy))
    return np.array(vertices)

class Hexagon:
    """Represents a regular hexagon with position and orientation"""

    def __init__(self, x: float, y: float, angle_deg: float, side_length: float = 1.0):
        self.x = x
        self.y = y
        self.angle_deg = angle_deg
        self.side_length = side_length
        self._polygon = None
        self._vertices = None

    @property
    def vertices(self) -> np.ndarray:
        """Cached vertices calculation"""
        if self._vertices is None:
            self._vertices = hexagon_vertices(self.x, self.y, self.angle_deg, self.side_length)
        return self._vertices

    @property
    def polygon(self) -> Polygon:
        """Cached polygon representation"""
        if self._polygon is None:
            self._polygon = Polygon(self.vertices)
        return self._polygon

    def reset_cache(self):
        """Clear cached values"""
        self._polygon = None
        self._vertices = None

    def update_position(self, x: float, y: float, angle_deg: float):
        """Update hexagon position and orientation"""
        self.x = x
        self.y = y
        self.angle_deg = angle_deg
        self.reset_cache()

class HexagonPack:
    """Manages collection of hexagons and spatial queries"""

    def __init__(self, hexagons: List[Hexagon]):
        self.hexagons = hexagons
        self._bounding_box = None
        self._spatial_grid = None
        self._grid_size = 5.0

    def get_all_vertices(self) -> List[Tuple[float, float]]:
        """Get all vertices from all hexagons"""
        vertices = []
        for hexagon in self.hexagons:
            vertices.extend(hexagon.vertices)
        return vertices

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get bounding box of all hexagons"""
        if self._bounding_box is None:
            if not self.hexagons:
                return (0, 0, 0, 0)

            all_vertices = self.get_all_vertices()
            if not all_vertices:
                return (0, 0, 0, 0)

            min_x = min(v[0] for v in all_vertices)
            min_y = min(v[1] for v in all_vertices)
            max_x = max(v[0] for v in all_vertices)
            max_y = max(v[1] for v in all_vertices)
            self._bounding_box = (min_x, min_y, max_x, max_y)
        return self._bounding_box

    def build_spatial_grid(self):
        """Build spatial grid for fast collision detection"""
        self._spatial_grid = defaultdict(list)
        bbox = self.get_bounding_box()
        if not bbox or bbox == (0, 0, 0, 0):
            return

        min_x, min_y, max_x, max_y = bbox
        for i, hexagon in enumerate(self.hexagons):
            bbox = hexagon.polygon.bounds
            min_x_h, min_y_h, max_x_h, max_y_h = bbox
            for x in range(int(min_x_h/self._grid_size), int(max_x_h/self._grid_size)+1):
                for y in range(int(min_y_h/self._grid_size), int(max_y_h/self._grid_size)+1):
                    self._spatial_grid[(x,y)].append(i)

    def get_collision_candidates(self, hex_index: int) -> List[int]:
        """Get potential collision candidates for a hexagon"""
        if self._spatial_grid is None:
            self.build_spatial_grid()

        candidates = []
        hexagon = self.hexagons[hex_index]
        bbox = hexagon.polygon.bounds
        min_x, min_y, max_x, max_y = bbox

        for x in range(int(min_x/self._grid_size), int(max_x/self._grid_size)+1):
            for y in range(int(min_y/self._grid_size), int(max_y/self._grid_size)+1):
                candidates.extend(self._spatial_grid.get((x,y), []))
        return [i for i in candidates if i != hex_index]

    def contains_all(self, outer_hexagon: Polygon) -> bool:
        """Check if all hexagons are contained within outer hexagon"""
        for hexagon in self.hexagons:
            if not outer_hexagon.contains(hexagon.polygon):
                if not (outer_hexagon.intersects(hexagon.polygon) and
                       outer_hexagon.intersection(hexagon.polygon).area == hexagon.polygon.area):
                    return False
        return True

    def has_collisions(self) -> bool:
        """Check if any hexagons collide with each other"""
        if not self.hexagons:
            return False

        # Use spatial indexing for faster collision detection
        self.build_spatial_grid()
        n = len(self.hexagons)

        for i in range(n):
            candidates = self.get_collision_candidates(i)
            for j in candidates:
                if self.hexagons[i].polygon.intersects(self.hexagons[j].polygon):
                    return True
        return False

class OptimizationEngine:
    """Handles the optimization process with configurable parameters"""

    def __init__(self, num_hexagons: int = 11, side_length: float = 1.0):
        self.num_hexagons = num_hexagons
        self.side_length = side_length
        self.max_iterations = 100
        self.population_size = 15
        self.seed_base = 42

    def calculate_outer_radius(self, positions: np.ndarray, angles: np.ndarray) -> float:
        """Calculate minimum radius needed to contain all inner hexagons"""
        max_dist = 0
        outer_center = (0, 0)

        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(len(positions)):
            x, y = positions[i]
            angle = angles[i]
            hex_vertices = hexagon_vertices(x, y, angle, self.side_length)
            all_vertices.extend(hex_vertices)

        # Find maximum distance from center
        for vertex in all_vertices:
            dist = np.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)

        # Add buffer for safety and account for hexagon shape
        return max_dist * 1.1  # Safety factor

    def create_outer_hexagon(self, positions: np.ndarray, angles: np.ndarray) -> Polygon:
        """Create outer hexagon polygon"""
        radius = self.calculate_outer_radius(positions, angles)
        vertices = hexagon_vertices(0, 0, 0, radius)
        return Polygon(vertices)

    def evaluate_solution(self, solution: np.ndarray, use_spatial_index: bool = True) -> float:
        """Evaluate a solution and return negative of objective (since we minimize)"""
        try:
            # Reshape solution into positions and angles
            positions = solution[:22].reshape(-1, 2)  # 11 hexagons * 2 coordinates each
            angles = solution[22:]  # 11 angles

            # Create hexagon objects
            hexagons = [Hexagon(x, y, angle) for x, y, angle in zip(positions[:, 0], positions[:, 1], angles)]
            hex_pack = HexagonPack(hexagons)

            # Check containment
            outer_hexagon = self.create_outer_hexagon(positions, angles)

            # Check containment for all inner hexagons
            if not hex_pack.contains_all(outer_hexagon):
                return 1e10  # Penalty for non-containment

            # Check for overlaps
            if hex_pack.has_collisions():
                return 1e10  # Penalty for overlap

            # Return negative of 1/outer_radius (we want to maximize 1/outer_radius)
            outer_radius = self.calculate_outer_radius(positions, angles)
            return -1.0 / outer_radius

        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return 1e10

    def generate_initial_population(self, num_starts: int = 5) -> List[np.ndarray]:
        """Generate multiple initial configurations"""
        initial_populations = []

        # Base pattern: hexagonal arrangement
        base_positions = []
        base_angles = []

        # Center hexagon
        base_positions.append([0.0, 0.0])
        base_angles.append(0.0)

        # Surrounding hexagons in ring
        for i in range(6):
            angle = i * 60
            radius = 2.0
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            base_positions.append([x, y])
            base_angles.append(0.0)

        # Additional positions for remaining hexagons
        # Use a more strategic layout based on hexagonal packing principles
        additional_positions = [
            (-3.0, 1.0), (3.0, 1.0),
            (-3.0, -1.0), (3.0, -1.0),
            (0.0, 3.0), (0.0, -3.0),
            (1.5, 2.6), (-1.5, -2.6),
            (-1.5, 2.6), (1.5, -2.6)
        ]

        for pos in additional_positions:
            if len(base_positions) < self.num_hexagons:
                base_positions.append(list(pos))
                base_angles.append(0.0)

        # Ensure we have exactly the right number of positions
        while len(base_positions) < self.num_hexagons:
            base_positions.append([0.0, 0.0])
            base_angles.append(0.0)

        # Generate different variations with better initial diversity
        for start in range(num_starts):
            # Create a slightly different initial configuration for each start
            initial_positions = [pos[:] for pos in base_positions]  # Copy
            initial_angles = [ang for ang in base_angles]  # Copy

            # Add more substantial random perturbations to encourage exploration
            for i in range(len(initial_positions)):
                if i > 0:  # Don't perturb center hexagon significantly
                    initial_positions[i][0] += random.uniform(-0.5, 0.5)
                    initial_positions[i][1] += random.uniform(-0.5, 0.5)
                    initial_angles[i] += random.uniform(-10, 10)

            # Flatten initial solution
            initial_solution = []
            for pos in initial_positions[:self.num_hexagons]:
                initial_solution.extend(pos)
            initial_solution.extend(initial_angles[:self.num_hexagons])
            initial_solution = np.array(initial_solution)

            initial_populations.append(initial_solution)

        return initial_populations

    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Main optimization function with multi-start approach"""
        # Generate multiple initial populations
        initial_populations = self.generate_initial_population(5)

        best_result = None
        best_score = float('inf')

        # Run optimization from multiple starting points
        for i, initial_solution in enumerate(initial_populations):
            try:
                logger.info(f"Starting optimization run {i+1}/{len(initial_populations)}")

                # Set bounds for optimization
                bounds = []
                # Position bounds
                for _ in range(22):
                    bounds.append((-15.0, 15.0))  # Extended bounds for better exploration
                # Angle bounds
                for _ in range(self.num_hexagons):
                    bounds.append((0.0, 360.0))   # Rotation angles

                # Custom adaptive differential evolution with improved adaptive mechanisms
                def adaptive_differential_evolution(func, bounds, maxiter, popsize, seed=None):
                    """Custom differential evolution with advanced adaptive mutation rate scheduling"""
                    # Initialize random number generator
                    rng = np.random.default_rng(seed)

                    # Generate initial population with better diversity
                    population = []
                    for _ in range(popsize):
                        individual = []
                        for (min_val, max_val) in bounds:
                            individual.append(rng.uniform(min_val, max_val))
                        population.append(individual)

                    # Track best solution
                    best_individual = None
                    best_fitness = float('inf')
                    previous_best = float('inf')
                    stagnation_count = 0
                    max_stagnation = 20
                    convergence_threshold = 1e-8

                    for generation in range(maxiter):
                        # Advanced adaptive mutation rate with dynamic adjustment
                        if generation == 0:
                            mutation_rate = 0.8
                        else:
                            # More sophisticated adaptation based on convergence progress
                            improvement = abs(previous_best - best_fitness)
                            
                            # If we're making significant progress, decrease mutation for exploitation
                            if improvement > convergence_threshold:
                                mutation_rate = max(0.1, 0.8 - (generation / maxiter) * 0.7)
                                stagnation_count = 0
                            else:
                                # If little progress, increase mutation to escape local optima
                                mutation_rate = min(0.9, 0.5 + (stagnation_count / max_stagnation) * 0.4)
                                stagnation_count += 1

                        # Evaluate current population
                        fitnesses = []
                        for individual in population:
                            fitness = func(np.array(individual))
                            fitnesses.append(fitness)
                            if fitness < best_fitness:
                                best_fitness = fitness
                                best_individual = individual[:]

                        # Track previous best for convergence monitoring
                        previous_best = best_fitness

                        # Create new population
                        new_population = []
                        for j in range(popsize):
                            # Select three different individuals
                            candidates = list(range(popsize))
                            candidates.remove(j)
                            selected = rng.choice(candidates, 3, replace=False)

                            # Differential evolution mutation with adaptive rate
                            mutant = []
                            for k in range(len(bounds)):
                                # Use adaptive mutation strategy
                                if rng.random() < mutation_rate or generation == 0:
                                    # Differential evolution mutation
                                    mutant.append(population[selected[0]][k] +
                                                0.8 * (population[selected[1]][k] - population[selected[2]][k]))
                                else:
                                    mutant.append(population[j][k])

                                # Ensure bounds are respected
                                min_val, max_val = bounds[k]
                                mutant[k] = np.clip(mutant[k], min_val, max_val)

                            # Adaptive crossover probability
                            # Higher probability early, lower later for focused exploitation
                            crossover_prob = 0.9 if generation < maxiter//3 else 0.7 if generation < 2*maxiter//3 else 0.5
                            trial = []
                            for k in range(len(bounds)):
                                if rng.random() < crossover_prob or k == 0:  # Keep some randomness
                                    trial.append(mutant[k])
                                else:
                                    trial.append(population[j][k])

                            new_population.append(trial)

                        population = new_population

                    return type('Result', (), {
                        'x': np.array(best_individual),
                        'fun': best_fitness
                    })()

                # Run adaptive differential evolution
                result = adaptive_differential_evolution(
                    lambda sol: self.evaluate_solution(sol, use_spatial_index=(i < 3)),  # Use spatial index for first 3 starts
                    bounds,
                    maxiter=self.max_iterations,
                    popsize=self.population_size,
                    seed=self.seed_base + i  # Different seed for each start
                )

                # Evaluate final result
                final_score = self.evaluate_solution(result.x, use_spatial_index=False)

                if final_score < best_score:
                    best_score = final_score
                    best_result = result

                logger.info(f"Run {i+1} completed with score: {final_score}")

            except Exception as e:
                logger.error(f"Start {i} failed: {e}")
                continue

        if best_result is None:
            # Fallback to simple solution
            raise RuntimeError("All optimization attempts failed")

        # Extract final solution
        final_positions = best_result.x[:22].reshape(-1, 2)
        final_angles = best_result.x[22:]

        # Refine the solution with enhanced local optimization
        refined_positions, refined_angles = self.local_refinement(final_positions, final_angles)

        return refined_positions, refined_angles

    def local_refinement(self, positions: np.ndarray, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply enhanced local refinement to improve the solution"""
        # Enhanced local refinement with multiple strategies
        best_positions = positions.copy()
        best_angles = angles.copy()
        best_score = self.evaluate_solution(np.concatenate([best_positions.flatten(), best_angles]))

        step_size = 0.05  # Increased initial step size
        max_iterations = 150  # More iterations for thorough refinement
        patience = 15  # More patience before stopping
        patience_counter = 0
        improvement_count = 0
        
        # Track improvement history for adaptive strategy
        recent_improvements = []

        for iteration in range(max_iterations):
            improved = False
            current_improvements = 0

            # Try multiple perturbation strategies
            for i in range(len(positions)):
                # Strategy 1: Try multiple step sizes for positions
                for dim in range(2):
                    old_val = best_positions[i][dim]
                    # Try several step sizes to find best direction
                    step_sizes = [step_size, step_size * 0.5, step_size * 2.0]
                    for delta_mult in [-1, 1]:  # Both directions
                        for s in step_sizes:
                            best_positions[i][dim] = old_val + delta_mult * s
                            new_score = self.evaluate_solution(np.concatenate([best_positions.flatten(), best_angles]))
                            if new_score < best_score:
                                best_score = new_score
                                improved = True
                                current_improvements += 1
                            else:
                                best_positions[i][dim] = old_val

                # Strategy 2: Try multiple angle perturbations
                old_angle = best_angles[i]
                angle_steps = [-10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0]
                for delta in angle_steps:
                    best_angles[i] = (old_angle + delta) % 360
                    new_score = self.evaluate_solution(np.concatenate([best_positions.flatten(), best_angles]))
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                        current_improvements += 1
                    else:
                        best_angles[i] = old_angle

            # Adaptive step size adjustment based on recent performance
            recent_improvements.append(current_improvements)
            if len(recent_improvements) > 5:
                recent_improvements.pop(0)
                
            # If we're consistently not improving, reduce step size
            if sum(recent_improvements[-5:]) == 0 and step_size > 0.001:
                step_size *= 0.9
            elif current_improvements > 2 and step_size < 0.2:
                step_size = min(0.2, step_size * 1.1)  # Increase if making good progress

            # Check for improvement
            if not improved:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            else:
                patience_counter = 0
                improvement_count += 1

        return best_positions, best_angles

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    try:
        # Initialize optimization engine
        optimizer = OptimizationEngine(num_hexagons=11, side_length=1.0)

        # Run optimization
        final_positions, final_angles = optimizer.optimize()

        # Create inner hex data
        inner_hex_data = np.column_stack([final_positions, final_angles])

        # Create outer hex data (centered)
        outer_hex_data = np.array([0, 0, 0])

        # Calculate outer hex side length
        # We need to calculate this based on the final solution
        max_dist = 0
        outer_center = (0, 0)

        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(len(final_positions)):
            x, y = final_positions[i]
            angle = final_angles[i]
            hex_vertices = hexagon_vertices(x, y, angle)
            all_vertices.extend(hex_vertices)

        # Find maximum distance from center
        for vertex in all_vertices:
            dist = np.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)

        outer_hex_side_length = max_dist / (np.sqrt(3) / 2) * 1.1  # Adding safety factor

        elapsed_time = time.time() - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds")

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        # Fallback to initial solution
        inner_hex_data = np.array([
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END