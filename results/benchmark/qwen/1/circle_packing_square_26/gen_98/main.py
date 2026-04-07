# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from typing import Tuple, List, Optional, Any
import time

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class OptimizationConfig:
    """Configuration manager for circle packing optimization parameters."""
    def __init__(self, n_circles: int = 26, pop_size: int = 100, 
                 gen_count: int = 80, mutpb: float = 0.3, cxpb: float = 0.5):
        self.N_CIRCLES = n_circles
        self.POP_SIZE = pop_size
        self.GEN_COUNT = gen_count
        self.MUTPB = mutpb
        self.CXPB = cxpb
        self.BOUNDARY_MARGIN = 0.005
        self.PENALTY_FACTOR = 1000
        self.MAX_LOCAL_ITERATIONS = 200

class InitializationEngine:
    """Handles various circle initialization strategies."""
    
    @staticmethod
    def create_hexagonal_grid(n_circles: int) -> List[Tuple[float, float]]:
        """Create hexagonal grid points for better spatial distribution."""
        sqrt_n = int(np.ceil(np.sqrt(n_circles)))
        rows = int(np.ceil(n_circles / sqrt_n))
        cols = int(np.ceil(n_circles / rows))
        
        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)
        
        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_circles:
                    break
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 1) * spacing_x + x_offset + 0.025
                y = (i + 1) * spacing_y + 0.025
                points.append((x, y))
        
        # Ensure we have enough points
        while len(points) < n_circles:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            points.append((x, y))
            
        return points[:n_circles]

    @classmethod
    def generate_voronoi_initial_config(cls, n_circles: int) -> np.ndarray:
        """Generate initial circle configuration using enhanced Voronoi-like distribution."""
        circles = np.zeros((n_circles, 3))
        points = cls.create_hexagonal_grid(n_circles)
        
        # Assign initial radii based on spatial density and boundary constraints
        for i, (x, y) in enumerate(points):
            # Calculate minimum distance to neighbors
            min_dist = float('inf')
            for j, (other_x, other_y) in enumerate(points):
                if i != j:
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_dist = min(min_dist, dist)

            # Determine initial radius
            initial_r = min(0.15, min_dist * 0.15) if min_dist < float('inf') else 0.07
            # Constrain to boundary limits
            boundary_radius = min(x, 1-x, y, 1-y)
            initial_r = min(initial_r, boundary_radius * 0.8)
            initial_r = max(0.005, min(0.15, initial_r))
            
            circles[i] = [x, y, initial_r]
            
        return circles

class ConstraintHandler:
    """Efficiently handles constraint validation and repair."""
    
    @staticmethod
    def is_contained(positions: np.ndarray, radii: np.ndarray, margin: float = 0.005) -> bool:
        """Check if all circles are fully contained in unit square."""
        return not np.any(
            (positions[:, 0] - radii < margin) | 
            (positions[:, 0] + radii > 1 - margin) | 
            (positions[:, 1] - radii < margin) | 
            (positions[:, 1] + radii > 1 - margin)
        )

    @staticmethod
    def check_overlaps_kdtree(positions: np.ndarray, radii: np.ndarray, 
                            penalty_factor: int = 1000) -> float:
        """Check overlaps efficiently using cKDTree."""
        try:
            tree = cKDTree(positions)
            pairs = tree.query_pairs(radii.sum() + 0.001, p=2)
            penalty = 0
            for i, j in pairs:
                r_i = radii[i]
                r_j = radii[j]
                pos_i = positions[i]
                pos_j = positions[j]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    penalty += penalty_factor * (r_i + r_j - dist)
            return penalty
        except Exception:
            # Fallback to brute force
            return ConstraintHandler._brute_force_overlap_check(positions, radii, penalty_factor)

    @staticmethod
    def _brute_force_overlap_check(positions: np.ndarray, radii: np.ndarray, 
                                 penalty_factor: int = 1000) -> float:
        """Brute force overlap checking for edge cases."""
        penalty = 0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                pos_i = positions[i]
                pos_j = positions[j]
                r_i = radii[i]
                r_j = radii[j]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    penalty += penalty_factor * (r_i + r_j - dist)
        return penalty

    @staticmethod
    def repair_containment(circles: np.ndarray, margin: float = 0.005) -> np.ndarray:
        """Repair containment constraint violations."""
        result = circles.copy()
        for i in range(len(result)):
            x, y, r = result[i]
            # Adjust if out of bounds
            if x - r < margin:
                x = r + margin
            elif x + r > 1 - margin:
                x = 1 - r - margin
            if y - r < margin:
                y = r + margin
            elif y + r > 1 - margin:
                y = 1 - r - margin
            result[i] = [x, y, r]
        return result

    @staticmethod
    def repair_overlaps(circles: np.ndarray) -> np.ndarray:
        """Repair overlap constraint violations."""
        result = circles.copy()
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                pos_i = result[i, :2]
                pos_j = result[j, :2]
                r_i = result[i, 2]
                r_j = result[j, 2]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    # Move one circle away from the other
                    dx, dy = pos_i - pos_j
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                    dx /= dist
                    dy /= dist
                    step = (r_i + r_j - dist) * 0.3
                    
                    # Only move if it stays within bounds
                    new_x = max(0.005, min(0.995, pos_i[0] + dx * step))
                    new_y = max(0.005, min(0.995, pos_i[1] + dy * step))
                    result[i] = [new_x, new_y, r_i]
        return result

class EvolutionaryOperators:
    """Modular implementation of evolutionary operators."""
    
    @staticmethod
    def create_individual(config: OptimizationConfig) -> Any:
        """Create a random valid individual."""
        circles = InitializationEngine.generate_voronoi_initial_config(config.N_CIRCLES)
        individual = circles.flatten().tolist()
        
        # Add small random perturbations
        for i in range(len(individual)):
            if i % 3 < 2:  # x or y coordinate
                individual[i] += random.uniform(-0.015, 0.015)
                individual[i] = max(0, min(1, individual[i]))
            else:  # radius
                individual[i] *= random.uniform(0.92, 1.08)
                individual[i] = max(0.001, min(0.5, individual[i]))
        return individual

    @staticmethod
    def mutate_individual(individual: list, config: OptimizationConfig) -> tuple:
        """Mutate individual with adaptive parameters."""
        individual_array = np.array(individual).reshape(-1, 3)
        radii = individual_array[:, 2]
        diversity = np.std(radii) / (np.mean(radii) + 1e-8) if np.mean(radii) > 1e-8 else 0

        # Adaptive mutation rate based on diversity
        adaptive_mutation_rate = config.MUTPB * (1 - min(0.8, diversity))

        for i in range(len(individual)):
            if random.random() < adaptive_mutation_rate:
                idx = i % 3
                if idx == 2:  # radius index
                    old_r = individual[i]
                    # Larger mutations for diverse populations
                    mutation_strength = 0.02 * (1 + diversity)
                    new_r = old_r + random.gauss(0, mutation_strength)
                    individual[i] = max(0.001, min(0.5, new_r))
                else:  # position indices (x, y)
                    old_val = individual[i]
                    # Mutate position with diversity scaling
                    mutation_strength = 0.03 * (1 + diversity)
                    new_val = old_val + random.gauss(0, mutation_strength)
                    individual[i] = max(0, min(1, new_val))
        return individual,

    @staticmethod
    def crossover_constraint_aware(ind1: list, ind2: list, config: OptimizationConfig) -> tuple:
        """Crossover that maintains constraints with repair mechanism."""
        # Perform standard uniform crossover
        tools.cxUniform(ind1, ind2, indpb=0.5)

        # Repair violated constraints in both children
        for ind in [ind1, ind2]:
            temp_ind = np.array(ind).reshape(-1, 3)
            
            # Fix containment issues
            temp_ind = ConstraintHandler.repair_containment(temp_ind)
            
            # Fix overlaps with smarter repair
            temp_ind = ConstraintHandler.repair_overlaps(temp_ind)
            
            ind[:] = temp_ind.flatten()
        return ind1, ind2

class LocalOptimizer:
    """Specialized local optimization module."""
    
    @staticmethod
    def optimize_circles(circles: np.ndarray, max_iterations: int = 200) -> np.ndarray:
        """Apply local optimization to refine solution."""
        result = circles.copy()
        
        # Multiple refinement passes
        for pass_num in range(3):
            improved = False
            
            # Pass 1: Increase radii where possible
            for i in range(len(result)):
                original_r = result[i, 2]
                max_increase = min(
                    result[i, 0], 1 - result[i, 0],
                    result[i, 1], 1 - result[i, 1]
                ) - original_r

                if max_increase > 0.001:
                    # Binary search for maximum safe increase
                    low = 0
                    high = max_increase
                    best_radius = original_r
                    
                    # Binary search iterations
                    for _ in range(8):
                        test_r = (low + high) / 2
                        test_r = min(test_r, max_increase)
                        
                        valid = True
                        test_pos = result[i, :2]
                        test_r_new = original_r + test_r
                        
                        # Check overlap with other circles
                        for j in range(len(result)):
                            if i != j:
                                pos_j = result[j, :2]
                                r_j = result[j, 2]
                                dist = np.sqrt(np.sum((test_pos - pos_j)**2))
                                if dist < (test_r_new + r_j):
                                    valid = False
                                    break
                        
                        if valid:
                            best_radius = original_r + test_r
                            low = test_r
                        else:
                            high = test_r
                    
                    if best_radius > original_r + 0.001:
                        result[i, 2] = best_radius
                        improved = True

            if not improved:
                # Pass 2: Position refinement
                for i in range(len(result)):
                    original_pos = result[i, :2].copy()
                    best_pos = original_pos.copy()
                    best_radius = result[i, 2]
                    best_score = best_radius
                    
                    # Try several positions around current location
                    for dx in [-0.015, -0.01, 0, 0.01, 0.015]:
                        for dy in [-0.015, -0.01, 0, 0.01, 0.015]:
                            test_x = max(0.005, min(0.995, result[i, 0] + dx))
                            test_y = max(0.005, min(0.995, result[i, 1] + dy))
                            
                            valid = True
                            test_r = result[i, 2]
                            
                            # Check overlap with other circles
                            for j in range(len(result)):
                                if i != j:
                                    pos_j = result[j, :2]
                                    r_j = result[j, 2]
                                    dist = np.sqrt((test_x - pos_j[0])**2 + (test_y - pos_j[1])**2)
                                    if dist < (test_r + r_j):
                                        valid = False
                                        break
                            
                            if valid:
                                score = test_r
                                if score > best_score:
                                    best_score = score
                                    best_pos = [test_x, test_y]
                    
                    # Apply best movement if found
                    if best_score > result[i, 2] or not np.array_equal(best_pos, original_pos):
                        result[i, :2] = best_pos
                        improved = True

            if not improved:
                break
                
        return result

class FitnessCalculator:
    """Dedicated fitness evaluation and penalty computation."""
    
    @staticmethod
    def calculate_fitness(circles: np.ndarray, config: OptimizationConfig) -> Tuple[float, float]:
        """Calculate fitness with penalty for constraints."""
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Calculate objective (sum of radii)
        total_radius = np.sum(radii)

        # Penalty for constraint violations
        penalty = 0

        # Check containment constraints efficiently
        if not ConstraintHandler.is_contained(positions, radii, config.BOUNDARY_MARGIN):
            penalty += 10000

        # Check overlap constraints
        penalty += ConstraintHandler.check_overlaps_kdtree(positions, radii, config.PENALTY_FACTOR)

        return (total_radius - penalty,)

class HeuristicFallback:
    """Handles fallback strategies when main optimization fails."""
    
    @staticmethod
    def generate_hexagonal_packing(n_circles: int) -> np.ndarray:
        """Fallback method using hexagonal packing."""
        circles = np.zeros((n_circles, 3))

        # Try a hexagonal packing pattern
        rows = 5
        cols = 5
        if n_circles < rows * cols:
            rows = int(np.ceil(n_circles / cols))

        # Create regular grid points with hexagonal offset
        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)

        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n_circles:
                    break
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 1) * spacing_x + x_offset
                y = (i + 1) * spacing_y
                # Set reasonable initial radius
                r = min(spacing_x, spacing_y) * 0.35
                circles[count] = [x, y, r]
                count += 1

        # Refine positions to avoid overlaps
        for _ in range(100):
            improved = False
            for i in range(n_circles):
                best_pos = circles[i, :2].copy()
                best_rad = circles[i, 2]
                best_score = -1000

                # Check nearby positions
                for dx in [-0.015, -0.01, 0, 0.01, 0.015]:
                    for dy in [-0.015, -0.01, 0, 0.01, 0.015]:
                        test_x = max(0.005, min(0.995, circles[i, 0] + dx))
                        test_y = max(0.005, min(0.995, circles[i, 1] + dy))
                        test_r = circles[i, 2]

                        valid = True
                        for j in range(n_circles):
                            if i != j:
                                dist = np.sqrt((test_x - circles[j, 0])**2 + (test_y - circles[j, 1])**2)
                                if dist < (test_r + circles[j, 2]):
                                    valid = False
                                    break

                        if valid:
                            score = test_r
                            if score > best_score:
                                best_score = score
                                best_pos = [test_x, test_y]

                if best_score > circles[i, 2]:
                    circles[i, :2] = best_pos
                    circles[i, 2] = best_score
                    improved = True

            if not improved:
                break

        return circles

class CirclePackingOptimizer:
    """Main optimizer class that orchestrates the entire optimization workflow."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._setup_deap()
        
    def _setup_deap(self):
        """Initialize DEAP framework components."""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    def _evaluate_fitness(self, individual):
        """Evaluate fitness of circle placement."""
        circles = np.array(individual).reshape(-1, 3)
        fitness_value, _ = FitnessCalculator.calculate_fitness(circles, self.config)
        return (fitness_value,)

    def optimize(self) -> np.ndarray:
        """Main optimization routine."""
        # Initialize toolbox
        toolbox = base.Toolbox()
        toolbox.register("individual", EvolutionaryOperators.create_individual, self.config)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_fitness)
        toolbox.register("mate", lambda ind1, ind2: EvolutionaryOperators.crossover_constraint_aware(ind1, ind2, self.config))
        toolbox.register("mutate", lambda ind: EvolutionaryOperators.mutate_individual(ind, self.config))
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create initial population
        population = toolbox.population(n=self.config.POP_SIZE)

        # Run evolution
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        try:
            population, logbook = algorithms.eaSimple(
                population, toolbox, cxpb=self.config.CXPB, mutpb=self.config.MUTPB,
                ngen=self.config.GEN_COUNT, stats=stats, halloffame=hof, verbose=False
            )
        except Exception as e:
            # Fallback to heuristic if GA fails
            print(f"GA failed with error: {e}")
            return HeuristicFallback.generate_hexagonal_packing(self.config.N_CIRCLES)

        # Return best solution
        best_individual = hof[0]
        result = np.array(best_individual).reshape(-1, 3)

        # Apply local optimization to refine further
        refined_result = LocalOptimizer.optimize_circles(result.copy(), self.config.MAX_LOCAL_ITERATIONS)
        
        return refined_result

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    config = OptimizationConfig(n_circles=26, pop_size=100, gen_count=80, mutpb=0.3, cxpb=0.5)
    optimizer = CirclePackingOptimizer(config)
    return optimizer.optimize()

# EVOLVE-BLOCK-END