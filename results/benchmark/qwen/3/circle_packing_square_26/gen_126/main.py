# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 300
TOURNAMENT_SIZE = 5
MUTATION_RATE_START = 0.25
MUTATION_RATE_END = 0.01
CROSSOVER_PROB = 0.8
INITIAL_REFINEMENT_STEPS = 10
FINAL_REFINEMENT_STEPS = 20

class AdaptiveSpatialIndexer:
    """Efficient spatial indexing with adaptive grid resolution"""
    
    def __init__(self, base_grid_size: int = 15):
        self.base_grid_size = base_grid_size
        self.grid_cells = {}
        self.current_grid_size = base_grid_size
    
    def _get_grid_key(self, x: float, y: float) -> Tuple[int, int]:
        """Convert coordinates to grid cell indices"""
        return (int(x * self.current_grid_size), int(y * self.current_grid_size))
    
    def build_index(self, circles: np.ndarray, dynamic_resolution: bool = True) -> dict:
        """Build spatial grid index with optional dynamic resolution"""
        self.grid_cells.clear()
        
        if dynamic_resolution:
            # Adjust grid size based on average circle radius
            avg_radius = np.mean(circles[:, 2])
            if avg_radius > 0.1:
                self.current_grid_size = max(10, int(self.base_grid_size / (avg_radius * 2)))
            else:
                self.current_grid_size = self.base_grid_size
                
        for i, (x, y, r) in enumerate(circles):
            cell = self._get_grid_key(x, y)
            if cell not in self.grid_cells:
                self.grid_cells[cell] = []
            self.grid_cells[cell].append(i)
        return self.grid_cells
    
    def get_neighbors(self, x: float, y: float, radius: float) -> List[int]:
        """Get candidate neighbors using adaptive grid"""
        neighbors = []
        center_cell = self._get_grid_key(x, y)
        
        # Check nearby cells in a 3x3 grid around center
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell in self.grid_cells:
                    neighbors.extend(self.grid_cells[cell])
        
        return neighbors

class GeometricConstraintValidator:
    """Advanced constraint validation using geometric insights"""
    
    @staticmethod
    def validate_containment(circles: np.ndarray) -> bool:
        """Check if all circles are fully contained"""
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        return True
    
    @staticmethod
    def validate_overlap(circles: np.ndarray, spatial_indexer: AdaptiveSpatialIndexer = None) -> bool:
        """Check for overlaps using spatial indexing"""
        if len(circles) <= 1:
            return True
            
        if spatial_indexer is not None:
            spatial_indexer.build_index(circles, dynamic_resolution=True)
            positions = [(x, y) for x, y, r in circles]
            tree = cKDTree(positions)
            
            for i, (xi, yi, ri) in enumerate(circles):
                indices = tree.query_ball_point([xi, yi], 2 * (ri + 0.01))
                
                for j in indices:
                    if i != j:
                        xj, yj, rj = circles[j]
                        distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                        
                        if distance < (ri + rj - 1e-8):
                            return False
        else:
            # Brute force fallback
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    xi, yi, ri = circles[i]
                    xj, yj, rj = circles[j]
                    distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                    
                    if distance < (ri + rj - 1e-8):
                        return False
                        
        return True
    
    @staticmethod
    def enforce_bounds(circles: np.ndarray) -> np.ndarray:
        """Enforce boundary constraints carefully"""
        result = circles.copy()
        
        for i in range(len(result)):
            x, y, r = result[i]
            
            # Adjust radius to fit bounds
            max_radius = min(x, 1-x, y, 1-y)
            r = min(r, max_radius)
            r = max(0.001, min(0.49, r))
            
            # Clamp coordinates
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            result[i] = [x, y, r]
        
        return result

class MultiObjectiveFitnessEvaluator:
    """Fitness evaluation that balances multiple objectives"""
    
    def __init__(self):
        self.boundary_penalty_weight = 10000.0
        self.overlap_penalty_weight = 100000.0
    
    def evaluate(self, circles: np.ndarray, spatial_indexer: AdaptiveSpatialIndexer = None, 
                generation: int = 0, total_generations: int = 100) -> float:
        """Evaluate fitness with multi-objective consideration"""
        # Check constraints
        if not GeometricConstraintValidator.validate_containment(circles):
            penalty = self._compute_boundary_penalty(circles)
            return -penalty
            
        if not GeometricConstraintValidator.validate_overlap(circles, spatial_indexer):
            penalty = self._compute_overlap_penalty(circles)
            return -penalty
        
        # Valid solution - use composite fitness
        total_radius = np.sum(circles[:, 2])
        
        # Add a packing density bonus to encourage tighter packings
        packing_density = self._calculate_packing_density(circles)
        density_bonus = packing_density * 100.0
        
        return total_radius + density_bonus
    
    def _compute_boundary_penalty(self, circles: np.ndarray) -> float:
        """Compute penalty based on boundary violations"""
        penalty = 0.0
        
        for x, y, r in circles:
            # Calculate violations
            left_violation = max(0, r - x)
            right_violation = max(0, r - (1 - x))
            bottom_violation = max(0, r - y)
            top_violation = max(0, r - (1 - y))
            
            penalty += (left_violation + right_violation + 
                       bottom_violation + top_violation) * self.boundary_penalty_weight
        
        return penalty
    
    def _compute_overlap_penalty(self, circles: np.ndarray) -> float:
        """Compute penalty based on overlap violations"""
        penalty = 0.0
        
        n = len(circles)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if distance < (r1 + r2):
                    overlap = (r1 + r2 - distance)
                    penalty += overlap * self.overlap_penalty_weight
                    
        return penalty
    
    def _calculate_packing_density(self, circles: np.ndarray) -> float:
        """Calculate a proxy for packing density"""
        if len(circles) < 2:
            return 0.0
            
        # Calculate average distance between centers
        positions = [(x, y) for x, y, r in circles]
        tree = cKDTree(positions)
        
        # Get nearest neighbors (excluding self)
        distances = []
        for i in range(len(positions)):
            # Query first 3 nearest neighbors
            indices = tree.query(positions[i], k=min(4, len(positions)))[1][1:]
            for idx in indices:
                if idx < len(positions):
                    dist = math.sqrt((positions[i][0] - positions[idx][0])**2 + 
                                   (positions[i][1] - positions[idx][1])**2)
                    distances.append(dist)
        
        if distances:
            avg_distance = np.mean(distances)
            # Normalize density based on average radius
            avg_radius = np.mean(circles[:, 2])
            if avg_radius > 0:
                # Density increases as average distance decreases relative to radius
                density = 1.0 / max(0.001, avg_distance / avg_radius)
                return min(density, 10.0)  # Cap at reasonable value
        
        return 0.0

class MaximalPackingInitializer:
    """Generates high-quality initial configurations using maximal packing principles"""
    
    @staticmethod
    def generate_optimized_seed(n: int) -> np.ndarray:
        """Generate an optimized initial seed using a greedy approach"""
        circles = np.zeros((n, 3))
        
        # Start with a simple square grid pattern
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)
        
        # Generate base positions
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) < n:
                    x = 0.05 + (j + 1) * spacing_x
                    y = 0.05 + (i + 1) * spacing_y
                    positions.append((x, y))
        
        # Fill remaining positions with random placement
        while len(positions) < n:
            positions.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))
        
        # Place circles with initial radii
        for i in range(n):
            x, y = positions[i]
            
            # Set initial radius based on proximity to other circles
            max_possible_radius = min(x, 1-x, y, 1-y)
            # Try to make it relatively large but not too crowded
            r = max(0.01, min(max_possible_radius, 0.05 + random.uniform(0, 0.03)))
            
            circles[i] = [x, y, r]
        
        # Perform initial refinement to remove overlaps
        circles = MaximalPackingInitializer._refine_initial_placement(circles)
        
        return circles
    
    @staticmethod
    def _refine_initial_placement(circles: np.ndarray) -> np.ndarray:
        """Refine initial placement using geometric constraints"""
        refined = circles.copy()
        
        # Multiple passes of overlap resolution
        for _ in range(15):
            changed = False
            for i in range(len(refined)):
                for j in range(len(refined)):
                    if i != j:
                        x1, y1, r1 = refined[i]
                        x2, y2, r2 = refined[j]
                        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        
                        if distance < (r1 + r2 - 1e-6):
                            # Move them apart
                            dx = x2 - x1
                            dy = y2 - y1
                            dist = max(1e-8, distance)
                            
                            dx /= dist
                            dy /= dist
                            
                            move_amount = (r1 + r2 - dist) * 0.5
                            
                            refined[i, 0] -= dx * move_amount * 0.3
                            refined[i, 1] -= dy * move_amount * 0.3
                            refined[j, 0] += dx * move_amount * 0.3
                            refined[j, 1] += dy * move_amount * 0.3
                            changed = True
            
            if not changed:
                break
        
        # Enforce bounds
        for i in range(len(refined)):
            x, y, r = refined[i]
            max_radius = min(x, 1-x, y, 1-y)
            r = min(r, max_radius)
            r = max(0.001, min(0.49, r))
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            refined[i] = [x, y, r]
            
        return refined

class HybridMutationOperator:
    """Advanced mutation operator with differential evolution principles"""
    
    @staticmethod
    def differential_mutation(parents: List[np.ndarray], individual: np.ndarray, 
                             mutation_rate: float) -> np.ndarray:
        """Apply differential mutation using multiple parent comparison"""
        mutated = individual.copy()
        n = len(mutated)
        
        if random.random() < mutation_rate:
            # Select a few parent individuals for differential operation
            selected_parents = random.sample(parents, min(3, len(parents)))
            
            # Compute difference vector from multiple parents
            diff_vector = np.zeros((n, 3))
            for parent in selected_parents:
                diff_vector += (parent - mutated) * random.uniform(0.5, 1.5)
            
            # Apply to individual
            for i in range(n):
                if random.random() < 0.6:  # Apply to 60% of genes
                    # Apply mutation to x, y, or r with different strengths
                    gene_type = random.randint(0, 2)
                    if gene_type == 0:  # x coordinate
                        mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + 
                                                    random.gauss(0, 0.02) + 
                                                    diff_vector[i, 0] * 0.1))
                    elif gene_type == 1:  # y coordinate
                        mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + 
                                                    random.gauss(0, 0.02) + 
                                                    diff_vector[i, 1] * 0.1))
                    else:  # radius
                        mutated[i, 2] = max(0.001, min(0.49, mutated[i, 2] + 
                                                     random.gauss(0, 0.01) + 
                                                     diff_vector[i, 2] * 0.1))
        
        return mutated

class NovelOptimizer:
    """Main optimization class with hybrid approach"""
    
    def __init__(self, population_size: int = 100, generations: int = 300):
        self.population_size = population_size
        self.generations = generations
        self.spatial_indexer = AdaptiveSpatialIndexer()
        self.validator = GeometricConstraintValidator()
        self.evaluator = MultiObjectiveFitnessEvaluator()
        self.initializer = MaximalPackingInitializer()
        self.mutator = HybridMutationOperator()
    
    def initialize_population(self, n: int) -> np.ndarray:
        """Initialize population with hybrid approach"""
        population = []
        
        # Create diverse initial configurations
        for i in range(self.population_size):
            # 40% Voronoi-like initialization
            if i % 5 == 0:
                circles = self.initializer.generate_optimized_seed(n)
            # 30% pure random with constraints
            elif i % 5 == 1:
                circles = np.zeros((n, 3))
                for j in range(n):
                    x = random.uniform(0.05, 0.95)
                    y = random.uniform(0.05, 0.95)
                    r = random.uniform(0.01, 0.08)
                    circles[j] = [x, y, r]
            # 30% grid-based
            else:
                circles = np.zeros((n, 3))
                rows = int(np.ceil(np.sqrt(n)))
                cols = int(np.ceil(n / rows))
                spacing_x = 0.9 / (cols + 1)
                spacing_y = 0.9 / (rows + 1)
                for j in range(n):
                    row = j // cols
                    col = j % cols
                    x = 0.05 + (col + 1) * spacing_x
                    y = 0.05 + (row + 1) * spacing_y
                    r = 0.02 + random.uniform(0, 0.03)
                    circles[j] = [x, y, r]
            
            # Apply boundary enforcement
            circles = self.validator.enforce_bounds(circles)
            population.append(circles)
        
        return np.array(population)
    
    def mutate(self, individual: np.ndarray, parents: List[np.ndarray], 
               generation: int, total_generations: int) -> np.ndarray:
        """Enhanced mutation using differential approach"""
        # Adaptive mutation rate
        progress = generation / total_generations
        mutation_rate = MUTATION_RATE_START + (MUTATION_RATE_END - MUTATION_RATE_START) * (1 - progress)
        
        # Apply differential mutation
        mutated = self.mutator.differential_mutation(parents, individual, mutation_rate)
        
        # Apply refinement with spatial indexing
        self.spatial_indexer.build_index(mutated)
        mutated = self._refine_after_mutation(mutated)
        
        return mutated
    
    def _refine_after_mutation(self, circles: np.ndarray) -> np.ndarray:
        """Refine mutated individuals with overlap resolution"""
        # Build spatial index
        self.spatial_indexer.build_index(circles)
        
        # Multiple refinement steps
        for _ in range(10):
            resolved = False
            for i in range(len(circles)):
                x, y, r = circles[i]
                neighbors = self.spatial_indexer.get_neighbors(x, y, r)
                
                for j in neighbors:
                    if i != j and j < len(circles):
                        x2, y2, r2 = circles[j]
                        distance = math.sqrt((x - x2)**2 + (y - y2)**2)
                        
                        if distance < (r + r2 - 1e-8):
                            dx = x2 - x
                            dy = y2 - y
                            dist = max(1e-8, distance)
                            
                            dx /= dist
                            dy /= dist
                            
                            move_amount = (r + r2 - dist) * 0.5
                            
                            # Apply with damping
                            damping_factor = 0.4
                            circles[i, 0] -= dx * move_amount * damping_factor * 0.3
                            circles[i, 1] -= dy * move_amount * damping_factor * 0.3
                            circles[j, 0] += dx * move_amount * damping_factor * 0.3
                            circles[j, 1] += dy * move_amount * damping_factor * 0.3
                            resolved = True
            
            if not resolved:
                break
        
        # Final bounds enforcement
        circles = self.validator.enforce_bounds(circles)
        return circles
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Improved crossover with blend and uniform strategies"""
        if random.random() > CROSSOVER_PROB:
            return parent1.copy() if random.random() < 0.5 else parent2.copy()
        
        n = len(parent1)
        child = np.zeros_like(parent1)
        
        # Blend crossover for continuous parameters
        alpha = random.random()
        for i in range(n):
            # Blend positions
            child[i, 0] = alpha * parent1[i, 0] + (1 - alpha) * parent2[i, 0]
            child[i, 1] = alpha * parent1[i, 1] + (1 - alpha) * parent2[i, 1]
            # Uniform crossover for radius
            if random.random() < 0.5:
                child[i, 2] = parent1[i, 2]
            else:
                child[i, 2] = parent2[i, 2]
        
        return child
    
    def evolve(self, n: int) -> np.ndarray:
        """Main evolution loop"""
        # Initialize population
        population = self.initialize_population(n)
        
        # Track best fitness
        best_fitness_history = []
        
        # Evolution loop
        for gen in range(self.generations):
            # Evaluate fitness for all individuals
            fitnesses = []
            for individual in population:
                fitness = self.evaluator.evaluate(individual, self.spatial_indexer, gen, self.generations)
                fitnesses.append(fitness)
            
            # Track best
            best_fitness = max(fitnesses)
            best_fitness_history.append(best_fitness)
            
            # Print progress
            if gen % 30 == 0:
                print(f"Generation {gen}: Best fitness = {best_fitness:.6f}")
            
            # Selection, crossover, and mutation
            new_population = []
            
            # Elitism: keep top 10%
            elite_count = max(1, self.population_size // 10)
            sorted_indices = np.argsort(fitnesses)[::-1][:elite_count]
            for idx in sorted_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_select(population, fitnesses)
                parent2 = self._tournament_select(population, fitnesses)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation with access to parents
                child = self.mutate(child, [parent1, parent2], gen, self.generations)
                
                new_population.append(child)
            
            # Trim to exact size
            population = new_population[:self.population_size]
        
        # Return best individual
        final_fitnesses = []
        for individual in population:
            fitness = self.evaluator.evaluate(individual, self.spatial_indexer, self.generations, self.generations)
            final_fitnesses.append(fitness)
        
        best_index = np.argmax(final_fitnesses)
        best_solution = population[best_index]
        
        return best_solution
    
    def _tournament_select(self, population: np.ndarray, fitnesses: List[float]) -> np.ndarray:
        """Tournament selection with larger tournaments"""
        tournament_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    optimizer = NovelOptimizer(population_size=100, generations=300)
    circles = optimizer.evolve(26)
    
    return circles

# EVOLVE-BLOCK-END