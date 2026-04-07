# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List, Optional
import math

class SpatialIndexer:
    """Efficient spatial indexing for circle collision detection using adaptive grid resolution"""
    
    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.grid_cells = {}
        
    def _get_grid_key(self, x: float, y: float) -> Tuple[int, int]:
        """Convert coordinates to grid cell indices"""
        return (int(x * self.grid_size), int(y * self.grid_size))
    
    def build_index(self, circles: np.ndarray) -> dict:
        """Build spatial grid index for efficient neighbor queries"""
        self.grid_cells.clear()
        for i, (x, y, r) in enumerate(circles):
            cell = self._get_grid_key(x, y)
            if cell not in self.grid_cells:
                self.grid_cells[cell] = []
            self.grid_cells[cell].append(i)
        return self.grid_cells
    
    def get_neighbors(self, x: float, y: float, radius: float) -> List[int]:
        """Get candidate neighbors within a search radius"""
        neighbors = []
        center_cell = self._get_grid_key(x, y)
        
        # Check nearby cells in a 3x3 grid around center
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell in self.grid_cells:
                    neighbors.extend(self.grid_cells[cell])
        
        return neighbors

class ConstraintValidator:
    """Handles all constraint validation and enforcement with optimized methods"""
    
    @staticmethod
    def validate_containment(circles: np.ndarray) -> bool:
        """Check if all circles are fully contained in the unit square"""
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        return True
    
    @staticmethod
    def validate_overlap(circles: np.ndarray, spatial_indexer: Optional[SpatialIndexer] = None) -> bool:
        """Check for circle overlaps using spatial indexing when available"""
        if len(circles) <= 1:
            return True
            
        # Use spatial indexing for efficiency
        if spatial_indexer is not None:
            # More efficient approach - only check neighbors in spatial grid
            positions = [(x, y) for x, y, r in circles]
            tree = cKDTree(positions)
            
            # Check each circle against its neighbors
            for i, (xi, yi, ri) in enumerate(circles):
                # Query nearby points with search radius based on sum of radii
                indices = tree.query_ball_point([xi, yi], ri + 0.01)
                
                for j in indices:
                    if i != j:
                        xj, yj, rj = circles[j]
                        distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                        
                        if distance < (ri + rj - 1e-6):  # Validity threshold
                            return False
        else:
            # Fallback to brute force for small populations
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    xi, yi, ri = circles[i]
                    xj, yj, rj = circles[j]
                    distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                    
                    if distance < (ri + rj - 1e-6):
                        return False
                        
        return True
    
    @staticmethod
    def enforce_bounds(circles: np.ndarray) -> np.ndarray:
        """Enforce boundary constraints by adjusting positions and radii"""
        result = circles.copy()
        
        for i in range(len(result)):
            x, y, r = result[i]
            
            # Ensure circle fits in the unit square
            max_radius = min(x, 1-x, y, 1-y)
            r = min(r, max_radius)
            r = max(0.001, min(0.49, r))
            
            # Clamp coordinates to valid range
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            result[i] = [x, y, r]
        
        return result

class FitnessEvaluator:
    """Evaluates fitness with adaptive penalty system"""
    
    def __init__(self, boundary_weight: float = 1000.0, overlap_weight: float = 100000.0):
        self.boundary_weight = boundary_weight
        self.overlap_weight = overlap_weight
    
    def evaluate(self, circles: np.ndarray, spatial_indexer: Optional[SpatialIndexer] = None) -> float:
        """Evaluate fitness with constraint penalties"""
        # Check constraints
        if not ConstraintValidator.validate_containment(circles):
            penalty = self._compute_boundary_penalty(circles)
            return -penalty
            
        if not ConstraintValidator.validate_overlap(circles, spatial_indexer):
            penalty = self._compute_overlap_penalty(circles)
            return -penalty
        
        # Valid solution - return sum of radii
        return float(np.sum(circles[:, 2]))
    
    def _compute_boundary_penalty(self, circles: np.ndarray) -> float:
        """Compute penalty based on boundary violations"""
        penalty = 0.0
        
        for x, y, r in circles:
            # Calculate boundary violations
            if x - r < 0:
                penalty += abs(x - r) * self.boundary_weight
            elif x + r > 1:
                penalty += abs(x + r - 1) * self.boundary_weight
            if y - r < 0:
                penalty += abs(y - r) * self.boundary_weight
            elif y + r > 1:
                penalty += abs(y + r - 1) * self.boundary_weight
                
        return penalty
    
    def _compute_overlap_penalty(self, circles: np.ndarray) -> float:
        """Compute penalty based on overlap violations"""
        penalty = 0.0
        # For simplicity, apply a high penalty for overlaps
        penalty = self.overlap_weight * 1000.0
        return penalty

class EvolutionaryOptimizer:
    """Main evolutionary optimization class with enhanced features"""
    
    def __init__(self, population_size: int = 100, generations: int = 300):
        self.population_size = population_size
        self.generations = generations
        self.spatial_indexer = SpatialIndexer()
        self.validator = ConstraintValidator()
        self.evaluator = FitnessEvaluator()
        
    def initialize_population(self, n: int) -> np.ndarray:
        """Initialize population using hexagonal packing for better starting configurations"""
        population = []
        
        # Hexagonal grid-based initialization for better spatial distribution
        rows, cols = 5, 5
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Create hexagonal layout with more uniform distribution
        for _ in range(self.population_size):
            circles = np.zeros((n, 3))
            
            # Distribute circles in a hexagonal pattern
            for i in range(n):
                row = i // cols
                col = i % cols
                
                # Offset odd rows for hexagonal packing
                x_base = (col + 1) * spacing_x
                y_base = (row + 1) * spacing_y
                
                if row % 2 == 1:
                    x_base += spacing_x * 0.5
                
                # Add more substantial random offset
                x = max(0.01, min(0.99, x_base + random.uniform(-0.03, 0.03)))
                y = max(0.01, min(0.99, y_base + random.uniform(-0.03, 0.03)))
                
                # Initial radius - start with larger values for better exploration
                r = 0.02 + random.uniform(0, 0.04)
                
                circles[i] = [x, y, r]
            
            # Apply constraint enforcement
            circles = self.validator.enforce_bounds(circles)
            population.append(circles)
        
        return np.array(population)
    
    def mutate(self, circles: np.ndarray, generation: int, total_generations: int) -> np.ndarray:
        """Enhanced mutation with adaptive rates and better refinement"""
        mutated = circles.copy()
        
        # Adaptive mutation rate - start high, decrease over time
        mutation_rate = 0.2 + 0.1 * (1 - generation / total_generations)
        
        # Different mutation step sizes for different parameters
        pos_step = 0.02 + 0.01 * (1 - generation / total_generations)
        rad_step = 0.01 + 0.005 * (1 - generation / total_generations)
        
        n = len(mutated)
        
        # Mutate circles with adaptive rate and step size
        for i in range(n):
            if random.random() < mutation_rate:
                # Choose which component to mutate with preference for position
                choice = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
                
                if choice == 0:  # X coordinate
                    mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + random.gauss(0, pos_step)))
                elif choice == 1:  # Y coordinate
                    mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + random.gauss(0, pos_step)))
                else:  # Radius
                    mutated[i, 2] = max(0.001, min(0.49, mutated[i, 2] + random.gauss(0, rad_step)))
        
        # Apply refinement steps
        mutated = self._refine_after_mutation(mutated)
        mutated = self.validator.enforce_bounds(mutated)
        
        return mutated
    
    def _refine_after_mutation(self, circles: np.ndarray) -> np.ndarray:
        """Refine mutated individuals to resolve potential issues"""
        # Build spatial index for efficient overlap checking
        self.spatial_indexer.build_index(circles)
        
        # Resolve overlaps using a simple iterative approach
        for _ in range(10):
            resolved = False
            for i in range(len(circles)):
                x, y, r = circles[i]
                
                # Find overlapping circles - use spatial index for efficiency
                neighbors = self.spatial_indexer.get_neighbors(x, y, r)
                
                for j in neighbors:
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = math.sqrt((x - x2)**2 + (y - y2)**2)
                        
                        # If overlap exists, adjust positions
                        if distance < (r + r2 - 1e-6):
                            # Move circles apart along displacement vector
                            dx = x2 - x
                            dy = y2 - y
                            dist = max(1e-6, distance)
                            
                            # Normalize and move apart
                            dx /= dist
                            dy /= dist
                            
                            move_amount = (r + r2 - dist) / 2.0
                            
                            circles[i, 0] -= dx * move_amount * 0.2
                            circles[i, 1] -= dy * move_amount * 0.2
                            circles[j, 0] += dx * move_amount * 0.2
                            circles[j, 1] += dy * move_amount * 0.2
                            resolved = True
            
            # If no changes made, stop iteration
            if not resolved:
                break
        
        return circles
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Improved crossover with selective recombination"""
        if random.random() > 0.7:  # Lower crossover probability for more diversity
            # Return one parent if crossover doesn't happen
            return parent1.copy() if random.random() < 0.5 else parent2.copy()
        
        n = len(parent1)
        child = np.zeros_like(parent1)
        
        # Uniform crossover - randomly select genes from parents
        for i in range(n):
            if random.random() < 0.5:
                child[i] = parent1[i].copy()
            else:
                child[i] = parent2[i].copy()
        
        # Apply refinement to ensure validity
        return self._refine_after_crossover(child)
    
    def _refine_after_crossover(self, child: np.ndarray) -> np.ndarray:
        """Refine offspring after crossover"""
        # Force boundary enforcement
        child = self.validator.enforce_bounds(child)
        
        # Quick overlap resolution
        self.spatial_indexer.build_index(child)
        if not ConstraintValidator.validate_overlap(child, self.spatial_indexer):
            # Simple resolution for immediate fixes
            for i in range(len(child)):
                for j in range(i+1, len(child)):
                    x1, y1, r1 = child[i]
                    x2, y2, r2 = child[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    if distance < (r1 + r2 - 1e-6):
                        # Simple adjustment to separate circles
                        dx = x2 - x1
                        dy = y2 - y1
                        dist = max(1e-6, distance)
                        
                        dx /= dist
                        dy /= dist
                        
                        move_amount = (r1 + r2 - dist) / 2.0
                        
                        child[i, 0] -= dx * move_amount * 0.05
                        child[i, 1] -= dy * move_amount * 0.05
                        child[j, 0] += dx * move_amount * 0.05
                        child[j, 1] += dy * move_amount * 0.05
        
        return child
    
    def evolve(self, n: int) -> np.ndarray:
        """Main evolution loop with enhanced convergence control"""
        # Initialize population
        population = self.initialize_population(n)
        
        # Track best fitness
        best_fitness_history = []
        
        # Evolution loop
        for gen in range(self.generations):
            # Evaluate fitness
            fitnesses = []
            for individual in population:
                fitness = self.evaluator.evaluate(individual, self.spatial_indexer)
                fitnesses.append(fitness)
            
            # Track best
            best_fitness = max(fitnesses)
            best_fitness_history.append(best_fitness)
            
            # Print progress
            if gen % 50 == 0:
                print(f"Generation {gen}: Best fitness = {best_fitness}")
            
            # Selection, crossover, and mutation
            new_population = []
            
            # Elitism: keep best individual
            best_idx = np.argmax(fitnesses)
            new_population.append(population[best_idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_select(population, fitnesses)
                parent2 = self._tournament_select(population, fitnesses)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutate(child, gen, self.generations)
                
                new_population.append(child)
            
            # Trim to exact population size
            population = new_population[:self.population_size]
        
        # Return the best individual
        final_fitnesses = []
        for individual in population:
            fitness = self.evaluator.evaluate(individual, self.spatial_indexer)
            final_fitnesses.append(fitness)
        
        best_index = np.argmax(final_fitnesses)
        best_solution = population[best_index]
        
        return best_solution
    
    def _tournament_select(self, population: np.ndarray, fitnesses: List[float], tournament_size: int = 5) -> np.ndarray:
        """Tournament selection with improved logic"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
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
    
    optimizer = EvolutionaryOptimizer(population_size=100, generations=300)
    circles = optimizer.evolve(26)
    
    return circles

# EVOLVE-BLOCK-END
