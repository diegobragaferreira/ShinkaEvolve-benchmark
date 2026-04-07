# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.optimize import minimize
import random
import math
from typing import Tuple, List

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

class AdaptiveVoronoiEvolution:
    def __init__(self):
        self.n_circles = 26
        self.population_size = 120
        self.generations = 150
        self.mutation_rate_start = 0.15
        self.mutation_rate_end = 0.01
        self.crossover_rate = 0.7
        self.elite_count = 8
        self.tournament_size = 5
        
    def create_voronoi_initialization(self) -> np.ndarray:
        """Create initial population using Voronoi-based approach for better spatial distribution"""
        # Generate points in a structured way to form Voronoi cells
        points = []
        
        # Create a hexagonal-like grid pattern for better initial distribution
        grid_size = int(np.ceil(np.sqrt(self.n_circles * 2)))
        spacing = 1.0 / (grid_size + 2)
        
        # Generate hexagonal grid points
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= self.n_circles * 2:
                    break
                # Offset every other row for hexagonal arrangement
                x = (j + 0.5 + (i % 2) * 0.5) * spacing
                y = (i + 0.5) * spacing * np.sqrt(3) / 2
                if x <= 1 and y <= 1 and x >= 0 and y >= 0:
                    points.append([x, y])
        
        # Trim to desired size
        points = points[:self.n_circles * 2]
        
        # Add jitter to points to avoid regular patterns
        for point in points:
            point[0] += random.uniform(-spacing/3, spacing/3)
            point[1] += random.uniform(-spacing/3, spacing/3)
            
        # Ensure points are within bounds
        points = [[max(0.01, min(0.99, p[0])), max(0.01, min(0.99, p[1]))] for p in points]
        
        # Create Voronoi diagram
        try:
            vor = Voronoi(points)
        except:
            # Fallback to simpler initialization if Voronoi fails
            return self.create_simple_grid_initialization()
        
        # Select valid Voronoi cells for circle placement
        circles = np.zeros((self.n_circles, 3))
        valid_cells = []
        
        # Find valid Voronoi regions
        for i in range(min(self.n_circles, len(vor.point_region))):
            region = vor.point_region[i]
            if region != -1:  # Valid region
                valid_cells.append(i)
        
        selected_indices = valid_cells[:self.n_circles]
        
        # Use selected Voronoi points for circle placement
        for i, idx in enumerate(selected_indices):
            center = vor.points[idx]
            x, y = center
            
            # Estimate appropriate radius based on Voronoi cell geometry
            radius_estimate = spacing / 2.5
            min_distance_to_boundary = min(x, y, 1-x, 1-y)
            final_radius = min(radius_estimate, min_distance_to_boundary * 0.7)
            final_radius = max(0.005, final_radius)
            
            circles[i] = [x, y, final_radius]
        
        return circles
    
    def create_simple_grid_initialization(self) -> np.ndarray:
        """Fallback initialization using regular grid"""
        circles = np.zeros((self.n_circles, 3))
        
        grid_size = int(np.ceil(np.sqrt(self.n_circles)))
        spacing = 1.0 / (grid_size + 1)
        
        idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                if idx >= self.n_circles:
                    break
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                # Add slight randomness to positions
                x += random.uniform(-spacing/4, spacing/4)
                y += random.uniform(-spacing/4, spacing/4)
                # Ensure within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                r = spacing / 3.5
                circles[idx] = [x, y, r]
                idx += 1
                
        return circles
    
    def validate_circles(self, circles: np.ndarray) -> bool:
        """Check if circles satisfy all constraints"""
        n = len(circles)
        
        # Check containment constraints
        for i in range(n):
            x, y, r = circles[i]
            if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
                return False

        # Check overlap constraints using efficient KDTree approach
        if n > 1:
            try:
                points = circles[:, :2]
                tree = cKDTree(points)
                
                for i in range(n):
                    x1, y1, r1 = circles[i]
                    # Find nearby points
                    neighbors = tree.query_ball_point([x1, y1], 2*(r1+0.01))
                    for j in neighbors:
                        if i != j:
                            x2, y2, r2 = circles[j]
                            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                            if distance < (r1 + r2):
                                return False
            except:
                # Fallback to direct calculation if tree fails
                for i in range(n):
                    x1, y1, r1 = circles[i]
                    for j in range(i+1, n):
                        x2, y2, r2 = circles[j]
                        distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        if distance < (r1 + r2):
                            return False
                            
        return True
    
    def calculate_fitness(self, circles: np.ndarray) -> float:
        """Calculate fitness as sum of radii with constraint penalties"""
        total_radius = np.sum(circles[:, 2])
        
        # Penalty for containment violations
        penalty = 0
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x < r or x > 1-r or y < r or y > 1-r:
                penalty += 1000
        
        # Penalty for overlap violations
        if len(circles) > 1:
            points = circles[:, :2]
            tree = cKDTree(points)
            
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                neighbors = tree.query_ball_point([x1, y1], 2*(r1+0.01))
                for j in neighbors:
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        if distance < (r1 + r2):
                            overlap = (r1 + r2) - distance
                            penalty += 10000 * overlap
        
        return total_radius - penalty
    
    def mutate_individual(self, individual: np.ndarray, generation: int) -> np.ndarray:
        """Apply adaptive mutation to individual"""
        mutated = individual.copy()
        mutation_rate = max(self.mutation_rate_end, 
                          self.mutation_rate_start * (self.mutation_rate_end/self.mutation_rate_start) ** (generation/self.generations))
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Mutate either position or radius
                if random.random() < 0.7:  # Mutate position
                    # Add Gaussian noise to position
                    if i % 3 == 0:  # x coordinate
                        mutated[i] = max(0.001, min(0.999, mutated[i] + random.gauss(0, 0.02)))
                    else:  # y coordinate  
                        mutated[i] = max(0.001, min(0.999, mutated[i] + random.gauss(0, 0.02)))
                else:  # Mutate radius
                    # Multiplicative mutation
                    mutated[i] = max(0.001, mutated[i] * random.uniform(0.8, 1.2))
        
        return mutated
    
    def crossover_individuals(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover with constraint awareness"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
            
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Uniform crossover
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        
        return child1, child2
    
    def repair_circles(self, circles: np.ndarray) -> np.ndarray:
        """Repair circles to satisfy constraints"""
        repaired = circles.copy()
        
        # First, fix containment
        for i in range(len(repaired)):
            x, y, r = repaired[i]
            # Ensure circles stay within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            repaired[i] = [x, y, r]
        
        # Then resolve overlaps by iterative adjustment
        for _ in range(20):  # Limited iterations to prevent infinite loops
            improved = False
            for i in range(len(repaired)):
                x1, y1, r1 = repaired[i]
                
                # Check for overlaps
                for j in range(len(repaired)):
                    if i != j:
                        x2, y2, r2 = repaired[j]
                        distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        
                        if distance < (r1 + r2):
                            # Resolve overlap by moving circles apart
                            dx = x2 - x1
                            dy = y2 - y1
                            dist = np.sqrt(dx*dx + dy*dy)
                            
                            if dist > 0:
                                # Normalize direction vector
                                dx /= dist
                                dy /= dist
                                
                                # Move both circles apart
                                overlap = (r1 + r2) - dist
                                move_amount = overlap * 0.5
                                
                                repaired[i][0] -= dx * move_amount
                                repaired[i][1] -= dy * move_amount
                                repaired[j][0] += dx * move_amount
                                repaired[j][1] += dy * move_amount
                                
                                improved = True
            
            if not improved:
                break
        
        # Final boundary correction
        for i in range(len(repaired)):
            x, y, r = repaired[i]
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            repaired[i] = [x, y, r]
            
        return repaired
    
    def adaptive_local_optimization(self, circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """Apply adaptive local optimization to improve circle placements"""
        optimized = circles.copy()
        
        # Try different optimization strategies
        strategies = [
            self._strategy_direct_radius_increase,
            self._strategy_coordinate_adjustment,
            self._strategy_hybrid_approach
        ]
        
        # Apply each strategy multiple times for better results
        for _ in range(3):
            for strategy in strategies:
                optimized = strategy(optimized, max_iterations // 3)
        
        return optimized
    
    def _strategy_direct_radius_increase(self, circles: np.ndarray, max_iter: int) -> np.ndarray:
        """Directly try to increase radii while maintaining constraints"""
        result = circles.copy()
        
        for _ in range(max_iter):
            improved = False
            for i in range(len(result)):
                x, y, r = result[i]
                
                # Calculate maximum possible radius
                max_radius = min(x, 1-x, y, 1-y)
                
                if max_radius > r + 1e-6:
                    # Try to increase radius
                    target_radius = min(r + 0.005, max_radius)
                    
                    # Check if this works without violating constraints
                    valid = True
                    for j in range(len(result)):
                        if i != j:
                            x2, y2, r2 = result[j]
                            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                            if distance < target_radius + r2:
                                valid = False
                                break
                    
                    if valid:
                        result[i, 2] = target_radius
                        improved = True
            
            if not improved:
                break
                
        return result
    
    def _strategy_coordinate_adjustment(self, circles: np.ndarray, max_iter: int) -> np.ndarray:
        """Adjust positions to reduce overlaps"""
        result = circles.copy()
        
        for _ in range(max_iter):
            improved = False
            
            # For each circle, try to improve its position
            for i in range(len(result)):
                x, y, r = result[i]
                
                # Calculate forces from nearby circles
                force_x, force_y = 0.0, 0.0
                
                for j in range(len(result)):
                    if i != j:
                        x2, y2, r2 = result[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        
                        if distance < (r + r2):
                            # Repulsive force
                            dx = x - x2
                            dy = y - y2
                            dist = np.sqrt(dx*dx + dy*dy)
                            
                            if dist > 1e-8:
                                force_x += dx / dist * (r + r2 - distance) * 0.1
                                force_y += dy / dist * (r + r2 - distance) * 0.1
                                
                # Apply forces
                new_x = x + force_x
                new_y = y + force_y
                
                # Keep within bounds
                new_x = max(r, min(1-r, new_x))
                new_y = max(r, min(1-r, new_y))
                
                # Check if this improves the configuration
                test_result = result.copy()
                test_result[i] = [new_x, new_y, r]
                
                # Validate the new configuration
                valid = True
                for k in range(len(test_result)):
                    if k != i:
                        x1, y1, r1 = test_result[k]
                        x2, y2, r2 = test_result[i]
                        distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        if distance < r1 + r2:
                            valid = False
                            break
                
                if valid:
                    result[i] = [new_x, new_y, r]
                    improved = True
            
            if not improved:
                break
                
        return result
    
    def _strategy_hybrid_approach(self, circles: np.ndarray, max_iter: int) -> np.ndarray:
        """Hybrid approach combining direct optimization and simulated annealing"""
        result = circles.copy()
        current_fitness = self.calculate_fitness(result)
        
        for _ in range(max_iter):
            # Create candidate by making small random changes
            candidate = result.copy()
            
            # Choose a random circle to modify
            idx = random.randint(0, len(candidate)-1)
            x, y, r = candidate[idx]
            
            # Apply small perturbation
            new_x = max(0.001, min(0.999, x + random.uniform(-0.01, 0.01)))
            new_y = max(0.001, min(0.999, y + random.uniform(-0.01, 0.01)))  
            new_r = max(0.001, min(0.4, r + random.uniform(-0.005, 0.005)))
            
            candidate[idx] = [new_x, new_y, new_r]
            
            # Repair and evaluate
            candidate = self.repair_circles(candidate)
            candidate_fitness = self.calculate_fitness(candidate)
            
            # Accept if better or with some probability
            if candidate_fitness > current_fitness:
                result = candidate
                current_fitness = candidate_fitness
            elif random.random() < 0.1:  # Small chance to accept worse solutions
                result = candidate
                current_fitness = candidate_fitness
                
        return result
    
    def run_evolution(self) -> np.ndarray:
        """Run the complete evolutionary algorithm"""
        # Initialize population
        population = []
        
        # Create diverse initial population
        for i in range(self.population_size):
            if i == 0:
                # First individual: Voronoi initialization
                circles = self.create_voronoi_initialization()
            elif i < self.population_size // 3:
                # Second third: grid initialization
                circles = self.create_simple_grid_initialization()
            else:
                # Last third: perturbed Voronoi
                base_circles = self.create_voronoi_initialization()
                circles = base_circles.copy()
                # Add some random perturbation
                for j in range(len(circles)):
                    circles[j, 0] += random.uniform(-0.02, 0.02)
                    circles[j, 1] += random.uniform(-0.02, 0.02)
                    circles[j, 2] *= random.uniform(0.9, 1.1)
            
            # Repair if needed
            circles = self.repair_circles(circles)
            population.append(circles)
        
        best_solution = None
        best_fitness = -float('inf')
        
        for generation in range(self.generations):
            # Evaluate fitness for all individuals
            fitnesses = []
            for circles in population:
                if self.validate_circles(circles):
                    fitness = self.calculate_fitness(circles)
                else:
                    fitness = -100000  # Very poor fitness for invalid solutions
                fitnesses.append(fitness)
            
            # Track best solution
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitnesses)[-self.elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1_idx = self._tournament_selection(population, fitnesses)
                parent2_idx = self._tournament_selection(population, fitnesses)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # Crossover
                child1, child2 = self.crossover_individuals(parent1, parent2)
                
                # Mutation
                child1 = self.mutate_individual(child1, generation)
                child2 = self.mutate_individual(child2, generation)
                
                # Repair
                child1 = self.repair_circles(child1)
                child2 = self.repair_circles(child2)
                
                # Add children to new population
                new_population.extend([child1, child2])
            
            # Trim to exact size
            population = new_population[:self.population_size]
        
        # Apply final local optimization to best solution
        if best_solution is not None:
            best_solution = self.adaptive_local_optimization(best_solution)
            
        return best_solution if best_solution is not None else self.create_simple_grid_initialization()
    
    def _tournament_selection(self, population: List[np.ndarray], fitnesses: List[float]) -> int:
        """Tournament selection with adaptive tournament size"""
        # Select tournament size based on diversity
        if len(fitnesses) > 1:
            diversity = np.std(fitnesses)
            # Smaller tournament size for high diversity, larger for low diversity
            tournament_size = max(3, min(7, int(5 - diversity * 20)))
        else:
            tournament_size = self.tournament_size
            
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return winner_index

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = AdaptiveVoronoiEvolution()
    return optimizer.run_evolution()

# EVOLVE-BLOCK-END