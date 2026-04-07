# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree, Voronoi
import random
from typing import Tuple, List
import math
from dataclasses import dataclass
import time

@dataclass
class OptimizerConfig:
    """Configuration for the hybrid circle packing optimizer."""
    population_size: int = 100
    generations: int = 150
    tournament_size: int = 5
    mutation_rate_start: float = 0.2
    mutation_rate_end: float = 0.005
    crossover_prob: float = 0.9
    validity_threshold: float = 1e-6
    benchmark_target: float = 2.6358627564136983
    max_local_iterations: int = 20
    sa_initial_temp: float = 1.0
    sa_cooling_rate: float = 0.95
    quadtree_resolution: int = 100

class QuadTreeNode:
    """Quadtree node for efficient spatial indexing."""
    def __init__(self, bounds, capacity=4):
        self.bounds = bounds  # (x_min, x_max, y_min, y_max)
        self.capacity = capacity
        self.points = []
        self.divided = False
        self.children = [None] * 4
        
    def subdivide(self):
        """Divide node into four quadrants."""
        x_min, x_max, y_min, y_max = self.bounds
        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2
        
        # Four quadrants: bottom-left, bottom-right, top-left, top-right
        self.children[0] = QuadTreeNode((x_min, mid_x, y_min, mid_y), self.capacity)
        self.children[1] = QuadTreeNode((mid_x, x_max, y_min, mid_y), self.capacity)
        self.children[2] = QuadTreeNode((x_min, mid_x, mid_y, y_max), self.capacity)
        self.children[3] = QuadTreeNode((mid_x, x_max, mid_y, y_max), self.capacity)
        self.divided = True
        
    def insert(self, point, index):
        """Insert a point into the quadtree."""
        x, y = point
        
        if not self.in_bounds(x, y):
            return False
            
        if len(self.points) < self.capacity and not self.divided:
            self.points.append((point, index))
            return True
            
        if not self.divided:
            self.subdivide()
            
        for child in self.children:
            if child.insert(point, index):
                return True
                
        return False
        
    def in_bounds(self, x, y):
        """Check if point is within bounds."""
        x_min, x_max, y_min, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max
        
    def query_range(self, range_bounds):
        """Query all points within a given range."""
        x_min, x_max, y_min, y_max = range_bounds
        result = []
        
        if not self.intersects_range(range_bounds):
            return result
            
        for point, index in self.points:
            x, y = point
            if x_min <= x <= x_max and y_min <= y <= y_max:
                result.append((point, index))
                
        if self.divided:
            for child in self.children:
                result.extend(child.query_range(range_bounds))
                
        return result
        
    def intersects_range(self, range_bounds):
        """Check if node intersects with query range."""
        x_min, x_max, y_min, y_max = range_bounds
        node_x_min, node_x_max, node_y_min, node_y_max = self.bounds
        
        return not (x_max < node_x_min or x_min > node_x_max or 
                   y_max < node_y_min or y_min > node_y_max)

class CirclePacker:
    """Hybrid optimizer for circle packing problem."""
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
        random.seed(42)
        np.random.seed(42)
        
    def voronoi_based_initialization(self, n: int) -> np.ndarray:
        """Initialize circles using Voronoi-based distribution with Lloyd relaxation."""
        # Generate initial points using a modified grid pattern
        points = []
        grid_size = int(math.ceil(math.sqrt(n)))
        
        # Create a structured grid pattern
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= n:
                    break
                x = 0.1 + 0.8 * i / (grid_size - 1) if grid_size > 1 else 0.5
                y = 0.1 + 0.8 * j / (grid_size - 1) if grid_size > 1 else 0.5
                points.append((x, y))
        
        # Add extra points if needed
        while len(points) < n:
            points.append((random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)))
            
        points = points[:n]
        
        # Apply Lloyd relaxation for better uniformity
        for _ in range(10):
            if len(points) < 2:
                break
            # Create Voronoi diagram
            vor = Voronoi(np.array(points))
            
            # Compute centroids of Voronoi cells
            new_points = []
            for region in vor.regions:
                if len(region) > 0 and -1 not in region:
                    # Compute centroid of polygon
                    xs = [vor.vertices[i][0] for i in region if i >= 0]
                    ys = [vor.vertices[i][1] for i in region if i >= 0]
                    if xs and ys:
                        centroid_x = sum(xs) / len(xs)
                        centroid_y = sum(ys) / len(ys)
                        # Keep within bounds
                        centroid_x = max(0.1, min(0.9, centroid_x))
                        centroid_y = max(0.1, min(0.9, centroid_y))
                        new_points.append((centroid_x, centroid_y))
                    else:
                        new_points.append(points[len(new_points)])
                else:
                    new_points.append(points[len(new_points)])
                    
            # Take only the required number of points
            points = new_points[:n]
            
        # Convert to circle configuration
        circles = np.zeros((n, 3))
        for i, (x, y) in enumerate(points):
            circles[i] = [x, y, 0.05]  # Initial radius
            
        return circles
    
    def initialize_population(self, n: int, population_size: int) -> List[np.ndarray]:
        """Initialize population with hybrid approach."""
        population = []
        
        # Create base population using Voronoi-based initialization
        base_pop = self.voronoi_based_initialization(n)
        
        # Generate variations using different strategies
        for i in range(population_size):
            circles = base_pop.copy()
            
            # Add some randomness to make it diverse
            if i > 0:
                # Apply slight perturbations with decreasing strength
                strength = 0.05 * (1 - i / population_size)
                for j in range(n):
                    circles[j, 0] += random.uniform(-strength, strength)
                    circles[j, 1] += random.uniform(-strength, strength)
                    circles[j, 2] += random.uniform(-0.01, 0.01)
                    
            # Ensure bounds
            circles = self.enforce_constraints(circles)
            
            # Resolve initial overlaps with force-based method
            circles = self.resolve_overlaps(circles)
            
            population.append(circles)
            
        return population
    
    def enforce_constraints(self, circles: np.ndarray) -> np.ndarray:
        """Enforce all constraints on circle positions and radii."""
        result = circles.copy()
        
        # Adjust positions and radii to satisfy bounds
        for i in range(len(result)):
            x, y, r = result[i]
            
            # Ensure circle fits in the unit square
            r = min(r, x, 1-x, y, 1-y)
            r = max(0.001, min(0.49, r))
            
            # Clamp coordinates to valid range
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            result[i] = [x, y, r]
        
        return result
    
    def get_quadtree(self, circles: np.ndarray) -> QuadTreeNode:
        """Create quadtree for efficient spatial queries."""
        root = QuadTreeNode((0, 1, 0, 1), capacity=4)
        for i, (x, y, r) in enumerate(circles):
            root.insert((x, y), i)
        return root
    
    def check_overlap_quadtree(self, circles: np.ndarray, quadtree: QuadTreeNode = None) -> bool:
        """Check overlaps using quadtree for efficiency."""
        if quadtree is None:
            quadtree = self.get_quadtree(circles)
            
        n = len(circles)
        for i in range(n):
            x_i, y_i, r_i = circles[i]
            
            # Query nearby cells with expanded range
            range_min = (x_i - r_i - 1e-10, x_i + r_i + 1e-10, 
                         y_i - r_i - 1e-10, y_i + r_i + 1e-10)
            nearby_points = quadtree.query_range(range_min)
            
            for (x_j, y_j), j in nearby_points:
                if i != j:
                    distance = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                    if distance < (r_i + r_j - self.config.validity_threshold):
                        return True
                        
        return False
    
    def resolve_overlaps(self, circles: np.ndarray, max_iterations: int = 20) -> np.ndarray:
        """Resolve overlaps using force-based method with improved convergence."""
        resolved = circles.copy()
        
        for iteration in range(max_iterations):
            changed = False
            tree = self.get_quadtree(resolved)
            
            # Check for overlaps and resolve them
            for i in range(len(resolved)):
                x_i, y_i, r_i = resolved[i]
                
                # Find potentially conflicting circles using quadtree
                range_min = (x_i - r_i - 1e-10, x_i + r_i + 1e-10, 
                             y_i - r_i - 1e-10, y_i + r_i + 1e-10)
                nearby_points = tree.query_range(range_min)
                
                for (x_j, y_j), j in nearby_points:
                    if i != j:
                        distance = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                        if distance < (r_i + r_j - self.config.validity_threshold):
                            # Resolve overlap with force-based approach
                            dx = x_j - x_i
                            dy = y_j - y_i
                            dist = max(self.config.validity_threshold, distance)
                            
                            # Normalize
                            dx /= dist
                            dy /= dist
                            
                            # Calculate overlap amount
                            overlap = (r_i + r_j - dist) * 0.5
                            
                            # Adjust based on relative sizes
                            scale_factor = min(1.0, r_i / (r_i + r_j + 0.001))
                            
                            resolved[i, 0] -= dx * overlap * scale_factor * 0.5
                            resolved[i, 1] -= dy * overlap * scale_factor * 0.5
                            resolved[j, 0] += dx * overlap * (1 - scale_factor) * 0.5
                            resolved[j, 1] += dy * overlap * (1 - scale_factor) * 0.5
                            changed = True
            
            # Ensure bounds
            for i in range(len(resolved)):
                x, y, r = resolved[i]
                max_radius = min(x, 1-x, y, 1-y)
                r = min(r, max_radius)
                r = max(0.001, r)
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                resolved[i] = [x, y, r]
                
            if not changed:
                break
                
        return resolved
    
    def check_containment(self, circles: np.ndarray) -> bool:
        """Check if all circles are fully contained in the unit square."""
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        return True
    
    def compute_penalty(self, circles: np.ndarray, generation: int = 0, 
                       total_generations: int = 100, penalty_scale: float = 1.0) -> float:
        """Compute penalty based on constraint violations."""
        penalty = 0.0
        
        # Check containment violations
        for x, y, r in circles:
            # Boundary violations  
            if x - r < 0:
                penalty += (abs(x - r) ** 2) * 10000 * penalty_scale
            elif x + r > 1:
                penalty += (abs(x + r - 1) ** 2) * 10000 * penalty_scale
            if y - r < 0:
                penalty += (abs(y - r) ** 2) * 10000 * penalty_scale
            elif y + r > 1:
                penalty += (abs(y + r - 1) ** 2) * 10000 * penalty_scale
        
        # Check overlap violations
        tree = self.get_quadtree(circles)
        if self.check_overlap_quadtree(circles, tree):
            penalty += 10000000.0 * penalty_scale
            
        return penalty
    
    def evaluate_fitness(self, circles: np.ndarray, generation: int = 0, 
                        total_generations: int = 100, use_penalty: bool = True) -> float:
        """Evaluate fitness of a circle configuration."""
        # If invalid, heavily penalize
        if not self.check_containment(circles) or self.check_overlap_quadtree(circles):
            if use_penalty:
                # Progressive scaling
                penalty_scale = 1.0 + (generation / total_generations) * 5.0
                penalty = self.compute_penalty(circles, generation, total_generations, penalty_scale)
                return -penalty
            else:
                return -float('inf')
        
        # Otherwise, return total radius
        total_radius = np.sum(circles[:, 2])
        return total_radius
    
    def mutate(self, circles: np.ndarray, generation: int, total_generations: int) -> np.ndarray:
        """Mutate a circle configuration with adaptive rates."""
        mutated = circles.copy()
        
        # Adaptive mutation rate using sigmoid decay
        mutation_rate = self.config.mutation_rate_start + (
            self.config.mutation_rate_end - self.config.mutation_rate_start
        ) * (1 / (1 + math.exp(-10 * (generation / total_generations - 0.5))))
        
        n = len(mutated)
        
        # Mutate some circles with different strategies
        for i in range(n):
            if random.random() < mutation_rate:
                # Choose mutation type
                mutation_type = random.choices(
                    ['position', 'radius'], 
                    weights=[0.7, 0.3]
                )[0]
                
                if mutation_type == 'position':
                    # Mutate position with varying strengths
                    strength = 0.05 * (1 - generation / total_generations)
                    mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + random.gauss(0, strength)))
                    mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + random.gauss(0, strength)))
                else:  # radius
                    # Mutate radius
                    mutated[i, 2] = max(0.001, min(0.49, mutated[i, 2] + random.gauss(0, 0.02)))
        
        # Ensure valid configuration after mutation
        mutated = self.enforce_constraints(mutated)
        return mutated
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform crossover between two parent configurations with enhanced recombination."""
        if random.random() > self.config.crossover_prob:
            # Return one of the parents randomly
            return parent1.copy() if random.random() < 0.5 else parent2.copy()
        
        n = len(parent1)
        child = np.zeros_like(parent1)
        
        # Use uniform crossover for better recombination
        for i in range(n):
            # Uniform crossover for each parameter
            if random.random() < 0.5:
                child[i] = parent1[i].copy()
            else:
                child[i] = parent2[i].copy()
            
            # Add some blending for better exploration
            if random.random() < 0.3:  # 30% chance of blending
                alpha = random.random()
                # Blend positions and radii
                child[i][0] = parent1[i][0] + alpha * (parent2[i][0] - parent1[i][0])
                child[i][1] = parent1[i][1] + alpha * (parent2[i][1] - parent1[i][1])
                child[i][2] = parent1[i][2] + alpha * (parent2[i][2] - parent1[i][2])
        
        # Ensure valid configuration after crossover
        child = self.enforce_constraints(child)
        return child
    
    def local_optimization(self, circles: np.ndarray, max_iterations: int = 20) -> np.ndarray:
        """Enhanced local optimization to refine circle placement."""
        optimized = circles.copy()
        
        # Gradient-based refinement with momentum tracking
        velocity = np.zeros_like(optimized)
        
        for iteration in range(max_iterations):
            tree = self.get_quadtree(optimized)
            changed = False
            
            # Update velocities and positions based on forces
            for i in range(len(optimized)):
                x_i, y_i, r_i = optimized[i]
                force_x, force_y = 0, 0
                
                # Find nearby circles and calculate forces
                range_min = (x_i - r_i - 1e-10, x_i + r_i + 1e-10, 
                             y_i - r_i - 1e-10, y_i + r_i + 1e-10)
                nearby_points = tree.query_range(range_min)
                
                for (x_j, y_j), j in nearby_points:
                    if i != j:
                        distance = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                        if distance < (r_i + r_j - self.config.validity_threshold):
                            dx = x_j - x_i
                            dy = y_j - y_i
                            dist = max(self.config.validity_threshold, distance)
                            
                            # Normalize
                            dx /= dist
                            dy /= dist
                            
                            # Repulsive force
                            overlap = (r_i + r_j - dist) * 0.5
                            force_x += dx * overlap * 0.1
                            force_y += dy * overlap * 0.1
                            changed = True
                
                # Boundary forces
                boundary_force = 0.1
                if x_i < r_i:
                    force_x += boundary_force * (r_i - x_i)
                elif x_i > 1 - r_i:
                    force_x += boundary_force * ((1 - r_i) - x_i)
                    
                if y_i < r_i:
                    force_y += boundary_force * (r_i - y_i)
                elif y_i > 1 - r_i:
                    force_y += boundary_force * ((1 - r_i) - y_i)
                
                # Update velocity with momentum
                momentum = 0.8
                velocity[i, 0] = momentum * velocity[i, 0] + 0.1 * force_x
                velocity[i, 1] = momentum * velocity[i, 1] + 0.1 * force_y
                
                # Update positions
                optimized[i, 0] += velocity[i, 0]
                optimized[i, 1] += velocity[i, 1]
            
            # Ensure bounds
            for i in range(len(optimized)):
                x, y, r = optimized[i]
                max_radius = min(x, 1-x, y, 1-y)
                r = min(r, max_radius)
                r = max(0.001, r)
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                optimized[i] = [x, y, r]
                
            if not changed:
                break
                
        return optimized
    
    def tournament_selection(self, population: List[np.ndarray], 
                           fitnesses: List[float], tournament_size: int) -> np.ndarray:
        """Select parent using tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index]
    
    def simulated_annealing(self, circles: np.ndarray, temperature: float) -> np.ndarray:
        """Simulated annealing approach for global optimization."""
        current = circles.copy()
        current_fitness = self.evaluate_fitness(current)
        
        # Try several random moves
        for _ in range(20):
            # Create candidate by slightly mutating
            candidate = self.mutate(current, 0, 1)  # dummy params
            candidate_fitness = self.evaluate_fitness(candidate)
            
            # Accept or reject based on SA criteria
            if candidate_fitness > current_fitness:
                current = candidate
                current_fitness = candidate_fitness
            else:
                # Accept with probability based on temperature and difference
                delta = current_fitness - candidate_fitness
                if random.random() < math.exp(-delta / max(temperature, 1e-10)):
                    current = candidate
                    current_fitness = candidate_fitness
        
        return current
    
    def evolve(self) -> np.ndarray:
        """Run the hybrid evolutionary optimization algorithm."""
        n = 26
        population = self.initialize_population(n, self.config.population_size)
        
        # Evaluate initial population
        fitnesses = [self.evaluate_fitness(individual) for individual in population]
        
        # Evolution loop
        for gen in range(self.config.generations):
            # Perform hierarchical optimization
            new_population = []
            
            for individual in population:
                # Level 1: Simulated Annealing for global exploration
                if gen < self.config.generations * 0.3:  # Early generations
                    temp = self.config.sa_initial_temp * (self.config.sa_cooling_rate ** gen)
                    individual = self.simulated_annealing(individual, temp)
                else:
                    # Level 2: Local optimization for exploitation
                    individual = self.local_optimization(individual, 10)
                
                # Level 3: Evolutionary refinement
                new_population.append(individual)
            
            # Selection, crossover, and mutation
            for _ in range(self.config.population_size // 2):
                # Tournament selection
                parent1 = self.tournament_selection(new_population, fitnesses, self.config.tournament_size)
                parent2 = self.tournament_selection(new_population, fitnesses, self.config.tournament_size)
                
                # Crossover
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                
                # Mutation
                child1 = self.mutate(child1, gen, self.config.generations)
                child2 = self.mutate(child2, gen, self.config.generations)
                
                # Local optimization
                child1 = self.local_optimization(child1)
                child2 = self.local_optimization(child2)
                
                # Add to population
                new_population.extend([child1, child2])
            
            # Trim to exact size
            new_population = new_population[:self.config.population_size]
            
            # Evaluate new population
            population = new_population
            fitnesses = [self.evaluate_fitness(individual, gen, self.config.generations) 
                        for individual in population]
            
            # Print progress
            best_fitness = max(fitnesses)
            if gen % 25 == 0:
                print(f"Generation {gen}: Best fitness = {best_fitness}")
        
        # Return the best individual
        best_index = np.argmax(fitnesses)
        best_solution = population[best_index]
        
        # Final refinement
        best_solution = self.local_optimization(best_solution, 30)
        best_solution = self.resolve_overlaps(best_solution, 10)
        
        return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    config = OptimizerConfig()
    packer = CirclePacker(config)
    return packer.evolve()

# EVOLVE-BLOCK-END