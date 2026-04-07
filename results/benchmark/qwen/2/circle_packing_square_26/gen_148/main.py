# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from typing import Tuple, List, Optional
import random
import time
from copy import deepcopy

# Global constants
POPULATION_SIZE = 80
GENERATIONS = 60
MUTATION_RATE_INITIAL = 0.12
CROSSOVER_RATE = 0.85
TOURNAMENT_SIZE = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

class PhysicsBasedCirclePackingOptimizer:
    def __init__(self):
        self.n_circles = 26
        self.max_iterations = 100
        self.force_iteration_limit = 50
        self.epsilon = 1e-8
        
    def is_valid_configuration(self, circles: np.ndarray) -> bool:
        """Check if the configuration satisfies all constraints."""
        if len(circles) != self.n_circles:
            return False
            
        # Check containment constraints using vectorized operations
        radii = circles[:, 2]
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        
        # Check if any radius violates containment
        containment_check = (
            (radii <= x_coords) & 
            (radii <= y_coords) & 
            (radii <= 1 - x_coords) & 
            (radii <= 1 - y_coords)
        )
        
        if not np.all(containment_check):
            return False

        # Check overlap constraints using pairwise distance matrix
        if self.n_circles > 1:
            distances = cdist(circles[:, :2], circles[:, :2])
            # Create upper triangular mask to avoid duplicate comparisons
            mask = np.triu(np.ones((self.n_circles, self.n_circles), dtype=bool), k=1)
            
            # Calculate minimum required distance
            min_distances = (circles[:, 2][:, np.newaxis] + circles[:, 2][np.newaxis, :]) * mask
            
            # Check for overlaps
            overlaps = distances < min_distances
            if np.any(overlaps):
                return False
                
        return True
    
    def calculate_sum_radii(self, circles: np.ndarray) -> float:
        """Calculate the sum of all radii."""
        return np.sum(circles[:, 2])
    
    def initialize_population(self, pop_size: int) -> List[np.ndarray]:
        """Initialize population using physics-inspired Voronoi-based approach."""
        population = []
        
        # Create diverse initial configurations using Voronoi and physics principles
        for i in range(pop_size):
            if i == 0:
                # Physics-inspired Voronoi initialization 
                circles = self.create_physics_initialization()
            elif i < pop_size // 3:
                # Random with physics-based constraint validation
                circles = self.create_constrained_random_initialization()
            else:
                # Modified Voronoi with physics-based refinement
                circles = self.create_refined_voronoi_initialization()
            
            # Ensure validity
            if self.is_valid_configuration(circles):
                population.append(circles.copy())
            else:
                # Fallback to valid configuration
                circles = self.create_simple_initialization()
                if self.is_valid_configuration(circles):
                    population.append(circles.copy())
                    
        return population
    
    def create_voronoi_points(self, n_points: int) -> np.ndarray:
        """Generate Voronoi points using hexagonal packing to ensure good distribution."""
        points = []
        rows = int(np.ceil(np.sqrt(n_points)))
        cols = int(np.ceil(n_points / rows))
        
        # Hexagonal spacing 
        spacing = 1.0 / (max(rows, cols) + 2)
        hex_height = spacing * np.sqrt(3) / 2
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                x = (j + 0.5 + (i % 2) * 0.5) * spacing
                y = (i + 0.5) * hex_height
                if x <= 1 and y <= 1:
                    points.append([x, y])
        
        # Trim to exact number needed and add jitter
        points = points[:n_points]
        for point in points:
            point[0] += np.random.uniform(-spacing/6, spacing/6)
            point[1] += np.random.uniform(-spacing/6, spacing/6)
        
        # Ensure bounds
        points = [[max(0.01, min(0.99, p[0])), max(0.01, min(0.99, p[1]))] for p in points]
        return np.array(points)
    
    def create_physics_initialization(self) -> np.ndarray:
        """Create initial configuration using Voronoi with physics-inspired refinement."""
        # Generate points using hexagonal pattern
        points = self.create_voronoi_points(self.n_circles + 10)
        
        # Create Voronoi diagram
        try:
            vor = Voronoi(points)
        except:
            # Fallback to simple initialization if Voronoi fails
            return self.create_simple_initialization()
        
        # Select points for circle centers with physics-based radius estimation
        circles = np.zeros((self.n_circles, 3))
        
        # Use first n_circles points from Voronoi
        valid_indices = list(range(min(self.n_circles, len(vor.points))))
        
        for i, idx in enumerate(valid_indices):
            center = vor.points[idx]
            x, y = center
            
            # Estimate radius based on Voronoi cell density and boundary constraints
            # Simplified: use distance to nearest neighbors, but also respect boundaries
            if len(vor.points) > 1:
                # Find nearby points to estimate cell area
                distances = np.sqrt(np.sum((vor.points - center)**2, axis=1))
                distances = distances[distances > 0]  # Exclude self
                if len(distances) > 0:
                    avg_distance = np.mean(distances)
                    estimated_radius = avg_distance / 3.0
                else:
                    estimated_radius = 0.1
            else:
                estimated_radius = 0.1
                
            # Respect boundary constraints
            min_dist_to_boundary = min(x, y, 1-x, 1-y)
            final_radius = min(estimated_radius, min_dist_to_boundary * 0.8)
            final_radius = max(0.01, min(final_radius, 0.2))  # Reasonable bounds
            
            circles[i] = [x, y, final_radius]
        
        # Apply physics-inspired optimization to initial configuration
        circles = self.physics_relaxation(circles)
        return circles
    
    def create_constrained_random_initialization(self) -> np.ndarray:
        """Create random initialization with better constraint handling."""
        circles = np.zeros((self.n_circles, 3))
        
        for i in range(self.n_circles):
            attempts = 0
            while attempts < 100:
                # Random placement
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                
                # Radius based on distance to closest boundary
                min_dist = min(x, y, 1-x, 1-y)
                r = np.random.uniform(0.01, min_dist/3)
                
                # Check overlap with existing circles
                overlap = False
                for j in range(i):
                    existing_x, existing_y, existing_r = circles[j]
                    dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                    if dist < r + existing_r:
                        overlap = True
                        break
                
                if not overlap:
                    circles[i] = [x, y, r]
                    break
                attempts += 1
                
            if attempts >= 100:
                # Fallback to grid if no valid position found
                grid_size = int(np.ceil(np.sqrt(self.n_circles)))
                spacing = 1.0 / (grid_size + 1)
                row = i // grid_size
                col = i % grid_size
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 4
                circles[i] = [x, y, r]
                
        return circles
    
    def create_refined_voronoi_initialization(self) -> np.ndarray:
        """Create refined Voronoi initialization with additional physics treatment."""
        # Start with Voronoi approach
        circles = self.create_physics_initialization()
        
        # Apply some initial physics relaxation
        circles = self.physics_relaxation(circles, iterations=20)
        
        # Add some randomness to diversify the population
        for i in range(self.n_circles):
            if np.random.random() < 0.3:
                # Slight perturbation
                circles[i][0] += np.random.uniform(-0.02, 0.02)
                circles[i][1] += np.random.uniform(-0.02, 0.02)
                circles[i][2] *= np.random.uniform(0.95, 1.05)
                
                # Clamp to boundaries
                circles[i][0] = np.clip(circles[i][0], 0, 1)
                circles[i][1] = np.clip(circles[i][1], 0, 1)
                circles[i][2] = np.clip(circles[i][2], 0.01, 0.5)
        
        return circles
    
    def create_simple_initialization(self) -> np.ndarray:
        """Create simple grid layout."""
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
                r = spacing / 4
                circles[idx] = [x, y, r]
                idx += 1
                
        return circles
    
    def physics_force_calculation(self, circles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate net forces on each circle due to overlaps and boundaries."""
        n = len(circles)
        forces_x = np.zeros(n)
        forces_y = np.zeros(n)
        
        # Boundary forces (repel from edges)
        for i in range(n):
            x, y, r = circles[i]
            # Force away from boundaries
            fx = 0.0
            fy = 0.0
            
            # Repel from left boundary
            if x <= r:
                fx += (r - x) * 100
            # Repel from right boundary  
            if x >= 1 - r:
                fx -= (x - (1 - r)) * 100
            # Repel from bottom boundary
            if y <= r:
                fy += (r - y) * 100
            # Repel from top boundary
            if y >= 1 - r:
                fy -= (y - (1 - r)) * 100
                
            forces_x[i] = fx
            forces_y[i] = fy
            
        # Overlap forces (repel from other circles)
        for i in range(n):
            x_i, y_i, r_i = circles[i]
            for j in range(i+1, n):
                x_j, y_j, r_j = circles[j]
                
                # Calculate distance
                dx = x_i - x_j
                dy = y_i - y_j
                dist = np.sqrt(dx*dx + dy*dy) + self.epsilon
                
                # If circles overlap or are too close
                if dist < (r_i + r_j):
                    # Repulsive force proportional to overlap
                    force_magnitude = 1.0 / (dist * dist + self.epsilon)
                    
                    # Normalize direction vector
                    fx = force_magnitude * dx / dist
                    fy = force_magnitude * dy / dist
                    
                    # Apply to both circles  
                    forces_x[i] += fx
                    forces_y[i] += fy
                    forces_x[j] -= fx
                    forces_y[j] -= fy
                    
        return forces_x, forces_y
    
    def physics_relaxation(self, circles: np.ndarray, iterations: int = None) -> np.ndarray:
        """Apply physics-inspired relaxation to remove overlaps and improve configuration."""
        if iterations is None:
            iterations = self.force_iteration_limit
            
        circles = circles.copy()
        n = len(circles)
        
        # Simple force-based relaxation
        for _ in range(iterations):
            # Calculate forces
            forces_x, forces_y = self.physics_force_calculation(circles)
            
            # Apply forces with damping
            damping = 0.1
            
            for i in range(n):
                # Limit force magnitude to prevent large jumps
                force_magnitude = np.sqrt(forces_x[i]**2 + forces_y[i]**2)
                if force_magnitude > 10.0:
                    forces_x[i] = forces_x[i] * 10.0 / force_magnitude
                    forces_y[i] = forces_y[i] * 10.0 / force_magnitude
                    
                # Apply force to position
                circles[i][0] += forces_x[i] * damping
                circles[i][1] += forces_y[i] * damping
                
                # Clamp to boundaries
                circles[i][0] = np.clip(circles[i][0], circles[i][2], 1 - circles[i][2])
                circles[i][1] = np.clip(circles[i][1], circles[i][2], 1 - circles[i][2])
                
            # Check if we're making progress
            if not self.is_valid_configuration(circles):
                # If violated, try to recover by shrinking radii slightly
                for i in range(n):
                    # Shrink radii slightly to make room
                    circles[i][2] *= 0.99
                
                # Clamp radii to minimum
                circles[:, 2] = np.maximum(circles[:, 2], 0.01)
                
        return circles
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Physics-inspired crossover with constraint-awareness."""
        if np.random.random() > CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()

        n = len(parent1)
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        # Physics-aware crossover: combine points using Voronoi-like logic
        # but prioritize configurations that maintain physical feasibility
        
        # For each circle, decide whether to take from parent1 or parent2
        for i in range(n):
            # Determine which parent is better for this circle
            parent1_radius = parent1[i][2]
            parent2_radius = parent2[i][2]
            
            # Prefer parent with larger radius if both are valid
            # Otherwise, random selection with preference for better fitness
            if np.random.random() < 0.5:
                child1[i] = parent1[i].copy()
                child2[i] = parent2[i].copy()
            else:
                child1[i] = parent2[i].copy()
                child2[i] = parent1[i].copy()
        
        # Apply physics relaxation to ensure validity
        child1 = self.physics_relaxation(child1)
        child2 = self.physics_relaxation(child2)
        
        return child1, child2

    def mutate(self, circles: np.ndarray, mutation_rate: float = MUTATION_RATE_INITIAL) -> np.ndarray:
        """Physics-informed mutation."""
        mutated = circles.copy()
        n = len(mutated)

        # Physics-based mutation: apply small physics forces to perturb circles
        for i in range(n):
            if np.random.random() < mutation_rate:
                # Apply small random displacements
                dx = np.random.normal(0, 0.02)
                dy = np.random.normal(0, 0.02)
                dr = np.random.normal(0, 0.005)
                
                # Apply displacement
                mutated[i][0] += dx
                mutated[i][1] += dy
                mutated[i][2] += dr
                
                # Clamp to bounds
                mutated[i][0] = np.clip(mutated[i][0], mutated[i][2], 1 - mutated[i][2])
                mutated[i][1] = np.clip(mutated[i][1], mutated[i][2], 1 - mutated[i][2])
                mutated[i][2] = np.clip(mutated[i][2], 0.01, 0.5)
                
        # Apply physics relaxation after mutation
        mutated = self.physics_relaxation(mutated)
        return mutated

    def select_tournament(self, population: List[np.ndarray], fitnesses: List[float], 
                         tournament_size: int = TOURNAMENT_SIZE) -> int:
        """Select an individual using tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return winner_index

    def compute_fitness(self, circles: np.ndarray) -> float:
        """Compute fitness with penalty for invalid configurations."""
        if self.is_valid_configuration(circles):
            return self.calculate_sum_radii(circles)
        else:
            return 0.0

    def run_evolution(self) -> np.ndarray:
        """Run the physics-based evolutionary algorithm."""
        # Initialize population
        population = self.initialize_population(POPULATION_SIZE)
        
        if not population:
            # Fallback to simple initialization
            return self.create_simple_initialization()
            
        best_solution = None
        best_fitness = -1

        for generation in range(GENERATIONS):
            # Adjust mutation rate based on generation (adaptive)
            mutation_rate = max(MUTATION_RATE_INITIAL * (1 - generation / GENERATIONS), 0.01)
            
            # Evaluate fitness for all individuals
            fitnesses = [self.compute_fitness(circles) for circles in population]
            
            # Track best solution
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()

            # Create new population
            new_population = []

            # Elitism: keep best individual
            new_population.append(best_solution.copy())

            # Generate offspring
            while len(new_population) < POPULATION_SIZE:
                # Tournament selection
                parent1_idx = self.select_tournament(population, fitnesses)
                parent2_idx = self.select_tournament(population, fitnesses)

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutation
                child1 = self.mutate(child1, mutation_rate)
                child2 = self.mutate(child2, mutation_rate)

                # Add children to new population
                new_population.extend([child1, child2])

            # Trim population to exact size
            population = new_population[:POPULATION_SIZE]

        # Return the best solution found
        if best_solution is None:
            # Fallback to a simple configuration if nothing worked
            return self.create_simple_initialization()

        return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = PhysicsBasedCirclePackingOptimizer()
    return optimizer.run_evolution()

# EVOLVE-BLOCK-END