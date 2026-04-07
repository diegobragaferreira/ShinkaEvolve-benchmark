# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi, Delaunay, KDTree
import time
from typing import Tuple, List
import random

class CirclePackingOptimizer:
    """
    An evolutionary optimizer for placing 26 non-overlapping circles in a unit square
    to maximize the sum of their radii.
    """
    
    def __init__(self):
        # Configuration parameters
        self.initial_population_size = 200
        self.max_generations = 500
        self.tournament_size = 5
        self.initial_mutation_rate = 0.15
        self.elitism_count = 10
        self.min_mutation_rate = 0.02
        self.boundary_margin = 0.01
        self.n_circles = 26
        self.eval_timeout_seconds = 55
        
    def validate_configuration(self, circles: np.ndarray) -> bool:
        """
        Check if a configuration of circles is valid (no overlaps, fully contained).
        """
        n = len(circles)
        
        # Check containment constraints
        for i in range(n):
            x, y, r = circles[i]
            if (r <= 0 or x < r + self.boundary_margin or 
                x > 1-r - self.boundary_margin or y < r + self.boundary_margin or 
                y > 1-r - self.boundary_margin):
                return False

        # Check overlap constraints with early termination
        try:
            # Use KDTree for efficient overlap checking
            points = circles[:, :2]
            radii = circles[:, 2]
            tree = KDTree(points)
            pairs = tree.query_pairs(0, return_distance=False)
            
            # Check pairs with early termination
            for i, j in pairs:
                if i < j:  # Avoid checking same pair twice
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2 + 1e-8:  # Small epsilon for numerical stability
                        return False
        except Exception:
            # Fallback to brute force if KDTree fails
            for i in range(n):
                x1, y1, r1 = circles[i]
                for j in range(i+1, n):
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2 + 1e-8:
                        return False

        return True

    def evaluate_fitness(self, circles: np.ndarray) -> float:
        """Evaluate fitness as the sum of radii."""
        return np.sum(circles[:, 2])

    def create_voronoi_initialization(self, n_circles: int) -> np.ndarray:
        """Create initial configuration using enhanced Voronoi-based spreading."""
        # Generate points using a sophisticated approach combining multiple strategies
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(grid_points) < n_circles:
                    x = (j + 0.5) / grid_size
                    y = (i + 0.5) / grid_size
                    grid_points.append([x, y])

        points = np.array(grid_points)

        # Add more sophisticated randomness and boundary points for better distribution
        noise_level = 0.05
        points += np.random.uniform(-noise_level, noise_level, points.shape)

        # Add boundary points to encourage better edge coverage
        boundary_points = []
        for _ in range(20):  # More boundary points for better coverage
            side = np.random.randint(0, 4)
            if side == 0:  # Top
                boundary_points.append([np.random.rand(), 1.0 - self.boundary_margin])
            elif side == 1:  # Bottom
                boundary_points.append([np.random.rand(), self.boundary_margin])
            elif side == 2:  # Left
                boundary_points.append([self.boundary_margin, np.random.rand()])
            else:  # Right
                boundary_points.append([1.0 - self.boundary_margin, np.random.rand()])

        points = np.vstack([points, boundary_points])

        # Clip points to ensure they're within bounds
        points = np.clip(points, self.boundary_margin, 1 - self.boundary_margin)

        # Compute Voronoi diagram
        try:
            vor = Voronoi(points)
            # Use Voronoi cell centers as initial circle positions
            centroids = vor.points[vor.point_region[:-1]]  # Exclude infinite region

            # Limit to number of circles needed
            if len(centroids) >= n_circles:
                selected_centroids = centroids[:n_circles]
            else:
                # If we don't have enough centroids, extend with additional points
                selected_centroids = centroids
                # Add more points in remaining space
                remaining = n_circles - len(selected_centroids)
                for _ in range(remaining):
                    # Add points based on Delaunay triangulation to maintain good spacing
                    try:
                        delaunay = Delaunay(selected_centroids)
                        # Sample points within the convex hull
                        centroid = np.mean(selected_centroids, axis=0)
                        # Add points with some offset from existing points
                        new_point = centroid + np.random.normal(0, 0.1, 2)
                        new_point = np.clip(new_point, self.boundary_margin, 1 - self.boundary_margin)
                        selected_centroids = np.vstack([selected_centroids, new_point])
                    except:
                        # Fallback to random point
                        new_point = np.random.uniform(self.boundary_margin, 1 - self.boundary_margin, 2)
                        selected_centroids = np.vstack([selected_centroids, new_point])
                selected_centroids = selected_centroids[:n_circles]

            # Create circles with initial radii that balance density and spacing
            circles = np.zeros((n_circles, 3))
            for i in range(n_circles):
                x, y = selected_centroids[i]

                # Calculate the minimum distance to any other point to determine appropriate initial radius
                distances = np.sqrt(np.sum((selected_centroids - [x, y])**2, axis=1))
                distances = distances[distances > 0]  # Exclude self-distance

                if len(distances) > 0:
                    # Use the median distance to determine radius, but scale appropriately
                    # This avoids very small radii that could lead to poor packing
                    min_distance = np.min(distances)
                    # Scale by packing density considerations for circle packings
                    radius = min(min_distance * 0.3, 0.2)
                else:
                    radius = 0.1

                # Ensure it's within bounds
                radius = min(radius, x - self.boundary_margin, 1 - x - self.boundary_margin,
                            y - self.boundary_margin, 1 - y - self.boundary_margin)

                # Use minimum radius to avoid degenerate cases
                circles[i] = [x, y, max(radius, 0.005)]

            return circles
        except Exception as e:
            # Fallback to random initialization if Voronoi fails
            return self.generate_random_initialization(n_circles)

    def generate_random_initialization(self, n_circles: int) -> np.ndarray:
        """Generate random initial configuration with improved strategy."""
        circles = np.zeros((n_circles, 3))

        # Try multiple attempts to place valid circles with better heuristics
        max_attempts = 2000
        for attempt in range(max_attempts):
            success = True
            circles = np.zeros((n_circles, 3))

            for i in range(n_circles):
                # Try to place circle without overlap
                placed = False
                inner_attempts = 0
                max_inner = 100

                while not placed and inner_attempts < max_inner:
                    x = np.random.uniform(self.boundary_margin, 1 - self.boundary_margin)
                    y = np.random.uniform(self.boundary_margin, 1 - self.boundary_margin)

                    # Try different radii sizes - biased towards smaller radii to avoid conflicts
                    r = np.random.uniform(0.001, 0.15)

                    # Check if it fits with existing circles
                    valid = True
                    for j in range(i):
                        prev_x, prev_y, prev_r = circles[j]
                        distance_squared = (x - prev_x)**2 + (y - prev_y)**2
                        min_distance_squared = (r + prev_r)**2

                        if distance_squared < min_distance_squared:
                            valid = False
                            break

                    # Check containment
                    if valid and (r > x or r > 1-x or r > y or r > 1-y):
                        valid = False

                    if valid:
                        circles[i] = [x, y, r]
                        placed = True
                    else:
                        inner_attempts += 1

                if not placed:
                    success = False
                    break

            if success:
                return circles

        # If we couldn't place circles, use a simpler method
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        spacing = 1.0 / grid_size
        r = spacing * 0.3

        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count < n_circles:
                    x = (j + 0.5) * spacing
                    y = (i + 0.5) * spacing
                    # Adjust for boundary constraints
                    x = np.clip(x, r + self.boundary_margin, 1 - r - self.boundary_margin)
                    y = np.clip(y, r + self.boundary_margin, 1 - r - self.boundary_margin)
                    circles[count] = [x, y, r]
                    count += 1

        return circles

    def initialize_population(self, pop_size: int, n_circles: int) -> List[np.ndarray]:
        """Create an initial population of valid circle configurations."""
        population = []

        # Use Voronoi-based initialization for first few individuals
        for i in range(min(50, pop_size)):
            circles = self.create_voronoi_initialization(n_circles)
            if self.validate_configuration(circles):
                population.append(circles)

        # Fill up with random initializations
        while len(population) < pop_size:
            circles = self.generate_random_initialization(n_circles)
            if self.validate_configuration(circles):
                population.append(circles)

        return population

    def tournament_selection(self, population: List[np.ndarray], fitnesses: List[float], tournament_size: int) -> np.ndarray:
        """Select an individual using tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parent configurations."""
        if np.random.random() > 0.8:  # Lower crossover rate for better preservation
            return parent1.copy(), parent2.copy()

        n = len(parent1)
        crossover_point = np.random.randint(1, n)

        child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])

        # Ensure children are valid
        child1 = self.enforce_boundaries(child1)
        child2 = self.enforce_boundaries(child2)

        return child1, child2

    def enforce_boundaries(self, circles: np.ndarray) -> np.ndarray:
        """Ensure circles respect boundary constraints."""
        result = circles.copy()
        for i in range(len(result)):
            x, y, r = result[i]
            # Clip position to stay within boundaries
            x = np.clip(x, r + self.boundary_margin, 1 - r - self.boundary_margin)
            y = np.clip(y, r + self.boundary_margin, 1 - r - self.boundary_margin)
            result[i] = [x, y, r]
        return result

    def mutate(self, circles: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Mutate a circle configuration with improved strategy."""
        mutated = circles.copy()

        # Mutate each circle with some probability
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Decide what to mutate with bias towards position
                mutation_type = np.random.choice(['position', 'radius'], p=[0.7, 0.3])

                if mutation_type == 'position':
                    # Slightly perturb position with bounded adjustment
                    dx = np.random.normal(0, 0.015)
                    dy = np.random.normal(0, 0.015)

                    mutated[i][0] = np.clip(mutated[i][0] + dx,
                                           mutated[i][2] + self.boundary_margin, 1 - mutated[i][2] - self.boundary_margin)
                    mutated[i][1] = np.clip(mutated[i][1] + dy,
                                           mutated[i][2] + self.boundary_margin, 1 - mutated[i][2] - self.boundary_margin)
                else:
                    # Mutate radius with careful bounds
                    dr = np.random.normal(0, 0.01)
                    new_radius = mutated[i][2] + dr
                    # Ensure radius stays positive and reasonable
                    mutated[i][2] = np.clip(new_radius, 0.001, min(0.4, 1 - mutated[i][0], mutated[i][0],
                                                                  1 - mutated[i][1], mutated[i][1]))

        return mutated

    def local_refinement(self, circles: np.ndarray) -> np.ndarray:
        """Apply local refinement to improve solution quality."""
        refined = circles.copy()
        
        # Binary search-based radius optimization for each circle
        for _ in range(20):
            improved = False
            
            for i in range(len(refined)):
                orig_x, orig_y, orig_r = refined[i]
                
                # Find minimum distance to other circles
                min_dist_to_others = float('inf')
                for j in range(len(refined)):
                    if i != j:
                        x2, y2, r2 = refined[j]
                        dist = np.sqrt((orig_x - x2)**2 + (orig_y - y2)**2)
                        min_dist_to_others = min(min_dist_to_others, dist)
                
                # Calculate maximum possible radius with binary search
                max_new_radius = min_dist_to_others - 1e-6 if min_dist_to_others > 1e-6 else orig_r
                
                if max_new_radius > orig_r:
                    # Binary search for optimal radius
                    low, high = orig_r, max_new_radius
                    best_radius = orig_r
                    
                    # Binary search loop
                    for _ in range(12):
                        test_r = (low + high) / 2
                        # Check if this radius is valid
                        temp_circles = refined.copy()
                        temp_circles[i][2] = test_r
                        
                        if self.validate_configuration(temp_circles):
                            best_radius = test_r
                            low = test_r
                        else:
                            high = test_r
                    
                    if best_radius > orig_r:
                        refined[i][2] = best_radius
                        improved = True
                
                # Try slight position adjustments
                for _ in range(2):
                    new_x = orig_x + np.random.uniform(-0.001, 0.001)
                    new_y = orig_y + np.random.uniform(-0.001, 0.001)
                    
                    # Clip to bounds using potential new radius
                    target_r = refined[i][2]
                    new_x = np.clip(new_x, target_r + self.boundary_margin, 1 - target_r - self.boundary_margin)
                    new_y = np.clip(new_y, target_r + self.boundary_margin, 1 - target_r - self.boundary_margin)
                    
                    # Check validity
                    temp_circles = refined.copy()
                    temp_circles[i][0] = new_x
                    temp_circles[i][1] = new_y
                    
                    if self.validate_configuration(temp_circles):
                        refined[i][0] = new_x
                        refined[i][1] = new_y
                        improved = True
                        break
            
            if not improved:
                break
        
        return refined

    def run_optimization(self) -> np.ndarray:
        """Run the complete optimization process."""
        # Initialize population
        population = self.initialize_population(self.initial_population_size, self.n_circles)

        # Remove invalid solutions
        valid_population = [ind for ind in population if self.validate_configuration(ind)]
        if not valid_population:
            # Fallback to simple initialization
            circles = np.zeros((self.n_circles, 3))
            grid_size = int(np.ceil(np.sqrt(self.n_circles)))
            spacing = 1.0 / grid_size
            r = spacing * 0.3
            count = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if count < self.n_circles:
                        x = (j + 0.5) * spacing
                        y = (i + 0.5) * spacing
                        circles[count] = [x, y, r]
                        count += 1
            return circles

        population = valid_population

        start_time = time.time()
        best_solution = None
        best_fitness = -np.inf

        for generation in range(self.max_generations):
            # Calculate adaptive mutation rate
            # Decrease over time to reduce exploration and increase exploitation
            adaptive_mutation_rate = self.max_generations * self.initial_mutation_rate / (self.max_generations + generation * 2)
            adaptive_mutation_rate = max(adaptive_mutation_rate, self.min_mutation_rate)

            # Evaluate fitness for all individuals
            fitnesses = [self.evaluate_fitness(ind) for ind in population]

            # Track best individual
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()

            # Create new population
            new_population = []

            # Elitism: keep best individuals
            elite_indices = np.argsort(fitnesses)[-self.elitism_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # Generate offspring
            while len(new_population) < self.initial_population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitnesses, self.tournament_size)
                parent2 = self.tournament_selection(population, fitnesses, self.tournament_size)

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutation
                child1 = self.mutate(child1, adaptive_mutation_rate)
                child2 = self.mutate(child2, adaptive_mutation_rate)

                # Ensure validity of children
                if self.validate_configuration(child1):
                    new_population.append(child1)
                if len(new_population) < self.initial_population_size and self.validate_configuration(child2):
                    new_population.append(child2)

            population = new_population[:self.initial_population_size]

            # Early stopping check
            if time.time() - start_time > self.eval_timeout_seconds:
                break

        # Apply final refinement to best solution
        if best_solution is not None:
            refined_solution = self.local_refinement(best_solution)
            if self.validate_configuration(refined_solution):
                return refined_solution
        
        # Return the best solution found
        if best_solution is not None:
            return best_solution
        else:
            # Fallback to simple initialization if no good solution was found
            circles = np.zeros((self.n_circles, 3))
            grid_size = int(np.ceil(np.sqrt(self.n_circles)))
            spacing = 1.0 / grid_size
            r = spacing * 0.3
            count = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if count < self.n_circles:
                        x = (j + 0.5) * spacing
                        y = (i + 0.5) * spacing
                        circles[count] = [x, y, r]
                        count += 1
            return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    random.seed(42)
    np.random.seed(42)
    
    optimizer = CirclePackingOptimizer()
    return optimizer.run_optimization()

# EVOLVE-BLOCK-END