# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import Voronoi, Delaunay, cKDTree
from scipy.spatial.distance import cdist
import time
from numba import jit

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class VoronoiCirclePacker:
    """Optimizes placement of 26 non-overlapping circles using Voronoi-inspired evolutionary approach."""
    
    def __init__(self, n_circles=26, pop_size=80, gen_count=60, mutpb=0.2, cxpb=0.6):
        self.N_CIRCLES = n_circles
        self.POP_SIZE = pop_size
        self.GEN_COUNT = gen_count
        self.MUTPB = mutpb
        self.CXPB = cxpb
        self.BENCHMARK = 2.6358627564136983
        self._setup_deap()
        
    def _setup_deap(self):
        """Initialize DEAP framework components."""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
    def _create_voronoi_initialization(self):
        """Create initial configuration using Voronoi-based approach with Delaunay triangulation."""
        # Generate points using a modified grid pattern to ensure good distribution
        points = []
        # Start with a structured grid pattern
        grid_size = int(np.ceil(np.sqrt(self.N_CIRCLES))) + 2
        spacing = 0.9 / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = 0.05 + i * spacing
                y = 0.05 + j * spacing
                if len(points) < self.N_CIRCLES:
                    points.append([x, y])
        
        # Ensure we have enough points
        while len(points) < self.N_CIRCLES:
            # Add random points in case we don't have enough from grid
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            points.append([x, y])
        
        points = points[:self.N_CIRCLES]
        
        # Use Voronoi diagram to get cell centers and determine radii
        vor = Voronoi(points)
        circles = []
        
        for i, (point, vertex_indices) in enumerate(zip(vor.points, vor.point_region)):
            if i >= self.N_CIRCLES:
                break
            
            # Get Voronoi region vertices for this point
            region_vertices = []
            for region_idx in vor.regions[vor.point_region[i]]:
                if region_idx != -1:  # Skip infinite regions
                    region_vertices.append(vor.vertices[region_idx])
            
            # Calculate maximum radius that fits in Voronoi cell
            if len(region_vertices) > 2:
                # Find minimum distance from center to cell boundary
                min_dist = float('inf')
                center = np.array(point)
                
                # Add padding to avoid edge issues
                for vertex in region_vertices:
                    dist = np.linalg.norm(center - np.array(vertex))
                    if dist < min_dist:
                        min_dist = dist
                
                # Use some fraction of this distance as radius
                max_radius = min_dist * 0.4
                
                # Ensure it doesn't exceed boundary constraints
                min_edge_dist = min(point[0], 1-point[0], point[1], 1-point[1])
                radius = min(max_radius, min_edge_dist * 0.8, 0.3)
                radius = max(0.005, min(0.5, radius))
                
                circles.append([point[0], point[1], radius])
            else:
                # Fallback for degenerate cases
                x, y = point
                min_edge_dist = min(x, 1-x, y, 1-y)
                radius = min(min_edge_dist * 0.4, 0.2)
                radius = max(0.005, min(0.5, radius))
                circles.append([x, y, radius])
        
        # If we didn't get enough circles, fill with more points
        while len(circles) < self.N_CIRCLES:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            min_edge_dist = min(x, 1-x, y, 1-y)
            radius = min(min_edge_dist * 0.2, 0.15)
            radius = max(0.005, min(0.5, radius))
            circles.append([x, y, radius])
            
        return np.array(circles[:self.N_CIRCLES])
    
    def _evaluate_fitness(self, individual):
        """Evaluate fitness of circle placement with penalty for constraints."""
        circles = np.array(individual).reshape(-1, 3)
        
        # Extract positions and radii
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Calculate objective (sum of radii)
        total_radius = np.sum(radii)
        
        # Penalty for constraint violations
        penalty = 0
        
        # Check containment constraints
        for i, (pos, r) in enumerate(zip(positions, radii)):
            x, y = pos
            # Circle must be fully inside unit square
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 10000
                
        # Check overlap constraints using efficient spatial data structure
        try:
            tree = cKDTree(positions)
            # Query pairs within reasonable range to reduce computation
            pairs = tree.query_pairs(2.0, p=2)
            # Filter actual overlapping pairs
            for i, j in pairs:
                if i < j:  # Avoid double counting
                    r_i = radii[i]
                    r_j = radii[j]
                    pos_i = positions[i]
                    pos_j = positions[j]
                    dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                    # Add penalty for overlaps
                    if dist < (r_i + r_j):
                        overlap = (r_i + r_j - dist)
                        penalty += 1000 * overlap
        except Exception as e:
            # Fallback to brute force for edge cases
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    pos_i = circles[i, :2]
                    pos_j = circles[j, :2]
                    r_i = circles[i, 2]
                    r_j = circles[j, 2]
                    dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                    if dist < (r_i + r_j):
                        penalty += 1000 * (r_i + r_j - dist)
        
        return (total_radius - penalty,)
    
    def _mutate_circle(self, individual):
        """Mutate circle placement with adaptive parameters."""
        individual_array = np.array(individual).reshape(-1, 3)
        radii = individual_array[:, 2]
        diversity = np.std(radii) / (np.mean(radii) + 1e-8) if np.mean(radii) > 1e-8 else 0
        
        # Adaptive mutation rate based on diversity and generation progress
        adaptive_mutation_rate = self.MUTPB * (1 - min(0.8, diversity))
        
        for i in range(len(individual)):
            if random.random() < adaptive_mutation_rate:
                idx = i % 3
                if idx == 2:  # radius index
                    old_r = individual[i]
                    # More aggressive mutation for radius to encourage growth
                    mutation_strength = 0.02 * (1 + diversity)
                    new_r = old_r + random.gauss(0, mutation_strength)
                    individual[i] = max(0.001, min(0.5, new_r))
                else:  # position indices (x, y)
                    old_val = individual[i]
                    # Larger mutation for positions to encourage exploration
                    mutation_strength = 0.025 * (1 + diversity)
                    new_val = old_val + random.gauss(0, mutation_strength)
                    individual[i] = max(0, min(1, new_val))
        return individual,
    
    def _create_individual(self):
        """Create a random valid individual using Voronoi initialization."""
        # Start with Voronoi-based configuration
        individual = self._create_voronoi_initialization().flatten().tolist()
        
        # Add small random perturbations
        for i in range(len(individual)):
            if i % 3 < 2:  # x or y coordinate
                individual[i] += random.uniform(-0.02, 0.02)
                individual[i] = max(0, min(1, individual[i]))
            else:  # radius
                individual[i] *= random.uniform(0.9, 1.1)
                individual[i] = max(0.001, min(0.5, individual[i]))
        return creator.Individual(individual)
    
    def _voronoi_crossover(self, ind1, ind2):
        """Specialized crossover that works with Voronoi cell relationships."""
        # Perform uniform crossover
        tools.cxUniform(ind1, ind2, indpb=0.5)
        
        # Repair violations by using Voronoi-based approach
        temp_ind = np.array(ind1).reshape(-1, 3)
        
        # Apply Voronoi-based constraint repair for better quality
        # First, get the Voronoi diagram of current configuration 
        positions = temp_ind[:, :2]
        
        # Repair containment
        for i in range(len(temp_ind)):
            x, y, r = temp_ind[i]
            # Keep circles within bounds
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            temp_ind[i] = [x, y, r]
        
        # Repair overlaps using iterative Voronoi-inspired approach
        for _ in range(3):
            improved = False
            for i in range(len(temp_ind)):
                # Try to expand radius safely in Voronoi context
                orig_r = temp_ind[i, 2]
                max_r = min(
                    temp_ind[i, 0], 1 - temp_ind[i, 0],
                    temp_ind[i, 1], 1 - temp_ind[i, 1]
                )
                
                # Only try to expand if there's room
                if max_r > orig_r + 0.001:
                    # Check neighbors in terms of Voronoi proximity
                    neighbors = []
                    for j in range(len(temp_ind)):
                        if i != j:
                            dist = np.sqrt(np.sum((temp_ind[i, :2] - temp_ind[j, :2])**2))
                            if dist < 1.5:  # Close proximity
                                neighbors.append(j)
                    
                    # Try to expand radius if not too close to neighbors
                    new_r = min(max_r, orig_r + 0.01)
                    valid = True
                    
                    # Check against neighbors
                    for j in neighbors:
                        dist = np.sqrt(np.sum((temp_ind[i, :2] - temp_ind[j, :2])**2))
                        if dist < (new_r + temp_ind[j, 2]):
                            valid = False
                            break
                    
                    if valid and new_r > orig_r:
                        temp_ind[i, 2] = new_r
                        improved = True
            
            if not improved:
                break
        
        # Return repaired individual
        ind1[:] = temp_ind.flatten()
        return ind1, ind2

    def _voronoi_local_optimization(self, circles):
        """Apply Voronoi-inspired local optimization."""
        # Work with Voronoi relationships
        for iteration in range(8):
            improved = False
            
            # Strategy 1: Radius maximization with Voronoi awareness
            for i in range(len(circles)):
                orig_r = circles[i, 2]
                # Maximum safe radius
                max_r = min(
                    circles[i, 0], 1 - circles[i, 0],
                    circles[i, 1], 1 - circles[i, 1]
                )
                
                if max_r > orig_r + 0.001:
                    # Binary search for best radius
                    low = 0
                    high = max_r - orig_r
                    best_r = orig_r
                    
                    for _ in range(10):
                        test_r = (low + high) / 2
                        test_r = min(test_r, high)
                        
                        # Check validity
                        valid = True
                        test_pos = circles[i, :2]
                        
                        for j in range(len(circles)):
                            if i != j:
                                pos_j = circles[j, :2]
                                r_j = circles[j, 2]
                                dist = np.sqrt(np.sum((test_pos - pos_j)**2))
                                if dist < (orig_r + test_r + r_j):
                                    valid = False
                                    break
                        
                        if valid:
                            best_r = orig_r + test_r
                            low = test_r
                        else:
                            high = test_r
                    
                    if best_r > orig_r:
                        circles[i, 2] = best_r
                        improved = True
            
            # Strategy 2: Position adjustment based on Voronoi influence
            if not improved:
                for i in range(len(circles)):
                    orig_pos = circles[i, :2].copy()
                    best_pos = orig_pos.copy()
                    best_r = circles[i, 2]
                    
                    # Check nearby positions based on Voronoi cell proximity
                    candidates = []
                    for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                        for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                            candidates.append((dx, dy))
                    
                    random.shuffle(candidates)
                    
                    for dx, dy in candidates[:8]:  # Test fewer candidates to save time
                        test_x = max(0.01, min(0.99, circles[i, 0] + dx))
                        test_y = max(0.01, min(0.99, circles[i, 1] + dy))
                        
                        valid = True
                        test_pos = [test_x, test_y]
                        
                        for j in range(len(circles)):
                            if i != j:
                                pos_j = circles[j, :2]
                                r_j = circles[j, 2]
                                dist = np.sqrt((test_x - pos_j[0])**2 + (test_y - pos_j[1])**2)
                                if dist < (best_r + r_j):
                                    valid = False
                                    break
                        
                        if valid:
                            score = best_r  # Just consider radius
                            if score > best_r:
                                best_r = score
                                best_pos = [test_x, test_y]
                    
                    if not np.array_equal(best_pos, orig_pos):
                        circles[i, :2] = best_pos
                        improved = True
            
            if not improved:
                break
                
        return circles

    def optimize(self):
        """Main optimization routine using Voronoi-inspired evolutionary approach."""
        # Initialize toolbox
        toolbox = base.Toolbox()
        toolbox.register("individual", self._create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_fitness)
        toolbox.register("mate", self._voronoi_crossover)
        toolbox.register("mutate", self._mutate_circle)
        toolbox.register("select", tools.selTournament, tournsize=4)

        # Create initial population
        population = toolbox.population(n=self.POP_SIZE)

        # Run evolution
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        try:
            population, logbook = algorithms.eaSimple(
                population, toolbox, cxpb=self.CXPB, mutpb=self.MUTPB,
                ngen=self.GEN_COUNT, stats=stats, halloffame=hof, verbose=False
            )
        except Exception as e:
            print(f"Evolution error: {e}")
            return self._fallback_heuristic()

        # Return best solution
        best_individual = hof[0]
        result = np.array(best_individual).reshape(-1, 3)
        
        # Apply Voronoi-inspired local optimization to refine further
        refined_result = self._voronoi_local_optimization(result.copy())
        
        return refined_result
    
    def _fallback_heuristic(self):
        """Fallback method using structured approach."""
        # Simple grid-based arrangement with refinement
        n = self.N_CIRCLES
        circles = np.zeros((n, 3))

        # Try a hexagonal packing pattern
        rows = 5
        cols = 5
        if n < rows * cols:
            rows = int(np.ceil(n / cols))

        # Create regular grid points
        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)

        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Set reasonable initial radius
                r = min(spacing_x, spacing_y) * 0.4
                circles[count] = [x, y, r]
                count += 1

        # Refine positions to avoid overlaps
        for _ in range(50):
            improved = False
            for i in range(n):
                best_pos = circles[i, :2].copy()
                best_rad = circles[i, 2]
                best_score = -1000

                # Check nearby positions
                for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                        test_x = max(0.01, min(0.99, circles[i, 0] + dx))
                        test_y = max(0.01, min(0.99, circles[i, 1] + dy))
                        test_r = circles[i, 2]

                        valid = True
                        for j in range(n):
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

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    optimizer = VoronoiCirclePacker()
    result = optimizer.optimize()
    
    # Validation and reporting
    total_radius = np.sum(result[:, 2])
    benchmark_ratio = total_radius / optimizer.BENCHMARK
    
    print(f"Total evaluation time: {time.time() - start_time:.2f}s")
    print(f"Sum of radii: {total_radius:.6f}")
    print(f"Benchmark ratio: {benchmark_ratio:.6f}")
    
    return result

# EVOLVE-BLOCK-END