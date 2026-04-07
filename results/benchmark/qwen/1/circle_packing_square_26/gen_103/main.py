# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import cKDTree
import time

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class CirclePackingOptimizer:
    """Optimizes placement of 26 non-overlapping circles in unit square to maximize sum of radii."""
    
    def __init__(self, n_circles=26, pop_size=100, gen_count=80, mutpb=0.2, cxpb=0.5):
        self.N_CIRCLES = n_circles
        self.POP_SIZE = pop_size
        self.GEN_COUNT = gen_count
        self.MUTPB = mutpb
        self.CXPB = cxpb
        self._setup_deap()
        
    def _setup_deap(self):
        """Initialize DEAP framework components."""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
    def _generate_golden_spiral_initial_config(self):
        """Generate initial circle configuration using golden spiral for better distribution."""
        circles = np.zeros((self.N_CIRCLES, 3))
        
        # Golden spiral distribution
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(self.N_CIRCLES):
            # Improved spiral layout with better distribution
            r = np.sqrt(i / (self.N_CIRCLES - 1)) if self.N_CIRCLES > 1 else 0
            theta = i * golden_angle
            
            # Convert to Cartesian coordinates in [0.1, 0.9] range to keep margin
            x = 0.4 * r * np.cos(theta) + 0.5
            y = 0.4 * r * np.sin(theta) + 0.5
            
            # Apply small random perturbations for diversity
            x += random.uniform(-0.015, 0.015)
            y += random.uniform(-0.015, 0.015)
            
            # Clamp to valid range
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            
            # Set initial radius based on distance to borders and available space
            min_dist_to_edge = min(x, 1-x, y, 1-y)
            # Base radius calculation with some randomness
            base_radius = min(min_dist_to_edge, 0.15) * random.uniform(0.6, 0.9)
            r = max(0.005, base_radius)
            
            circles[i] = [x, y, r]
            
        return circles
    
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
            # Query pairs within max possible distance to reduce computation
            pairs = tree.query_pairs(2.0, p=2)
            # Filter actual overlapping pairs
            for i, j in pairs:
                r_i = radii[i]
                r_j = radii[j]
                pos_i = positions[i]
                pos_j = positions[j]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                # Add penalty for overlaps
                if dist < (r_i + r_j):
                    overlap = (r_i + r_j - dist)
                    penalty += 1000 * overlap
        except Exception:
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
        
        # Adaptive mutation rate based on diversity
        adaptive_mutation_rate = self.MUTPB * (1 - min(0.8, diversity))
        
        for i in range(len(individual)):
            if random.random() < adaptive_mutation_rate:
                idx = i % 3
                if idx == 2:  # radius index
                    old_r = individual[i]
                    # More aggressive mutation for radius to encourage growth
                    mutation_strength = 0.01 * (1 + diversity)
                    new_r = old_r + random.gauss(0, mutation_strength)
                    individual[i] = max(0.001, min(0.5, new_r))
                else:  # position indices (x, y)
                    old_val = individual[i]
                    # Slightly smaller mutation for positions to maintain stability
                    mutation_strength = 0.015 * (1 + diversity)
                    new_val = old_val + random.gauss(0, mutation_strength)
                    individual[i] = max(0, min(1, new_val))
        return individual,
    
    def _create_individual(self):
        """Create a random valid individual."""
        # Start with golden spiral-based configuration
        individual = self._generate_golden_spiral_initial_config().flatten().tolist()
        
        # Add small random perturbations
        for i in range(len(individual)):
            if i % 3 < 2:  # x or y coordinate
                individual[i] += random.uniform(-0.01, 0.01)
                individual[i] = max(0, min(1, individual[i]))
            else:  # radius
                individual[i] *= random.uniform(0.95, 1.05)
                individual[i] = max(0.001, min(0.5, individual[i]))
        return creator.Individual(individual)
    
    def _crossover_constraint_aware(self, ind1, ind2):
        """Crossover that maintains constraints with repair mechanism."""
        # Perform standard uniform crossover
        tools.cxUniform(ind1, ind2, indpb=0.5)
        
        # Repair violated constraints
        temp_ind = np.array(ind1).reshape(-1, 3)
        
        # Fix containment issues
        for i in range(len(temp_ind)):
            x, y, r = temp_ind[i]
            adjusted = False
            if x - r < 0:
                x = r
                adjusted = True
            elif x + r > 1:
                x = 1 - r
                adjusted = True
            if y - r < 0:
                y = r
                adjusted = True
            elif y + r > 1:
                y = 1 - r
                adjusted = True
            if adjusted:
                temp_ind[i] = [x, y, r]
        
        # Fix overlaps with more sophisticated approach
        for i in range(len(temp_ind)):
            for j in range(i+1, len(temp_ind)):
                pos_i = temp_ind[i, :2]
                pos_j = temp_ind[j, :2]
                r_i = temp_ind[i, 2]
                r_j = temp_ind[j, 2]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    # Apply force-based separation
                    dx = pos_i[0] - pos_j[0]
                    dy = pos_i[1] - pos_j[1]
                    if dx == 0 and dy == 0:
                        # If same position, move randomly
                        angle = random.uniform(0, 2*np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                    dx /= dist
                    dy /= dist
                    step = (r_i + r_j - dist) * 0.3
                    # Apply only to one of them to maintain diversity
                    if random.random() < 0.5:
                        new_x = max(0.01, min(0.99, pos_i[0] + dx * step))
                        new_y = max(0.01, min(0.99, pos_i[1] + dy * step))
                        temp_ind[i] = [new_x, new_y, r_i]
                    else:
                        new_x = max(0.01, min(0.99, pos_j[0] - dx * step))
                        new_y = max(0.01, min(0.99, pos_j[1] - dy * step))
                        temp_ind[j] = [new_x, new_y, r_j]
        
        # Return repaired individuals
        ind1[:] = temp_ind.flatten()
        return ind1, ind2

    def _improved_local_optimization(self, circles):
        """Apply advanced local optimization to refine solution."""
        # Try a more systematic approach with multiple optimization passes
        for iteration in range(15):
            improved = False
            
            # Pass 1: Try to increase each circle's radius
            for i in range(len(circles)):
                original_r = circles[i, 2]
                # Calculate maximum possible increase
                max_increase = min(
                    circles[i, 0], 1 - circles[i, 0],
                    circles[i, 1], 1 - circles[i, 1]
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
                        test_r_new = original_r + test_r
                        
                        valid = True
                        test_pos = circles[i, :2]
                        
                        # Check overlap with other circles
                        for j in range(len(circles)):
                            if i != j:
                                pos_j = circles[j, :2]
                                r_j = circles[j, 2]
                                dist = np.sqrt(np.sum((test_pos - pos_j)**2))
                                if dist < (test_r_new + r_j):
                                    valid = False
                                    break
                        
                        if valid:
                            best_radius = original_r + test_r
                            low = test_r
                        else:
                            high = test_r
                    
                    if best_radius > original_r:
                        circles[i, 2] = best_radius
                        improved = True
            
            # Pass 2: Optimize positions for existing radii
            if not improved:
                for i in range(len(circles)):
                    original_pos = circles[i, :2].copy()
                    best_pos = original_pos.copy()
                    best_radius = circles[i, 2]
                    best_score = best_radius
                    
                    # Try several positions for optimization
                    candidates = []
                    for dx in [-0.015, -0.01, 0, 0.01, 0.015]:
                        for dy in [-0.015, -0.01, 0, 0.01, 0.015]:
                            candidates.append((dx, dy))
                    
                    # Shuffle candidates for exploration
                    random.shuffle(candidates)
                    
                    for dx, dy in candidates[:10]:  # Test top candidates
                        test_x = max(0.01, min(0.99, circles[i, 0] + dx))
                        test_y = max(0.01, min(0.99, circles[i, 1] + dy))
                        test_r = circles[i, 2]
                        
                        valid = True
                        # Check overlap with other circles
                        for j in range(len(circles)):
                            if i != j:
                                pos_j = circles[j, :2]
                                r_j = circles[j, 2]
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
                    if best_score > circles[i, 2] or not np.array_equal(best_pos, original_pos):
                        circles[i, :2] = best_pos
                        improved = True
            
            if not improved:
                break
                
        return circles

    def optimize(self):
        """Main optimization routine."""
        # Initialize toolbox
        toolbox = base.Toolbox()
        toolbox.register("individual", self._create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_fitness)
        toolbox.register("mate", self._crossover_constraint_aware)
        toolbox.register("mutate", self._mutate_circle)
        toolbox.register("select", tools.selTournament, tournsize=5)  # Increase tournament size

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
            return self._heuristic_circle_packing()

        # Return best solution
        best_individual = hof[0]
        result = np.array(best_individual).reshape(-1, 3)
        
        # Apply advanced local optimization to refine further
        refined_result = self._improved_local_optimization(result.copy())
        
        return refined_result
    
    def _heuristic_circle_packing(self):
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
    optimizer = CirclePackingOptimizer()
    return optimizer.optimize()

# EVOLVE-BLOCK-END