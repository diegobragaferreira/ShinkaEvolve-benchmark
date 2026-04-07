# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from collections import defaultdict
import heapq

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class CirclePackingOptimizer:
    """Optimizes placement of 26 non-overlapping circles in unit square to maximize sum of radii."""

    def __init__(self, n_circles=26, pop_size=120, gen_count=100, mutpb=0.3, cxpb=0.5):
        self.N_CIRCLES = n_circles
        self.POP_SIZE = pop_size
        self.GEN_COUNT = gen_count
        self.MUTPB = mutpb
        self.CXPB = cxpb
        self.SPECIES_COUNT = 4  # Number of species for speciation
        self._setup_deap()

    def _setup_deap(self):
        """Initialize DEAP framework components."""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    def _create_priority_voronoi_initialization(self):
        """Generate initial circle configuration using enhanced Voronoi-like distribution with priority ordering."""
        circles = np.zeros((self.N_CIRCLES, 3))

        # Create initial hexagonal grid points
        sqrt_n = int(np.ceil(np.sqrt(self.N_CIRCLES)))
        rows = int(np.ceil(self.N_CIRCLES / sqrt_n))
        cols = int(np.ceil(self.N_CIRCLES / rows))

        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)

        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.N_CIRCLES:
                    break
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 1) * spacing_x + x_offset
                y = (i + 1) * spacing_y
                points.append([x, y])

        # Ensure we have enough points
        while len(points) < self.N_CIRCLES:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            points.append([x, y])

        points = points[:self.N_CIRCLES]
        
        # Priority-based assignment using Voronoi cell area estimation
        # Calculate radii based on Voronoi cells, starting with largest
        # Create priority queue to assign larger radii first
        priorities = []
        for i, (x, y) in enumerate(points):
            # Estimate Voronoi cell size by finding minimum distance to neighbors
            min_dist = float('inf')
            for j, (other_x, other_y) in enumerate(points):
                if i != j:
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_dist = min(min_dist, dist)
            
            # Priority is inversely related to neighbor distance (larger cells = higher priority)
            priority = -min_dist if min_dist < float('inf') else -0.1
            priorities.append((priority, i))
        
        # Sort by priority (descending, so highest priority first)
        priorities.sort(reverse=True)
        
        # Assign radii and positions from highest to lowest priority
        assigned_positions = []
        assigned_radii = []
        
        for priority, idx in priorities:
            x, y = points[idx]
            # Calculate safe radius based on neighbors and boundaries
            min_dist = float('inf')
            for j, (other_x, other_y) in enumerate(points):
                if j != idx:
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_dist = min(min_dist, dist)
            
            # Set initial radius to 25% of minimum neighbor distance or reasonable value
            initial_r = min(0.15, min_dist * 0.25) if min_dist < float('inf') else 0.07
            # Constrain to boundary limits
            boundary_radius = min(x, 1-x, y, 1-y)
            initial_r = min(initial_r, boundary_radius * 0.8)
            initial_r = max(0.005, min(0.15, initial_r))
            
            assigned_positions.append([x, y])
            assigned_radii.append(initial_r)
        
        # Fill circles array with assigned values
        for i in range(self.N_CIRCLES):
            circles[i] = [assigned_positions[i][0], assigned_positions[i][1], assigned_radii[i]]

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

        # Check overlap constraints efficiently using cKDTree
        try:
            tree = cKDTree(positions)
            pairs = tree.query_pairs(radii.sum() + 0.001, p=2)
            overlap_penalty = 0
            for i, j in pairs:
                r_i = radii[i]
                r_j = radii[j]
                pos_i = positions[i]
                pos_j = positions[j]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    overlap_penalty += 1000 * (r_i + r_j - dist)
            penalty += overlap_penalty
        except:
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
                    mutation_strength = 0.02 * (1 + diversity)
                    new_r = old_r + random.gauss(0, mutation_strength)
                    individual[i] = max(0.001, min(0.5, new_r))
                else:  # position indices (x, y)
                    old_val = individual[i]
                    mutation_strength = 0.03 * (1 + diversity)
                    new_val = old_val + random.gauss(0, mutation_strength)
                    individual[i] = max(0, min(1, new_val))
        return individual,

    def _create_individual(self):
        """Create a random valid individual."""
        # Start with priority Voronoi-based configuration
        individual = self._create_priority_voronoi_initialization().flatten().tolist()

        # Add small random perturbations
        for i in range(len(individual)):
            if i % 3 < 2:  # x or y coordinate
                individual[i] += random.uniform(-0.015, 0.015)
                individual[i] = max(0, min(1, individual[i]))
            else:  # radius
                individual[i] *= random.uniform(0.92, 1.08)
                individual[i] = max(0.001, min(0.5, individual[i]))
        return creator.Individual(individual)

    def _speciate_population(self, population):
        """Group individuals into species based on geometric similarity."""
        if len(population) < 2:
            return [population]
            
        # Use cKDTree to find similar individuals
        species = []
        unassigned = list(range(len(population)))
        
        while unassigned:
            # Pick a representative from unassigned group
            rep_idx = unassigned[0]
            rep_individual = np.array(population[rep_idx]).reshape(-1, 3)
            
            # Find similar individuals (using Euclidean distance in position space)
            rep_positions = rep_individual[:, :2]
            species_members = [rep_idx]
            unassigned.remove(rep_idx)
            
            # Compare with remaining individuals
            for idx in unassigned[:]:  # Make a copy to iterate
                candidate_individual = np.array(population[idx]).reshape(-1, 3)
                candidate_positions = candidate_individual[:, :2]
                
                # Calculate average distance between representatives
                distances = cdist(rep_positions, candidate_positions, metric='euclidean')
                avg_distance = np.mean(distances)
                
                # If within threshold, add to species
                if avg_distance < 0.03:  # Threshold for species similarity
                    species_members.append(idx)
                    unassigned.remove(idx)
            
            species.append([population[i] for i in species_members])
        
        # If we didn't create enough species, fill with random subsets
        if len(species) < self.SPECIES_COUNT:
            # Distribute remaining individuals
            remaining_individuals = [item for sublist in species for item in sublist]
            for i in range(len(remaining_individuals), len(population)):
                remaining_individuals.append(population[i])
            
            # Create additional species from remaining individuals
            individual_chunks = [remaining_individuals[i::self.SPECIES_COUNT] 
                               for i in range(self.SPECIES_COUNT)]
            for chunk in individual_chunks:
                if chunk and len(species) < self.SPECIES_COUNT:
                    species.append(chunk)
        
        return species[:self.SPECIES_COUNT]

    def _crossover_constraint_aware(self, ind1, ind2):
        """Crossover that maintains constraints with repair mechanism."""
        # Perform standard uniform crossover
        tools.cxUniform(ind1, ind2, indpb=0.5)

        # Repair violated constraints in both children
        for ind in [ind1, ind2]:
            temp_ind = np.array(ind).reshape(-1, 3)
            
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

            # Fix overlaps with smarter repair
            for i in range(len(temp_ind)):
                for j in range(i+1, len(temp_ind)):
                    pos_i = temp_ind[i, :2]
                    pos_j = temp_ind[j, :2]
                    r_i = temp_ind[i, 2]
                    r_j = temp_ind[j, 2]
                    dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                    if dist < (r_i + r_j):
                        # Move one circle away from the other
                        # Prefer to adjust the circle that's more constrained
                        dx, dy = pos_i - pos_j
                        dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                        dx /= dist
                        dy /= dist
                        step = (r_i + r_j - dist) * 0.3
                        
                        # Only move if it stays within bounds
                        new_x = max(0.005, min(0.995, pos_i[0] + dx * step))
                        new_y = max(0.005, min(0.995, pos_i[1] + dy * step))
                        temp_ind[i] = [new_x, new_y, r_i]
            
            ind[:] = temp_ind.flatten()
        return ind1, ind2

    def _constraint_aware_local_search(self, circles):
        """Apply enhanced local optimization to refine solution using gradient-based approach."""
        # Multiple refinement passes
        for pass_num in range(3):
            improved = False
            
            # Pass 1: Increase radii where possible
            for i in range(len(circles)):
                original_r = circles[i, 2]
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
                        
                        valid = True
                        test_pos = circles[i, :2]
                        test_r_new = original_r + test_r
                        
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
                    
                    if best_radius > original_r + 0.001:
                        circles[i, 2] = best_radius
                        improved = True

            if not improved:
                # Pass 2: Position refinement with gradient-based approach
                for i in range(len(circles)):
                    original_pos = circles[i, :2].copy()
                    best_pos = original_pos.copy()
                    best_radius = circles[i, 2]
                    best_score = best_radius
                    
                    # Calculate forces from neighboring circles
                    forces = np.array([0.0, 0.0])
                    for j in range(len(circles)):
                        if i != j:
                            pos_i = circles[i, :2]
                            pos_j = circles[j, :2]
                            r_i = circles[i, 2]
                            r_j = circles[j, 2]
                            dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                            
                            if dist < (r_i + r_j) and dist > 0:
                                # Repulsion force
                                force_dir = pos_i - pos_j
                                force_magnitude = (r_i + r_j - dist) / dist
                                force_dir = force_dir / (dist + 1e-8)
                                forces += force_dir * force_magnitude * 0.5
                    
                    # Try several positions around current location
                    for dx in [-0.015, -0.01, 0, 0.01, 0.015]:
                        for dy in [-0.015, -0.01, 0, 0.01, 0.015]:
                            test_x = max(0.005, min(0.995, circles[i, 0] + dx))
                            test_y = max(0.005, min(0.995, circles[i, 1] + dy))
                            
                            valid = True
                            test_r = circles[i, 2]
                            
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

    def _heuristic_circle_packing(self):
        """Fallback method using structured approach."""
        n = self.N_CIRCLES
        circles = np.zeros((n, 3))

        # Try a hexagonal packing pattern
        rows = 5
        cols = 5
        if n < rows * cols:
            rows = int(np.ceil(n / cols))

        # Create regular grid points with hexagonal offset
        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)

        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
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
            for i in range(n):
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

    def optimize(self):
        """Main optimization routine with speciation and adaptive parameters."""
        # Initialize toolbox
        toolbox = base.Toolbox()
        toolbox.register("individual", self._create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_fitness)
        toolbox.register("mate", self._crossover_constraint_aware)
        toolbox.register("mutate", self._mutate_circle)
        toolbox.register("select", tools.selTournament, tournsize=5)  # Increased tournament size for better selection pressure

        # Create initial population
        population = toolbox.population(n=self.POP_SIZE)

        # Run evolution with speciation
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        try:
            # Split population into species
            species = self._speciate_population(population)
            
            # Evolution loop with species dynamics
            for gen in range(self.GEN_COUNT):
                # Adaptive mutation rate based on generation progress
                adaptive_mutpb = self.MUTPB * (1 - gen / self.GEN_COUNT)
                adaptive_mutpb = max(adaptive_mutpb, 0.05)  # Minimum mutation rate
                
                # Process each species individually
                for s_idx, specie in enumerate(species):
                    if len(specie) < 2:
                        continue
                        
                    # Update toolbox with species-specific parameters
                    toolbox.register("mutate", self._mutate_circle)
                    toolbox.register("select", tools.selTournament, tournsize=5)
                    
                    # Create subpopulation for this species
                    subpop = specie[:]
                    
                    # Perform evolution on this species
                    try:
                        # Simple EA for species to promote diversity
                        for _ in range(2):  # Two generations per species
                            # Fitness evaluation
                            fitnesses = list(map(toolbox.evaluate, subpop))
                            for ind, fit in zip(subpop, fitnesses):
                                ind.fitness.values = fit
                            
                            # Selection
                            selected = toolbox.select(subpop, len(subpop))
                            
                            # Crossover and mutation
                            offspring = []
                            for i in range(0, len(selected), 2):
                                if i+1 < len(selected):
                                    child1, child2 = toolbox.mate(selected[i], selected[i+1])
                                    child1 = toolbox.mutate(child1)[0]
                                    child2 = toolbox.mutate(child2)[0]
                                    offspring.extend([child1, child2])
                            
                            # Replace subpopulation with offspring
                            subpop = offspring[:len(subpop)]
                            
                    except Exception as e:
                        # Fallback to simple mutation
                        for ind in subpop:
                            toolbox.mutate(ind)
                
                # Reintegrate species back into population
                recombined_pop = []
                for specie in species:
                    recombined_pop.extend(specie)
                
                population = recombined_pop[:self.POP_SIZE]
                
                # Regenerate species for next generation
                species = self._speciate_population(population)
                
                # Update Hall of Fame with best overall individual
                for ind in population:
                    if len(hof) == 0 or ind.fitness.values[0] > hof[0].fitness.values[0]:
                        hof.update([ind])
                        
                # Periodic speciation update
                if gen % 10 == 0:
                    species = self._speciate_population(population)

            # Final refinement using enhanced local search
            best_individual = hof[0]
            result = np.array(best_individual).reshape(-1, 3)
            refined_result = self._constraint_aware_local_search(result.copy())
            
        except Exception as e:
            # Fallback to heuristic method
            print(f"Error in evolution: {e}")
            return self._heuristic_circle_packing()

        return refined_result

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer()
    return optimizer.optimize()

# EVOLVE-BLOCK-END