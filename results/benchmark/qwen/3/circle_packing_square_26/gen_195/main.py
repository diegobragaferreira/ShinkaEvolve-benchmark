# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree, Voronoi
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time
import math

# Global constants for optimization
POPULATION_SIZE = 150
GENERATIONS = 600
INITIAL_MUTATION_RATE = 0.3
FINAL_MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.85
TOURNAMENT_SIZE = 6
BOUNDARY_PENALTY_BASE = 500.0
OVERLAP_PENALTY_BASE = 8000.0
ELITISM_COUNT = 8
MIN_RADIUS = 0.001
MAX_RADIUS = 0.49

class AdaptiveVoronoiEvolutionOptimizer:
    """Advanced evolutionary optimizer with multi-scale Voronoi initialization and dynamic adaptation"""

    def __init__(self):
        self.generation_history = []

    def generate_multi_scale_voronoi_points(self, n_circles: int) -> List[Tuple[float, float]]:
        """Generate points using multi-scale Voronoi approach for better distribution"""
        points = []
        
        # Multi-resolution approach: coarse grid → fine refinement
        scales = [3, 5, 7]  # Different grid densities
        
        for scale in scales:
            if len(points) >= n_circles:
                break
                
            grid_size = max(scale, int(np.ceil(np.sqrt(n_circles))))
            step = 1.0 / (grid_size + 1)
            
            for i in range(1, grid_size):
                for j in range(1, grid_size):
                    if len(points) >= n_circles:
                        break
                    x = i * step + random.uniform(-step/4, step/4)
                    y = j * step + random.uniform(-step/4, step/4)
                    # Ensure points are within bounds
                    x = max(0.05, min(0.95, x))
                    y = max(0.05, min(0.95, y))
                    points.append((x, y))
        
        # Fill remaining points with random sampling
        while len(points) < n_circles:
            points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))
            
        return points[:n_circles]

    def initialize_population(self, pop_size: int, n_circles: int) -> List[np.ndarray]:
        """Initialize population with advanced multi-scale approach"""
        population = []
        
        # Generate points using multi-scale Voronoi
        voronoi_points = self.generate_multi_scale_voronoi_points(n_circles)
        
        for _ in range(pop_size):
            individual = np.zeros((n_circles, 3))
            
            # Assign positions with adaptive perturbation
            for i in range(n_circles):
                if i < len(voronoi_points):
                    x, y = voronoi_points[i]
                else:
                    x = random.uniform(0.05, 0.95)
                    y = random.uniform(0.05, 0.95)
                
                # Adaptive perturbation based on generation stage
                perturbation_scale = 0.05
                x += random.uniform(-perturbation_scale, perturbation_scale)
                y += random.uniform(-perturbation_scale, perturbation_scale)
                
                # Clip to valid range
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                individual[i, 0] = x
                individual[i, 1] = y
                
                # Assign radius based on proximity to other points and boundaries
                margin = min(x, y, 1 - x, 1 - y)
                
                # Calculate proximity to other points to estimate good radius
                min_dist_to_others = float('inf')
                for j in range(min(i, 5)):  # Check only recent points for efficiency
                    if j < len(voronoi_points):
                        other_x, other_y = voronoi_points[j]
                        dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                        min_dist_to_others = min(min_dist_to_others, dist)
                
                if min_dist_to_others < float('inf'):
                    base_radius = min(0.15, min(margin, min_dist_to_others/3.0))
                else:
                    base_radius = min(0.15, margin / 2.0)
                
                # Add randomness to radius
                individual[i, 2] = max(MIN_RADIUS, base_radius * random.uniform(0.7, 1.3))
            
            # Refine solution to ensure validity
            individual = self.refine_solution(individual, is_initial=True)
            population.append(individual)
        
        return population

    def is_valid_position(self, x: float, y: float, r: float) -> bool:
        """Check if a circle position is valid (within bounds)"""
        return (r <= x <= 1 - r and r <= y <= 1 - r)

    def calculate_penalty(self, circles: np.ndarray, generation: int = 0) -> Tuple[float, float, float]:
        """Calculate penalty with adaptive scaling based on optimization progress"""
        penalty = 0.0
        boundary_violations = 0.0
        overlap_violations = 0.0
        
        n = len(circles)
        
        # Adaptive penalty scaling
        progress = min(1.0, generation / GENERATIONS)
        penalty_multiplier = 1.0 + progress * 3.0  # Increase penalties as optimization progresses
        
        # Check containment penalties
        for circle in circles:
            x, y, r = circle
            if not self.is_valid_position(x, y, r):
                # Calculate violation amounts with penalty multiplier
                left_violation = max(0, r - x)
                right_violation = max(0, r - (1 - x))
                bottom_violation = max(0, r - y)
                top_violation = max(0, r - (1 - y))
                boundary_violations += (left_violation + right_violation + 
                                     bottom_violation + top_violation)
        
        # Use spatial indexing with adaptive threshold
        valid_circles = [c for c in circles if c[2] > MIN_RADIUS]
        
        if len(valid_circles) > 1:
            # Use efficient KDTree for large populations
            if len(valid_circles) > 50:
                positions = np.array([[c[0], c[1]] for c in valid_circles])
                tree = cKDTree(positions)
                # Query pairs within reasonable distance
                pairs = tree.query_pairs(0.05, p=np.inf)
                
                for i, j in pairs:
                    if i != j:
                        c1 = valid_circles[i]
                        c2 = valid_circles[j]
                        distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                        if distance < (c1[2] + c2[2]):
                            overlap_violations += (c1[2] + c2[2]) - distance
            else:
                # For smaller groups, brute force
                for i in range(len(valid_circles)):
                    for j in range(i+1, len(valid_circles)):
                        c1 = valid_circles[i]
                        c2 = valid_circles[j]
                        distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                        if distance < (c1[2] + c2[2]):
                            overlap_violations += (c1[2] + c2[2]) - distance
        
        penalty = (BOUNDARY_PENALTY_BASE * boundary_violations + 
                  OVERLAP_PENALTY_BASE * overlap_violations) * penalty_multiplier
        
        return penalty, boundary_violations, overlap_violations

    def evaluate_fitness(self, circles: np.ndarray, generation: int = 0) -> Tuple[float, float, float]:
        """Evaluate the fitness of a solution"""
        # Sum of radii (primary objective)
        total_radius = np.sum(circles[:, 2])
        
        # Penalty for constraint violations
        penalty, _, _ = self.calculate_penalty(circles, generation)
        
        # Fitness is total radius minus penalty
        fitness = total_radius - penalty
        
        return fitness, total_radius, penalty

    def tournament_selection(self, population: List[np.ndarray], fitness_scores: List[float], 
                           tournament_size: int = TOURNAMENT_SIZE) -> np.ndarray:
        """Select an individual using tournament selection with diversity consideration"""
        # Sample with replacement
        selected_indices = random.sample(range(len(population)), tournament_size)
        selected_fitness = [fitness_scores[i] for i in selected_indices]
        
        # Prefer higher fitness, but occasionally select from middle
        winner_idx = selected_indices[np.argmax(selected_fitness)]
        return population[winner_idx].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray, 
                 generation: int = 0, crossover_rate: float = CROSSOVER_RATE) -> np.ndarray:
        """Perform adaptive crossover with enhanced mixing"""
        if random.random() > crossover_rate:
            return parent1.copy()
            
        n = len(parent1)
        child = np.zeros_like(parent1)
        
        # Adaptive crossover point selection based on generation
        if generation < GENERATIONS * 0.3:  # Early generations: more random
            num_crossover_points = max(1, random.randint(1, min(4, n//2)))
        else:  # Later generations: more structured
            num_crossover_points = max(1, min(3, n//4))
            
        crossover_points = sorted(random.sample(range(1, n), num_crossover_points))
        
        # Alternate between parents for segments with adaptive pattern
        last_point = 0
        use_parent1 = True if random.random() < 0.6 else False  # Slight bias
        
        for point in crossover_points:
            if use_parent1:
                child[last_point:point, :] = parent1[last_point:point, :]
            else:
                child[last_point:point, :] = parent2[last_point:point, :]
            last_point = point
            use_parent1 = not use_parent1
        
        # Handle final segment
        if use_parent1:
            child[last_point:, :] = parent1[last_point:, :]
        else:
            child[last_point:, :] = parent2[last_point:, :]
            
        # Local refinement to fix any constraint violations
        child = self.refine_solution(child)
        return child

    def refine_solution(self, circles: np.ndarray, is_initial: bool = False) -> np.ndarray:
        """Advanced refinement with physics-inspired overlap resolution"""
        refined = circles.copy()
        
        # Phase 1: Boundary correction
        for i in range(len(refined)):
            x, y, r = refined[i]
            if not self.is_valid_position(x, y, r):
                # Adjust to nearest valid position
                if r > x:
                    x = r + 0.001
                if r > y:
                    y = r + 0.001
                if r > (1 - x):
                    x = 1 - r - 0.001
                if r > (1 - y):
                    y = 1 - r - 0.001
                refined[i, 0] = x
                refined[i, 1] = y
        
        # Phase 2: Overlap resolution with iterative force-based approach
        max_iter = 50 if not is_initial else 20  # Less iterations for initial
        for iteration in range(max_iter):
            changed = False
            valid_circles = [c for c in refined if c[2] > MIN_RADIUS]
            
            if len(valid_circles) <= 1:
                break
                
            # Use fast spatial indexing for overlap detection
            if len(valid_circles) > 30:
                positions = np.array([[c[0], c[1]] for c in valid_circles])
                tree = cKDTree(positions)
                pairs = tree.query_pairs(0.02, p=np.inf)  # Small radius for efficiency
                
                for i, j in pairs:
                    if i != j:
                        c1 = valid_circles[i]
                        c2 = valid_circles[j]
                        distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                        
                        if distance < (c1[2] + c2[2]):
                            # Force-based adjustment
                            dx = c2[0] - c1[0]
                            dy = c2[1] - c1[1]
                            dist = max(0.0001, distance)
                            
                            # Normalize
                            dx /= dist
                            dy /= dist
                            
                            # Move circles apart proportionally to their radii
                            separation = (c1[2] + c2[2] - dist) * 0.5
                            
                            # Apply force
                            force1 = separation * (c2[2] / (c1[2] + c2[2]))
                            force2 = separation * (c1[2] / (c1[2] + c2[2]))
                            
                            refined[i, 0] -= dx * force1
                            refined[i, 1] -= dy * force1
                            refined[j, 0] += dx * force2
                            refined[j, 1] += dy * force2
                            changed = True
            else:
                # Brute force for small populations
                for i in range(len(valid_circles)):
                    for j in range(i+1, len(valid_circles)):
                        c1 = valid_circles[i]
                        c2 = valid_circles[j]
                        distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                        
                        if distance < (c1[2] + c2[2]):
                            # Force-based adjustment
                            dx = c2[0] - c1[0]
                            dy = c2[1] - c1[1]
                            dist = max(0.0001, distance)
                            
                            # Normalize
                            dx /= dist
                            dy /= dist
                            
                            # Move circles apart
                            separation = (c1[2] + c2[2] - dist) * 0.5
                            
                            refined[i, 0] -= dx * separation * 0.5
                            refined[i, 1] -= dy * separation * 0.5
                            refined[j, 0] += dx * separation * 0.5
                            refined[j, 1] += dy * separation * 0.5
                            changed = True
            
            # Boundary check after adjustments
            for i in range(len(refined)):
                x, y, r = refined[i]
                r = max(MIN_RADIUS, min(MAX_RADIUS, r))
                x = np.clip(x, r, 1 - r)
                y = np.clip(y, r, 1 - r)
                refined[i] = [x, y, r]
            
            if not changed:
                break
                
        return refined

    def adaptive_mutation_rate(self, generation: int, progress_ratio: float) -> float:
        """Adaptive mutation rate with learning-based adjustments"""
        # Base mutation rate that decreases gradually
        base_rate = INITIAL_MUTATION_RATE - (INITIAL_MUTATION_RATE - FINAL_MUTATION_RATE) * progress_ratio
        
        # Add dynamic adjustment based on performance
        if len(self.generation_history) > 5:
            recent_improvements = [self.generation_history[-i][1] - self.generation_history[-i-1][1] 
                                 for i in range(1, min(5, len(self.generation_history)))]
            avg_improvement = np.mean(recent_improvements) if recent_improvements else 0
            
            # If improvement is slow, increase mutation rate
            if avg_improvement < 0.001:
                base_rate *= 1.2
                
        return max(0.01, base_rate)

    def smart_mutate(self, individual: np.ndarray, generation: int, 
                    progress_ratio: float, mutation_rate: float) -> np.ndarray:
        """Advanced mutation with phase-aware strategy"""
        mutated = individual.copy()
        n = len(mutated)
        
        # Determine mutation phase based on generation
        if generation < GENERATIONS * 0.4:  # Exploration phase
            mutation_strength = 0.07
            pos_weight = 0.7
            rad_weight = 0.3
        elif generation < GENERATIONS * 0.7:  # Balance phase
            mutation_strength = 0.03
            pos_weight = 0.5
            rad_weight = 0.5
        else:  # Exploitation phase
            mutation_strength = 0.01
            pos_weight = 0.3
            rad_weight = 0.7
        
        for i in range(n):
            if random.random() < mutation_rate:
                # Choose mutation type with phase-appropriate weighting
                mutation_type = random.choices(
                    [0, 1, 2],  # 0:pos_x, 1:pos_y, 2:radius
                    weights=[pos_weight, pos_weight, rad_weight]
                )[0]
                
                if mutation_type == 0:  # Mutate x position
                    mutated[i, 0] = np.clip(mutated[i, 0] + random.gauss(0, mutation_strength), 
                                          0.05, 0.95)
                elif mutation_type == 1:  # Mutate y position
                    mutated[i, 1] = np.clip(mutated[i, 1] + random.gauss(0, mutation_strength), 
                                          0.05, 0.95)
                else:  # Mutate radius
                    mutated[i, 2] = np.clip(mutated[i, 2] + random.gauss(0, mutation_strength*0.5), 
                                          MIN_RADIUS, MAX_RADIUS)
        
        # Apply refinement after mutation
        mutated = self.refine_solution(mutated)
        return mutated

    def evolve_population(self, population: List[np.ndarray], generation: int) -> Tuple[List[np.ndarray], float, float, float]:
        """Evolve the population for one generation with adaptive strategies"""
        # Evaluate fitness
        fitness_scores = []
        total_radii = []
        penalties = []

        for individual in population:
            fitness, total_radius, penalty = self.evaluate_fitness(individual, generation)
            fitness_scores.append(fitness)
            total_radii.append(total_radius)
            penalties.append(penalty)

        # Track best individual
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        best_total_radius = total_radii[best_idx]
        best_penalty = penalties[best_idx]
        
        # Record history for adaptive strategies
        self.generation_history.append((best_fitness, best_total_radius, best_penalty))
        if len(self.generation_history) > 100:
            self.generation_history.pop(0)

        # Create new population
        new_population = []

        # Elitism: keep the best individuals
        elite_indices = np.argsort(fitness_scores)[-ELITISM_COUNT:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Generate rest of population
        while len(new_population) < len(population):
            # Selection
            parent1 = self.tournament_selection(population, fitness_scores)
            parent2 = self.tournament_selection(population, fitness_scores)

            # Crossover with generation-aware parameters
            child = self.crossover(parent1, parent2, generation)
            
            # Mutation with adaptive rate and smart strategy
            progress_ratio = generation / GENERATIONS
            mut_rate = self.adaptive_mutation_rate(generation, progress_ratio)
            child = self.smart_mutate(child, generation, progress_ratio, mut_rate)

            new_population.append(child)

        return new_population, best_fitness, best_total_radius, best_penalty

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    optimizer = AdaptiveVoronoiEvolutionOptimizer()
    n = 26
    population = optimizer.initialize_population(POPULATION_SIZE, n)

    best_total_radius = 0.0
    best_individual = None
    best_penalty = float('inf')

    # Evolution loop
    start_time = time.time()
    for generation in range(GENERATIONS):
        population, gen_fitness, gen_radius, gen_penalty = optimizer.evolve_population(population, generation)

        if gen_radius > best_total_radius:
            best_total_radius = gen_radius
            best_individual = population[0]  # Keep track of best individual
            best_penalty = gen_penalty

        # Print progress every 50 generations
        if generation % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Generation {generation}: Best radius sum = {gen_radius:.6f} (penalty={gen_penalty:.2f}) Time: {elapsed:.2f}s")

    elapsed = time.time() - start_time
    print(f"Final result: Best radius sum = {best_total_radius:.6f} (penalty={best_penalty:.2f}) Time: {elapsed:.2f}s")
    print(f"Benchmark ratio: {best_total_radius / 2.6358627564136983:.6f}")

    # Return the best solution found
    if best_individual is not None:
        return best_individual
    else:
        # Fallback to returning first individual if something went wrong
        return population[0]

# EVOLVE-BLOCK-END