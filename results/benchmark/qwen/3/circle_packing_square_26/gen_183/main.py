# EVOLVE-BLOCK-START
import numpy as np
import random
from copy import deepcopy
from typing import Tuple, List
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import math

class VoronoiSpatialIndex:
    """Voronoi-based spatial indexing for efficient overlap detection"""
    
    def __init__(self):
        self.voronoi_cells = []
        self.points = []
        
    def update_points(self, points: np.ndarray):
        """Update Voronoi diagram with new points"""
        if len(points) < 2:
            self.voronoi_cells = []
            self.points = points.tolist() if isinstance(points, np.ndarray) else points
            return
            
        try:
            # Use Voronoi with margin to avoid edge issues
            points_ext = np.vstack([points, [[-1,-1], [2,-1], [-1,2], [2,2]]])
            vor = Voronoi(points_ext)
            self.voronoi_cells = vor.vertices.tolist()
            self.points = points.tolist() if isinstance(points, np.ndarray) else points
        except:
            # Fallback to basic approach
            self.voronoi_cells = []
            self.points = points.tolist() if isinstance(points, np.ndarray) else points
    
    def get_potential_neighbors(self, x: float, y: float, radius: float, 
                              candidate_indices: List[int] = None) -> List[int]:
        """Get potential neighbors using Voronoi-like spatial relationships"""
        if not self.points or len(self.points) < 2:
            return list(range(len(self.points))) if candidate_indices is None else candidate_indices
            
        # Use distance-based approach with Voronoi concept
        neighbors = []
        query_point = [x, y]
        
        # Calculate distances to all points
        distances = []
        for i, point in enumerate(self.points):
            if i != -1:  # Skip self
                d = distance.euclidean(query_point, point)
                distances.append((d, i))
        
        # Sort by distance and return closest few
        distances.sort(key=lambda x: x[0])
        
        # Return neighbors within a certain distance threshold
        threshold = radius * 3.0  # Adjust based on expected packing density
        for dist, idx in distances:
            if dist <= threshold:
                if candidate_indices is None or idx in candidate_indices:
                    neighbors.append(idx)
            else:
                break
                
        return neighbors

def get_voronoi_points(n_points: int, max_iter: int = 1000) -> np.ndarray:
    """Generate points using Voronoi-based optimization for better distribution"""
    points = np.random.rand(n_points, 2) * 0.8 + 0.1  # Keep away from edges
    
    for iteration in range(max_iter):
        # Calculate pairwise distances
        distances = cdist(points, points)
        
        # Set diagonal to large value to skip self-comparisons
        np.fill_diagonal(distances, np.inf)
        
        # Find minimum distances for each point
        min_distances = np.min(distances, axis=1)
        
        # Update points to spread them out
        for i in range(n_points):
            # Calculate forces from neighbors
            force_x, force_y = 0.0, 0.0
            
            for j in range(n_points):
                if i != j:
                    dx = points[i, 0] - points[j, 0]
                    dy = points[i, 1] - points[j, 1]
                    dist = max(1e-6, np.sqrt(dx*dx + dy*dy))
                    
                    # Repulsive force (inverse square law)
                    force_x += dx / (dist * dist)
                    force_y += dy / (dist * dist)
            
            # Apply force with damping
            points[i, 0] += force_x * 0.001
            points[i, 1] += force_y * 0.001
            
            # Keep within bounds
            points[i, 0] = np.clip(points[i, 0], 0.01, 0.99)
            points[i, 1] = np.clip(points[i, 1], 0.01, 0.99)
    
    return points

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n_circles = 26
    max_generations = 500
    population_size = 75
    
    # Initialize Voronoi spatial index
    spatial_index = VoronoiSpatialIndex()
    
    def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
        """Initialize population with Voronoi-based approach"""
        population = []
        
        for _ in range(pop_size):
            # Generate Voronoi-style points for better distribution
            points = get_voronoi_points(n_circles)
            
            circles = np.zeros((n_circles, 3))
            
            # Create circles with calculated radii
            for i in range(n_circles):
                x, y = points[i]
                
                # Calculate safe radius based on proximity to other circles
                min_dist = float('inf')
                for other_x, other_y in points[:i]:
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_dist = min(min_dist, dist)
                
                # Set radius with boundary constraints and neighbor distances
                boundary_dist = min(x, 1-x, y, 1-y)
                # Use more sophisticated radius calculation
                if i < 5:
                    radius = min(0.1, boundary_dist, min_dist/2)
                elif i < 10:
                    radius = min(0.08, boundary_dist, min_dist/2)
                else:
                    radius = min(0.05, boundary_dist, min_dist/2)
                
                if radius <= 0:
                    radius = 0.01
                
                circles[i] = [x, y, radius]
            
            population.append(circles)
            
        return population
    
    def is_valid(circles: np.ndarray) -> bool:
        """Check if all circles are within bounds and non-overlapping"""
        # Check boundary constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlap constraints
        n = len(circles)
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    return False
                    
        return True
    
    def evaluate_fitness(circles: np.ndarray) -> float:
        """Evaluate fitness of a solution"""
        if not is_valid(circles):
            # Penalty based on constraint violations
            total_penalty = 0
            
            # Boundary violations
            for i in range(len(circles)):
                x, y, r = circles[i]
                boundary_violation = 0
                if x - r < 0:
                    boundary_violation += abs(x - r)
                if x + r > 1:
                    boundary_violation += abs(x + r - 1)
                if y - r < 0:
                    boundary_violation += abs(y - r)
                if y + r > 1:
                    boundary_violation += abs(y + r - 1)
                total_penalty += boundary_violation * 100000
            
            # Overlap violations
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    overlap = max(0, r1 + r2 - distance)
                    total_penalty += overlap * 1000000
            
            return -total_penalty
        
        # Valid configuration: maximize sum of radii
        return np.sum(circles[:, 2])
    
    def mutate(circles: np.ndarray, generation: int, max_generations: int, 
              phase: str = "global") -> np.ndarray:
        """Apply mutation to circles with constraint-aware approach"""
        mutated = deepcopy(circles)
        
        # Adaptive mutation rate
        if phase == "global":
            mutation_rate = 0.3 - (0.2 * generation / max_generations)
            mutation_strength = 0.05
        else:  # local
            mutation_rate = 0.15 - (0.1 * generation / max_generations) 
            mutation_strength = 0.02
        
        # Mutate each circle with some probability
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Mutate position
                mutated[i, 0] += np.random.normal(0, mutation_strength)
                mutated[i, 1] += np.random.normal(0, mutation_strength)
                
                # Clamp to unit square with safety margin
                mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0]))
                mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1]))
                
                # Mutate radius with constraint awareness
                if phase == "global":
                    mutated[i, 2] += np.random.normal(0, mutation_strength * 0.5)
                else:  # local
                    mutated[i, 2] += np.random.normal(0, mutation_strength * 0.2)
                
                mutated[i, 2] = max(0.001, mutated[i, 2])
                
                # Adjust radius to respect boundaries and constraints
                x, y, r = mutated[i]
                max_radius = min(x, 1-x, y, 1-y)
                mutated[i, 2] = min(r, max_radius)
                
        return mutated
    
    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Create offspring via crossover of two parents"""
        child = deepcopy(parent1)
        n = len(parent1)
        
        # Two-point crossover
        crossover_point1 = random.randint(1, n//2)
        crossover_point2 = random.randint(crossover_point1, n-1)
        
        for i in range(crossover_point1, crossover_point2):
            child[i] = parent2[i].copy()
            
        return child
    
    def tournament_selection(population: List[np.ndarray], k: int = 3) -> np.ndarray:
        """Select individual using tournament selection"""
        selected = random.sample(population, k)
        return max(selected, key=evaluate_fitness)
    
    def local_improvement(circles: np.ndarray, max_iter: int = 50) -> np.ndarray:
        """Apply local geometric improvement to circles"""
        refined = deepcopy(circles)
        
        for iteration in range(max_iter):
            improved = False
            
            # Try to increase radii where possible
            for i in range(len(refined)):
                x, y, r = refined[i]
                
                # Calculate maximum possible radius
                max_possible_r = min(x, 1-x, y, 1-y)
                
                # Check overlap constraints
                for j in range(len(refined)):
                    if i != j:
                        x2, y2, r2 = refined[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        max_r_from_j = distance - r2
                        if max_r_from_j > 0:
                            max_possible_r = min(max_possible_r, max_r_from_j)
                
                # Increase radius if beneficial
                new_r = min(max_possible_r, r + 0.002)
                if new_r > r:
                    refined[i, 2] = new_r
                    improved = True
            
            if not improved:
                break
                
        return refined
    
    def voronoi_refinement(circles: np.ndarray) -> np.ndarray:
        """Use Voronoi refinement to improve circle packing"""
        refined = deepcopy(circles)
        
        # Use Voronoi-based repulsion to spread circles
        points = refined[:, :2]
        radii = refined[:, 2]
        
        # Calculate forces between circles
        n = len(refined)
        forces = np.zeros_like(points)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = points[i, 0] - points[j, 0]
                    dy = points[i, 1] - points[j, 1]
                    dist = max(1e-6, np.sqrt(dx*dx + dy*dy))
                    r_i = radii[i]
                    r_j = radii[j]
                    
                    # Repulsion force
                    if dist < (r_i + r_j):
                        # Overlapping - strong repulsion
                        force = (1.0 / (dist * dist)) * 0.001
                    else:
                        # Non-overlapping - attraction to optimal spacing
                        force = -0.0001 * (dist - (r_i + r_j)) / (dist + 1e-6)
                    
                    forces[i, 0] += force * dx / dist
                    forces[i, 1] += force * dy / dist
        
        # Apply forces
        for i in range(n):
            points[i, 0] += forces[i, 0]
            points[i, 1] += forces[i, 1]
            
            # Keep within bounds
            points[i, 0] = np.clip(points[i, 0], 0.01, 0.99)
            points[i, 1] = np.clip(points[i, 1], 0.01, 0.99)
        
        refined[:, :2] = points
        return refined
    
    # Initialize population
    population = initialize_population(population_size, n_circles)
    
    # Evolve with phases
    best_fitness = float('-inf')
    best_solution = None
    
    # Phase 1: Global exploration (first 300 generations)
    for generation in range(300):
        # Evaluate fitness for entire population
        fitness_scores = [evaluate_fitness(individual) for individual in population]
        
        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_solution = deepcopy(population[max_fitness_idx])
        
        # Elitism: keep top 20%
        elite_count = max(1, population_size // 5)
        sorted_indices = np.argsort(fitness_scores)[::-1][:elite_count]
        elite = [population[i] for i in sorted_indices]
        
        # Create new population
        new_population = deepcopy(elite)
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection for parents
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation (global phase)
            mutated_child = mutate(child, generation, 300, "global")
            
            # Local refinement
            refined_child = local_improvement(mutated_child)
            
            new_population.append(refined_child)
        
        population = new_population[:population_size]
    
    # Phase 2: Local optimization (remaining 200 generations)
    for generation in range(200):
        # Evaluate fitness for entire population
        fitness_scores = [evaluate_fitness(individual) for individual in population]
        
        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_solution = deepcopy(population[max_fitness_idx])
        
        # Elitism: keep top 20%
        elite_count = max(1, population_size // 5)
        sorted_indices = np.argsort(fitness_scores)[::-1][:elite_count]
        elite = [population[i] for i in sorted_indices]
        
        # Create new population
        new_population = deepcopy(elite)
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection for parents
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation (local phase)
            mutated_child = mutate(child, generation, 200, "local")
            
            # Voronoi-based refinement
            refined_child = voronoi_refinement(mutated_child)
            
            # Final local improvement
            refined_child = local_improvement(refined_child)
            
            new_population.append(refined_child)
        
        population = new_population[:population_size]
    
    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to first individual if nothing was found
        return population[0]

# EVOLVE-BLOCK-END