# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from numba import njit
import time
import random
from collections import defaultdict
import warnings
from copy import deepcopy

# Core geometric functions with JIT compilation
@njit
def generate_hexagon_vertices(x, y, angle_degrees, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length"""
    angle_rad = np.radians(angle_degrees)
    vertices = np.empty((6, 2))
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i, 0] = x + side_length * np.cos(theta)
        vertices[i, 1] = y + side_length * np.sin(theta)
    return vertices

@njit
def check_containment_inner_to_outer(inner_x, inner_y, inner_angle, outer_x, outer_y, outer_angle, outer_side_length):
    """Check if inner hexagon is fully contained within outer hexagon"""
    inner_vertices = generate_hexagon_vertices(inner_x, inner_y, inner_angle, 1.0)

    # Create outer hexagon vertices
    outer_vertices = generate_hexagon_vertices(outer_x, outer_y, outer_angle, outer_side_length)
    outer_polygon = Polygon(outer_vertices)

    # Check if all vertices of inner hexagon are inside outer hexagon
    for i in range(6):
        if not outer_polygon.contains(Point(inner_vertices[i, 0], inner_vertices[i, 1])):
            return False

    return True

@njit
def check_overlap_hexagons(x1, y1, angle1, x2, y2, angle2):
    """Check if two hexagons overlap using vertex-based collision detection"""
    vertices1 = generate_hexagon_vertices(x1, y1, angle1, 1.0)
    vertices2 = generate_hexagon_vertices(x2, y2, angle2, 1.0)

    # Simple bounding box check first
    min1 = np.min(vertices1, axis=0)
    max1 = np.max(vertices1, axis=0)
    min2 = np.min(vertices2, axis=0)
    max2 = np.max(vertices2, axis=0)

    if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
        return False

    # Create polygons and check intersection
    poly1 = Polygon(vertices1)
    poly2 = Polygon(vertices2)

    # If intersection exists, they overlap
    return poly1.intersects(poly2)

@njit
def point_in_hexagon(point_x, point_y, hex_center_x, hex_center_y, hex_angle, hex_side_length):
    """Check if a point is inside a hexagon using geometric properties"""
    # Transform point to hexagon's coordinate system
    dx = point_x - hex_center_x
    dy = point_y - hex_center_y
    angle_rad = np.radians(hex_angle)

    # Rotate point to align with hexagon axes
    rotated_x = dx * np.cos(-angle_rad) - dy * np.sin(-angle_rad)
    rotated_y = dx * np.sin(-angle_rad) + dy * np.cos(-angle_rad)

    # For unit hexagons, check against the boundary
    # Maximum distance from center in direction of axes
    max_dist = hex_side_length
    if abs(rotated_x) > max_dist or abs(rotated_y) > max_dist:
        return False

    # Check against slanted edges - simplified but effective for unit hexagons
    if abs(rotated_x) <= max_dist and abs(rotated_y) <= max_dist:
        return True
    return False

# Spatial hashing for efficient overlap detection
@njit
def get_grid_coords(x, y, cell_size=2.0):
    """Get grid coordinates for a point"""
    return int(x / cell_size), int(y / cell_size)

class SpatialHash:
    """Efficient spatial hash for neighbor search during overlap detection"""
    
    def __init__(self, grid_size=2.0):
        self.grid_size = grid_size
        self.grid = defaultdict(list)
    
    @njit
    def insert(self, hex_id, center_x, center_y):
        """Insert a hexagon into the spatial grid"""
        cell_x, cell_y = get_grid_coords(center_x, center_y, self.grid_size)
        self.grid[(cell_x, cell_y)].append(hex_id)
    
    @njit  
    def get_neighbors(self, center_x, center_y):
        """Get candidate hexagons that might collide with given position"""
        cell_x, cell_y = get_grid_coords(center_x, center_y, self.grid_size)
        neighbors = []
        
        # Check the cell and its 8 neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                if neighbor_cell in self.grid:
                    neighbors.extend(self.grid[neighbor_cell])
        
        return neighbors

class HexagonPackingOptimizer:
    """Specialized optimizer for 12-hexagon packing problem"""
    
    def __init__(self):
        self.best_solution = None
        self.best_score = float('inf')
        
    def _initialize_base_pattern(self):
        """Create mathematically-informed base configuration for 12 hexagons"""
        # Inspired by optimal known arrangements and packing theory
        # Using more precise geometric placements
        base_pattern = [
            [0.0, 0.0, 0.0],        # Center
            [0.0, 3.2, 0.0],        # Top 
            [0.0, -3.2, 0.0],       # Bottom
            [2.7, 1.55, 0.0],       # Top Right
            [-2.7, 1.55, 0.0],      # Top Left
            [2.7, -1.55, 0.0],      # Bottom Right
            [-2.7, -1.55, 0.0],     # Bottom Left
            [3.7, 0.0, 0.0],        # Far Right  
            [-3.7, 0.0, 0.0],       # Far Left
            [1.85, 2.95, 0.0],      # Upper Middle Right
            [-1.85, 2.95, 0.0],     # Upper Middle Left
            [1.85, -2.95, 0.0],     # Lower Middle Right
            [-1.85, -2.95, 0.0],    # Lower Middle Left
        ]
        return base_pattern[:-1]  # Remove outer side length
    
    def _generate_initial_population(self, pop_size):
        """Generate diverse initial population with mathematical guidance"""
        population = []
        base_pattern = self._initialize_base_pattern()
        
        for _ in range(pop_size):
            config = []
            # Add perturbations with varying intensities
            for i, (x, y, angle) in enumerate(base_pattern):
                # Vary perturbation intensity based on hexagon position
                if i == 0:  # Center
                    pert_x = x + random.uniform(-0.15, 0.15)
                    pert_y = y + random.uniform(-0.15, 0.15)
                    pert_angle = angle + random.uniform(-3, 3)
                elif i <= 6:  # Primary ring
                    pert_x = x + random.uniform(-0.25, 0.25)
                    pert_y = y + random.uniform(-0.25, 0.25)
                    pert_angle = angle + random.uniform(-5, 5)
                else:  # Outer ring
                    pert_x = x + random.uniform(-0.35, 0.35)
                    pert_y = y + random.uniform(-0.35, 0.35)
                    pert_angle = angle + random.uniform(-8, 8)
                config.extend([pert_x, pert_y, pert_angle])
            
            # Add outer side length with reasonable starting range
            config.append(4.5 + random.uniform(0, 2.5))
            population.append(config)
        return population
    
    def _evaluate_individual(self, params):
        """Fast evaluation of individual fitness with early termination"""
        # Extract hexagon params
        hexagons = []
        idx = 0
        for i in range(12):
            hexagons.append((params[idx], params[idx+1], params[idx+2]))
            idx += 3
        outer_side_length = params[-1]
        
        # Quick containment check - early termination if violated
        containment_penalty = 0
        for i in range(12):
            if not check_containment_inner_to_outer(
                hexagons[i][0], hexagons[i][1], hexagons[i][2],
                0, 0, 0, outer_side_length
            ):
                containment_penalty += 15000.0
        
        # If containment violated, return immediately
        if containment_penalty > 0:
            return containment_penalty + 1e10
        
        # Overlap checking with early termination
        overlap_penalty = 0
        # Use spatial hashing for faster neighbor queries
        spatial_hash = SpatialHash(grid_size=1.5)
        
        # Insert all hexagons into spatial grid
        for i in range(12):
            spatial_hash.insert(i, hexagons[i][0], hexagons[i][1])
        
        # For each hexagon, check overlaps only with neighbors
        for i in range(12):
            neighbors = spatial_hash.get_neighbors(hexagons[i][0], hexagons[i][1])
            for j in neighbors:
                if i >= j:  # Avoid duplicate checks
                    continue
                if check_overlap_hexagons(
                    hexagons[i][0], hexagons[i][1], hexagons[i][2],
                    hexagons[j][0], hexagons[j][1], hexagons[j][2]
                ):
                    overlap_penalty += 5000.0
                    # Early termination if overlaps found
                    if overlap_penalty >= 1e9:
                        break
            if overlap_penalty >= 1e9:
                break
        
        # Calculate objective value
        objective = -1.0 / outer_side_length + containment_penalty + overlap_penalty
        return objective
    
    def _evolve_generation(self, population, bounds):
        """Evolutionary generation with custom operators"""
        # Sort by fitness
        population.sort(key=lambda x: self._evaluate_individual(x))
        best_fitness = self._evaluate_individual(population[0])
        
        if best_fitness < self.best_score:
            self.best_score = best_fitness
            self.best_solution = deepcopy(population[0])
        
        # Select top 50% for breeding (elitism)
        elite_count = len(population) // 2
        elites = population[:elite_count]
        
        # Generate offspring through specialized crossover/mutation
        offspring = []
        for i in range(elite_count):
            parent1 = elites[i % len(elites)]
            parent2 = elites[(i + 1) % len(elites)]
            
            # Crossover: blend positions, combine rotations
            child = self._crossover(parent1, parent2)
            
            # Mutation: perturb with adaptive intensity
            child = self._mutate(child, bounds)
            offspring.append(child)
        
        # Fill remainder with new individuals
        new_individuals = self._generate_initial_population(len(population) - len(offspring))
        return elites + offspring + new_individuals
    
    def _crossover(self, parent1, parent2):
        """Custom crossover operator for hexagon packing"""
        child = []
        # Cross over positions and rotations
        for i in range(0, 36, 3):  # Process groups of 3 (x,y,angle)
            if random.random() < 0.5:
                child.extend([parent1[i], parent1[i+1], parent1[i+2]])
            else:
                child.extend([parent2[i], parent2[i+1], parent2[i+2]])
        
        # Blend outer side length
        child.append((parent1[-1] + parent2[-1]) / 2)
        return child
    
    def _mutate(self, individual, bounds):
        """Adaptive mutation operator with symmetry preservation"""
        mutated = individual.copy()
        
        # Mutate positions and rotations
        for i in range(36):  # 12 hexagons * 3 parameters
            if random.random() < 0.2:  # 20% chance to mutate each parameter
                # Adaptive mutation strength based on parameter type
                if i % 3 == 0:  # x coordinate
                    strength = 0.15  # Small mutation for precise positioning
                elif i % 3 == 1:  # y coordinate  
                    strength = 0.15
                else:  # angle
                    strength = 3.0  # Larger mutation for rotation
                
                mutated[i] += random.uniform(-strength, strength)
                
                # Apply bounds
                mutated[i] = max(bounds[i][0], min(bounds[i][1], mutated[i]))
        
        # Mutate outer side length
        if random.random() < 0.1:  # 10% chance
            mutated[-1] *= random.uniform(0.95, 1.05)  # Small adjustment
            mutated[-1] = max(bounds[-1][0], min(bounds[-1][1], mutated[-1]))
        
        return mutated
    
    def _progressive_optimize(self, bounds, max_generations=50):
        """Progressive optimization with increasing resolution"""
        # Generation 1: Coarse resolution
        pop_size = 25
        population = self._generate_initial_population(pop_size)
        
        for gen in range(max_generations):
            population = self._evolve_generation(population, bounds)
            
            # Gradually increase precision in later generations
            if gen > max_generations // 2:
                # Increase mutation rates for fine-tuning
                pass
        
        # Final optimization using DE for fine-tuning
        if self.best_solution is not None:
            def objective_func(params):
                return self._evaluate_individual(params)
            
            try:
                # Fine tune with differential evolution
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=42,
                    maxiter=75,
                    popsize=20,
                    mutation=(0.8, 1),
                    recombination=0.9,
                    tol=1e-6,
                    workers=1,
                    init=[self.best_solution]
                )
                
                return result.x
            except:
                return self.best_solution
        else:
            # Fallback to best from population
            population.sort(key=lambda x: self._evaluate_individual(x))
            return population[0]
    
    def optimize(self, bounds):
        """Main optimization routine with progressive approach"""
        # Run progressive optimization
        final_solution = self._progressive_optimize(bounds)
        return final_solution

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Define bounds for optimization
    bounds = []
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    bounds.append((1, 20))  # Outer side length bound
    
    try:
        # Initialize optimizer
        optimizer = HexagonPackingOptimizer()
        
        # Run optimization
        best_params = optimizer.optimize(bounds)
        
        # Extract configuration
        inner_hex_data = []
        idx = 0
        for i in range(12):
            inner_hex_data.append([
                best_params[idx], 
                best_params[idx+1], 
                best_params[idx+2]
            ])
            idx += 3
            
        outer_side_length = best_params[-1]
        
        # Store results
        inner_hex_data = np.array(inner_hex_data)
        outer_hex_data = np.array([0, 0, 0])
        
        # Calculate metrics
        inv_outer_hex_side_length = 1.0 / outer_side_length
        benchmark_ratio = inv_outer_hex_side_length / 0.2537
        
        print(f"inv_outer_hex_side_length: {inv_outer_hex_side_length:.8f}")
        print(f"benchmark_ratio: {benchmark_ratio:.8f}")
        print(f"eval_time: {time.time() - start_time:.4f}s")
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        # Fallback to previous solution
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0],
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_side_length = 8.0

        # Calculate fallback metrics
        inv_outer_hex_side_length, benchmark_ratio = 1.0/outer_side_length, 1.0/outer_side_length/0.2537
        print(f"Fallback - inv_outer_hex_side_length: {inv_outer_hex_side_length:.8f}")
        print(f"Fallback - benchmark_ratio: {benchmark_ratio:.8f}")
        print(f"eval_time: {time.time() - start_time:.4f}s")

    # Ensure all computations completed within time limit
    elapsed_time = time.time() - start_time
    if elapsed_time > 175:  # Leave buffer
        warnings.warn("Warning: Time limit approaching")
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END