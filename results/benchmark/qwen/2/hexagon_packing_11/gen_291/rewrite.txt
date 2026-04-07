# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from scipy.optimize import differential_evolution, minimize
import time
import math
from typing import Tuple, Optional, List
import warnings
from joblib import Parallel, delayed
import multiprocessing
import random
from numba import jit, prange

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3)/2
BENCHMARK_RATIO = 0.2544

@jit(nopython=True)
def hexagon_vertices_numba(center_x, center_y, rotation_deg, side_length):
    """Fast computation of hexagon vertices using numba"""
    vertices = np.empty((6, 2), dtype=np.float64)
    angle_offset = rotation_deg * np.pi / 180.0
    for i in range(6):
        angle = angle_offset + i * np.pi / 3.0
        vertices[i, 0] = center_x + side_length * np.cos(angle)
        vertices[i, 1] = center_y + side_length * np.sin(angle)
    return vertices

@jit(nopython=True)
def point_in_polygon_fast(point_x, point_y, polygon_vertices):
    """Fast point-in-polygon test using ray casting algorithm"""
    n = len(polygon_vertices)
    inside = False
    
    p1x, p1y = polygon_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_vertices[i % n]
        if point_y > min(p1y, p2y):
            if point_y <= max(p1y, p2y):
                if point_x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (point_y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or point_x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@jit(nopython=True)
def project_polygon_onto_axis_numba(vertices, axis):
    """Project polygon vertices onto axis and return min/max projections"""
    projections = np.empty(len(vertices), dtype=np.float64)
    for i in range(len(vertices)):
        projections[i] = vertices[i, 0] * axis[0] + vertices[i, 1] * axis[1]
    return np.min(projections), np.max(projections)

@jit(nopython=True)
def get_hexagon_axes_numba(vertices):
    """Get axes (normals) of a hexagon for SAT collision detection"""
    axes = np.empty((6, 2), dtype=np.float64)
    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i+1) % 6]
        edge = p2 - p1
        # Normal vector (perpendicular to edge)
        normal = np.array([-edge[1], edge[0]])
        # Normalize
        norm = np.sqrt(normal[0]*normal[0] + normal[1]*normal[1])
        if norm > 1e-10:
            normal = normal / norm
        axes[i] = normal
    return axes

@jit(nopython=True)
def sat_collision_check_numba(hex1_vertices, hex2_vertices):
    """Use Separating Axis Theorem to detect collision between hexagons"""
    # Get axes for both polygons
    axes1 = get_hexagon_axes_numba(hex1_vertices)
    axes2 = get_hexagon_axes_numba(hex2_vertices)
    all_axes = np.vstack([axes1, axes2])

    # Check each axis
    for axis in all_axes:
        min1, max1 = project_polygon_onto_axis_numba(hex1_vertices, axis)
        min2, max2 = project_polygon_onto_axis_numba(hex2_vertices, axis)

        # Check for separation
        if max1 < min2 or max2 < min1:
            return False  # No overlap on this axis, so they don't collide

    return True  # Overlap on all axes, so they collide

class HexagonPacker:
    def __init__(self):
        self.num_inner = 11
        self.unit_hex_radius = UNIT_HEX_RADIUS
        self.unit_hex_apogee = UNIT_HEX_APOGEE
        self.max_iterations = 1000
        self.population_size = 50
        self.local_refinements = 3

    def create_unit_hexagon_vertices(self, center=(0,0), rotation=0):
        """Create unit hexagon vertices efficiently"""
        return hexagon_vertices_numba(center[0], center[1], rotation, self.unit_hex_radius)

    def validate_hexagon_placement(self, inner_hex_data, outer_radius):
        """High-performance constraint validation with early termination"""
        # Precompute all hexagon vertices
        hex_vertices = []
        for i in range(self.num_inner):
            center = (inner_hex_data[i][0], inner_hex_data[i][1])
            rotation = inner_hex_data[i][2]
            vertices = self.create_unit_hexagon_vertices(center, rotation)
            hex_vertices.append(vertices)

        # Create outer hexagon vertices
        outer_vertices = self.create_unit_hexagon_vertices((0, 0), 0)
        scaled_outer_vertices = outer_vertices * outer_radius

        # Check containment for all hexagons (early termination)
        for i in range(self.num_inner):
            vertices = hex_vertices[i]
            # Check if any vertex lies outside outer hexagon
            for vertex in vertices:
                if not point_in_polygon_fast(vertex[0], vertex[1], scaled_outer_vertices):
                    return False, 0.0  # containment violated

        # Check overlaps (early termination)
        for i in range(self.num_inner):
            for j in range(i+1, self.num_inner):
                if sat_collision_check_numba(hex_vertices[i], hex_vertices[j]):
                    return False, 0.0  # overlap violated

        # Valid configuration
        return True, 1.0 / outer_radius

    def _hexagonal_cluster_initialization(self):
        """Initialize population with hexagonal clustering pattern"""
        configs = []
        
        # Base hexagonal cluster pattern
        base_patterns = [
            # Central hexagon
            [(0, 0, 0)],
            # Ring around center
            [(0, 0, 0), (2.0, 0, 0), (-2.0, 0, 0), (1.0, 1.732, 0), (-1.0, 1.732, 0), 
             (1.0, -1.732, 0), (-1.0, -1.732, 0)],
            # Double ring
            [(0, 0, 0), (2.0, 0, 0), (-2.0, 0, 0), (1.0, 1.732, 0), (-1.0, 1.732, 0), 
             (1.0, -1.732, 0), (-1.0, -1.732, 0), (3.0, 0, 0), (-3.0, 0, 0), 
             (2.0, 3.464, 0), (-2.0, 3.464, 0), (2.0, -3.464, 0), (-2.0, -3.464, 0)]
        ]
        
        # Generate diverse configurations from base patterns
        for base_pattern in base_patterns:
            for _ in range(5):  # Create multiple variants
                config = []
                for i, (cx, cy, rot) in enumerate(base_pattern):
                    # Add jitter to positions and rotations
                    jitter_x = np.random.normal(0, 0.2)
                    jitter_y = np.random.normal(0, 0.2)
                    jitter_rot = np.random.normal(0, 10)
                    config.append((cx + jitter_x, cy + jitter_y, (rot + jitter_rot) % 360))
                configs.append(config)
        
        # Add some random configurations
        for _ in range(20):
            config = []
            for i in range(self.num_inner):
                config.append((
                    np.random.uniform(-3, 3),
                    np.random.uniform(-3, 3),
                    np.random.uniform(0, 360)
                ))
            configs.append(config)
            
        return configs

    def _initialize_population(self):
        """Create diverse initial population using hexagonal clustering"""
        configs = self._hexagonal_cluster_initialization()
        
        # Convert to numpy arrays with proper structure
        population = []
        for config in configs:
            individual = []
            for cx, cy, rot in config:
                individual.extend([cx, cy, rot])
            # Add outer radius estimate - based on maximum distance from center
            max_dist = 0
            for cx, cy, _ in config:
                dist = np.sqrt(cx*cx + cy*cy) + self.unit_hex_apogee
                max_dist = max(max_dist, dist)
            individual.append(max_dist + 0.5)  # Add margin
            population.append(np.array(individual))
            
        return population

    def _evaluate_population_parallel(self, population):
        """Parallel evaluation of population fitness"""
        def evaluate_individual(individual):
            # Extract parameters
            inner_params = individual[:-1]
            outer_radius = individual[-1]
            
            # Reshape inner parameters
            inner_data = np.zeros((self.num_inner, 3))
            for i in range(self.num_inner):
                inner_data[i] = [inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]]
            
            # Validate and return fitness
            valid, fitness = self.validate_hexagon_placement(inner_data, outer_radius)
            if valid:
                return fitness
            else:
                return -1e10  # Penalty for invalid configurations
                
        # Parallel evaluation
        fitness_scores = Parallel(n_jobs=min(multiprocessing.cpu_count(), 8))(
            delayed(evaluate_individual)(individual) for individual in population
        )
        
        return fitness_scores

    def _quantum_mutation(self, individual, generation, max_generations):
        """Quantum-inspired mutation with adaptive parameters"""
        mutated = individual.copy()
        # Adaptive mutation rate decreases over time
        mutation_rate = 0.1 * (1.0 - generation / max_generations)
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                if i < len(mutated) - 1:  # Position or rotation parameters
                    if i % 3 == 0:  # x position
                        mutated[i] += np.random.normal(0, 0.2)
                    elif i % 3 == 1:  # y position
                        mutated[i] += np.random.normal(0, 0.2)
                    else:  # rotation
                        mutated[i] += np.random.normal(0, 15)
                        mutated[i] = mutated[i] % 360
                else:  # outer radius
                    mutated[i] += np.random.normal(0, 0.1)
                    mutated[i] = max(2.0, mutated[i])
                    
        return mutated

    def _quantum_crossover(self, parent1, parent2):
        """Quantum-inspired crossover operation"""
        # Quantum-like uniform crossover
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
                
        return child1, child2

    def _hybrid_evolutionary_search(self):
        """Main hybrid evolutionary search with quantum-inspired operators"""
        # Initialize population
        population = self._initialize_population()
        best_individual = None
        best_fitness = -1e10
        
        # Evolutionary parameters
        max_generations = 100
        elite_size = 3
        
        for generation in range(max_generations):
            # Evaluate population
            fitness_scores = self._evaluate_population_parallel(population)
            
            # Track best solution
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_individual = population[best_idx].copy()
            
            # Selection - tournament selection
            selected_parents = []
            for _ in range(self.population_size // 2):
                tournament_indices = np.random.choice(len(population), 3)
                tournament_fitnesses = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
                selected_parents.append(population[winner_idx].copy())
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = selected_parents[np.random.randint(len(selected_parents))]
                parent2 = selected_parents[np.random.randint(len(selected_parents))]
                
                child1, child2 = self._quantum_crossover(parent1, parent2)
                child1 = self._quantum_mutation(child1, generation, max_generations)
                child2 = self._quantum_mutation(child2, generation, max_generations)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:self.population_size]
            
            # Early stopping if improvement is minimal
            if generation > 20 and generation % 10 == 0:
                # Check recent improvement
                recent_scores = fitness_scores[-20:] if len(fitness_scores) >= 20 else fitness_scores
                if len(recent_scores) > 1:
                    improvement = max(recent_scores) - min(recent_scores)
                    if improvement < 1e-6:
                        break
        
        return best_individual

    def _geometric_local_refinement(self, initial_individual):
        """Specialized geometric local refinement"""
        # Convert to structured format
        inner_params = initial_individual[:-1]
        outer_radius = initial_individual[-1]
        
        # Reshape inner parameters
        inner_data = np.zeros((self.num_inner, 3))
        for i in range(self.num_inner):
            inner_data[i] = [inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]]
        
        # Geometric optimization using local search
        def objective_function(params):
            # Reshape parameters
            data = params.reshape(self.num_inner, 3)
            outer_rad = params[-1]  # Last element is outer radius
            
            # Validate and return fitness
            valid, fitness = self.validate_hexagon_placement(data, outer_rad)
            if valid:
                return -fitness  # Negative for minimization
            else:
                return 1e10  # Large penalty
        
        # Start with initial solution
        flat_params = inner_params.copy()
        flat_params = np.append(flat_params, outer_radius)
        
        # Local optimization with L-BFGS-B
        try:
            bounds = []
            # Position bounds
            for _ in range(self.num_inner):
                bounds.extend([(-10.0, 10.0), (-10.0, 10.0), (0.0, 360.0)])
            # Outer radius bound
            bounds.append((2.0, 15.0))
            
            result = minimize(
                objective_function,
                flat_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 100}
            )
            
            if result.success:
                refined_params = result.x
                refined_inner = refined_params[:-1].reshape(self.num_inner, 3)
                refined_outer = refined_params[-1]
                
                # Validate refined solution
                valid, _ = self.validate_hexagon_placement(refined_inner, refined_outer)
                if valid:
                    return np.append(refined_inner.flatten(), refined_outer)
                    
        except Exception:
            pass
            
        return initial_individual

    def optimize_with_hierarchical_approach(self):
        """Main optimization using hierarchical approach"""
        # Phase 1: Global search with hybrid evolutionary algorithm
        print("Starting global evolutionary search...")
        global_best = self._hybrid_evolutionary_search()
        
        if global_best is None:
            raise RuntimeError("No valid solution found from global search")
        
        # Phase 2: Local geometric refinement
        print("Starting local geometric refinement...")
        refined_solution = self._geometric_local_refinement(global_best)
        
        # Phase 3: SAT-based tightening (if needed)
        print("Performing SAT-based validation and tightening...")
        # Convert back to structured format for validation
        inner_params = refined_solution[:-1]
        outer_radius = refined_solution[-1]
        
        # Reshape inner parameters
        inner_data = np.zeros((self.num_inner, 3))
        for i in range(self.num_inner):
            inner_data[i] = [inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]]
        
        # Final validation
        valid, final_fitness = self.validate_hexagon_placement(inner_data, outer_radius)
        
        if not valid:
            # If still invalid, use the best global solution anyway for fallback
            return global_best
        
        return refined_solution

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses hybrid quantum-inspired evolutionary algorithm with hexagonal clustering initialization, 
    geometric local refinement, and specialized SAT-based validation.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        packer = HexagonPacker()
        
        # Run hierarchical optimization
        final_solution = packer.optimize_with_hierarchical_approach()
        
        # Extract results
        inner_params = final_solution[:-1]
        outer_radius = final_solution[-1]
        
        # Reshape inner parameters to proper format
        inner_hex_data = np.zeros((11, 3))
        for i in range(11):
            inner_hex_data[i] = [inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]]
        
        outer_hex_data = np.array([0, 0, 0])
        
        return inner_hex_data, outer_hex_data, outer_radius
        
    except Exception as e:
        warnings.warn(f"Optimization failed with error: {str(e)}")
        pass
    
    # Fallback to original approach if optimization fails
    inner_hex_data = np.array([
        [0, 0, 0],        # center
        [-2.5, 0, 0],     # left
        [2.5, 0, 0],      # right
        [-1.25, 2.17, 0], # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0], # bottom-left
        [1.25, -2.17, 0], # bottom-right
        [-3.75, 2.17, 0], # far top-left
        [3.75, 2.17, 0],  # far top-right
        [-3.75, -2.17, 0], # far bottom-left
        [3.75, -2.17, 0], # far bottom-right
    ])
    
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 8  # large enough to contain all inner hexagons
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END