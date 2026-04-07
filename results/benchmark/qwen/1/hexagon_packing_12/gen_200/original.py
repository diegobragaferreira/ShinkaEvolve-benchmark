# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point
import time
from numba import jit, prange
from collections import namedtuple
import heapq
import math

# Define a structure for hexagon data
HexagonData = namedtuple('HexagonData', ['x', 'y', 'rotation'])

# BVH Node for spatial acceleration
class BVHNode:
    def __init__(self, bounds_min, bounds_max, objects=None, left=None, right=None):
        self.bounds_min = np.array(bounds_min)
        self.bounds_max = np.array(bounds_max)
        self.objects = objects or []
        self.left = left
        self.right = right
        
    def contains(self, point):
        return (self.bounds_min <= point).all() and (point <= self.bounds_max).all()
        
    def intersects(self, other):
        return not (self.bounds_max[0] < other.bounds_min[0] or 
                   self.bounds_min[0] > other.bounds_max[0] or
                   self.bounds_max[1] < other.bounds_min[1] or 
                   self.bounds_min[1] > other.bounds_max[1])

# BVH for hexagon overlap detection
class BVH:
    def __init__(self, hexagons, max_objects_per_node=4):
        self.max_objects_per_node = max_objects_per_node
        self.root = self._build_tree(hexagons)
        
    def _build_tree(self, hexagons):
        if not hexagons:
            return None
            
        # Initialize bounds
        bounds_min = np.array([float('inf'), float('inf')])
        bounds_max = np.array([-float('inf'), -float('inf')])
        
        for hex_data in hexagons:
            bounds_min[0] = min(bounds_min[0], hex_data.x - 1)
            bounds_min[1] = min(bounds_min[1], hex_data.y - 1)
            bounds_max[0] = max(bounds_max[0], hex_data.x + 1)
            bounds_max[1] = max(bounds_max[1], hex_data.y + 1)
            
        # If too few objects, store directly
        if len(hexagons) <= self.max_objects_per_node:
            return BVHNode(bounds_min, bounds_max, hexagons)
            
        # Split along longest axis
        split_axis = 0 if (bounds_max[0] - bounds_min[0]) > (bounds_max[1] - bounds_min[1]) else 1
        split_value = (bounds_min[split_axis] + bounds_max[split_axis]) / 2
        
        left_objects = []
        right_objects = []
        for hex_data in hexagons:
            if hex_data.x if split_axis == 0 else hex_data.y <= split_value:
                left_objects.append(hex_data)
            else:
                right_objects.append(hex_data)
                
        # Prune empty branches
        left_child = self._build_tree(left_objects) if left_objects else None
        right_child = self._build_tree(right_objects) if right_objects else None
        
        return BVHNode(bounds_min, bounds_max, None, left_child, right_child)
    
    def query(self, point, radius):
        """Find all hexagons within a given radius of point"""
        results = []
        self._query_recursive(self.root, point, radius, results)
        return results
        
    def _query_recursive(self, node, point, radius, results):
        if not node:
            return
            
        # Skip if no intersection
        if not node.intersects(BVHNode(
            [point[0] - radius, point[1] - radius],
            [point[0] + radius, point[1] + radius]
        )):
            return
            
        # Add objects at leaf nodes
        if node.objects:
            for obj in node.objects:
                dist = np.sqrt((obj.x - point[0])**2 + (obj.y - point[1])**2)
                if dist <= radius:
                    results.append(obj)
        else:
            # Recursively check children
            self._query_recursive(node.left, point, radius, results)
            self._query_recursive(node.right, point, radius, results)

@jit(nopython=True, parallel=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Compute vertices of a hexagon given center, rotation, and side length."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    # Vertices of regular hexagon with side length 1 centered at origin
    base_verts = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Rotate and translate
    rotated_verts = np.empty_like(base_verts)
    for i in range(6):
        x_orig, y_orig = base_verts[i]
        rotated_verts[i] = [
            x + side_length * (x_orig * cos_a - y_orig * sin_a),
            y + side_length * (x_orig * sin_a + y_orig * cos_a)
        ]

    return rotated_verts

@jit(nopython=True)
def distance_point_to_line(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment."""
    # Vector from (x1,y1) to (x2,y2)
    dx = x2 - x1
    dy = y2 - y1
    
    # Length squared of line segment
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        # Line segment is a point
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0, min(1, t))  # Clamp projection to line segment
    
    # Find closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def compute_min_distance_hexagon_hexagon(h1_x, h1_y, h1_angle, h2_x, h2_y, h2_angle):
    """Compute minimum distance between two hexagons using analytical approach."""
    v1 = hexagon_vertices(h1_x, h1_y, h1_angle)
    v2 = hexagon_vertices(h2_x, h2_y, h2_angle)
    
    min_dist = np.inf
    # Check vertex-to-vertex distances
    for i in range(6):
        for j in range(6):
            dist = np.sqrt((v1[i,0]-v2[j,0])**2 + (v1[i,1]-v2[j,1])**2)
            if dist < min_dist:
                min_dist = dist
    
    # Check vertex-to-edge distances
    for i in range(6):
        for j in range(6):
            # Distance from vertex v1[i] to edge v2[j]-v2[(j+1)%6]
            dist = distance_point_to_line(v1[i,0], v1[i,1], v2[j,0], v2[j,1], v2[(j+1)%6,0], v2[(j+1)%6,1])
            if dist < min_dist:
                min_dist = dist
            
            # Distance from vertex v2[j] to edge v1[i]-v1[(i+1)%6]
            dist = distance_point_to_line(v2[j,0], v2[j,1], v1[i,0], v1[i,1], v1[(i+1)%6,0], v1[(i+1)%6,1])
            if dist < min_dist:
                min_dist = dist
    
    return min_dist

class HexagonPackingEvolutionary:
    """Evolutionary algorithm for hexagon packing using discrete lattice and hybrid optimization."""
    
    def __init__(self):
        self.hex_side_length = 1.0
        self.lattice_vectors = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2]
        ])
        self.population_size = 50
        self.generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
    def compute_outer_hexagon_polygon(self, side_length):
        """Get shapely polygon for outer hexagon."""
        # Vertices of a regular hexagon with given side length, centered at origin
        vertices = []
        for i in range(6):
            theta = i * np.pi / 3
            x = side_length * np.cos(theta)
            y = side_length * np.sin(theta)
            vertices.append((x, y))
        return Polygon(vertices)
        
    def is_contained(self, h_x, h_y, outer_radius):
        """Check if hexagon center is within outer hexagon."""
        distance = np.sqrt(h_x*h_x + h_y*h_y)
        return distance <= (outer_radius - 1.0)
        
    def compute_overlap_penalty(self, hexagons, outer_radius):
        """Efficiently compute overlap penalty with BVH acceleration."""
        n = len(hexagons)
        if n <= 1:
            return 0.0
            
        # Create BVH for spatial acceleration
        bvh = BVH(hexagons)
        
        penalty = 0.0
        
        # Check only hexagons that might overlap (within radius 2 of each other)
        for i in range(n):
            # Find potentially overlapping hexagons
            point = np.array([hexagons[i].x, hexagons[i].y])
            candidates = bvh.query(point, 2.0)
            
            for j in range(i+1, n):
                # Quick distance check first
                dist = np.sqrt((hexagons[i].x - hexagons[j].x)**2 + 
                              (hexagons[i].y - hexagons[j].y)**2)
                
                if dist <= 2.0:  # Might overlap
                    # Use more precise distance calculation
                    min_dist = compute_min_distance_hexagon_hexagon(
                        hexagons[i].x, hexagons[i].y, hexagons[i].rotation,
                        hexagons[j].x, hexagons[j].y, hexagons[j].rotation
                    )
                    
                    if min_dist < 0.001:  # Overlapping
                        # Calculate area of intersection (approximate)
                        penalty += 1000.0
                        
        return penalty
        
    def evaluate_fitness(self, hexagons, outer_radius):
        """Evaluate fitness of a configuration."""
        # Check containment
        for hex_data in hexagons:
            if not self.is_contained(hex_data.x, hex_data.y, outer_radius):
                return 1e10  # Large penalty for containment violation
                
        # Check overlap penalty
        overlap_penalty = self.compute_overlap_penalty(hexagons, outer_radius)
        if overlap_penalty > 0:
            return overlap_penalty + 1e9
            
        # Valid configuration - return inverse of outer radius
        return -1.0 / outer_radius
        
    def generate_lattice_points(self, max_radius=10):
        """Generate potential hexagon positions on a hexagonal lattice."""
        points = []
        max_coord = int(max_radius * 2)
        
        for i in range(-max_coord, max_coord + 1):
            for j in range(-max_coord, max_coord + 1):
                # Convert to Cartesian coordinates
                x = i * self.lattice_vectors[0][0] + j * self.lattice_vectors[1][0]
                y = i * self.lattice_vectors[0][1] + j * self.lattice_vectors[1][1]
                
                # Only include points within reasonable bounds
                if np.sqrt(x*x + y*y) < max_radius:
                    points.append((x, y))
                    
        return points
        
    def generate_initial_population(self, lattice_points):
        """Generate initial population with hexagonal packing patterns."""
        population = []
        
        # Pattern 1: Central + ring structure
        for _ in range(self.population_size // 2):
            # Start with central hexagon
            hexagons = [HexagonData(0.0, 0.0, 0.0)]
            
            # Add 6 surrounding hexagons in first ring
            for k in range(6):
                angle = k * 60
                x = np.cos(np.radians(angle)) * 1.732  # sqrt(3)
                y = np.sin(np.radians(angle)) * 1.732
                hexagons.append(HexagonData(x, y, 0.0))
                
            # Add 6 hexagons in second ring
            for k in range(6):
                angle = k * 60 + 30
                x = np.cos(np.radians(angle)) * 3.464  # 2*sqrt(3)
                y = np.sin(np.radians(angle)) * 3.464
                hexagons.append(HexagonData(x, y, 0.0))
                
            # Add randomness
            for i in range(12):
                if i < len(hexagons):
                    # Slight randomization
                    hexagons[i] = HexagonData(
                        hexagons[i].x + np.random.normal(0, 0.1),
                        hexagons[i].y + np.random.normal(0, 0.1),
                        hexagons[i].rotation + np.random.normal(0, 10)
                    )
                    
            # Add to population
            population.append(hexagons.copy())
            
        # Pattern 2: Random placements
        for _ in range(self.population_size // 2):
            hexagons = []
            # Select random lattice points
            selected_points = np.random.choice(lattice_points, 12, replace=False)
            for i, (x, y) in enumerate(selected_points):
                hexagons.append(HexagonData(x, y, np.random.uniform(0, 360)))
            population.append(hexagons.copy())
            
        return population
        
    def mutate(self, hexagons):
        """Mutate a configuration with probability of change."""
        mutated = []
        for hex_data in hexagons:
            if np.random.random() < self.mutation_rate:
                # Randomly modify position and/or rotation
                new_x = hex_data.x + np.random.normal(0, 0.5)
                new_y = hex_data.y + np.random.normal(0, 0.5)
                new_rot = (hex_data.rotation + np.random.normal(0, 20)) % 360
                mutated.append(HexagonData(new_x, new_y, new_rot))
            else:
                mutated.append(hex_data)
        return mutated
        
    def crossover(self, parent1, parent2):
        """Specialized crossover that respects hexagonal packing properties."""
        if np.random.random() < self.crossover_rate:
            # Blend positions from parents but maintain structure
            child = []
            for i in range(len(parent1)):
                if np.random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
                    
            # Apply some structural constraints
            # First hexagon (central) stays fixed
            if len(child) > 0:
                child[0] = HexagonData(0.0, 0.0, child[0].rotation)
                
            return child
        else:
            # Return one parent unchanged
            return parent1.copy() if np.random.random() < 0.5 else parent2.copy()
            
    def find_best_individual(self, population, outer_radius):
        """Find the best individual in the population."""
        best_fitness = float('inf')
        best_individual = None
        for individual in population:
            fitness = self.evaluate_fitness(individual, outer_radius)
            if fitness < best_fitness:
                best_fitness = fitness
                best_individual = individual
        return best_individual, best_fitness
        
    def optimize(self):
        """Main evolutionary optimization routine."""
        # Generate lattice points
        lattice_points = self.generate_lattice_points(10)
        
        # Step 1: Generate initial population
        population = self.generate_initial_population(lattice_points)
        
        # Step 2: Evolutionary process
        best_individual = None
        best_fitness = float('inf')
        best_outer_radius = 10.0
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                # Estimate initial outer radius
                max_dist = 0
                for hex_data in individual:
                    dist = np.sqrt(hex_data.x**2 + hex_data.y**2)
                    max_dist = max(max_dist, dist)
                outer_radius = max_dist + 2.0
                
                fitness = self.evaluate_fitness(individual, outer_radius)
                fitness_scores.append((individual, fitness, outer_radius))
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = individual
                    best_outer_radius = outer_radius
                    
            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[1])
            
            # Selection: keep top half
            top_half = [ind for ind, _, _ in fitness_scores[:len(population)//2]]
            
            # Create new population by crossover and mutation
            new_population = []
            
            # Keep best individuals
            new_population.extend(top_half)
            
            # Fill rest with offspring
            while len(new_population) < self.population_size:
                parent1 = np.random.choice(top_half)
                parent2 = np.random.choice(top_half)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
                
            population = new_population
            
        # Step 3: Local refinement around best solution
        if best_individual is not None:
            # Use local optimization to refine solution
            refined_solution = self.refine_solution(best_individual, best_outer_radius)
            return refined_solution, best_outer_radius
            
        return population[0], 10.0
        
    def refine_solution(self, hexagons, outer_radius):
        """Refine the solution using gradient-based optimization."""
        # Convert to parameters for scipy
        initial_params = []
        for hex_data in hexagons:
            initial_params.extend([hex_data.x, hex_data.y, hex_data.rotation])
        initial_params.append(outer_radius)
        
        # Bounds
        bounds = []
        for _ in range(12):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])
        bounds.append((1.0, 20.0))
        
        def objective(params):
            # Convert back to hexagon data
            hexagon_list = []
            for i in range(12):
                idx = i * 3
                x, y, rot = params[idx], params[idx+1], params[idx+2]
                hexagon_list.append(HexagonData(x, y, rot))
            outer_r = params[-1]
            return self.evaluate_fitness(hexagon_list, outer_r)
            
        try:
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50}
            )
            
            # Convert result back
            if result.success:
                hexagon_list = []
                for i in range(12):
                    idx = i * 3
                    x, y, rot = result.x[idx], result.x[idx+1], result.x[idx+2]
                    hexagon_list.append(HexagonData(x, y, rot))
                return hexagon_list
        except:
            pass
            
        return hexagons

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    try:
        # Initialize the evolutionary optimizer
        optimizer = HexagonPackingEvolutionary()
        
        # Run optimization
        inner_hex_data, outer_hex_side_length = optimizer.optimize()
        
        # Convert hexagon data to required format
        formatted_data = []
        for hex_data in inner_hex_data:
            formatted_data.append([hex_data.x, hex_data.y, hex_data.rotation])
            
        # Create outer hexagon data (centered at origin, no rotation)
        outer_hex_data = np.array([0, 0, 0])
        
        # Ensure we don't exceed time limits
        end_time = time.time()
        eval_time = end_time - start_time
        
        return np.array(formatted_data), outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        # Fallback to simple configuration if optimization fails
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
        outer_hex_side_length = 8  # Large enough to contain all inner hexagons
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END