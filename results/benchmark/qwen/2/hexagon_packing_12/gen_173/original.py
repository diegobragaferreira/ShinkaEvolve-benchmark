# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
from numba import njit, prange
import random
from collections import defaultdict
import heapq

# Numba-compiled functions for performance
@njit
def hexagon_vertices_numba(center_x, center_y, angle_deg, side_length=1):
    """Fast computation of hexagon vertices using Numba JIT."""
    angle_rad = np.radians(angle_deg)
    # Vertices of a regular hexagon with side length 1, centered at origin
    base_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ], dtype=np.float64)

    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotated_vertices = np.empty_like(base_vertices)

    for i in range(6):
        rotated_vertices[i] = np.array([
            base_vertices[i][0] * cos_a - base_vertices[i][1] * sin_a,
            base_vertices[i][0] * sin_a + base_vertices[i][1] * cos_a
        ])

    return rotated_vertices + np.array([center_x, center_y], dtype=np.float64)

@njit
def distance_point_to_segment(px, py, x1, y1, x2, y2):
    """Compute distance from point to line segment."""
    # Vector from (x1,y1) to (x2,y2)
    dx, dy = x2 - x1, y2 - y1
    # Length squared of segment
    length_sq = dx*dx + dy*dy
    if length_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto segment
    t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / length_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

@njit
def point_in_hexagon_fast(px, py, center_x, center_y, angle_deg, side_length=1):
    """Fast point-in-hexagon check using distance from center."""
    # Distance from point to center
    dx, dy = px - center_x, py - center_y
    dist = np.sqrt(dx*dx + dy*dy)
    
    # For a unit hexagon, maximum distance from center to any vertex is 1
    # But we account for rotation in our geometry
    return dist <= side_length

@njit
def compute_hexagon_bounds(center_x, center_y, angle_deg, side_length=1):
    """Compute tight bounding box for hexagon."""
    vertices = hexagon_vertices_numba(center_x, center_y, angle_deg, side_length)
    min_x, max_x = vertices[:, 0].min(), vertices[:, 0].max()
    min_y, max_y = vertices[:, 1].min(), vertices[:, 1].max()
    return min_x, max_x, min_y, max_y

@njit
def check_overlap_sat(vertices1, vertices2):
    """Separating Axis Theorem for hexagon overlap."""
    # Get all edges of both hexagons
    edges1 = np.empty((6, 2), dtype=np.float64)
    edges2 = np.empty((6, 2), dtype=np.float64)

    for i in range(6):
        edges1[i] = vertices1[i] - vertices1[(i+1)%6]
        edges2[i] = vertices2[i] - vertices2[(i+1)%6]

    # Project both hexagons onto each edge direction
    all_axes = np.vstack([edges1, edges2])

    for axis in all_axes:
        # Normalize axis
        axis_norm = np.sqrt(axis[0]**2 + axis[1]**2)
        if axis_norm == 0:
            continue
        norm_axis = axis / axis_norm

        # Project both polygons onto this axis
        proj1 = np.empty(6, dtype=np.float64)
        proj2 = np.empty(6, dtype=np.float64)

        for i in range(6):
            proj1[i] = vertices1[i][0] * norm_axis[0] + vertices1[i][1] * norm_axis[1]
            proj2[i] = vertices2[i][0] * norm_axis[0] + vertices2[i][1] * norm_axis[1]

        # Check for overlap
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()

        # If no overlap, then they don't intersect
        if max1 < min2 or max2 < min1:
            return False

    return True

class QuadTree:
    """Quadtree for efficient spatial queries."""
    
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary  # [x_min, y_min, x_max, y_max]
        self.capacity = capacity
        self.points = []
        self.divided = False
        
    def insert(self, point):
        """Insert a point into the quadtree."""
        if not self._in_boundary(point):
            return False
            
        if len(self.points) < self.capacity and not self.divided:
            self.points.append(point)
            return True
            
        if not self.divided:
            self._subdivide()
            
        for quadrant in self.quadrants:
            if quadrant.insert(point):
                return True
        return False
        
    def _in_boundary(self, point):
        """Check if point is within boundary."""
        x, y = point
        return (self.boundary[0] <= x <= self.boundary[2] and 
                self.boundary[1] <= y <= self.boundary[3])
                
    def _subdivide(self):
        """Divide the quadtree into quadrants."""
        x_min, y_min, x_max, y_max = self.boundary
        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2
        
        self.quadrants = []
        
        # Create four quadrants
        self.quadrants.append(QuadTree([x_min, y_min, mid_x, mid_y], self.capacity))
        self.quadrants.append(QuadTree([mid_x, y_min, x_max, mid_y], self.capacity))
        self.quadrants.append(QuadTree([x_min, mid_y, mid_x, y_max], self.capacity))
        self.quadrants.append(QuadTree([mid_x, mid_y, x_max, y_max], self.capacity))
        
        # Redistribute points
        for point in self.points:
            for quadrant in self.quadrants:
                if quadrant.insert(point):
                    break
                    
        self.divided = True
        
    def query_range(self, range_rect):
        """Query all points within a rectangular range."""
        found_points = []
        
        if not self._intersects(range_rect):
            return found_points
            
        for point in self.points:
            x, y = point
            if (range_rect[0] <= x <= range_rect[2] and 
                range_rect[1] <= y <= range_rect[3]):
                found_points.append(point)
                
        if self.divided:
            for quadrant in self.quadrants:
                found_points.extend(quadrant.query_range(range_rect))
                
        return found_points
        
    def _intersects(self, range_rect):
        """Check if rectangle intersects with boundary."""
        x_min, y_min, x_max, y_max = self.boundary
        rx_min, ry_min, rx_max, ry_max = range_rect
        
        return not (x_max < rx_min or rx_max < x_min or y_max < ry_min or ry_max < y_min)

def create_hexagon_polygon(center_x, center_y, angle_deg, side_length=1):
    """Create a Shapely polygon for a hexagon."""
    vertices = hexagon_vertices_numba(center_x, center_y, angle_deg, side_length)
    return Polygon(vertices)

def check_containment_shapely(inner_poly, outer_poly):
    """Check containment using Shapely operations."""
    return outer_poly.contains(inner_poly)

def check_overlap_shapely(poly1, poly2):
    """Check overlap using Shapely operations."""
    return poly1.intersects(poly2)

def calculate_min_enclosing_hexagon_fast(inner_hex_data, scale_factor=1.05):
    """Fast calculation of minimum enclosing hexagon using bounding circle."""
    # Get all vertices of all inner hexagons
    all_vertices = np.empty((0, 2), dtype=np.float64)

    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = hexagon_vertices_numba(center_x, center_y, angle)
        all_vertices = np.vstack([all_vertices, vertices])

    if len(all_vertices) == 0:
        return 1.0, np.array([0., 0.])

    # Find bounding circle radius
    centroid = np.mean(all_vertices, axis=0)
    distances = np.sqrt(np.sum((all_vertices - centroid)**2, axis=1))
    max_distance = np.max(distances)

    # For a regular hexagon, side length = max_distance * sqrt(3)/2
    side_length = max_distance * 2 / np.sqrt(3) * scale_factor

    return side_length, centroid

def generate_symmetric_initial_population(pop_size, num_hexagons=12):
    """Generate intelligent initial population with symmetry patterns."""
    population = []

    # Base symmetric configuration: hexagonal pattern around center
    # Pattern: one center + 6 surrounding + 5 additional
    base_angles = np.linspace(0, 360, 7, endpoint=False)  # 6 surrounding + 1 center

    for _ in range(pop_size):
        # Start with a symmetric layout
        hex_config = []

        # Central hexagon
        hex_config.append([0, 0, np.random.uniform(0, 360)])

        # Surrounding hexagons forming a ring
        for i in range(1, 7):  # First 6 surrounding
            radius = 2.0
            angle_rad = np.radians(base_angles[i] + np.random.uniform(-30, 30))
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            angle = np.random.uniform(0, 360)
            hex_config.append([x, y, angle])

        # Add remaining 5 hexagons with additional symmetry
        for i in range(7, 12):
            # Distribute around a larger ring
            radius = 3.5
            angle_rad = np.radians(base_angles[i % 6] + np.random.uniform(-15, 15))
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            angle = np.random.uniform(0, 360)
            hex_config.append([x, y, angle])

        # Add small random perturbations to break exact symmetry
        for i in range(len(hex_config)):
            hex_config[i][0] += np.random.normal(0, 0.1)
            hex_config[i][1] += np.random.normal(0, 0.1)
            hex_config[i][2] += np.random.uniform(-5, 5)

        population.append(np.array(hex_config).flatten())

    return population

def setup_spatial_index(inner_hex_data):
    """Setup spatial index for efficient overlap checks."""
    # Determine bounding box
    all_bounds = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        min_x, max_x, min_y, max_y = compute_hexagon_bounds(center_x, center_y, angle)
        all_bounds.append([min_x, min_y, max_x, max_y])
    
    if not all_bounds:
        return None
        
    # Create bounding box for quadtree
    bbox = [
        min(bound[0] for bound in all_bounds),
        min(bound[1] for bound in all_bounds),
        max(bound[2] for bound in all_bounds),
        max(bound[3] for bound in all_bounds)
    ]
    
    # Create quadtree with slight padding
    padding = 2.0
    bbox = [
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding
    ]
    
    qt = QuadTree(bbox, capacity=4)
    
    # Insert centroids into quadtree
    for i in range(len(inner_hex_data)):
        center_x, center_y, _ = inner_hex_data[i]
        qt.insert((center_x, center_y))
        
    return qt

def check_overlaps_efficient(inner_hex_data, spatial_tree):
    """Check overlaps using spatial tree for O(log n) lookups."""
    # For each hexagon, find potential neighbors via spatial tree
    num_hex = len(inner_hex_data)
    
    if spatial_tree is None:
        # Fall back to brute force for very small sets
        for i in range(num_hex):
            for j in range(i+1, num_hex):
                center_x1, center_y1, angle1 = inner_hex_data[i]
                center_x2, center_y2, angle2 = inner_hex_data[j]
                
                # Simple distance check first
                dist = np.sqrt((center_x1 - center_x2)**2 + (center_y1 - center_y2)**2)
                if dist < 2.0:  # Threshold for potential overlap
                    vertices1 = hexagon_vertices_numba(center_x1, center_y1, angle1)
                    vertices2 = hexagon_vertices_numba(center_x2, center_y2, angle2)
                    if check_overlap_sat(vertices1, vertices2):
                        return True
        return False
    
    for i in range(num_hex):
        center_x1, center_y1, angle1 = inner_hex_data[i]
        
        # Query nearby points in spatial tree
        # Define search area as potential overlap region
        search_radius = 2.0
        search_rect = [
            center_x1 - search_radius,
            center_y1 - search_radius,
            center_x1 + search_radius,
            center_y1 + search_radius
        ]
        
        # Find potential neighbors
        neighbors = spatial_tree.query_range(search_rect)
        
        for neighbor_point in neighbors:
            if neighbor_point[0] == center_x1 and neighbor_point[1] == center_y1:
                continue  # Skip self
                
            # Check if this neighbor is actually close enough
            center_x2, center_y2 = neighbor_point
            dist = np.sqrt((center_x1 - center_x2)**2 + (center_y1 - center_y2)**2)
            if dist >= 2.0:
                continue  # Too far away
                
            # Perform detailed overlap check
            vertices1 = hexagon_vertices_numba(center_x1, center_y1, angle1)
            vertices2 = hexagon_vertices_numba(center_x2, center_y2, 0)  # No rotation for simplicity in neighbor check
            if check_overlap_sat(vertices1, vertices2):
                return True
                
    return False

def evaluate_fitness(solution_array, penalty_weights=None):
    """Improved evaluation with multi-criteria optimization."""
    if penalty_weights is None:
        penalty_weights = {'containment': 100000, 'overlap': 1000000, 'boundary': 5000}

    # Reshape solution array into 12 hexagons with (x, y, angle) each
    inner_hex_data = solution_array.reshape(-1, 3)

    # Calculate the minimum enclosing hexagon
    min_side_length, centroid = calculate_min_enclosing_hexagon_fast(inner_hex_data, 1.05)

    # Check all constraints
    num_hex = len(inner_hex_data)
    penalty = 0.0

    # Create outer hexagon polygon for containment checks
    outer_hex = create_hexagon_polygon(centroid[0], centroid[1], 0, min_side_length)

    # Check containment for all hexagons - use fast geometric method
    for i in range(num_hex):
        center_x, center_y, angle = inner_hex_data[i]
        # Fast containment check: distance from center to center of outer hexagon
        dist = np.sqrt((center_x - centroid[0])**2 + (center_y - centroid[1])**2)
        # Max distance from center of inner hexagon to its furthest vertex in outer hexagon
        max_dist_from_center = min_side_length * np.sqrt(3) / 2
        # If inner hexagon's center is outside the max containment distance, penalize
        if dist + 1.0 > max_dist_from_center:  # Inner hex side length is 1
            penalty += penalty_weights['containment']
    
    # Setup spatial index for overlap checks
    spatial_tree = setup_spatial_index(inner_hex_data)
    
    # Check overlaps using spatial indexing for efficiency
    if check_overlaps_efficient(inner_hex_data, spatial_tree):
        penalty += penalty_weights['overlap']

    # Multi-criterion optimization: balance inverse side length with packing density
    # We want to maximize 1/outer_side_length, subject to constraints
    # This can be viewed as minimizing outer_side_length
    objective_value = min_side_length + penalty  # Minimize side length + penalty
    
    return objective_value, min_side_length

def construct_symmetric_template():
    """Construct a carefully designed symmetric template."""
    # Known good symmetric configuration for 12 hexagons
    template = np.array([
        [0.0, 0.0, 0.0],          # Center
        [0.0, 3.0, 0.0],          # Top
        [0.0, -3.0, 0.0],         # Bottom
        [2.6, 1.5, 0.0],          # Top Right
        [-2.6, 1.5, 0.0],         # Top Left
        [2.6, -1.5, 0.0],         # Bottom Right
        [-2.6, -1.5, 0.0],        # Bottom Left
        [3.5, 0.0, 0.0],          # Far Right
        [-3.5, 0.0, 0.0],         # Far Left
        [1.75, 3.03, 0.0],        # Upper Middle Right
        [-1.75, 3.03, 0.0],       # Upper Middle Left
        [1.75, -3.03, 0.0],       # Lower Middle Right
        [-1.75, -3.03, 0.0],      # Lower Middle Left
    ])
    
    # Remove the last element which was for outer side length and add it separately
    template = template[:-1]
    
    # Add some variation in rotations to avoid exact symmetry
    for i in range(len(template)):
        template[i][2] += np.random.uniform(-10, 10)
    
    return template.flatten()

def hexagon_packing_12():
    """
    Constructs an optimized packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Define bounds for optimization
    # Each hexagon has 3 parameters: x, y, angle; plus outer side length (but we'll optimize that implicitly)
    bounds = []
    for _ in range(12):
        bounds.extend([(-8, 8), (-8, 8), (0, 360)])
    
    # Multi-start optimization with multiple strategies
    strategies = [
        {"popsize": 25, "maxiter": 60, "mutation": (0.5, 1.0), "recombination": 0.7},
        {"popsize": 30, "maxiter": 80, "mutation": (0.7, 1.0), "recombination": 0.8},
        {"popsize": 35, "maxiter": 100, "mutation": (0.8, 1.0), "recombination": 0.85}
    ]
    
    best_solution = None
    best_side_length = float('inf')
    best_objective = float('inf')
    
    # Strategy 1: Start with a symmetric template
    try:
        template = construct_symmetric_template()
        initial_pop = [template]
        
        # Generate additional diverse starting points
        for _ in range(5):
            # Generate slightly different variants
            variant = template + np.random.uniform(-0.5, 0.5, len(template))
            initial_pop.append(variant)
            
        result = differential_evolution(
            lambda x: evaluate_fitness(x)[0],
            bounds,
            maxiter=strategies[0]["maxiter"],
            popsize=strategies[0]["popsize"],
            mutation=strategies[0]["mutation"],
            recombination=strategies[0]["recombination"],
            seed=42,
            disp=False,
            init=initial_pop
        )
        
        final_objective, side_length = evaluate_fitness(result.x)
        
        if final_objective < best_objective:
            best_objective = final_objective
            best_side_length = side_length
            best_solution = result.x.copy()
            
    except Exception as e:
        print(f"Template strategy failed: {e}")
        
    # Strategy 2: Run multiple runs with different random seeds
    for strategy_idx, strategy in enumerate(strategies[:2]):  # Use first two strategies
        try:
            # Generate better initial population for this run
            initial_pop = generate_symmetric_initial_population(strategy["popsize"] - 1)
            
            # Add a fresh random solution to ensure variety
            random_solution = np.random.uniform(-8, 8, 36)
            initial_pop.append(random_solution)
            
            result = differential_evolution(
                lambda x: evaluate_fitness(x)[0],
                bounds,
                maxiter=strategy["maxiter"],
                popsize=strategy["popsize"],
                mutation=strategy["mutation"],
                recombination=strategy["recombination"],
                seed=42 + strategy_idx,
                disp=False,
                init=initial_pop
            )
            
            final_objective, side_length = evaluate_fitness(result.x)
            
            if final_objective < best_objective:
                best_objective = final_objective
                best_side_length = side_length
                best_solution = result.x.copy()
                
        except Exception as e:
            print(f"Strategy {strategy_idx + 1} failed: {e}")
            continue
    
    # Final refinement
    if best_solution is not None:
        # Use local search around best result
        try:
            # Refine with local search on key parameters
            refined_solution = best_solution.copy()
            
            # Perturb solution slightly to see if we can improve
            for _ in range(20):
                # Create small perturbations
                perturbed = refined_solution + np.random.normal(0, 0.1, len(refined_solution))
                
                # Check if this improves fitness
                obj, side = evaluate_fitness(perturbed)
                if obj < best_objective:
                    best_objective = obj
                    best_side_length = side
                    refined_solution = perturbed.copy()
                    
            best_solution = refined_solution
        except Exception as e:
            print(f"Local refinement failed: {e}")
    
    print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
    print(f"Best objective value: {best_objective}")
    
    if best_solution is None:
        # Fallback to previous solution
        inner_hex_data = np.array([
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right
            [0, -4, 0],  # far bottom-center
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Extract the best solution
    inner_hex_data = best_solution.reshape(-1, 3)
    
    # Verify final result
    final_obj, final_side = evaluate_fitness(best_solution)
    
    # Center the outer hexagon at the centroid of inner hexagons
    _, centroid = calculate_min_enclosing_hexagon_fast(inner_hex_data, 1.05)
    outer_hex_data = np.array([centroid[0], centroid[1], 0])
    
    return inner_hex_data, outer_hex_data, final_side

# EVOLVE-BLOCK-END
