# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from numba import jit, prange
import time
from collections import namedtuple
import math

# Named tuple for hexagon data
HexagonData = namedtuple('HexagonData', ['x', 'y', 'rotation'])

# Efficient hexagon vertex computation using Numba
@jit(nopython=True)
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
    """Fast distance from point to line segment."""
    dx = x2 - x1
    dy = y2 - y1
    
    if dx*dx + dy*dy == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
    t = max(0, min(1, t))
    
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def compute_min_distance_hexagon_hexagon(h1_x, h1_y, h1_angle, h2_x, h2_y, h2_angle):
    """Compute minimum distance between two hexagons efficiently."""
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
            dist = distance_point_to_line(v1[i,0], v1[i,1], v2[j,0], v2[j,1], v2[(j+1)%6,0], v2[(j+1)%6,1])
            if dist < min_dist:
                min_dist = dist
                
            dist = distance_point_to_line(v2[j,0], v2[j,1], v1[i,0], v1[i,1], v1[(i+1)%6,0], v1[(i+1)%6,1])
            if dist < min_dist:
                min_dist = dist
    
    return min_dist

def compute_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Convert hexagon parameters to shapely polygon efficiently."""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def create_base_hexagonal_grid():
    """Create a base hexagonal packing arrangement."""
    # Central hexagon
    positions = [[0, 0, 0]]
    
    # First ring - 6 hexagons around center
    ring1_radius = 2.0
    for i in range(6):
        angle = 2 * np.pi * i / 6
        x = ring1_radius * np.cos(angle)
        y = ring1_radius * np.sin(angle)
        positions.append([x, y, 0])
        
    # Second ring - 5 hexagons (leaving space for optimization)
    ring2_radius = 3.0
    for i in range(5):
        angle = 2 * np.pi * i / 5 + np.pi/10  # Offset to optimize space
        x = ring2_radius * np.cos(angle)
        y = ring2_radius * np.sin(angle)
        positions.append([x, y, 0])
        
    return np.array(positions)[:12]

def estimate_outer_hexagon_radius(positions, angles):
    """Quick estimation of required outer hexagon size."""
    if len(positions) == 0:
        return 10.0
        
    # Get all vertices of all hexagons
    all_vertices = []
    for pos, angle in zip(positions, angles):
        vertices = hexagon_vertices(pos[0], pos[1], angle)
        all_vertices.extend(vertices)
        
    if len(all_vertices) == 0:
        return 10.0
        
    all_coords = np.array(all_vertices)
    min_x, max_x = all_coords[:, 0].min(), all_coords[:, 0].max()
    min_y, max_y = all_coords[:, 1].min(), all_coords[:, 1].max()
    
    # Center of bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Maximum distance from center to any vertex (plus safety margin)
    max_dist = 0
    for vx, vy in all_vertices:
        dist = np.sqrt((vx - center_x)**2 + (vy - center_y)**2)
        max_dist = max(max_dist, dist)
        
    return max_dist * 1.1

class HexagonalPackingOptimizer:
    """Advanced optimizer using geometric insights and hybrid strategies."""
    
    def __init__(self):
        self.hex_side_length = 1.0
        self.lattice_vectors = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2]
        ])
        
    def compute_outer_hexagon_polygon(self, side_length):
        """Get shapely polygon for outer hexagon."""
        vertices = []
        for i in range(6):
            theta = i * np.pi / 3
            x = side_length * np.cos(theta)
            y = side_length * np.sin(theta)
            vertices.append((x, y))
        return Polygon(vertices)
        
    def is_contained(self, h_x, h_y, outer_radius):
        """Quick containment check."""
        distance = np.sqrt(h_x*h_x + h_y*h_y)
        return distance <= (outer_radius - 1.0)
        
    def fast_overlap_detection(self, hexagons, outer_radius):
        """Fast overlap detection using geometric bounds."""
        n = len(hexagons)
        if n <= 1:
            return False
            
        # Early bound check: if centers are too far apart, no overlap
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((hexagons[i].x - hexagons[j].x)**2 + 
                              (hexagons[i].y - hexagons[j].y)**2)
                # If centers are more than 2 units apart, definitely no overlap
                if dist >= 2.0:
                    continue
                    
                # More expensive check if needed
                min_dist = compute_min_distance_hexagon_hexagon(
                    hexagons[i].x, hexagons[i].y, hexagons[i].rotation,
                    hexagons[j].x, hexagons[j].y, hexagons[j].rotation
                )
                
                if min_dist < 0.001:  # Overlapping
                    return True
                    
        return False
        
    def evaluate_fitness(self, hexagons, outer_radius):
        """Evaluate fitness with geometric constraints."""
        # Check containment
        for hex_data in hexagons:
            if not self.is_contained(hex_data.x, hex_data.y, outer_radius):
                return 1e10
                
        # Check overlap using fast method
        hexagon_tuple_list = [(h.x, h.y, h.rotation) for h in hexagons]
        positions = np.array([[h.x, h.y] for h in hexagons])
        angles = np.array([h.rotation for h in hexagons])
        
        if self.fast_overlap_detection(hexagons, outer_radius):
            return 1e10
            
        # Valid configuration - return inverse of outer radius
        return -1.0 / outer_radius
        
    def generate_initial_solution(self):
        """Generate a high-quality initial solution."""
        # Start with our optimized base hexagonal grid
        positions_angles = create_base_hexagonal_grid()
        
        # Add small random perturbations to break degeneracies
        np.random.seed(42)
        for i in range(len(positions_angles)):
            positions_angles[i][0] += np.random.normal(0, 0.1)
            positions_angles[i][1] += np.random.normal(0, 0.1)
            positions_angles[i][2] += np.random.normal(0, 10)
            
        return positions_angles
        
    def optimize_with_stages(self):
        """Multi-stage optimization with increasing precision."""
        # Stage 1: Coarse optimization with relaxed constraints
        initial_guess = self.generate_initial_solution()
        x0 = initial_guess.flatten()
        
        bounds = []
        for _ in range(12):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])
            
        # Coarse optimization with reduced penalties
        result1 = differential_evolution(
            self.objective_function,
            bounds,
            args=(0.5, False),  # Reduced penalty, no fine-tuning yet
            maxiter=100,
            popsize=20,
            seed=42,
            disp=False
        )
        
        # Stage 2: Fine tuning with full constraints
        refined_params = result1.x.reshape(-1, 3)
        
        # Full optimization with proper penalties
        result2 = differential_evolution(
            self.objective_function,
            bounds,
            args=(1.0, True),  # Full penalty, fine-tuning enabled
            maxiter=150,
            popsize=25,
            seed=42,
            disp=False
        )
        
        # Final refinement using local optimization
        final_params = result2.x.reshape(-1, 3)
        refined_solution = self.local_refinement(final_params)
        
        # Extract final result
        positions = refined_solution[:, :2]
        angles = refined_solution[:, 2]
        outer_radius = estimate_outer_hexagon_radius(positions, angles)
        
        return refined_solution, outer_radius
        
    def objective_function(self, params, penalty_multiplier=1.0, enable_fine_tuning=True):
        """Objective function for optimization."""
        positions_angles = params.reshape(-1, 3)
        positions = positions_angles[:, :2]
        angles = positions_angles[:, 2]
        
        # Estimate outer radius
        estimated_radius = estimate_outer_hexagon_radius(positions, angles)
        
        # Create hexagon objects
        hexagons = [HexagonData(float(pos[0]), float(pos[1]), float(angle)) 
                   for pos, angle in zip(positions, angles)]
        
        # Evaluate fitness with penalty multiplier
        fitness = self.evaluate_fitness(hexagons, estimated_radius)
        
        # Apply penalty multiplier
        if enable_fine_tuning:
            fitness *= penalty_multiplier
            
        return fitness
        
    def local_refinement(self, positions_angles):
        """Use local optimization to refine solution."""
        initial_params = positions_angles.flatten()
        bounds = []
        for _ in range(12):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])
            
        # Wrap objective for local minimizer
        def objective(params):
            positions_angles_test = params.reshape(-1, 3)
            positions = positions_angles_test[:, :2]
            angles = positions_angles_test[:, 2]
            
            # Estimate outer radius
            estimated_radius = estimate_outer_hexagon_radius(positions, angles)
            
            # Create hexagon objects
            hexagons = [HexagonData(float(pos[0]), float(pos[1]), float(angle)) 
                       for pos, angle in zip(positions, angles)]
            
            fitness = self.evaluate_fitness(hexagons, estimated_radius)
            return fitness
            
        try:
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50}
            )
            
            if result.success:
                return result.x.reshape(-1, 3)
        except:
            pass
            
        return positions_angles

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
        optimizer = HexagonalPackingOptimizer()
        inner_hex_data, outer_hex_side_length = optimizer.optimize_with_stages()
        
        # Create outer hexagon data
        outer_hex_data = np.array([0, 0, 0])
        
        # Validate solution
        try:
            all_hexagons = []
            for i, (pos, angle) in enumerate(zip(inner_hex_data[:, :2], inner_hex_data[:, 2])):
                h = compute_hexagon_polygon(pos[0], pos[1], angle)
                all_hexagons.append(h)
                
            # Check all pairwise overlaps
            for i in range(len(all_hexagons)):
                for j in range(i+1, len(all_hexagons)):
                    if all_hexagons[i].intersects(all_hexagons[j]):
                        raise ValueError("Overlapping hexagons detected")
                        
            # Check containment in outer hexagon
            outer_hex = compute_hexagon_polygon(0, 0, 0, outer_hex_side_length)
            for hexagon in all_hexagons:
                if not outer_hex.contains(hexagon):
                    raise ValueError("Some hexagons outside outer hexagon")
                    
        except ValueError as e:
            print(f"Validation error: {e}")
            # Fallback to working configuration
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
            outer_hex_side_length = 8.0
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to default configuration
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
        outer_hex_side_length = 8.0
        
    end_time = time.time()
    eval_time = end_time - start_time
    
    return inner_hex_data, np.array([0, 0, 0]), outer_hex_side_length

# EVOLVE-BLOCK-END