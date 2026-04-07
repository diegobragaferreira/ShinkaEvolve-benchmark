# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time
from numba import njit, prange
from collections import defaultdict
import warnings

# Numba-compiled geometric operations for maximum performance
@njit
def generate_hexagon_vertices_numba(center_x: float, center_y: float, angle_deg: float, side_length: float = 1.0) -> np.ndarray:
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = np.radians(angle_deg)
    vertices = np.empty((6, 2))
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i, 0] = center_x + side_length * np.cos(theta)
        vertices[i, 1] = center_y + side_length * np.sin(theta)
    return vertices

@njit
def point_in_polygon(point_x: float, point_y: float, polygon_vertices: np.ndarray) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm."""
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

@njit
def check_containment_numba(inner_vertices: np.ndarray, outer_vertices: np.ndarray) -> bool:
    """Check if inner hexagon is fully contained within outer hexagon."""
    # Check that all vertices of inner hexagon are inside outer hexagon
    for i in range(6):
        if not point_in_polygon(inner_vertices[i, 0], inner_vertices[i, 1], outer_vertices):
            return False
    return True

@njit
def check_overlap_numba(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
    """Fast overlap check using bounding rectangle and point-in-polygon."""
    # Quick bounding box check
    min1 = np.min(hex1_vertices, axis=0)
    max1 = np.max(hex1_vertices, axis=0)
    min2 = np.min(hex2_vertices, axis=0)
    max2 = np.max(hex2_vertices, axis=0)
    
    if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
        return False
    
    # Check if any vertex of hex1 is inside hex2
    for i in range(6):
        if point_in_polygon(hex1_vertices[i, 0], hex1_vertices[i, 1], hex2_vertices):
            return True
    
    # Check if any vertex of hex2 is inside hex1
    for i in range(6):
        if point_in_polygon(hex2_vertices[i, 0], hex2_vertices[i, 1], hex1_vertices):
            return True
    
    return False

# Spatial hashing for efficient neighbor search
class SpatialHashGrid:
    def __init__(self, cell_size: float = 2.0):
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    def _hash_cell(self, x: float, y: float) -> tuple:
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, hex_id: int, center_x: float, center_y: float):
        cell = self._hash_cell(center_x, center_y)
        self.grid[cell].append(hex_id)

    def get_candidates(self, center_x: float, center_y: float) -> list:
        cell = self._hash_cell(center_x, center_y)
        candidates = []

        # Check the cell and its 8 neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                candidates.extend(self.grid[neighbor_cell])

        return candidates

class HexagonPacker:
    def __init__(self):
        self.side_length = 1.0
        self.hex_radius = 1.0
        self.apothem = np.sqrt(3) / 2
        self.n_hexagons = 12
        
    def generate_symmetric_initial_config(self) -> np.ndarray:
        """Generate an initial symmetric configuration based on known patterns."""
        # Pattern based on known good symmetric arrangements
        positions = np.array([
            [0.0, 0.0, 0.0],        # Center
            [-2.1, 0.0, 0.0],       # Left
            [2.1, 0.0, 0.0],        # Right
            [-1.05, 1.82, 0.0],     # Top-left
            [1.05, 1.82, 0.0],      # Top-right
            [-1.05, -1.82, 0.0],    # Bottom-left
            [1.05, -1.82, 0.0],     # Bottom-right
            [-3.15, 1.82, 0.0],     # Far-top-left
            [3.15, 1.82, 0.0],      # Far-top-right
            [-3.15, -1.82, 0.0],    # Far-bottom-left
            [3.15, -1.82, 0.0],     # Far-bottom-right
            [0.0, -3.64, 0.0],      # Far-bottom
        ])
        
        # Perturb slightly for diversity while maintaining symmetry structure
        noise = np.random.normal(0, 0.1, (12, 3))
        positions = positions + noise
        
        return positions.flatten()
    
    def get_outer_hexagon_vertices(self, center_x: float, center_y: float, outer_radius: float) -> np.ndarray:
        """Generate vertices for the outer hexagon."""
        return generate_hexagon_vertices_numba(center_x, center_y, 0.0, outer_radius)
    
    def calculate_min_enclosing_radius(self, hex_positions: np.ndarray) -> tuple:
        """Calculate the minimum enclosing circle radius."""
        # Calculate centroids of all hexagons
        centers = hex_positions.reshape(-1, 3)[:, :2]
        centroid = np.mean(centers, axis=0)
        
        # Calculate maximum distance from centroid to any hexagon center
        distances = np.sqrt(np.sum((centers - centroid)**2, axis=1))
        max_distance = np.max(distances)
        
        # For outer hexagon, we add a small margin to ensure full containment
        # Outer hexagon radius = max_distance + hex_radius + small padding
        outer_radius = max_distance + self.hex_radius + 0.1
        
        return outer_radius, centroid
    
    def evaluate_solution(self, params: np.ndarray) -> float:
        """Evaluate the quality of a solution with penalty terms."""
        # Reshape parameters
        hex_positions = params.reshape(-1, 3)
        
        # Calculate outer hexagon properties
        outer_radius, centroid = self.calculate_min_enclosing_radius(hex_positions)
        outer_vertices = self.get_outer_hexagon_vertices(centroid[0], centroid[1], outer_radius)
        
        penalty = 0.0
        
        # Check containment and overlap
        spatial_grid = SpatialHashGrid(cell_size=2.0)
        
        # Insert all hexagons into spatial grid
        for i in range(len(hex_positions)):
            center_x, center_y, _ = hex_positions[i]
            spatial_grid.insert(i, center_x, center_y)
        
        # Check containment
        for i in range(len(hex_positions)):
            center_x, center_y, angle = hex_positions[i]
            inner_vertices = generate_hexagon_vertices_numba(center_x, center_y, angle, self.side_length)
            
            # Check containment
            if not check_containment_numba(inner_vertices, outer_vertices):
                penalty += 15000.0  # Strong penalty for containment violations
        
        # Check overlaps using spatial hashing
        for i in range(len(hex_positions)):
            center_x, center_y, angle = hex_positions[i]
            inner_vertices = generate_hexagon_vertices_numba(center_x, center_y, angle, self.side_length)
            candidates = spatial_grid.get_candidates(center_x, center_y)
            
            for j in candidates:
                if i >= j:  # Avoid duplicate checks
                    continue
                center_x2, center_y2, angle2 = hex_positions[j]
                inner_vertices2 = generate_hexagon_vertices_numba(center_x2, center_y2, angle2, self.side_length)
                
                if check_overlap_numba(inner_vertices, inner_vertices2):
                    penalty += 50000.0  # Strong penalty for overlap violations
        
        # Objective: maximize 1/outer_radius = minimize -1/outer_radius
        objective = -1.0 / outer_radius + penalty
        
        return objective

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    packer = HexagonPacker()
    
    # Setup bounds for optimization
    bounds = []
    for i in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    
    # Generate initial population with symmetry
    initial_pop = []
    for _ in range(10):
        initial_pop.append(packer.generate_symmetric_initial_config())
    
    # Run optimization with improved settings
    try:
        result = differential_evolution(
            packer.evaluate_solution,
            bounds,
            seed=42,
            maxiter=150,
            popsize=20,
            mutation=(0.8, 1.0),
            recombination=0.7,
            tol=1e-6,
            workers=1,
            init=initial_pop
        )
        
        best_params = result.x
        inner_hex_data = best_params.reshape(-1, 3)
        
        # Final refinement
        outer_radius, centroid = packer.calculate_min_enclosing_radius(inner_hex_data)
        outer_hex_data = np.array([centroid[0], centroid[1], 0])
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        # Fallback solution with better known symmetric pattern
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.1, 0, 0],
            [2.1, 0, 0],
            [-1.05, 1.82, 0],
            [1.05, 1.82, 0],
            [-1.05, -1.82, 0],
            [1.05, -1.82, 0],
            [-3.15, 1.82, 0],
            [3.15, 1.82, 0],
            [-3.15, -1.82, 0],
            [3.15, -1.82, 0],
            [0, -3.64, 0],
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_radius = 7.0
    
    # Calculate final metrics
    inv_outer_hex_side_length = 1.0 / outer_radius
    benchmark_ratio = inv_outer_hex_side_length / 0.2537
    eval_time = time.time() - start_time
    
    # Print metrics
    print(f"inv_outer_hex_side_length: {inv_outer_hex_side_length:.8f}")
    print(f"benchmark_ratio: {benchmark_ratio:.8f}")
    print(f"eval_time: {eval_time:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END