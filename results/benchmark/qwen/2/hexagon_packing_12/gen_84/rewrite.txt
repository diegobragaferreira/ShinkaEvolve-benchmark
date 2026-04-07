# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
import time
from numba import jit, prange
import warnings

# Global constants
HEX_RADIUS = 1.0
HEX_APO = HEX_RADIUS * np.sqrt(3) / 2

class HexagonGeometry:
    """Handles geometric computations for hexagons with optimized JIT compilation"""
    
    @staticmethod
    @jit(nopython=True)
    def vertices(center_x, center_y, angle_degrees):
        """Generate vertices of a unit regular hexagon given center and rotation - JIT compiled."""
        angle_rad = np.radians(angle_degrees)
        vertices = np.empty((6, 2), dtype=np.float64)
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            x = HEX_RADIUS * np.cos(theta)
            y = HEX_RADIUS * np.sin(theta)
            # Apply rotation and translation
            vertices[i, 0] = x * np.cos(angle_rad) - y * np.sin(angle_rad) + center_x
            vertices[i, 1] = x * np.sin(angle_rad) + y * np.cos(angle_rad) + center_y
        return vertices

    @staticmethod
    @jit(nopython=True)
    def point_in_polygon(point, polygon_vertices):
        """Fast point-in-polygon test - JIT compiled."""
        x, y = point
        n = len(polygon_vertices)
        inside = False
        
        p1x, p1y = polygon_vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

class SpatialHash:
    """Spatial hash grid for efficient neighbor lookups"""
    
    def __init__(self, cell_size=2.0):
        self.cell_size = cell_size
        self.grid = {}
    
    def clear(self):
        self.grid.clear()
    
    def _hash(self, x, y):
        """Hash coordinates to grid cell"""
        return (int(x // self.cell_size), int(y // self.cell_size))
    
    def insert(self, hex_id, center_x, center_y):
        """Insert hexagon into spatial hash grid"""
        cell_x, cell_y = self._hash(center_x, center_y)
        if (cell_x, cell_y) not in self.grid:
            self.grid[(cell_x, cell_y)] = []
        self.grid[(cell_x, cell_y)].append(hex_id)
    
    def get_neighbors(self, center_x, center_y):
        """Get all hexagons in the same and neighboring cells"""
        cell_x, cell_y = self._hash(center_x, center_y)
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cell_x + dx, cell_y + dy)
                if key in self.grid:
                    neighbors.extend(self.grid[key])
        return neighbors

class HexagonValidator:
    """Validates hexagon configurations for containment and overlap"""
    
    @staticmethod
    def compute_outer_hex_side_length(inner_positions):
        """Compute minimum outer hexagon side length using vectorized operations."""
        if len(inner_positions) == 0:
            return 1e6
            
        # Preallocate arrays for performance
        all_vertices = np.empty((len(inner_positions) * 6, 2), dtype=np.float64)
        centers = np.empty((len(inner_positions), 2), dtype=np.float64)
        
        # Generate all vertices and centers
        for i in range(len(inner_positions)):
            center_x, center_y, angle_deg = inner_positions[i]
            centers[i] = [center_x, center_y]
            
            vertices = HexagonGeometry.vertices(center_x, center_y, angle_deg)
            for j in range(6):
                all_vertices[i*6 + j] = vertices[j]
        
        # Compute bounding circle center and radius
        center = np.mean(all_vertices, axis=0)
        
        # Calculate maximum distance from center to any vertex
        max_dist_sq = 0.0
        for i in range(len(all_vertices)):
            dx = all_vertices[i, 0] - center[0]
            dy = all_vertices[i, 1] - center[1]
            dist_sq = dx*dx + dy*dy
            max_dist_sq = max(max_dist_sq, dist_sq)
        
        # Side length = sqrt(max_dist) * 2 / sqrt(3)
        return np.sqrt(max_dist_sq) * 2.0 / np.sqrt(3)

    @staticmethod
    def check_containment_all(hex_positions, outer_polygon):
        """Check if all hexagons are fully contained within outer hexagon."""
        for i, (center_x, center_y, angle_deg) in enumerate(hex_positions):
            vertices = HexagonGeometry.vertices(center_x, center_y, angle_deg)
            for x, y in vertices:
                if not outer_polygon.contains(Point(x, y)):
                    return False
        return True

    @staticmethod
    def check_overlap_fast(hex_positions, spatial_hash=None):
        """Fast overlap check using spatial hashing."""
        num_hexagons = len(hex_positions)
        
        # Early termination for single hexagon
        if num_hexagons <= 1:
            return False
            
        # Create spatial hash for neighbor lookup
        if spatial_hash is None:
            spatial_hash = SpatialHash()
            for i, (cx, cy, _) in enumerate(hex_positions):
                spatial_hash.insert(i, cx, cy)
        
        # Check overlaps using spatial hashing
        for i in range(num_hexagons):
            cx, cy, _ = hex_positions[i]
            neighbors = spatial_hash.get_neighbors(cx, cy)
            
            for j in neighbors:
                if i >= j:  # Avoid checking same pair twice and self
                    continue
                    
                # Quick bounding box check first
                cx1, cy1, _ = hex_positions[i]
                cx2, cy2, _ = hex_positions[j]
                
                # Skip if bounding boxes don't overlap
                if abs(cx1 - cx2) > 3.0 or abs(cy1 - cy2) > 3.0:
                    continue
                    
                # Precise overlap test
                verts1 = HexagonGeometry.vertices(cx1, cy1, 0)
                verts2 = HexagonGeometry.vertices(cx2, cy2, 0)
                
                # Simple polygon intersection check
                poly1 = Polygon(verts1)
                poly2 = Polygon(verts2)
                if poly1.intersects(poly2) and not poly1.touches(poly2):
                    return True
                    
        return False

class Optimizer:
    """Handles all optimization procedures for finding optimal hexagon packing"""
    
    @staticmethod
    def create_outer_hexagon_polygon(side_length):
        """Create Shapely polygon representation of outer hexagon."""
        vertices = HexagonGeometry.vertices(0, 0, 0)
        return Polygon(vertices)
    
    @staticmethod
    def objective_function(solution):
        """Objective function to maximize 1/outer_hex_side_length."""
        # Reshape solution back to 12 hexagons with (x, y, angle)
        positions = solution.reshape(-1, 3)
        
        # Compute outer hexagon side length
        side_length = HexagonValidator.compute_outer_hex_side_length(positions)
        
        # Check constraints with early termination
        outer_poly = Optimizer.create_outer_hexagon_polygon(side_length)
        
        # Containment check
        if not HexagonValidator.check_containment_all(positions, outer_poly):
            return 1e6  # Heavy penalty for containment violations
            
        # Overlap check
        if HexagonValidator.check_overlap_fast(positions):
            return 1e5  # Heavy penalty for overlaps
            
        # Return negative of 1/size to maximize 1/size
        if side_length < 1e-6:
            return 1e6
        return -1.0 / side_length

    @staticmethod
    def get_symmetric_initial_config():
        """Generate initial configuration with enhanced symmetry properties."""
        # Generate a more sophisticated symmetric configuration
        positions = []
        
        # Central hexagon
        positions.append([0, 0, 0])
        
        # First ring: 6 hexagons around center
        angles = np.arange(0, 360, 60)
        radius = 2.0
        for angle in angles:
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions.append([x, y, 0])
        
        # Second ring: 6 hexagons at greater distance  
        radius = 3.5
        for angle in angles:
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions.append([x, y, 0])
        
        # Additional strategic placements
        additional_positions = [
            [0, -4.5, 0],  # Bottom center
            [4.5, 0, 0],   # Right center
            [-4.5, 0, 0],  # Left center
            [0, 4.5, 0],   # Top center
        ]
        
        positions.extend(additional_positions)
        
        return np.array(positions).flatten()

    @staticmethod
    def multistage_optimization():
        """Multi-stage optimization approach."""
        start_time = time.time()
        
        # Stage 1: Global optimization with differential evolution
        bounds = [(-15, 15)] * 36  # 12 hexagons * 3 parameters each
        
        # Initial configuration
        initial_positions = Optimizer.get_symmetric_initial_config()
        
        try:
            # Global search with reduced complexity
            result = differential_evolution(
                Optimizer.objective_function,
                bounds,
                maxiter=30,
                popsize=10,
                seed=42,
                disp=False,
                strategy='best1bin'
            )
            
            if result.success:
                final_positions = result.x.reshape(-1, 3)
                side_length = HexagonValidator.compute_outer_hex_side_length(final_positions)
                
                # Stage 2: Local refinement
                refined_positions = Optimizer.local_refinement(result.x, side_length)
                final_positions = refined_positions.reshape(-1, 3)
                side_length = HexagonValidator.compute_outer_hex_side_length(final_positions)
                
                return final_positions, side_length
                
        except Exception as e:
            warnings.warn(f"Optimization error: {e}")
            
        # Fallback to initial configuration
        positions = initial_positions.reshape(-1, 3)
        side_length = HexagonValidator.compute_outer_hex_side_length(positions)
        return positions, side_length
    
    @staticmethod
    def local_refinement(initial_solution, current_side_length):
        """Use local optimization to refine the solution."""
        def objective(x):
            _, side_length = Optimizer.evaluate_with_validation(x)
            return -np.log(side_length)  # Minimize negative log side length
            
        bounds = [(-20, 20)] * len(initial_solution)
        
        try:
            result = minimize(objective, initial_solution, method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 20}, tol=1e-6)
            return result.x if result.success else initial_solution
        except:
            return initial_solution

    @staticmethod
    def evaluate_with_validation(solution):
        """Validate solution and return fitness and side length."""
        positions = solution.reshape(-1, 3)
        side_length = HexagonValidator.compute_outer_hex_side_length(positions)
        
        outer_poly = Optimizer.create_outer_hexagon_polygon(side_length)
        
        # Check containment and overlaps
        if not HexagonValidator.check_containment_all(positions, outer_poly):
            return 1e6, side_length
            
        if HexagonValidator.check_overlap_fast(positions):
            return 1e5, side_length
            
        return -1.0 / side_length, side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Execute multi-stage optimization
    inner_hex_data, outer_hex_side_length = Optimizer.multistage_optimization()
    
    # Set outer hexagon parameters (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    # Calculate benchmark ratio
    inv_side_length = 1.0 / outer_hex_side_length
    benchmark_ratio = inv_side_length / 0.2537
    
    # Print diagnostic info
    print(f"inv_outer_hex_side_length: {inv_side_length:.8f}")
    print(f"benchmark_ratio: {benchmark_ratio:.8f}")
    print(f"eval_time: {eval_time:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END