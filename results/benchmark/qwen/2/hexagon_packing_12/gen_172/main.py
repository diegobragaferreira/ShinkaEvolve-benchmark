# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
from numba import jit, prange
import time
import warnings

@jit(nopython=True)
def hexagon_vertices_jit(x, y, angle_deg, side_length=1):
    """Fast generation of hexagon vertices using numba"""
    angle_rad = np.radians(angle_deg)
    angles = np.arange(0, 6) * np.pi / 3
    vertices = np.zeros((6, 2))
    for i in range(6):
        vertices[i, 0] = x + side_length * np.cos(angles[i] + angle_rad)
        vertices[i, 1] = y + side_length * np.sin(angles[i] + angle_rad)
    return vertices

@jit(nopython=True)
def point_in_polygon_fast(point, polygon):
    """Fast point-in-polygon test using ray casting"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

class HexGridOptimizer:
    """Hexagonal grid-based optimizer for hexagon packing"""
    
    def __init__(self):
        self.hex_radius = 1.0
        self.hex_apothem = self.hex_radius * np.sqrt(3) / 2
        self.hex_height = 2 * self.hex_apothem
        self.hex_width = 2 * self.hex_radius
        self.grid_spacing = self.hex_width * 1.05  # Slightly larger than hex width to ensure separation
        self.grid_offsets = np.array([
            [0, 0],
            [self.hex_width * 0.5, self.hex_apothem],
            [self.hex_width, 0],
            [self.hex_width * 0.5, -self.hex_apothem]
        ])
        
    def create_outer_hexagon(self, side_length: float, center_x: float = 0, center_y: float = 0) -> Polygon:
        """Create outer hexagon as Shapely polygon"""
        vertices = hexagon_vertices_jit(center_x, center_y, 0, side_length)
        return Polygon(vertices)
    
    def get_grid_point(self, row, col):
        """Get the grid point coordinates for a hexagonal grid"""
        x = col * self.hex_width + (row % 2) * self.hex_width / 2
        y = row * self.hex_height
        return x, y
    
    def generate_hexagon_positions(self, center_pos, radius=2.5):
        """Generate initial hexagon placements in a hexagonal pattern"""
        positions = []
        
        # Central hexagon
        positions.append([center_pos[0], center_pos[1], 0.0])
        
        # First ring - 6 hexagons
        for i in range(6):
            angle = i * 60
            rad = radius
            x = center_pos[0] + rad * np.cos(np.radians(angle))
            y = center_pos[1] + rad * np.sin(np.radians(angle))
            positions.append([x, y, 0.0])
            
        # Second ring - 6 hexagons
        radius *= 1.2
        for i in range(6):
            angle = i * 60 + 30  # Offset each ring
            rad = radius
            x = center_pos[0] + rad * np.cos(np.radians(angle))
            y = center_pos[1] + rad * np.sin(np.radians(angle))
            positions.append([x, y, 0.0])
        
        # Additional strategic positions
        additional = [
            [center_pos[0] - 2.5, center_pos[1] + 2.0, 0.0],
            [center_pos[0] + 2.5, center_pos[1] + 2.0, 0.0],
            [center_pos[0] - 2.5, center_pos[1] - 2.0, 0.0],
            [center_pos[0] + 2.5, center_pos[1] - 2.0, 0.0],
        ]
        
        positions.extend(additional)
        
        return positions
    
    def compute_outer_hex_side_length(self, hex_positions):
        """Compute minimum outer hexagon side length required to contain all inner hexagons"""
        # Get all vertices from all hexagons
        all_vertices = []
        for pos in hex_positions:
            x, y, angle = pos
            vertices = hexagon_vertices_jit(x, y, angle)
            all_vertices.extend(vertices)
        
        all_vertices = np.array(all_vertices)
        
        # Find bounding circle center and radius
        center = np.mean(all_vertices, axis=0)
        
        # Calculate maximum distance from center to any vertex
        distances = np.linalg.norm(all_vertices - center, axis=1)
        max_distance = np.max(distances)
        
        # For a hexagon, we need side length >= max_distance * 2 / sqrt(3)
        side_length = max_distance * 2 / np.sqrt(3)
        
        return side_length
    
    def check_containment(self, hex_position, outer_polygon):
        """Check if all vertices of hexagon are inside outer polygon"""
        x, y, angle = hex_position
        vertices = hexagon_vertices_jit(x, y, angle)
        for vertex in vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False
        return True
    
    def calculate_penalty(self, hex_positions, outer_side_length):
        """Calculate penalty based on constraints"""
        penalty = 0.0
        
        # Create outer polygon
        outer_polygon = self.create_outer_hexagon(outer_side_length)
        
        # Check containment penalties
        for pos in hex_positions:
            if not self.check_containment(pos, outer_polygon):
                penalty += 1e8
        
        # Check overlap penalties using fast approximation
        n = len(hex_positions)
        for i in range(n):
            for j in range(i+1, n):
                pos1 = hex_positions[i]
                pos2 = hex_positions[j]
                
                # Fast bounding box check first
                x1, y1, _ = pos1
                x2, y2, _ = pos2
                
                # Approximate distance using hexagonal grid
                dx = abs(x1 - x2)
                dy = abs(y1 - y2)
                
                if dx < 2.5 and dy < 2.5:
                    vertices1 = hexagon_vertices_jit(pos1[0], pos1[1], pos1[2])
                    vertices2 = hexagon_vertices_jit(pos2[0], pos2[1], pos2[2])
                    
                    # Use Shapely for precise overlap check
                    poly1 = Polygon(vertices1)
                    poly2 = Polygon(vertices2)
                    
                    if poly1.intersects(poly2):
                        penalty += 1e7
                        
        return penalty
    
    def evaluate_objective(self, params, outer_side_length=10.0):
        """Evaluate the objective function"""
        # Reshape params to hexagon positions
        positions = params.reshape(-1, 3)
        
        # Calculate penalty
        penalty = self.calculate_penalty(positions, outer_side_length)
        
        # Calculate actual outer side length
        actual_side_length = self.compute_outer_hex_side_length(positions)
        
        if actual_side_length > outer_side_length:
            penalty += 1e8
            
        # Objective: maximize 1/actual_side_length
        # So we minimize -1/actual_side_length + penalty
        return -1.0 / actual_side_length + penalty
    
    def get_initial_solution(self):
        """Generate a good initial solution"""
        # Generate hexagonal pattern
        positions = self.generate_hexagon_positions([0, 0], radius=2.2)
        
        # Add some randomness for diversity
        for i in range(len(positions)):
            if i > 0:  # Don't perturb center
                positions[i][0] += np.random.normal(0, 0.1)
                positions[i][1] += np.random.normal(0, 0.1)
                positions[i][2] += np.random.normal(0, 5)  # Small rotation perturbations
        
        # Flatten for optimization
        flat_params = np.array(positions).flatten()
        return flat_params

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
        optimizer = HexGridOptimizer()
        
        # Get initial solution
        initial_params = optimizer.get_initial_solution()
        
        # Define bounds for optimization
        bounds = []
        for i in range(12):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])
        bounds.append((1.0, 20.0))  # Outer side length bound
        
        # Phase 1: Coarse optimization to find good configuration
        def coarse_objective(params):
            return optimizer.evaluate_objective(params, outer_side_length=8.0)
        
        # Use L-BFGS-B for coarse optimization
        try:
            result_coarse = minimize(
                coarse_objective,
                initial_params,
                method='L-BFGS-B',
                bounds=[(-10, 10), (-10, 10), (0, 360)] * 12 + [(1.0, 20.0)],
                options={'maxiter': 20, 'ftol': 1e-6, 'gtol': 1e-6},
                tol=1e-6
            )
            
            if result_coarse.success:
                final_params = result_coarse.x
            else:
                final_params = initial_params
        except:
            final_params = initial_params
        
        # Phase 2: Refine with better optimization
        def refined_objective(params):
            return optimizer.evaluate_objective(params, outer_side_length=8.0)
        
        try:
            result_fine = minimize(
                refined_objective,
                final_params,
                method='L-BFGS-B',
                bounds=[(-10, 10), (-10, 10), (0, 360)] * 12 + [(1.0, 20.0)],
                options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-8},
                tol=1e-8
            )
            
            if result_fine.success:
                final_params = result_fine.x
        except:
            pass
        
        # Extract final configuration
        positions = final_params.reshape(-1, 3)
        outer_side_length = optimizer.compute_outer_hex_side_length(positions)
        
        # Create inner hex data
        inner_hex_data = positions.copy()
        outer_hex_data = np.array([0, 0, 0])
        
        # Final validation
        penalty = optimizer.calculate_penalty(positions, outer_side_length)
        if penalty > 1e6:
            # Revert to better known configuration if final result is invalid
            warnings.warn("Final solution had constraint violations, using fallback.")
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
            outer_side_length = 8.0
            outer_hex_data = np.array([0, 0, 0])
        
        return inner_hex_data, outer_hex_data, outer_side_length
        
    except Exception as e:
        warnings.warn(f"Error in hexagon packing: {str(e)}")
        # Fallback to simple grid if optimization fails
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
        
        return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END