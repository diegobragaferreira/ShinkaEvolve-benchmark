# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree
import time
from typing import Tuple, List, Optional, Any
import warnings
from numba import njit

class SymmetricHexagonPackerOptimized:
    """An optimized symmetric approach to hexagon packing with performance enhancements"""

    def __init__(self):
        # Precomputed constants for hexagon geometry
        self.hex_radius = 1.0  # Unit hexagon radius
        self.hex_diameter = 2.0  # Unit hexagon diameter
        self.circumradius = 1.0  # Circumradius of unit hexagon
        
    @staticmethod
    @njit
    def _create_hexagon_vertices_numba(center_x, center_y, side_length, rotation_degrees):
        """Fast vertex creation using numba JIT compilation"""
        angle_rad = np.radians(rotation_degrees)
        angle_step = 2 * np.pi / 6
        vertices = np.empty((6, 2))
        for i in range(6):
            angle = angle_step * i + angle_rad
            x = center_x + side_length * np.cos(angle)
            y = center_y + side_length * np.sin(angle)
            vertices[i] = [x, y]
        return vertices

    def create_hexagon_vertices(self, center, side_length, rotation_degrees):
        """Create vertices of a regular hexagon with numba acceleration"""
        return self._create_hexagon_vertices_numba(center[0], center[1], side_length, rotation_degrees)

    @staticmethod
    @njit
    def _compute_distance_squared(x1, y1, x2, y2):
        """Fast squared distance calculation"""
        dx = x1 - x2
        dy = y1 - y2
        return dx*dx + dy*dy

    @staticmethod
    @njit
    def _is_point_in_hexagon_numba(px, py, hex_vertices):
        """Fast point-in-polygon check for hexagon using ray casting"""
        n = len(hex_vertices)
        inside = False
        p1x, p1y = hex_vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = hex_vertices[i % n]
            if py > min(p1y, p2y):
                if py <= max(p1y, p2y):
                    if px <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or px <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def get_outer_hex_side_from_config(self, inner_hex_data, center=(0,0)):
        """Compute the minimum required outer hexagon side length from current configuration."""
        if len(inner_hex_data) == 0:
            return 100.0

        max_dist = 0.0
        for i in range(len(inner_hex_data)):
            cx, cy, _ = inner_hex_data[i]
            dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
            dist_to_edge = dist + self.circumradius
            max_dist = max(max_dist, dist_to_edge)

        return max_dist * 2.0  # Diameter gives us the side length for a hexagon

    def check_containment_all_vertices(self, hex_vertices, outer_hex_center, outer_hex_side_length):
        """Check if all vertices of a hexagon are inside the outer hexagon."""
        outer_vertices = self.create_hexagon_vertices(outer_hex_center, outer_hex_side_length, 0)
        outer_polygon = Polygon(outer_vertices)

        for vertex in hex_vertices:
            point = Point(vertex)
            if not outer_polygon.contains(point):
                return False
        return True

    def check_overlap_pair(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons overlap using Shapely."""
        hex1_polygon = Polygon(hex1_vertices)
        hex2_polygon = Polygon(hex2_vertices)
        return hex1_polygon.intersects(hex2_polygon)

    def evaluate_configuration_fast(self, inner_hex_data, outer_hex_center=(0,0)):
        """Fast evaluation of configuration with early termination and optimized checks"""
        if len(inner_hex_data) != 12:
            return 1e-10

        # Quick bounding circle check to reject obviously invalid configurations
        outer_side_length = self.get_outer_hex_side_from_config(inner_hex_data, outer_hex_center)
        
        # Fast check: all hexagon centers should be within the outer hexagon's bounding circle
        # Using squared distance to avoid sqrt computation
        outer_radius_sq = (outer_side_length/2)**2  # Approximate outer hexagon radius
        for i in range(len(inner_hex_data)):
            cx, cy, _ = inner_hex_data[i]
            dist_sq = cx*cx + cy*cy
            if dist_sq > outer_radius_sq:
                return 1e-10

        # Create hexagon polygons
        hex_polygons = []
        for i in range(len(inner_hex_data)):
            cx, cy, angle = inner_hex_data[i]
            vertices = self.create_hexagon_vertices((cx, cy), 1.0, angle)
            hex_polygons.append(vertices)

        # Create outer hexagon polygon
        outer_vertices = self.create_hexagon_vertices(outer_hex_center, outer_side_length, 0)
        outer_polygon = Polygon(outer_vertices)

        # Check containment for all vertices efficiently
        # First check if any vertex is outside using fast point-in-polygon
        for i in range(len(inner_hex_data)):
            vertices = hex_polygons[i]
            for vertex in vertices:
                if not self._is_point_in_hexagon_numba(vertex[0], vertex[1], outer_vertices):
                    return 1e-10

        # Spatial acceleration for overlap detection using KDTree
        try:
            centers = np.array([[h[0], h[1]] for h in inner_hex_data])
            tree = cKDTree(centers)
            
            # Find nearby pairs to check for overlaps (distance less than 2 units)
            pairs = tree.query_pairs(r=self.hex_diameter, p=np.inf)
            
            # Only check actual overlaps for pairs that might intersect
            for i, j in pairs:
                if i < j:  # Avoid double checking
                    if self.check_overlap_pair(hex_polygons[i], hex_polygons[j]):
                        return 1e-10
                        
        except Exception:
            # Fallback to brute force if spatial indexing fails
            for i in range(len(inner_hex_data)):
                for j in range(i+1, len(inner_hex_data)):
                    if self.check_overlap_pair(hex_polygons[i], hex_polygons[j]):
                        return 1e-10

        # If we reach here, the configuration is valid
        return 1.0 / outer_side_length

    def generate_improved_initial_placement(self):
        """Generate initial placement using triangular lattice principles"""
        # Create a hexagonal lattice pattern that naturally packs well
        # This uses a triangular arrangement to maximize density
        
        positions = []
        
        # Central hexagon
        positions.append([0, 0, 0])
        
        # First ring: 6 hexagons in a triangular pattern
        # Distance of 2 units (circumradius of hexagons plus gap)
        for i in range(6):
            angle = i * 60  # 60 degree steps
            x = 2.0 * np.cos(np.radians(angle))
            y = 2.0 * np.sin(np.radians(angle))
            positions.append([x, y, 0])
            
        # Second ring: 5 hexagons in triangular pattern (total 12)
        # Place them in a way that fills gaps from first ring
        # Using a more compact arrangement
        angles = [0, 120, 240, 60, 180]  # Carefully chosen angles for tight packing
        radius = 3.5  # Larger radius to allow for better packing
        
        for i, angle in enumerate(angles):
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions.append([x, y, 0])
            
        # Ensure exactly 12 positions
        while len(positions) < 12:
            positions.append([0, -4, 0])
        positions = positions[:12]
        
        # Add small random perturbations to break perfect symmetry and avoid plateaus
        # This helps escape local optima
        np.random.seed(42)
        for i in range(len(positions)):
            positions[i][0] += np.random.normal(0, 0.08)  # Slightly larger noise
            positions[i][1] += np.random.normal(0, 0.08)
            
        return np.array(positions)

    def encode_individual(self, params):
        """Encode symmetric parameters into full hexagon configuration"""
        # params = [ring1_spacing, ring2_spacing, ring_rotation, angle_offset1, angle_offset2, angle_offset3]
        ring1_spacing, ring2_spacing, ring_rotation, offset1, offset2, offset3 = params
        
        # Central hexagon
        positions = [[0, 0, 0]]
        
        # Ring 1 hexagons (6 hexagons)
        for i in range(6):
            angle = 60 * i + ring_rotation + offset1
            x = ring1_spacing * np.cos(np.radians(angle))
            y = ring1_spacing * np.sin(np.radians(angle))
            positions.append([x, y, 0])
            
        # Ring 2 hexagons (5 hexagons) 
        for i in range(5):
            angle = 72 * i + ring_rotation + offset2
            x = ring2_spacing * np.cos(np.radians(angle))
            y = ring2_spacing * np.sin(np.radians(angle))
            positions.append([x, y, 0])
            
        # Ensure exactly 12 positions
        while len(positions) < 12:
            positions.append([0, -4, 0])
        positions = positions[:12]
        
        return np.array(positions)

    def decode_individual(self, inner_hex_data):
        """Decode full configuration back to symmetric representation"""
        # Calculate average distances from center for each ring
        distances = []
        for i in range(len(inner_hex_data)):
            cx, cy, _ = inner_hex_data[i]
            dist = np.sqrt(cx**2 + cy**2)
            distances.append(dist)
            
        # Group distances into approximate rings
        ring1_distances = [d for d in distances if 1.5 <= d <= 2.5]
        ring2_distances = [d for d in distances if 3.0 <= d <= 4.5]
        
        # Simple heuristics for symmetric parameters
        avg_ring1 = np.mean(ring1_distances) if ring1_distances else 2.0
        avg_ring2 = np.mean(ring2_distances) if ring2_distances else 3.5
        
        # Assume ring rotation is 0 for simplicity in decoding
        return [avg_ring1, avg_ring2, 0.0, 0.0, 0.0, 0.0]

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    packer = SymmetricHexagonPackerOptimized()
    
    # Generate initial symmetric configuration
    initial_guess = packer.generate_improved_initial_placement()
    
    # Define bounds for our reduced parameter space (6 parameters)
    # [ring1_spacing, ring2_spacing, ring_rotation, offset1, offset2, offset3]
    bounds = [
        (1.5, 3.0),      # ring1_spacing: distance from center for first ring
        (3.0, 5.0),      # ring2_spacing: distance from center for second ring  
        (-180, 180),     # ring_rotation: overall rotation angle
        (-30, 30),       # offset1: phase adjustment for ring 1
        (-30, 30),       # offset2: phase adjustment for ring 2
        (-30, 30)        # offset3: additional freedom
    ]
    
    def objective(params):
        # Convert symmetric parameters to full configuration
        hex_data = packer.encode_individual(params)
        
        # Evaluate the configuration with fast evaluation
        score = packer.evaluate_configuration_fast(hex_data)
        return -score  # Negative because we want to maximize

    # Try evolutionary approach with symmetry preservation
    try:
        # Use a simpler constrained optimization for better performance within time limit
        # Reduce iterations and population size to meet time requirements
        result = differential_evolution(
            objective,
            bounds,
            maxiter=30,         # Reduced iterations to save time
            popsize=8,          # Smaller population
            seed=42,
            strategy='best1bin'
        )
        
        # Convert best symmetric parameters back to full hexagon configuration  
        optimized_hex_data = packer.encode_individual(result.x)
        
        # Final evaluation to ensure validity with fast evaluation
        final_score = packer.evaluate_configuration_fast(optimized_hex_data)
        
        if result.success and final_score > 1e-5:
            # Compute the outer hexagon parameters
            outer_side_length = 1.0 / final_score
            outer_hex_center = (0, 0)
            
            # Create outer hexagon data (centered at origin, no rotation)
            outer_hex_data = np.array([0, 0, 0])
            
            return optimized_hex_data, outer_hex_data, outer_side_length
            
    except Exception as e:
        warnings.warn(f"Symmetric optimization failed: {str(e)}")
        pass

    # Fall back to the original reasonable configuration 
    inner_hex_data = np.array([
        [0, 0, 0],  # center
        [0, 2, 0],  # top
        [0, -2, 0],  # bottom  
        [1.732, 1, 0],  # top right
        [-1.732, 1, 0],  # top left
        [1.732, -1, 0],  # bottom right
        [-1.732, -1, 0],  # bottom left
        [3.464, 0, 0],  # far right
        [-3.464, 0, 0],  # far left
        [1.732, 3, 0],  # top far right
        [-1.732, 3, 0],  # top far left
        [1.732, -3, 0],  # bottom far right
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 6.928  # approximated value (1/0.1443 ~= 6.928)

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END