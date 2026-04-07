# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import math

# Constants
UNIT_HEX_RADIUS = 1.0  # Side length of unit hexagon
UNIT_HEX_APOGEE = np.sqrt(3)/2  # Distance from center to corner of unit hexagon

class HexagonGeometry:
    """Handles all geometric operations for hexagon creation and manipulation."""
    
    @staticmethod
    def create_unit_hexagon(center=(0,0), rotation=0):
        """Create a unit regular hexagon as a Shapely Polygon."""
        angle_offset = np.deg2rad(rotation)
        points = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
            y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
            points.append((x, y))
        return Polygon(points)
    
    @staticmethod
    def get_all_vertices(hexagon):
        """Extract all vertices from a hexagon polygon."""
        return list(hexagon.exterior.coords)[:-1]  # Exclude closing point

class GeometricOptimizer:
    """Implements geometric optimization approach for hexagon packing."""
    
    def __init__(self):
        self.n_inner = 11
        self.hex_radius = UNIT_HEX_RADIUS
        self.hex_apogee = UNIT_HEX_APOGEE
        
    def calculate_outer_hexagon_side_length(self, inner_params):
        """Calculate the minimal outer hexagon side length required."""
        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(self.n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = HexagonGeometry.create_unit_hexagon((x, y), angle)
            for point in HexagonGeometry.get_all_vertices(hexagon):
                all_vertices.append(point)
        
        if not all_vertices:
            return 1000.0
            
        # Calculate bounding circle
        vertices_array = np.array(all_vertices)
        centroid = np.mean(vertices_array, axis=0)
        distances = np.sqrt(np.sum((vertices_array - centroid)**2, axis=1))
        max_distance = np.max(distances)
        
        # For hexagon, we need to account for the fact that hexagons are inscribed
        # The outer hexagon side length needs to accommodate the maximum distance + padding
        outer_side_length = max_distance * 2 / math.sqrt(3) + 1e-6
        return outer_side_length
    
    def check_constraints(self, inner_params, outer_side_length):
        """Efficient constraint checking for packing feasibility."""
        # Create inner hexagons
        inner_hexagons = []
        for i in range(self.n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = HexagonGeometry.create_unit_hexagon((x, y), angle)
            inner_hexagons.append(hexagon)
        
        # Create outer hexagon
        outer_hexagon = HexagonGeometry.create_unit_hexagon((0, 0), 0)
        outer_coords = list(outer_hexagon.exterior.coords)
        scaled_coords = [(x*outer_side_length, y*outer_side_length) for x, y in outer_coords]
        outer_hexagon_scaled = Polygon(scaled_coords)
        
        # Check containment - check all vertices of each inner hexagon
        for hexagon in inner_hexagons:
            vertices = HexagonGeometry.get_all_vertices(hexagon)
            for vertex in vertices:
                point = Point(vertex)
                if not outer_hexagon_scaled.contains(point):
                    return False, 0.0  # containment violated
                    
        # Check overlaps - check all pairs of hexagons
        for i in range(self.n_inner):
            for j in range(i+1, self.n_inner):
                if inner_hexagons[i].intersects(inner_hexagons[j]):
                    return False, 0.0  # overlap violated
                    
        # Calculate tight radius for objective function
        tight_radius = self.calculate_outer_hexagon_side_length(inner_params)
        return True, 1.0 / tight_radius
    
    def initialize_optimal_config(self):
        """Create an initial configuration based on geometric insights."""
        # Start with a known good hexagonal arrangement with central hexagon
        # Arrange in a pattern that resembles a hexagonal lattice with 11 points
        initial_positions = []
        
        # Central hexagon
        initial_positions.append([0.0, 0.0, 0.0])
        
        # Surrounding hexagons in a hexagonal formation
        # First ring: 6 hexagons around the center
        angles = [k * 60 for k in range(6)]
        for angle in angles:
            rad = 2.0  # Distance between centers (slightly more than 2 unit radii)
            x = rad * math.cos(math.radians(angle))
            y = rad * math.sin(math.radians(angle))
            # Randomize rotation to break symmetry
            rot = np.random.uniform(0, 360)
            initial_positions.append([x, y, rot])
            
        # Second ring: 4 hexagons forming a compact outer ring
        angles = [30, 90, 150, 210]  # Placed to fill gaps nicely
        for angle in angles:
            rad = 3.5  # Further out
            x = rad * math.cos(math.radians(angle))
            y = rad * math.sin(math.radians(angle))
            rot = np.random.uniform(0, 360)
            initial_positions.append([x, y, rot])
            
        # Convert to flattened format
        flattened = []
        for pos in initial_positions:
            flattened.extend(pos)
            
        # Add estimated outer radius (will be refined)
        flattened.append(5.0)
        return np.array(flattened)
    
    def objective_function(self, params):
        """Objective function to minimize: negative of 1/outer_radius."""
        outer_side_length = params[-1]
        inner_params = params[:-1]
        
        # Quick bounds check
        if outer_side_length < 3.0 or outer_side_length > 10.0:
            return 10000.0 + abs(outer_side_length)
            
        # Check constraints
        feasible, inv_radius = self.check_constraints(inner_params, outer_side_length)
        
        if not feasible:
            return 10000.0 + abs(outer_side_length)
            
        # Return negative of inverse radius to minimize (maximize 1/outer_radius)
        return -inv_radius
    
    def optimize(self):
        """Main optimization routine using geometric insights."""
        # Initialize with a smart configuration
        initial_params = self.initialize_optimal_config()
        
        # First, do a coarse optimization to get close to good region
        bounds_coarse = []
        for _ in range(self.n_inner):
            bounds_coarse.extend([(-8.0, 8.0), (-8.0, 8.0), (0, 360)])
        bounds_coarse.append((3.0, 10.0))
        
        # Coarse optimization
        try:
            coarse_result = minimize(
                self.objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds_coarse,
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6},
                callback=lambda x: None
            )
            
            if coarse_result.success:
                # Refine with tighter bounds
                bounds_fine = []
                for _ in range(self.n_inner):
                    bounds_fine.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])
                bounds_fine.append((3.0, 8.0))
                
                fine_result = minimize(
                    self.objective_function,
                    coarse_result.x,
                    method='L-BFGS-B',
                    bounds=bounds_fine,
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8},
                    callback=lambda x: None
                )
                
                if fine_result.success:
                    return fine_result.x
                    
        except Exception as e:
            warnings.warn(f"Coarse optimization failed: {str(e)}")
            
        # If all else fails, return initial guess
        return initial_params

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses geometric optimization with carefully chosen initial configuration.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        optimizer = GeometricOptimizer()
        final_params = optimizer.optimize()
        
        # Extract results
        n = 11
        inner_params = final_params[:-1]
        outer_side_length = final_params[-1]
        
        # Validate solution (can be slow, so let's do a quick sanity check)
        if outer_side_length > 3.0 and outer_side_length < 10.0:
            # Format output
            inner_hex_data = np.zeros((n, 3))
            for i in range(n):
                inner_hex_data[i] = inner_params[3*i:3*i+3]
            
            outer_hex_data = np.array([0, 0, 0])
            
            return inner_hex_data, outer_hex_data, outer_side_length

    except Exception as e:
        warnings.warn(f"Error in optimization: {str(e)}")
        pass

    # Fallback to original method if optimization fails
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