# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import time

# Constants
UNIT_HEX_RADIUS = 1.0  # Side length of unit hexagon
UNIT_HEX_APOGEE = np.sqrt(3)/2  # Distance from center to corner of unit hexagon

class HexagonPacker:
    def __init__(self):
        self.n_inner = 11
        self.unit_hex_radius = UNIT_HEX_RADIUS
        self.unit_hex_apogee = UNIT_HEX_APOGEE
        
    def create_unit_hexagon(self, center=(0,0), rotation=0):
        """Create a unit regular hexagon as a Shapely Polygon"""
        angle_offset = np.deg2rad(rotation)
        points = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + self.unit_hex_radius * np.cos(angle)
            y = center[1] + self.unit_hex_radius * np.sin(angle)
            points.append((x, y))
        return Polygon(points)
    
    def validate_polygon(self, polygon):
        """Ensure polygon is valid for geometric operations"""
        if not polygon.is_valid:
            return make_valid(polygon)
        return polygon
    
    def check_containment(self, inner_hexagon, outer_hexagon):
        """Check if inner hexagon is fully contained within outer hexagon"""
        # Check if all vertices of inner hexagon are inside outer hexagon
        for point in inner_hexagon.exterior.coords[:-1]:
            if not outer_hexagon.contains(Point(point)):
                return False
        return True
    
    def check_overlap(self, hex1, hex2):
        """Check if two hexagons overlap"""
        return hex1.intersects(hex2)
    
    def calculate_outer_hex_radius(self, inner_hex_data, outer_center=(0,0)):
        """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
        max_dist = 0
        for i in range(len(inner_hex_data)):
            center = inner_hex_data[i][:2]
            dist = np.linalg.norm(np.array(center) - np.array(outer_center))
            # Add distance from center to corner of unit hexagon
            dist += self.unit_hex_apogee
            max_dist = max(max_dist, dist)
        return max_dist

    def evaluate_constraints(self, inner_params, outer_radius):
        """Comprehensive constraint evaluation with early termination"""
        inner_hexagons = []
        
        # Create inner hexagons
        for i in range(self.n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = self.create_unit_hexagon((x, y), angle)
            inner_hexagons.append(hexagon)
            
        # Create outer hexagon
        outer_hexagon = self.create_unit_hexagon((0, 0), 0)
        outer_coords = list(outer_hexagon.exterior.coords)
        scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
        outer_hexagon_scaled = Polygon(scaled_coords)
        
        # Check containment (early termination)
        for hexagon in inner_hexagons:
            if not self.check_containment(hexagon, outer_hexagon_scaled):
                return False, False, 0.0  # containment violated
            
        # Check overlaps (early termination)
        for i in range(self.n_inner):
            for j in range(i+1, self.n_inner):
                if self.check_overlap(inner_hexagons[i], inner_hexagons[j]):
                    return False, False, 0.0  # overlap violated
                    
        return True, True, 1.0 / outer_radius  # valid solution

    def objective_function(self, params):
        """Objective function to minimize: negative of 1/outer_radius (i.e., maximize 1/outer_radius)"""
        # params: [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11, outer_radius]
        n = self.n_inner
        outer_radius = params[-1]
        
        # Extract inner hexagon parameters
        inner_params = params[:-1]
        
        # Check constraints
        containment_ok, overlap_ok, inv_radius = self.evaluate_constraints(inner_params, outer_radius)
        
        # If any constraint violated, return large penalty
        if not (containment_ok and overlap_ok):
            return 10000.0 + abs(outer_radius)  # penalty for constraint violations

        # Return negative of inverse radius to minimize (maximize 1/outer_radius)
        return -inv_radius

    def generate_initial_population(self, pop_size=50):
        """Generate diverse initial configurations using geometric heuristics"""
        initial_configs = []
        
        # Generate multiple starting configurations
        for _ in range(pop_size):
            config = []
            
            # Center hexagon
            config.extend([0.0, 0.0, np.random.uniform(0, 360)])
            
            # Surrounding hexagons in honeycomb pattern with slight variations
            base_positions = [
                (-2.0, 0.0),
                (2.0, 0.0),
                (0.0, 2.0),
                (0.0, -2.0),
                (-1.0, 1.0),
                (1.0, 1.0),
                (-1.0, -1.0),
                (1.0, -1.0),
                (-2.5, 1.5),
                (2.5, 1.5),
                (-2.5, -1.5),
                (2.5, -1.5)
            ]
            
            # Select 11 positions with some jitter
            selected_positions = base_positions[:11]
            for i, (cx, cy) in enumerate(selected_positions):
                # Add small random variation
                jitter_x = np.random.normal(0, 0.3)
                jitter_y = np.random.normal(0, 0.3)
                config.extend([cx + jitter_x, cy + jitter_y, np.random.uniform(0, 360)])
                
            # Add outer radius estimate
            config.append(5.0 + np.random.uniform(0, 2.0))
            initial_configs.append(config)
            
        return initial_configs

    def optimize_with_local_search(self, initial_params):
        """Refine solution using local optimization after global search"""
        bounds = []
        # Bounds for inner hexagon positions and rotations
        for _ in range(self.n_inner):
            bounds.extend([(-8.0, 8.0), (-8.0, 8.0), (0, 360)])  # x, y, angle
        # Bound for outer radius
        bounds.append((3.0, 12.0))  # Reasonable range for outer radius

        options = {'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
        
        try:
            # Use L-BFGS-B for fine-tuning
            result = minimize(
                self.objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options=options,
                callback=lambda x: None  # Empty callback
            )
            
            if result.success:
                return result.x
                
        except Exception as e:
            warnings.warn(f"Local search failed: {str(e)}")
            
        return initial_params

    def optimize_solution(self):
        """Main optimization routine using differential evolution followed by local search"""
        # Generate bounds for optimization
        bounds = []
        
        # Bounds for inner hexagon positions and rotations
        for _ in range(self.n_inner):
            bounds.extend([(-8.0, 8.0), (-8.0, 8.0), (0, 360)])  # x, y, angle
            
        # Bound for outer radius
        bounds.append((3.0, 12.0))  # Reasonable range for outer radius

        # Initial guess from heuristic placement
        initial_guess = []
        
        # Honeycomb-like arrangement with some randomness
        centers = [
            (0, 0),           # center
            (-2, 0),          # left
            (2, 0),           # right
            (0, 2),           # top
            (0, -2),          # bottom
            (-1, 1),          # top-left
            (1, 1),           # top-right
            (-1, -1),         # bottom-left
            (1, -1),          # bottom-right
            (-2.5, 1.5),      # far top-left
            (2.5, 1.5),       # far top-right
        ]
        
        for i, (cx, cy) in enumerate(centers):
            initial_guess.extend([cx, cy, np.random.uniform(0, 60)])
            
        initial_guess.append(5.0)  # Initial outer radius estimate

        # Optimization settings
        try:
            # Use differential evolution for global optimization
            result = differential_evolution(
                self.objective_function,
                bounds,
                seed=42,
                maxiter=150,
                popsize=20,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )
            
            if result.success:
                # Refine with local search
                refined_params = self.optimize_with_local_search(result.x)
                return refined_params
                
        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")
            
        # Return initial guess if optimization fails
        return initial_guess

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary optimization to find the best arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    packer = HexagonPacker()
    
    try:
        # Run optimization
        final_params = packer.optimize_solution()
        
        # Extract results
        n = 11
        inner_params = final_params[:-1]
        outer_radius = final_params[-1]
        
        # Validate solution
        containment_ok, overlap_ok, inv_radius = packer.evaluate_constraints(inner_params, outer_radius)
        
        if containment_ok and overlap_ok:
            # Format output
            inner_hex_data = np.zeros((n, 3))
            for i in range(n):
                inner_hex_data[i] = inner_params[3*i:3*i+3]
            
            outer_hex_data = np.array([0, 0, 0])
            
            return inner_hex_data, outer_hex_data, outer_radius
            
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