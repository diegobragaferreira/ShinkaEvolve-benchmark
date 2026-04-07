# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import time
from joblib import Parallel, delayed

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
        """Check if inner hexagon is fully contained within outer hexagon with buffer"""
        # Apply small buffer to handle floating point precision issues
        buffered_hexagon = inner_hexagon.buffer(-1e-8)
        buffered_outer = outer_hexagon.buffer(1e-8)
        return buffered_outer.contains(buffered_hexagon)

    def check_overlap(self, hex1, hex2):
        """Check if two hexagons overlap with buffer"""
        # Apply small buffer to handle floating point precision issues
        buffered_hex1 = hex1.buffer(1e-8)
        buffered_hex2 = hex2.buffer(1e-8)
        return buffered_hex1.intersects(buffered_hex2)

    def calculate_tight_outer_radius(self, inner_params):
        """Calculate tightest possible outer hexagon radius using actual vertex positions"""
        # Get all hexagon vertices and find bounding circle
        all_vertices = []

        for i in range(self.n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = self.create_unit_hexagon((x, y), angle)
            # Get all vertices of this hexagon
            for point in hexagon.exterior.coords[:-1]:  # exclude closing point
                all_vertices.append(point)

        if not all_vertices:
            return 1.0

        # Convert to numpy array for easier computation
        vertices_array = np.array(all_vertices)

        # Find centroid of all vertices
        centroid = np.mean(vertices_array, axis=0)

        # Calculate distances from centroid to all vertices
        distances = np.sqrt(np.sum((vertices_array - centroid)**2, axis=1))

        # Outer radius is the maximum distance plus a small margin for numerical stability
        outer_radius = np.max(distances) + 1e-6

        return outer_radius

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

        # Calculate actual tight radius for better objective function
        actual_tight_radius = self.calculate_tight_outer_radius(inner_params)
        return True, True, 1.0 / actual_tight_radius  # valid solution

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

    def generate_initial_configurations(self, num_configs=15):
        """Generate diverse initial configurations using multiple hexagonal patterns"""
        initial_configs = []
        
        # Pattern 1: Honeycomb-like with slight perturbation
        pattern1_centers = [
            (0.0, 0.0),       # center
            (-1.9, 0.0),      # left
            (1.9, 0.0),       # right
            (0.0, 1.9),       # top
            (0.0, -1.9),      # bottom
            (-1.4, 1.4),      # top-left
            (1.4, 1.4),       # top-right
            (-1.4, -1.4),     # bottom-left
            (1.4, -1.4),      # bottom-right
            (-2.3, 0.0),      # further left
            (2.3, 0.0),       # further right
        ]
        
        # Pattern 2: More spread out hexagonal pattern
        pattern2_centers = [
            (0.0, 0.0),       # center
            (-2.1, 0.0),      # left
            (2.1, 0.0),       # right
            (0.0, 2.1),       # top
            (0.0, -2.1),      # bottom
            (-1.6, 1.6),      # top-left
            (1.6, 1.6),       # top-right
            (-1.6, -1.6),     # bottom-left
            (1.6, -1.6),      # bottom-right
            (-2.5, 0.0),      # further left
            (2.5, 0.0),       # further right
        ]
        
        # Pattern 3: Compact arrangement
        pattern3_centers = [
            (0.0, 0.0),       # center
            (-1.7, 0.0),      # left
            (1.7, 0.0),       # right
            (0.0, 1.7),       # top
            (0.0, -1.7),      # bottom
            (-1.2, 1.2),      # top-left
            (1.2, 1.2),       # top-right
            (-1.2, -1.2),     # bottom-left
            (1.2, -1.2),      # bottom-right
            (-2.0, 0.0),      # further left
            (2.0, 0.0),       # further right
        ]
        
        patterns = [pattern1_centers, pattern2_centers, pattern3_centers]
        
        # Generate configs for each pattern
        for i, centers in enumerate(patterns):
            for j in range(5):  # 5 configs per pattern
                config = []
                for k, (cx, cy) in enumerate(centers):
                    # Add small random variation to avoid symmetry issues
                    jitter_x = np.random.normal(0, 0.15)
                    jitter_y = np.random.normal(0, 0.15)
                    angle = np.random.uniform(0, 360)
                    config.extend([cx + jitter_x, cy + jitter_y, angle])
                
                # Add outer radius estimate
                max_dist = max(np.sqrt(cx**2 + cy**2) + self.unit_hex_apogee 
                             for cx, cy in centers)
                config.append(max_dist + 0.4)
                initial_configs.append(config)
        
        return initial_configs

    def optimize_with_local_search(self, initial_params):
        """Refine solution using local optimization with multiple attempts"""
        bounds = []
        # Bounds for inner hexagon positions and rotations - tightened for better convergence
        for _ in range(self.n_inner):
            bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])  # x, y, angle
        # Bound for outer radius - tightened for better convergence
        bounds.append((3.0, 10.0))  # Reasonable range for outer radius

        options = {'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}

        try:
            # First attempt with L-BFGS-B
            result = minimize(
                self.objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options=options,
                callback=lambda x: None  # Empty callback
            )
            
            if result.success and result.fun < -0.25:  # Only return if significantly better
                return result.x
                
        except Exception as e:
            warnings.warn(f"Local search failed: {str(e)}")

        return initial_params

    def optimize_with_de_and_local_search(self, seed_val):
        """Run one complete optimization cycle with DE followed by local search"""
        # Generate bounds for optimization
        bounds = []
        # Bounds for inner hexagon positions and rotations
        for _ in range(self.n_inner):
            bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])  # x, y, angle
        # Bound for outer radius
        bounds.append((3.0, 10.0))  # Reasonable range for outer radius

        # Generate initial population
        initial_configs = self.generate_initial_configurations()
        np.random.seed(seed_val)
        initial_guess = initial_configs[np.random.randint(0, len(initial_configs))]
        
        # Optimization settings with enhanced parameters
        try:
            # Use differential evolution for global optimization with better parameters
            result = differential_evolution(
                self.objective_function,
                bounds,
                seed=seed_val,
                maxiter=200,  # Increase iterations
                popsize=30,   # Increase population size
                tol=1e-8,     # Tighter tolerance
                mutation=(0.7, 1.2),  # More aggressive mutation
                recombination=0.8,    # Higher recombination rate
                disp=False
            )

            if result.success:
                # Refine with local search
                refined_params = self.optimize_with_local_search(result.x)
                return refined_params
                
        except Exception as e:
            warnings.warn(f"Optimization failed with seed {seed_val}: {str(e)}")
            
        return initial_guess

    def optimize_solution(self):
        """Main optimization routine using parallel differential evolution"""
        # Run multiple optimization attempts in parallel
        seeds = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]
        
        # Run parallel optimization
        results = Parallel(n_jobs=-1)(
            delayed(self.optimize_with_de_and_local_search)(seed)
            for seed in seeds
        )
        
        # Find the best result among all attempts
        best_result = None
        best_score = float('-inf')
        
        for result in results:
            if result is not None:
                # Evaluate the fitness of this solution
                n = self.n_inner
                inner_params = result[:-1]
                outer_radius = result[-1]
                
                # Quick constraint check without expensive full validation
                try:
                    _, _, inv_radius = self.evaluate_constraints(inner_params, outer_radius)
                    if inv_radius > best_score and inv_radius > 0.20:  # Only consider promising solutions
                        best_score = inv_radius
                        best_result = result
                except:
                    continue
        
        if best_result is not None:
            return best_result
        else:
            # Fallback to a single run if nothing worked
            return self.optimize_with_de_and_local_search(42)

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