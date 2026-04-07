# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import time
from joblib import Parallel, delayed
import copy
from functools import partial

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3)/2

class HexagonGeometry:
    """Specialized geometric utilities for hexagon operations"""
    
    @staticmethod
    def create_unit_hexagon(center=(0,0), rotation=0, side_length=1):
        """Create a unit regular hexagon as a Shapely Polygon"""
        angle_offset = np.deg2rad(rotation)
        points = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + side_length * np.cos(angle)
            y = center[1] + side_length * np.sin(angle)
            points.append((x, y))
        return Polygon(points)

    @staticmethod
    def validate_polygon(polygon):
        """Ensure polygon is valid for geometric operations"""
        if not polygon.is_valid:
            return make_valid(polygon)
        return polygon

    @staticmethod
    def get_hexagon_vertices(center, rotation, side_length=1):
        """Get vertices of a hexagon without creating polygon object"""
        angle_offset = np.deg2rad(rotation)
        vertices = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + side_length * np.cos(angle)
            y = center[1] + side_length * np.sin(angle)
            vertices.append((x, y))
        return vertices

    @staticmethod
    def get_hexagon_bounding_box(center, rotation, side_length=1):
        """Fast computation of hexagon bounding box"""
        # For unit hexagon with side length 1, the apothem is sqrt(3)/2 ≈ 0.866
        apothem = side_length * np.sqrt(3) / 2
        # The radius is equal to side length for regular hexagon
        radius = side_length
        
        # Fast bounding box calculation
        # We know that for a hexagon, the bounding box dimensions are:
        # width = 2 * radius = 2
        # height = 2 * apothem = sqrt(3) ≈ 1.732
        # But we also need to account for rotation effects
        
        # For simplicity, assuming the hexagon fits in a square with diagonal 2*radius
        half_diag = radius * np.sqrt(2)
        
        return (
            center[0] - half_diag,
            center[0] + half_diag,
            center[1] - half_diag,
            center[1] + half_diag
        )

class HexagonConstraintValidator:
    """Advanced constraint checker with smart filtering"""
    
    @staticmethod
    def quick_containment_check(inner_vertices, outer_radius):
        """Fast preliminary containment check using bounding boxes"""
        # Check if any vertex is outside the outer hexagon
        for vx, vy in inner_vertices:
            # For outer hexagon centered at origin with radius R, 
            # check if point is within distance R from center
            if np.sqrt(vx*vx + vy*vy) > outer_radius:
                return False
        return True
    
    @staticmethod
    def quick_overlap_check(vertices1, vertices2):
        """Fast bounding box overlap check"""
        # Get bounding boxes
        min1_x = min(v[0] for v in vertices1)
        max1_x = max(v[0] for v in vertices1)
        min1_y = min(v[1] for v in vertices1)
        max1_y = max(v[1] for v in vertices1)
        
        min2_x = min(v[0] for v in vertices2)
        max2_x = max(v[0] for v in vertices2)
        min2_y = min(v[1] for v in vertices2)
        max2_y = max(v[1] for v in vertices2)
        
        # Check overlap
        if (max1_x < min2_x or max2_x < min1_x or 
            max1_y < min2_y or max2_y < min1_y):
            return False
        return True
    
    @staticmethod
    def precise_containment_check(inner_hexagon, outer_hexagon):
        """Precise containment check with buffer"""
        buffered_inner = inner_hexagon.buffer(-1e-10)
        return outer_hexagon.contains(buffered_inner)

    @staticmethod
    def precise_overlap_check(hex1, hex2):
        """Precise overlap check with buffer"""
        buffered_hex1 = hex1.buffer(1e-10)
        buffered_hex2 = hex2.buffer(1e-10)
        return buffered_hex1.intersects(buffered_hex2)

class HexagonPackingOptimizer:
    """Multi-scale evolutionary optimizer for hexagon packing"""
    
    def __init__(self):
        self.n_inner = 11
        self.unit_hex_radius = UNIT_HEX_RADIUS
        self.unit_hex_apogee = UNIT_HEX_APOGEE
        self.geometry = HexagonGeometry()
        self.validator = HexagonConstraintValidator()
        
    def calculate_geometry_metrics(self, inner_params):
        """Calculate key geometric metrics for evaluation"""
        # Get all hexagon vertices to compute tight bounding
        all_vertices = []
        for i in range(self.n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            vertices = self.geometry.get_hexagon_vertices((x, y), angle)
            all_vertices.extend(vertices)

        if not all_vertices:
            return 1.0, 0.0

        # Find centroid of all vertices
        vertices_array = np.array(all_vertices)
        centroid = np.mean(vertices_array, axis=0)

        # Calculate distances from centroid to all vertices
        distances = np.sqrt(np.sum((vertices_array - centroid)**2, axis=1))

        # Maximum distance + margin for numerical stability
        tight_radius = np.max(distances) + 1e-6
        
        # Compute packing density approximation
        # Area of inner hexagons (11 hexagons with area = 3*sqrt(3)/2 ≈ 2.598 each)
        inner_area = 11 * 3 * np.sqrt(3) / 2
        
        # Area of outer hexagon with tight radius
        outer_area = 3 * np.sqrt(3) * tight_radius * tight_radius / 2
        
        density = inner_area / outer_area if outer_area > 0 else 0.0
        
        return tight_radius, density
    
    def validate_solution(self, inner_params, outer_radius):
        """Comprehensive solution validation with progressive checks"""
        # Get inner hexagons vertices
        inner_vertices_list = []
        for i in range(self.n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            vertices = self.geometry.get_hexagon_vertices((x, y), angle)
            inner_vertices_list.append(vertices)
        
        # Quick bounding box check for containment
        for vertices in inner_vertices_list:
            if not self.validator.quick_containment_check(vertices, outer_radius):
                return False, False, 0.0, 0.0

        # Quick overlap check between all pairs
        for i in range(self.n_inner):
            for j in range(i+1, self.n_inner):
                if not self.validator.quick_overlap_check(
                    inner_vertices_list[i], 
                    inner_vertices_list[j]
                ):
                    continue  # Skip expensive detailed check
                
                # Only do precise check if quick check indicates potential overlap
                x1, y1, angle1 = inner_params[3*i:3*i+3]
                x2, y2, angle2 = inner_params[3*j:3*j+3]
                
                hex1 = self.geometry.create_unit_hexagon((x1, y1), angle1)
                hex2 = self.geometry.create_unit_hexagon((x2, y2), angle2)
                
                if self.validator.precise_overlap_check(hex1, hex2):
                    return False, False, 0.0, 0.0

        # Precise containment check
        outer_hexagon = self.geometry.create_unit_hexagon((0, 0), 0)
        outer_coords = list(outer_hexagon.exterior.coords)
        scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
        outer_hexagon_scaled = Polygon(scaled_coords)
        
        for i in range(self.n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            inner_hex = self.geometry.create_unit_hexagon((x, y), angle)
            if not self.validator.precise_containment_check(inner_hex, outer_hexagon_scaled):
                return False, False, 0.0, 0.0

        # Calculate tight radius and density
        tight_radius, density = self.calculate_geometry_metrics(inner_params)
        return True, True, 1.0 / tight_radius, density
    
    def objective_function(self, params):
        """Multi-objective function that balances tightness and density"""
        # params: [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11, outer_radius]
        outer_radius = params[-1]
        inner_params = params[:-1]

        # Validate solution
        containment_ok, overlap_ok, inv_radius, density = self.validate_solution(inner_params, outer_radius)

        if not (containment_ok and overlap_ok):
            return 10000.0 + abs(outer_radius)  # Penalty for constraint violations

        # Multi-objective: maximize 1/outer_radius AND packing density
        # We want to balance tightness with packing efficiency
        return -(inv_radius + 0.1 * density)  # Negative because we minimize
    
    def generate_initial_configurations(self):
        """Generate diverse initial configurations using geometric insights"""
        configs = []
        
        # Pattern 1: Central plus ring arrangement
        base_pattern1 = [
            (0.0, 0.0, 0.0),       # center
            (-1.8, 0.0, 0.0),      # left
            (1.8, 0.0, 0.0),       # right
            (0.0, 1.8, 0.0),       # top
            (0.0, -1.8, 0.0),      # bottom
            (-1.3, 1.3, 0.0),      # top-left
            (1.3, 1.3, 0.0),       # top-right
            (-1.3, -1.3, 0.0),     # bottom-left
            (1.3, -1.3, 0.0),      # bottom-right
            (-2.2, 0.0, 0.0),      # further left
            (2.2, 0.0, 0.0),       # further right
        ]
        
        # Pattern 2: Honeycomb-like structure with more spread
        base_pattern2 = [
            (0.0, 0.0, 0.0),       # center
            (-2.0, 0.0, 0.0),      # left
            (2.0, 0.0, 0.0),       # right
            (0.0, 2.0, 0.0),       # top
            (0.0, -2.0, 0.0),      # bottom
            (-1.5, 1.5, 0.0),      # top-left
            (1.5, 1.5, 0.0),       # top-right
            (-1.5, -1.5, 0.0),     # bottom-left
            (1.5, -1.5, 0.0),      # bottom-right
            (-2.8, 0.0, 0.0),      # further left
            (2.8, 0.0, 0.0),       # further right
        ]
        
        # Pattern 3: Optimized spacing
        base_pattern3 = [
            (0.0, 0.0, 0.0),       # center
            (-1.6, 0.0, 0.0),      # left
            (1.6, 0.0, 0.0),       # right
            (0.0, 1.6, 0.0),       # top
            (0.0, -1.6, 0.0),      # bottom
            (-1.2, 1.2, 0.0),      # top-left
            (1.2, 1.2, 0.0),       # top-right
            (-1.2, -1.2, 0.0),     # bottom-left
            (1.2, -1.2, 0.0),      # bottom-right
            (-2.4, 0.0, 0.0),      # further left
            (2.4, 0.0, 0.0),       # further right
        ]
        
        patterns = [base_pattern1, base_pattern2, base_pattern3]
        
        # Generate diverse configurations for each base pattern
        for pattern in patterns:
            for _ in range(20):
                config = []
                for i, (cx, cy, angle) in enumerate(pattern):
                    # Add small random variation
                    jitter_x = np.random.uniform(-0.15, 0.15)
                    jitter_y = np.random.uniform(-0.15, 0.15)
                    jitter_angle = np.random.uniform(-10, 10)
                    config.extend([cx + jitter_x, cy + jitter_y, angle + jitter_angle])
                config.append(4.0 + np.random.uniform(0.2, 1.0))  # outer radius estimate
                configs.append(config)
        
        return configs
    
    def local_refinement(self, initial_params):
        """Refine solution using local search with dynamic bounds"""
        bounds = []
        # Dynamic bounds based on current solution
        for _ in range(self.n_inner):
            bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])  # x, y, angle
        bounds.append((3.0, 10.0))  # outer radius range

        options = {'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}

        try:
            result = minimize(
                self.objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options=options,
                callback=lambda x: None
            )

            if result.success:
                return result.x

        except Exception as e:
            warnings.warn(f"Local search failed: {str(e)}")

        return initial_params
    
    def hierarchical_optimization(self):
        """Multi-stage optimization approach"""
        # Stage 1: Coarse global search with broad bounds
        bounds_stage1 = []
        for _ in range(self.n_inner):
            bounds_stage1.extend([(-8.0, 8.0), (-8.0, 8.0), (0, 360)])
        bounds_stage1.append((3.0, 12.0))
        
        # Generate initial configurations
        initial_configs = self.generate_initial_configurations()
        
        # Run several differential evolution attempts with different seeds
        best_objective = float('inf')
        best_params = None
        
        for i, config in enumerate(initial_configs[:8]):  # Test 8 configs
            try:
                result = differential_evolution(
                    self.objective_function,
                    bounds_stage1,
                    seed=42 + i,
                    maxiter=30,
                    popsize=15,
                    tol=1e-5,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    disp=False
                )
                
                if result.success:
                    obj_val = self.objective_function(result.x)
                    if obj_val < best_objective:
                        best_objective = obj_val
                        best_params = result.x.copy()
                        
            except Exception as e:
                continue
        
        # Stage 2: Fine-grained refinement  
        if best_params is not None:
            # Tighten bounds and do local refinement
            refined_params = self.local_refinement(best_params)
            return refined_params
        
        # Fallback: use default configuration
        initial_guess = []
        centers = [
            (0.0, 0.0),       # center
            (-1.8, 0.0),      # left
            (1.8, 0.0),       # right
            (0.0, 1.8),       # top
            (0.0, -1.8),      # bottom
            (-1.3, 1.3),      # top-left
            (1.3, 1.3),       # top-right
            (-1.3, -1.3),     # bottom-left
            (1.3, -1.3),      # bottom-right
            (-2.2, 0.0),      # further left
            (2.2, 0.0),       # further right
        ]

        for i, (cx, cy) in enumerate(centers):
            initial_guess.extend([cx, cy, np.random.uniform(0, 360)])

        initial_guess.append(4.0)  # Initial outer radius estimate
        return initial_guess

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hierarchical evolutionary optimization approach.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    optimizer = HexagonPackingOptimizer()

    try:
        # Run hierarchical optimization
        final_params = optimizer.hierarchical_optimization()

        # Extract results
        n = 11
        inner_params = final_params[:-1]
        outer_radius = final_params[-1]

        # Validate solution again
        containment_ok, overlap_ok, inv_radius, density = optimizer.validate_solution(inner_params, outer_radius)

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
