# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from numba import jit
import warnings
from joblib import Parallel, delayed
import random

# Constants
NUM_INNER_HEX = 11
UNIT_HEX_RADIUS = 1.0

class HexagonGeometry:
    """Efficient geometric operations for hexagon computations"""
    
    @staticmethod
    @jit(nopython=True)
    def get_hexagon_vertices(x, y, angle_deg, radius=1.0):
        """Get vertices of a regular hexagon given position and angle"""
        vertices = np.zeros((6, 2))
        angle_rad = np.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
        return vertices

    @staticmethod
    @jit(nopython=True)
    def get_hexagon_edges(vertices):
        """Get list of edges from hexagon vertices"""
        edges = np.zeros((6, 2))
        for i in range(6):
            edges[i] = vertices[(i+1)%6] - vertices[i]
        return edges

    @staticmethod
    @jit(nopython=True)
    def get_separating_axes(edges):
        """Get normals to all edges for SAT separation test"""
        axes = np.zeros((6, 2))
        for i in range(6):
            edge = edges[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.sqrt(normal[0]**2 + normal[1]**2)
            if norm > 1e-10:
                normal = normal / norm
            axes[i] = normal
        return axes

    @staticmethod
    @jit(nopython=True)
    def project_polygon_onto_axis(vertices, axis):
        """Project all vertices of a polygon onto an axis"""
        projections = np.zeros(6)
        for i in range(6):
            projections[i] = vertices[i, 0] * axis[0] + vertices[i, 1] * axis[1]
        return np.min(projections), np.max(projections)

    @staticmethod
    def hexagons_overlap_fast(hex1_vertices, hex2_vertices):
        """Fast SAT-based overlap detection for hexagons"""
        # Get edges and axes
        edges1 = HexagonGeometry.get_hexagon_edges(hex1_vertices)
        edges2 = HexagonGeometry.get_hexagon_edges(hex2_vertices)
        all_edges = np.vstack([edges1, edges2])
        axes = HexagonGeometry.get_separating_axes(all_edges)
        
        # Test each axis
        for axis in axes:
            min1, max1 = HexagonGeometry.project_polygon_onto_axis(hex1_vertices, axis)
            min2, max2 = HexagonGeometry.project_polygon_onto_axis(hex2_vertices, axis)
            
            # If projections don't overlap, there's a separating axis
            if max1 < min2 or max2 < min1:
                return False
                
        return True

class HexagonValidator:
    """Validates hexagon configurations with performance optimizations"""
    
    @staticmethod
    def validate_hexagon_placement(hex_vertices, outer_vertices):
        """Validate a single hexagon's placement"""
        # Check containment in outer hexagon quickly
        outer_center = np.array([0.0, 0.0])
        outer_radius = np.linalg.norm(outer_vertices[0])
        for vertex in hex_vertices:
            distance = np.linalg.norm(np.array(vertex) - outer_center)
            if distance > outer_radius * 0.99:  # Allow for small numerical errors
                return False
        return True

    @staticmethod
    def validate_configuration_fast(inner_hex_data, outer_side_length):
        """Fast validation of entire configuration"""
        outer_vertices = HexagonGeometry.get_hexagon_vertices(0, 0, 0, outer_side_length)
        
        n = len(inner_hex_data)
        
        # Check all hexagons for containment
        for i in range(n):
            x, y, angle = inner_hex_data[i]
            hex_vertices = HexagonGeometry.get_hexagon_vertices(x, y, angle)
            
            if not HexagonValidator.validate_hexagon_placement(hex_vertices, outer_vertices):
                return False, 0.0

        # Check pairwise overlaps (efficiently)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, angle1 = inner_hex_data[i]
                x2, y2, angle2 = inner_hex_data[j]
                
                hex1_vertices = HexagonGeometry.get_hexagon_vertices(x1, y1, angle1)
                hex2_vertices = HexagonGeometry.get_hexagon_vertices(x2, y2, angle2)
                
                if HexagonGeometry.hexagons_overlap_fast(hex1_vertices, hex2_vertices):
                    return False, 0.0
                    
        return True, 1.0 / outer_side_length

class HexagonPackingOptimizer:
    """Optimized hexagon packing optimization class"""
    
    def __init__(self):
        self.best_inner_data = None
        self.best_outer_side_length = float('inf')
        self.best_score = -float('inf')
        
    def generate_initial_guess(self, method='hybrid'):
        """Generate high-quality initial configuration using multiple strategies"""
        if method == 'hexagonal':
            # Hexagonal pattern based on literature
            base_positions = [
                [0, 0, 0],  # center
                [-2.0, 0, 0],  # left
                [2.0, 0, 0],  # right
                [0, 2.0, 0],  # top
                [0, -2.0, 0],  # bottom
                [-1.0, 1.0, 0],  # top-left
                [1.0, 1.0, 0],  # top-right
                [-1.0, -1.0, 0],  # bottom-left
                [1.0, -1.0, 0],  # bottom-right
                [-2.0, 1.0, 0],  # far top-left
                [2.0, 1.0, 0],  # far top-right
            ]
        elif method == 'spiral':
            # Spiral pattern for diversity
            base_positions = []
            for i in range(11):
                angle = i * 0.5
                radius = i * 0.5
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                rotation = np.random.uniform(0, 360)
                base_positions.append([x, y, rotation])
        else:  # random
            # Random pattern with better distribution
            base_positions = []
            for i in range(11):
                x = np.random.uniform(-3, 3)
                y = np.random.uniform(-3, 3)
                rotation = np.random.uniform(0, 360)
                base_positions.append([x, y, rotation])
        
        # Add small jitter to escape local minima
        base_positions = np.array(base_positions)
        for i in range(len(base_positions)):
            base_positions[i][0] += np.random.normal(0, 0.05)
            base_positions[i][1] += np.random.normal(0, 0.05)
            base_positions[i][2] += np.random.normal(0, 2)
            
        return base_positions

    def evaluate_objective(self, params, outer_side_length_guess=None):
        """Evaluate objective function for optimization"""
        # Reshape parameters into hexagon data
        hex_data = params.reshape(11, 3)
        
        # Use provided outer side length or estimate it
        if outer_side_length_guess is not None:
            outer_side_length = outer_side_length_guess
        else:
            # Quick estimate
            max_dist = 0
            for i in range(11):
                x, y, _ = hex_data[i]
                dist = np.sqrt(x*x + y*y) + UNIT_HEX_RADIUS
                max_dist = max(max_dist, dist)
            outer_side_length = max_dist * 1.1  # Safety margin
            
        # Validate configuration
        is_valid, objective_value = HexagonValidator.validate_configuration_fast(hex_data, outer_side_length)
        
        # Return negative for minimization (we want to maximize 1/outer_side_length)
        if is_valid:
            return -objective_value
        else:
            # Large penalty for invalid configurations
            return -1e10

    def optimize_with_de(self, initial_guess):
        """Use Differential Evolution for global optimization"""
        # Flatten the initial guess
        flat_initial = initial_guess.flatten()
        
        # Define bounds for optimization
        bounds = []
        for i in range(11):  # 11 hexagons
            # x coordinates
            bounds.append((-10, 10))
            # y coordinates  
            bounds.append((-10, 10))
            # rotations - 0 to 360 degrees
            bounds.append((0, 360))
            
        try:
            # Run differential evolution with optimized parameters
            result = differential_evolution(
                self.evaluate_objective,
                bounds,
                maxiter=150,
                popsize=8,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )
            
            # Extract solution
            if result.success:
                final_params = result.x
                final_hex_data = final_params.reshape(11, 3)
                
                # Recalculate the best outer side length
                max_dist = 0
                for i in range(11):
                    x, y, _ = final_hex_data[i]
                    dist = np.sqrt(x*x + y*y) + UNIT_HEX_RADIUS
                    max_dist = max(max_dist, dist)
                best_outer_side_length = max_dist * 1.1
                
                # Verify final configuration
                is_valid, objective_value = HexagonValidator.validate_configuration_fast(final_hex_data, best_outer_side_length)
                
                if is_valid and objective_value > self.best_score:
                    self.best_score = objective_value
                    self.best_inner_data = final_hex_data.copy()
                    self.best_outer_side_length = best_outer_side_length
                    
        except Exception as e:
            warnings.warn(f"Differential evolution failed: {e}")

    def local_refinement(self):
        """Apply local refinement to improve solution"""
        if self.best_inner_data is None:
            return
            
        # Simple local optimization using gradient ascent approach
        step_size = 0.05
        max_iterations = 30  # Reduced iterations for time efficiency
        
        for _ in range(max_iterations):
            improved = False
            current_params = self.best_inner_data.flatten()
            
            # Try small perturbations to each parameter
            for i in range(len(current_params)):
                # Save current parameters
                old_val = current_params[i]
                
                # Try moving in positive direction
                current_params[i] = old_val + step_size
                test_score = self.evaluate_objective(current_params)
                
                if test_score < -self.best_score:  # Better score (lower negative value)
                    self.best_score = -test_score
                    improved = True
                else:
                    # Revert
                    current_params[i] = old_val
                    
                # Try moving in negative direction
                current_params[i] = old_val - step_size
                test_score = self.evaluate_objective(current_params)
                
                if test_score < -self.best_score:  # Better score
                    self.best_score = -test_score
                    improved = True
                else:
                    # Revert
                    current_params[i] = old_val
                    
            if not improved:
                break
                
        # Update final data
        if self.best_inner_data is not None:
            self.best_inner_data = current_params.reshape(11, 3)

def optimize_single_instance(optimizer, init_method, seed=None):
    """Run optimization for a single instance with specific initialization"""
    if seed is not None:
        np.random.seed(seed)

    try:
        # Create initial configuration using specified method
        initial_config = optimizer.generate_initial_guess(method=init_method)

        # Perform optimization with DE first
        optimizer.optimize_with_de(initial_config)
        
        # Apply local refinement
        optimizer.local_refinement()
        
        return optimizer.best_score, optimizer.best_inner_data, optimizer.best_outer_side_length
    except Exception as e:
        return -1.0, None, None

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    optimizer = HexagonPackingOptimizer()

    # Run parallel optimization with different initializations
    # Use 3 different initialization methods and 3 seeds each = 9 parallel runs
    init_methods = ['hexagonal', 'spiral', 'random']
    seeds = [42, 123, 456]

    # Prepare jobs
    jobs = []
    for method in init_methods:
        for seed in seeds:
            jobs.append(delayed(optimize_single_instance)(optimizer, method, seed))

    # Execute parallel jobs
    results = Parallel(n_jobs=4, verbose=0)(jobs)

    # Find the best result across all parallel runs
    best_score = -1.0
    best_result = None

    for score, config, side_length in results:
        if score > best_score and config is not None:
            best_score = score
            best_result = (config, side_length)

    # If we found a good solution, return it
    if best_result is not None and best_score > 0.1:  # Only accept reasonable solutions
        best_config, best_side_length = best_result
        # Verify final solution
        if HexagonValidator.validate_configuration_fast(best_config, best_side_length)[0]:
            outer_hex_data = np.array([0, 0, 0])
            return best_config, outer_hex_data, best_side_length

    # Fallback to original approach with better validation
    initial_config = np.array([
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
    ])

    # Set reasonable initial outer hexagon size based on configuration
    max_dist_from_center = 0
    for i in range(len(initial_config)):
        center_x, center_y, _ = initial_config[i]
        dist = np.sqrt(center_x**2 + center_y**2)
        max_dist_from_center = max(max_dist_from_center, dist + 1.0)  # Add radius margin

    # Outer hexagon should have side length slightly larger than max distance
    outer_hex_side_length = max_dist_from_center * 1.2  # 20% margin

    # Evaluate this configuration
    valid, _ = HexagonValidator.validate_configuration_fast(initial_config, outer_hex_side_length)

    # If initial configuration is invalid due to overlap or containment,
    # we fall back to the simpler approach but with better validation
    if not valid:
        # Fallback to a basic valid configuration
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
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0  # fallback value
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Since we've confirmed initial config works, we can return it
    inner_hex_data = initial_config.copy()
    outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END