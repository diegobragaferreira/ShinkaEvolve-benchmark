# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree
import time
from typing import Tuple, List, Optional
import random

class HexagonPackerMonteCarlo:
    """Monte Carlo approach to hexagon packing with symmetry awareness."""
    
    def __init__(self, n_hexagons: int = 12):
        self.n_hexagons = n_hexagons
        self.hex_side_length = 1.0
        self.best_score = 0.0
        self.best_config = None
        self.best_outer_radius = float('inf')
        self.max_iterations = 100000
        self.validation_attempts = 1000

    @staticmethod
    def _hexagon_vertices(center_x: float, center_y: float, angle_deg: float,
                         side_length: float = 1.0) -> np.ndarray:
        """Compute vertices of a regular hexagon."""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            x = center_x + side_length * np.cos(theta)
            y = center_y + side_length * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    @staticmethod
    def _compute_hexagon_circumradius(side_length: float = 1.0) -> float:
        """Return the circumradius of a regular hexagon."""
        return side_length

    @staticmethod
    def _compute_hexagon_inradius(side_length: float = 1.0) -> float:
        """Return the inradius of a regular hexagon."""
        return side_length * np.sqrt(3) / 2

    def _check_containment_all_vertices(self, hex_vertices: np.ndarray, 
                                       outer_center: Tuple[float, float], 
                                       outer_radius: float) -> bool:
        """Check if all vertices of hexagon are within outer hexagon."""
        outer_vertices = self._hexagon_vertices(outer_center[0], outer_center[1], 0, outer_radius)
        outer_polygon = Polygon(outer_vertices)

        for vertex in hex_vertices:
            point = Point(vertex)
            if not outer_polygon.contains(point):
                return False
        return True

    def _check_overlap_pair(self, hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Check if two hexagons overlap."""
        hex1_polygon = Polygon(hex1_vertices)
        hex2_polygon = Polygon(hex2_vertices)
        return hex1_polygon.intersects(hex2_polygon)

    def _fast_collision_check(self, positions: np.ndarray, angles: np.ndarray) -> bool:
        """Fast collision detection using bounding circles."""
        # Precompute centers for all hexagons
        centers = positions.copy()
        
        # For each pair, check if their circumradii overlap
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist_sq = np.sum((centers[i] - centers[j]) ** 2)
                # Two unit hexagons can overlap if distance between centers < 2
                if dist_sq < 4.0:
                    return True  # Potentially overlapping
        return False

    def _estimate_outer_radius(self, positions: np.ndarray, angles: np.ndarray) -> float:
        """Efficiently estimate required outer hexagon radius."""
        # Get all vertices of all hexagons
        all_vertices = []
        for pos, angle in zip(positions, angles):
            vertices = self._hexagon_vertices(pos[0], pos[1], angle)
            all_vertices.extend(vertices)

        if len(all_vertices) == 0:
            return 10.0

        # Compute bounding box
        coords = np.array(all_vertices)
        min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
        min_y, max_y = coords[:, 1].min(), coords[:, 1].max()

        # Center of bounding box
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Maximum distance from center to any vertex
        max_dist = 0
        for vx, vy in all_vertices:
            dist = np.sqrt((vx - center_x)**2 + (vy - center_y)**2)
            max_dist = max(max_dist, dist)
            
        # Add safety margin
        return max_dist * 1.05

    def _is_valid_configuration(self, positions: np.ndarray, angles: np.ndarray,
                               outer_radius: float, outer_center=(0, 0)) -> bool:
        """Full validation of a configuration."""
        # Quick preliminary check using bounding circles
        if self._fast_collision_check(positions, angles):
            # Do detailed overlap checking only if necessary
            for i in range(len(positions)):
                hex1_vertices = self._hexagon_vertices(positions[i][0], positions[i][1], angles[i])
                if not self._check_containment_all_vertices(hex1_vertices, outer_center, outer_radius):
                    return False
                for j in range(i+1, len(positions)):
                    hex2_vertices = self._hexagon_vertices(positions[j][0], positions[j][1], angles[j])
                    if self._check_overlap_pair(hex1_vertices, hex2_vertices):
                        return False
        else:
            # If no collisions detected by fast method, check containment only
            for i in range(len(positions)):
                hex1_vertices = self._hexagon_vertices(positions[i][0], positions[i][1], angles[i])
                if not self._check_containment_all_vertices(hex1_vertices, outer_center, outer_radius):
                    return False
        
        return True

    def _generate_symmetric_initial_config(self) -> np.ndarray:
        """Generate a symmetric configuration based on mathematical insights."""
        positions_angles = np.zeros((self.n_hexagons, 3))

        # Central hexagon
        positions_angles[0] = [0, 0, 0]

        # First ring - 6 hexagons around center at radius 2.0
        ring1_radius = 2.0
        for i in range(1, 7):
            angle = 2 * np.pi * (i-1) / 6
            positions_angles[i] = [ring1_radius * np.cos(angle), ring1_radius * np.sin(angle), 0]

        # Second ring - 5 hexagons arranged optimally
        ring2_radius = 3.0
        for i in range(7, self.n_hexagons):
            # Distribute evenly around the ring
            angle = 2 * np.pi * (i-7) / 5 + np.pi/10  # offset to break symmetry
            positions_angles[i] = [ring2_radius * np.cos(angle), ring2_radius * np.sin(angle), 0]

        # Add slight randomness but maintain structure
        np.random.seed(42)
        for i in range(self.n_hexagons):
            positions_angles[i][0] += np.random.uniform(-0.2, 0.2)
            positions_angles[i][1] += np.random.uniform(-0.2, 0.2)
            positions_angles[i][2] += np.random.uniform(-5, 5)

        return positions_angles

    def _random_sample_config(self, search_bounds=None) -> np.ndarray:
        """Generate a random valid configuration."""
        if search_bounds is None:
            # Set reasonable bounds for the search space
            bounds = [(-10, 10), (-10, 10), (0, 360)] * self.n_hexagons
        else:
            bounds = search_bounds

        positions_angles = np.zeros((self.n_hexagons, 3))
        
        for i in range(self.n_hexagons):
            # Sample x coordinate
            positions_angles[i][0] = np.random.uniform(bounds[i*3][0], bounds[i*3][1])
            # Sample y coordinate  
            positions_angles[i][1] = np.random.uniform(bounds[i*3+1][0], bounds[i*3+1][1])
            # Sample rotation angle
            positions_angles[i][2] = np.random.uniform(bounds[i*3+2][0], bounds[i*3+2][1])
            
        return positions_angles

    def _refine_best_config(self, initial_config: np.ndarray, 
                          max_iter: int = 1000) -> np.ndarray:
        """Refine the best configuration using local search."""
        current_config = initial_config.copy()
        best_positions = current_config[:, :2].copy()
        best_angles = current_config[:, 2].copy()
        
        for iteration in range(max_iter):
            # Perturb one element at a time
            hex_idx = np.random.randint(0, self.n_hexagons)
            param_type = np.random.randint(0, 3)  # 0=x, 1=y, 2=angle
            
            # Store old values
            old_val = current_config[hex_idx, param_type]
            
            # Perturb
            if param_type < 2:  # x or y coordinate
                current_config[hex_idx, param_type] += np.random.normal(0, 0.1)
            else:  # angle
                current_config[hex_idx, param_type] += np.random.normal(0, 5)
            
            # Ensure angle stays in bounds
            current_config[hex_idx, 2] = current_config[hex_idx, 2] % 360
            
            # Test new configuration
            new_positions = current_config[:, :2]
            new_angles = current_config[:, 2]
            new_radius = self._estimate_outer_radius(new_positions, new_angles)
            
            if self._is_valid_configuration(new_positions, new_angles, new_radius):
                # Accept if improvement
                old_radius = self._estimate_outer_radius(best_positions, best_angles)
                if new_radius < old_radius:
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
            else:
                # Restore old value if invalid
                current_config[hex_idx, param_type] = old_val
                
        # Final update to best config
        refined_config = np.column_stack([best_positions, best_angles])
        return refined_config

    def _monte_carlo_optimization(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Perform Monte Carlo optimization with smart sampling."""
        best_positions = None
        best_angles = None
        best_radius = float('inf')
        best_score = 0.0
        
        # Generate initial configurations using multiple methods
        configs_to_try = []
        
        # Method 1: Symmetric initial configuration
        sym_config = self._generate_symmetric_initial_config()
        configs_to_try.append(sym_config)
        
        # Method 2: Random configurations
        for _ in range(20):
            rand_config = self._random_sample_config()
            configs_to_try.append(rand_config)
            
        # Try each initial configuration
        for config in configs_to_try:
            positions = config[:, :2]
            angles = config[:, 2]
            radius = self._estimate_outer_radius(positions, angles)
            
            if self._is_valid_configuration(positions, angles, radius):
                inv_radius = 1.0 / radius
                if inv_radius > best_score:
                    best_score = inv_radius
                    best_positions = positions.copy()
                    best_angles = angles.copy()
                    best_radius = radius
        
        # If we didn't find anything good, try full Monte Carlo
        if best_positions is None:
            # Sample randomly
            for iteration in range(self.max_iterations):
                config = self._random_sample_config()
                positions = config[:, :2]
                angles = config[:, 2]
                radius = self._estimate_outer_radius(positions, angles)
                
                if self._is_valid_configuration(positions, angles, radius):
                    inv_radius = 1.0 / radius
                    if inv_radius > best_score:
                        best_score = inv_radius
                        best_positions = positions.copy()
                        best_angles = angles.copy()
                        best_radius = radius
                        
        # Refine the best configuration found
        if best_positions is not None:
            refined_config = self._refine_best_config(
                np.column_stack([best_positions, best_angles]),
                max_iter=2000
            )
            best_positions = refined_config[:, :2]
            best_angles = refined_config[:, 2]
            best_radius = self._estimate_outer_radius(best_positions, best_angles)
            best_score = 1.0 / best_radius
            
        return best_positions, best_angles, best_radius

    def solve(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Main solving method that returns the optimized packing."""
        # Start with a good initial configuration
        positions, angles, outer_radius = self._monte_carlo_optimization()
        
        # If we still don't have a good result, fallback to baseline
        if positions is None or angles is None:
            # Fallback to a basic symmetric configuration
            positions = np.array([
                [0, 0], [0, 2], [0, -2], [1.732, 1], [-1.732, 1],
                [1.732, -1], [-1.732, -1], [3.464, 0], [-3.464, 0],
                [1.732, 3], [-1.732, 3], [1.732, -3]
            ])
            angles = np.zeros(12)
            outer_radius = 6.928  # Approximate
            
        # Format final output
        inner_hex_data = np.column_stack([positions, angles])
        outer_hex_data = np.array([0, 0, 0])  # Centered at origin
        
        return inner_hex_data, outer_hex_data, outer_radius

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Create packer instance and solve
    packer = HexagonPackerMonteCarlo(n_hexagons=12)
    inner_hex_data, outer_hex_data, outer_hex_side_length = packer.solve()
    
    end_time = time.time()
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END