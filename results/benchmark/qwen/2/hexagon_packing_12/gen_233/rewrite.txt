# EVOLVE-BLOCK-START
import numpy as np
from numba import jit
import time
import heapq
from collections import deque
import warnings

# Core geometric utilities
class HexagonGeometry:
    """Efficient geometric computations for hexagon operations."""
    
    def __init__(self):
        self.side_length = 1.0
        self.apothem = np.sqrt(3) / 2
        self.height = 2 * self.apothem
        self.width = 2 * self.side_length
    
    @staticmethod
    @jit(nopython=True)
    def vertices_jit(center_x: float, center_y: float, rotation_deg: float) -> np.ndarray:
        """JIT compiled hexagon vertex calculation."""
        angle_rad = np.radians(rotation_deg)
        # Unit hexagon vertices centered at origin
        base_vertices = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        
        # Rotate and translate
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_vertices = base_vertices @ rotation_matrix.T
        return rotated_vertices + np.array([center_x, center_y])
    
    def vertices(self, center_x: float, center_y: float, rotation_deg: float) -> np.ndarray:
        """Get hexagon vertices."""
        return self.vertices_jit(center_x, center_y, rotation_deg)

# Fast geometric operations
@jit(nopython=True)
def _get_edges(vertices: np.ndarray) -> np.ndarray:
    """Get edges from vertices."""
    edges = np.empty((len(vertices), 2))
    for i in range(len(vertices)):
        edges[i] = vertices[i] - vertices[(i+1) % len(vertices)]
    return edges

@jit(nopython=True)
def _project_polygon_onto_axis(vertices: np.ndarray, axis: np.ndarray) -> tuple:
    """Project polygon vertices onto an axis."""
    projections = np.empty(len(vertices))
    for i in range(len(vertices)):
        projections[i] = vertices[i, 0] * axis[0] + vertices[i, 1] * axis[1]
    return np.min(projections), np.max(projections)

@jit(nopython=True)
def _hexagon_overlap_sat_jit(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
    """Separating Axis Theorem for hexagon overlap detection."""
    # Get edges of both hexagons
    edges1 = _get_edges(hex1_vertices)
    edges2 = _get_edges(hex2_vertices)
    
    # Combine all axes (edges perpendicular to edges)
    all_axes = np.empty((len(edges1) + len(edges2), 2))
    for i in range(len(edges1)):
        # Normal vector to edge (perpendicular)
        all_axes[i] = np.array([-edges1[i, 1], edges1[i, 0]])
        # Normalize
        norm = np.sqrt(all_axes[i, 0]**2 + all_axes[i, 1]**2)
        if norm > 1e-10:
            all_axes[i] /= norm
    for i in range(len(edges2)):
        # Normal vector to edge (perpendicular)
        all_axes[len(edges1) + i] = np.array([-edges2[i, 1], edges2[i, 0]])
        # Normalize
        norm = np.sqrt(all_axes[len(edges1) + i, 0]**2 + all_axes[len(edges1) + i, 1]**2)
        if norm > 1e-10:
            all_axes[len(edges1) + i] /= norm
    
    # Check each axis
    for axis in all_axes:
        min1, max1 = _project_polygon_onto_axis(hex1_vertices, axis)
        min2, max2 = _project_polygon_onto_axis(hex2_vertices, axis)
        
        # If no overlap on this axis, polygons don't overlap
        if max1 < min2 or max2 < min1:
            return False
    
    return True

@jit(nopython=True)
def point_in_hexagon(x: float, y: float, center_x: float, center_y: float, 
                     side_length: float, rotation_deg: float) -> bool:
    """Fast point-in-hexagon test."""
    # Convert to hexagon's local coordinate system
    angle_rad = np.radians(rotation_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    local_x = (x - center_x) * cos_a + (y - center_y) * sin_a
    local_y = -(x - center_x) * sin_a + (y - center_y) * cos_a
    
    # Scale by hexagon side length
    local_x /= side_length
    local_y /= side_length
    
    # Check if within unit hexagon bounds
    # For unit hexagon, boundaries are: 
    # |x| <= 1, |y| <= sqrt(3)/2, and |x| + |y| <= sqrt(3)
    abs_x = abs(local_x)
    abs_y = abs(local_y)
    
    # More precise boundary check
    if abs_x > 1.0:
        return False
    if abs_y > np.sqrt(3) / 2:
        return False
    if abs_x + abs_y > np.sqrt(3):
        return False
    
    return True

class HexagonPacker:
    """Monte Carlo based hexagon packing solver."""
    
    def __init__(self):
        self.geo = HexagonGeometry()
        
    def is_contained(self, hex_vertices: np.ndarray, outer_center_x: float,
                     outer_center_y: float, outer_side_length: float) -> bool:
        """Check if all hexagon vertices are within outer hexagon."""
        # Simplified test - check if all vertices are within bounding circle
        dist_from_center = np.sqrt((hex_vertices[:, 0] - outer_center_x)**2 +
                                  (hex_vertices[:, 1] - outer_center_y)**2)
        # Maximum distance from center in a regular hexagon is side_length * sqrt(3)/2
        max_radius = outer_side_length * np.sqrt(3) / 2
        return np.all(dist_from_center <= max_radius)
    
    def has_overlap(self, hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Check if two hexagons overlap using SAT."""
        return _hexagon_overlap_sat_jit(hex1_vertices, hex2_vertices)
    
    def calculate_enclosing_hexagon(self, inner_hex_data: np.ndarray, 
                                   scale_factor: float = 1.0) -> tuple:
        """Calculate minimum enclosing hexagon side length."""
        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle = inner_hex_data[i]
            vertices = self.geo.vertices(center_x, center_y, angle)
            all_vertices.extend(vertices)

        all_vertices = np.array(all_vertices)

        # Find bounding circle radius
        centroid = np.mean(all_vertices, axis=0)
        distances = np.sqrt(np.sum((all_vertices - centroid)**2, axis=1))
        max_distance = np.max(distances)

        # For a regular hexagon, side length = max_distance * sqrt(3)/2
        side_length = max_distance * 2 / np.sqrt(3) * scale_factor
        return side_length, centroid
    
    def check_validity(self, hex_data: np.ndarray, outer_side_length: float) -> bool:
        """Check if the configuration meets all constraints."""
        # Check containment for all hexagons
        num_hex = len(hex_data)
        for i in range(num_hex):
            center_x, center_y, angle = hex_data[i]
            vertices = self.geo.vertices(center_x, center_y, angle)
            
            if not self.is_contained(vertices, 0, 0, outer_side_length):
                return False
        
        # Check overlaps between all pairs
        for i in range(num_hex):
            for j in range(i+1, num_hex):
                vertices1 = self.geo.vertices(*hex_data[i])
                vertices2 = self.geo.vertices(*hex_data[j])
                
                if self.has_overlap(vertices1, vertices2):
                    return False
        
        return True
    
    def generate_candidate_config(self, max_attempts: int = 1000) -> np.ndarray:
        """Generate a random valid configuration with some symmetry considerations."""
        # Try to place hexagons in a symmetric pattern
        for attempt in range(max_attempts):
            # Start with a symmetric base
            # We'll use a 6-fold symmetric pattern with additional randomness
            config = np.zeros((12, 3))  # (x, y, angle)
            
            # Base positions for symmetric 12-hex pattern
            base_positions = [
                [0, 0],           # Center
                [-2.0, 0],        # Left
                [2.0, 0],         # Right
                [-1.0, 1.73],     # Top-left
                [1.0, 1.73],      # Top-right
                [-1.0, -1.73],    # Bottom-left
                [1.0, -1.73],     # Bottom-right
                [-3.0, 1.73],     # Far top-left
                [3.0, 1.73],      # Far top-right
                [-3.0, -1.73],    # Far bottom-left
                [3.0, -1.73],     # Far bottom-right
                [0, -3.46],       # Far bottom
            ]
            
            # Add small random perturbations
            for i in range(12):
                config[i][0] = base_positions[i][0] + np.random.normal(0, 0.3)
                config[i][1] = base_positions[i][1] + np.random.normal(0, 0.3)
                config[i][2] = np.random.uniform(0, 360)  # Random rotation
            
            # Early validity check
            if self.check_validity(config, 10.0):
                return config
        
        # Return default configuration if nothing works
        config = np.array([
            [0, 0, 0],
            [-2.0, 0, 0],
            [2.0, 0, 0],
            [-1.0, 1.73, 0],
            [1.0, 1.73, 0],
            [-1.0, -1.73, 0],
            [1.0, -1.73, 0],
            [-3.0, 1.73, 0],
            [3.0, 1.73, 0],
            [-3.0, -1.73, 0],
            [3.0, -1.73, 0],
            [0, -3.46, 0],
        ])
        return config
    
    def refine_config(self, current_config: np.ndarray, max_iterations: int = 500) -> np.ndarray:
        """Refine configuration using local search."""
        current_score = -1.0 / self.calculate_enclosing_hexagon(current_config)[0]
        
        # Hill climbing with random perturbations
        for iteration in range(max_iterations):
            # Generate neighbor configuration
            neighbor = current_config.copy()
            
            # Perturb one hexagon slightly
            hex_idx = np.random.randint(0, 12)
            neighbor[hex_idx, 0] += np.random.normal(0, 0.1)
            neighbor[hex_idx, 1] += np.random.normal(0, 0.1)
            neighbor[hex_idx, 2] += np.random.normal(0, 5)
            
            # Check validity and accept if better
            if self.check_validity(neighbor, 10.0):
                new_score = -1.0 / self.calculate_enclosing_hexagon(neighbor)[0]
                if new_score > current_score:
                    current_config = neighbor
                    current_score = new_score
        
        return current_config

def hexagon_packing_12():
    """
    Constructs an optimized packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    try:
        packer = HexagonPacker()
        best_config = None
        best_side_length = float('inf')
        best_score = float('-inf')
        
        # Use Monte Carlo sampling with multiple trials
        num_trials = 500
        for trial in range(num_trials):
            # Generate a candidate configuration
            config = packer.generate_candidate_config(1000)
            
            # Refine the configuration locally
            refined_config = packer.refine_config(config, 100)
            
            # Check if this is valid and better
            if packer.check_validity(refined_config, 15.0):
                side_length, centroid = packer.calculate_enclosing_hexagon(refined_config, 1.05)
                score = 1.0 / side_length
                
                if score > best_score:
                    best_score = score
                    best_side_length = side_length
                    best_config = refined_config.copy()
        
        if best_config is None:
            # Fallback to basic symmetric arrangement
            best_config = np.array([
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
            side_length, _ = packer.calculate_enclosing_hexagon(best_config, 1.05)
            best_side_length = side_length
        
        # Final evaluation
        outer_hex_data = np.array([0, 0, 0])
        inv_outer_hex_side_length = 1.0 / best_side_length
        benchmark_ratio = inv_outer_hex_side_length / 0.2537
        
        # Output metrics for verification
        print(f"inv_outer_hex_side_length: {inv_outer_hex_side_length:.8f}")
        print(f"benchmark_ratio: {benchmark_ratio:.8f}")
        print(f"eval_time: {time.time() - start_time:.4f}s")
        
        return best_config, outer_hex_data, best_side_length
        
    except Exception as e:
        warnings.warn(f"Error in hexagon packing: {e}")
        # Fallback to simple symmetric arrangement
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
        outer_hex_side_length = 8.0
        
        # Calculate fallback metrics
        inv_outer_hex_side_length = 1.0 / outer_hex_side_length
        benchmark_ratio = inv_outer_hex_side_length / 0.2537
        print(f"Fallback - inv_outer_hex_side_length: {inv_outer_hex_side_length:.8f}")
        print(f"Fallback - benchmark_ratio: {benchmark_ratio:.8f}")
        print(f"eval_time: {time.time() - start_time:.4f}s")
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END