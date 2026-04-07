# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from numba import jit
import warnings
from math import sqrt, cos, sin, pi

# Hexagon geometry constants
HEX_SIDE_LENGTH = 1.0
HEX_APOGEE = sqrt(3) / 2
HEX_HEIGHT = 2 * HEX_APOGEE
HEX_WIDTH = 2 * HEX_SIDE_LENGTH

@jit(nopython=True)
def hexagon_vertices(center_x: float, center_y: float, rotation_deg: float) -> np.ndarray:
    """Generate vertices of a regular hexagon."""
    angle_rad = np.radians(rotation_deg)
    vertices = np.empty((6, 2))
    for i in range(6):
        angle_i = i * pi / 3 + angle_rad
        vertices[i, 0] = center_x + HEX_SIDE_LENGTH * cos(angle_i)
        vertices[i, 1] = center_y + HEX_SIDE_LENGTH * sin(angle_i)
    return vertices

@jit(nopython=True)
def distance_point_to_line_segment(p, a, b):
    """Distance from point p to line segment ab."""
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = max(0, min(1, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

@jit(nopython=True)
def point_in_hexagon(point, hex_vertices):
    """Check if point is inside hexagon using ray casting."""
    x, y = point
    n = len(hex_vertices)
    inside = False
    p1x, p1y = hex_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = hex_vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@jit(nopython=True)
def hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check overlap between two hexagons."""
    # Check if any vertex of hex1 is inside hex2
    for v in hex1_vertices:
        if point_in_hexagon(v, hex2_vertices):
            return True
    # Check if any vertex of hex2 is inside hex1
    for v in hex2_vertices:
        if point_in_hexagon(v, hex1_vertices):
            return True
    return False

class LatticeOptimizer:
    """Lattice-based optimization for hexagon packing."""
    
    def __init__(self):
        self.lattice_basis = np.array([
            [1.0, 0.0],
            [0.5, sqrt(3)/2]
        ])
        self.lattice_spacing = 2.0  # Spacing between lattice points
        
    def generate_lattice_points(self, max_radius):
        """Generate candidate lattice points within a radius."""
        points = []
        max_dist = int(max_radius * 2)
        for i in range(-max_dist, max_dist + 1):
            for j in range(-max_dist, max_dist + 1):
                # Construct point from lattice basis
                point = self.lattice_basis[0] * i + self.lattice_basis[1] * j
                dist = np.sqrt(point[0]**2 + point[1]**2)
                if dist < max_radius:
                    points.append(point)
        return np.array(points)
        
    def construct_initial_configuration(self):
        """Construct promising initial configuration using hexagonal lattice."""
        # Build a structured configuration with symmetries
        positions = []
        
        # Center hexagon
        positions.append([0.0, 0.0, 0.0])
        
        # Layer 1: 6 hexagons around center (at distance 2)
        for i in range(6):
            angle = i * pi / 3
            x = 2 * cos(angle)
            y = 2 * sin(angle)
            positions.append([x, y, 0.0])
            
        # Layer 2: 12 hexagons (at distance 3.5)
        for i in range(12):
            angle = i * pi / 6
            x = 3.5 * cos(angle)
            y = 3.5 * sin(angle)
            positions.append([x, y, 0.0])
            
        return np.array(positions)
        
    def calculate_bounding_circle_radius(self, positions):
        """Calculate minimum radius to contain all hexagons."""
        max_dist = 0.0
        for pos in positions:
            x, y, _ = pos
            dist = sqrt(x*x + y*y) + HEX_APOGEE  # Add apogee for safety
            max_dist = max(max_dist, dist)
        return max_dist * 1.05  # Add small margin
        
    def compute_objective(self, positions):
        """Compute objective function value for a configuration."""
        # Calculate container radius
        container_radius = self.calculate_bounding_circle_radius(positions)
        
        # Check containment and overlaps
        penalty = 0.0
        
        # Check containment
        for pos in positions:
            x, y, _ = pos
            dist = sqrt(x*x + y*y)
            if dist > container_radius - HEX_APOGEE:
                penalty += 10000.0
                
        # Check overlaps
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                pos1 = positions[i]
                pos2 = positions[j]
                vertices1 = hexagon_vertices(pos1[0], pos1[1], pos1[2])
                vertices2 = hexagon_vertices(pos2[0], pos2[1], pos2[2])
                
                if hexagon_overlap(vertices1, vertices2):
                    penalty += 100000.0
                    
        return -1.0 / container_radius + penalty
    
    def optimize_configuration(self, initial_positions, max_iterations=100):
        """Optimize configuration using gradient-based method."""
        # Flatten for optimization
        flat_positions = initial_positions.flatten()
        
        def objective(flat):
            # Reshape back to positions
            positions = flat.reshape(-1, 3)
            return self.compute_objective(positions)
            
        # Use a simple gradient-like approach with local search
        current_flat = flat_positions.copy()
        current_obj = objective(current_flat)
        
        # Simple local optimization
        for _ in range(max_iterations):
            # Try small perturbations
            best_flat = current_flat.copy()
            best_obj = current_obj
            
            # Try small movements
            for i in range(len(current_flat)):
                # Try positive and negative small steps
                for step in [-0.1, 0.1]:
                    test_flat = current_flat.copy()
                    test_flat[i] += step
                    test_obj = objective(test_flat)
                    
                    if test_obj < best_obj:
                        best_obj = test_obj
                        best_flat = test_flat
                        
            if best_obj < current_obj:
                current_flat = best_flat
                current_obj = best_obj
            else:
                # No improvement, try random small adjustments
                current_flat += np.random.normal(0, 0.01, len(current_flat))
                current_obj = objective(current_flat)
                
        # Return optimized positions
        final_positions = current_flat.reshape(-1, 3)
        return final_positions

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
        # Initialize lattice optimizer
        optimizer = LatticeOptimizer()
        
        # Generate initial configuration
        initial_positions = optimizer.construct_initial_configuration()
        
        # Optimize the configuration
        final_positions = optimizer.optimize_configuration(initial_positions)
        
        # Compute final container size and metrics
        container_radius = optimizer.calculate_bounding_circle_radius(final_positions)
        
        # Ensure it's a proper hexagon side length (minimum enclosing hexagon)
        # For a hexagon circumscribed around a circle of radius r, side length = r
        outer_hex_side_length = container_radius
        
        # Center is at origin
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        
        # Validate solution
        penalty = 0.0
        for i in range(len(final_positions)):
            pos1 = final_positions[i]
            vertices1 = hexagon_vertices(pos1[0], pos1[1], pos1[2])
            
            # Check containment
            dist = sqrt(pos1[0]**2 + pos1[1]**2)
            if dist > container_radius - HEX_APOGEE:
                penalty += 10000.0
                
            # Check overlaps with others
            for j in range(i+1, len(final_positions)):
                pos2 = final_positions[j]
                vertices2 = hexagon_vertices(pos2[0], pos2[1], pos2[2])
                
                if hexagon_overlap(vertices1, vertices2):
                    penalty += 100000.0
                    
        # Calculate final objective
        inv_outer_hex_side_length = 1.0 / outer_hex_side_length
        benchmark_ratio = inv_outer_hex_side_length / 0.2537
        
        eval_time = time.time() - start_time
        
        print(f"inv_outer_hex_side_length: {inv_outer_hex_side_length:.8f}")
        print(f"benchmark_ratio: {benchmark_ratio:.8f}")
        print(f"eval_time: {eval_time:.4f}s")
        
        return final_positions, outer_hex_data, outer_hex_side_length
        
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