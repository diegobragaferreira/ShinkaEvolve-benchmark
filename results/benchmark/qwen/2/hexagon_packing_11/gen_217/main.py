# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import time
from numba import jit
import random

# Constants
UNIT_HEX_RADIUS = 1.0  # Side length of unit hexagon
UNIT_HEX_APOGEE = np.sqrt(3)/2  # Distance from center to corner of unit hexagon

@jit(nopython=True)
def distance_point_to_line_segment(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment efficiently"""
    # Vector from (x1,y1) to (x2,y2)
    dx = x2 - x1
    dy = y2 - y1
    
    # Length squared of line segment
    len_sq = dx*dx + dy*dy
    
    # Avoid division by zero
    if len_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line segment
    t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / len_sq))
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def distance_hexagon_to_hexagon(h1_center_x, h1_center_y, h1_angle, 
                               h2_center_x, h2_center_y, h2_angle):
    """Fast distance calculation between two hexagons (simplified)"""
    # Approximate using distance between centers minus apogees
    dx = h1_center_x - h2_center_x
    dy = h1_center_y - h2_center_y
    center_distance = np.sqrt(dx*dx + dy*dy)
    return center_distance - 2.0 * UNIT_HEX_APOGEE

@jit(nopython=True)
def point_in_hexagon_fast(px, py, hex_center_x, hex_center_y, hex_angle):
    """Fast point-in-hexagon test using dot products"""
    # Convert to hexagon coordinate system
    cos_a = np.cos(hex_angle)
    sin_a = np.sin(hex_angle)
    rel_x = px - hex_center_x
    rel_y = py - hex_center_y
    rot_x = rel_x * cos_a + rel_y * sin_a
    rot_y = -rel_x * sin_a + rel_y * cos_a
    
    # Check if point is inside hexagon (simplified)
    # We only check if it's within the bounds of the hexagon
    # Since we're using fixed rotation, we can do approximate check
    return abs(rot_x) < 1.5 and abs(rot_y) < 1.5

def create_unit_hexagon(center=(0,0), rotation=0):
    """Create a unit regular hexagon as a Shapely Polygon"""
    angle_offset = np.deg2rad(rotation)
    points = []
    for i in range(6):
        angle = angle_offset + i * np.pi/3
        x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
        y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
        points.append((x, y))
    return Polygon(points)

def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer_hexagon"""
    # Check if all vertices of inner hexagon are inside outer hexagon
    for point in hexagon.exterior.coords[:-1]:
        if not outer_hexagon.contains(Point(point)):
            return False
    return True

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    return hex1.intersects(hex2)

class HexagonMonteCarloPacker:
    def __init__(self):
        self.n_inner = 11
        self.unit_hex_radius = UNIT_HEX_RADIUS
        self.unit_hex_apogee = UNIT_HEX_APOGEE
        self.max_iterations = 50000
        self.temperature_start = 1.0
        self.temperature_end = 0.001
        self.cooling_rate = 0.9999
        self.acceptance_threshold = 0.8
        
    def create_outer_hexagon(self, outer_radius):
        """Create outer hexagon of specified radius"""
        return create_unit_hexagon((0, 0), 0)
    
    def calculate_energy(self, positions, angles, outer_radius):
        """Calculate total energy of the system"""
        # Create hexagons
        hexagons = []
        for i in range(self.n_inner):
            hexagon = create_unit_hexagon((positions[i][0], positions[i][1]), angles[i])
            hexagons.append(hexagon)
        
        # Create outer hexagon
        outer_hexagon = self.create_outer_hexagon(outer_radius)
        
        # Energy components
        energy = 0.0
        
        # Containment energy - penalty for any vertex outside outer hexagon
        for i, hexagon in enumerate(hexagons):
            for point in hexagon.exterior.coords[:-1]:
                if not outer_hexagon.contains(Point(point)):
                    # Calculate how far outside
                    dist = Point(point).distance(Point(0, 0))
                    energy += 1000.0 * (dist - outer_radius)**2
        
        # Overlap energy - penalty for overlaps between hexagons
        for i in range(self.n_inner):
            for j in range(i+1, self.n_inner):
                if check_overlap(hexagons[i], hexagons[j]):
                    # Simple penalty for overlap
                    energy += 10000.0
        
        # Packing density energy - encourage compact packing
        # Calculate average distance between centers
        if self.n_inner > 1:
            total_dist = 0.0
            count = 0
            for i in range(self.n_inner):
                for j in range(i+1, self.n_inner):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    total_dist += np.sqrt(dx*dx + dy*dy)
                    count += 1
            if count > 0:
                avg_dist = total_dist / count
                # Reward configurations with appropriate spacing
                if avg_dist < 1.5:  # Too close
                    energy += 100.0 * (1.5 - avg_dist)**2
                elif avg_dist > 3.0:  # Too far apart
                    energy += 100.0 * (avg_dist - 3.0)**2
        
        return energy
    
    def calculate_tight_outer_radius(self, positions, angles):
        """Calculate tightest possible outer hexagon radius using actual vertex positions"""
        # Get all hexagon vertices and find bounding circle
        all_vertices = []

        for i in range(self.n_inner):
            hexagon = create_unit_hexagon((positions[i][0], positions[i][1]), angles[i])
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
    
    def generate_initial_config(self):
        """Generate a good initial configuration"""
        # Start with a honeycomb-like pattern
        positions = []
        angles = []
        
        # Center hexagon
        positions.append((0.0, 0.0))
        angles.append(0.0)
        
        # Surrounding hexagons in a pattern that works well for 11 hexagons
        # Based on known good configurations
        surrounding_positions = [
            (-1.8, 0.0),
            (1.8, 0.0),
            (0.0, 1.8),
            (0.0, -1.8),
            (-1.3, 1.3),
            (1.3, 1.3),
            (-1.3, -1.3),
            (1.3, -1.3),
            (-2.2, 0.0),
            (2.2, 0.0),
        ]
        
        for i, pos in enumerate(surrounding_positions):
            positions.append(pos)
            angles.append(0.0)
        
        # Add some small randomness
        for i in range(len(positions)):
            positions[i] = (
                positions[i][0] + np.random.uniform(-0.1, 0.1),
                positions[i][1] + np.random.uniform(-0.1, 0.1)
            )
        
        # Random angles for variety
        for i in range(len(angles)):
            angles[i] = np.random.uniform(0, 360)
        
        return positions, angles
    
    def propose_move(self, positions, angles, temperature):
        """Propose a new configuration by perturbing one hexagon"""
        # Choose which hexagon to move
        idx = random.randint(0, self.n_inner - 1)
        
        # Create new positions and angles
        new_positions = [pos[:] for pos in positions]  # shallow copy
        new_angles = angles[:]
        
        # Perturb position slightly
        new_positions[idx] = (
            new_positions[idx][0] + np.random.uniform(-0.2, 0.2),
            new_positions[idx][1] + np.random.uniform(-0.2, 0.2)
        )
        
        # Perturb angle slightly
        new_angles[idx] = (new_angles[idx] + np.random.uniform(-30, 30)) % 360
        
        return new_positions, new_angles
    
    def monte_carlo_optimize(self, max_iter=10000):
        """Perform Monte Carlo optimization"""
        # Generate initial configuration
        positions, angles = self.generate_initial_config()
        
        # Estimate initial outer radius
        initial_radius = self.calculate_tight_outer_radius(positions, angles)
        outer_radius = initial_radius
        
        # Calculate initial energy
        current_energy = self.calculate_energy(positions, angles, outer_radius)
        best_energy = current_energy
        best_positions = [pos[:] for pos in positions]
        best_angles = angles[:]
        best_radius = outer_radius
        
        # Temperature schedule
        temperature = self.temperature_start
        
        # Track acceptance rate
        accepted_moves = 0
        total_moves = 0
        
        for iteration in range(max_iter):
            # Propose new configuration
            new_positions, new_angles = self.propose_move(positions, angles, temperature)
            
            # Calculate new energy
            new_energy = self.calculate_energy(new_positions, new_angles, outer_radius)
            
            # Metropolis criterion
            delta_energy = new_energy - current_energy
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temperature):
                positions = new_positions
                angles = new_angles
                current_energy = new_energy
                accepted_moves += 1
                
                # Update best solution if improved
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_positions = [pos[:] for pos in positions]
                    best_angles = angles[:]
                    best_radius = self.calculate_tight_outer_radius(best_positions, best_angles)
            
            total_moves += 1
            
            # Cool down temperature
            if iteration % 100 == 0:
                temperature *= self.cooling_rate
                
                # Reset temperature if cooling too fast
                if temperature < self.temperature_end:
                    temperature = self.temperature_end
                
                # Adaptive cooling based on acceptance rate
                if total_moves > 0:
                    acceptance_rate = accepted_moves / total_moves
                    if acceptance_rate < 0.2 and temperature > 0.01:
                        temperature *= 0.9
                    elif acceptance_rate > 0.8 and temperature < 1.0:
                        temperature *= 1.1
                        
                    accepted_moves = 0
                    total_moves = 0
        
        return best_positions, best_angles, best_radius
    
    def optimize_solution(self):
        """Main optimization routine"""
        best_positions, best_angles, best_radius = self.monte_carlo_optimize()
        
        # Refine using local search with gradient-free method
        # Try several random restarts to avoid local optima
        for _ in range(5):
            pos, ang, rad = self.monte_carlo_optimize()
            if rad < best_radius:
                best_positions = pos
                best_angles = ang
                best_radius = rad
        
        return best_positions, best_angles, best_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses Monte Carlo optimization to find the best arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        # Initialize the packer
        packer = HexagonMonteCarloPacker()
        
        # Run optimization
        positions, angles, outer_radius = packer.optimize_solution()
        
        # Validate solution
        # Create hexagons
        hexagons = []
        for i in range(11):
            hexagon = create_unit_hexagon((positions[i][0], positions[i][1]), angles[i])
            hexagons.append(hexagon)
        
        # Create outer hexagon
        outer_hexagon = packer.create_outer_hexagon(outer_radius)
        
        # Check constraints
        containment_ok = True
        overlap_ok = True
        
        for hexagon in hexagons:
            if not check_containment(hexagon, outer_hexagon):
                containment_ok = False
                break
        
        if containment_ok:
            for i in range(11):
                for j in range(i+1, 11):
                    if check_overlap(hexagons[i], hexagons[j]):
                        overlap_ok = False
                        break
                if not overlap_ok:
                    break
        
        if containment_ok and overlap_ok:
            # Format output
            inner_hex_data = np.zeros((11, 3))
            for i in range(11):
                inner_hex_data[i] = [positions[i][0], positions[i][1], angles[i]]
            
            outer_hex_data = np.array([0, 0, 0])
            
            return inner_hex_data, outer_hex_data, outer_radius
            
    except Exception as e:
        warnings.warn(f"Error in Monte Carlo optimization: {str(e)}")
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