# EVOLVE-BLOCK-START
import numpy as np
import math
from shapely.geometry import Polygon, Point
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import time

class HexagonPackingGravity:
    def __init__(self):
        self.side_length = 1.0
        self.num_hexagons = 11
        
    def create_hexagon_vertices(self, center_x, center_y, angle_deg):
        """Create vertices of a regular hexagon given center, rotation, and side length"""
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            angle = angle_rad + i * math.pi / 3
            x = center_x + self.side_length * math.cos(angle)
            y = center_y + self.side_length * math.sin(angle)
            vertices.append((x, y))
        return vertices
    
    def check_hexagon_containment(self, hex_vertices, outer_hex_vertices):
        """Check if all vertices of inner hexagon are within outer hexagon with buffer"""
        outer_polygon = Polygon(outer_hex_vertices)
        buffered_outer = outer_polygon.buffer(1e-6)
        for vertex in hex_vertices:
            if not buffered_outer.contains(Point(vertex[0], vertex[1])):
                return False
        return True
    
    def check_hexagon_overlap(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons overlap using Shapely with buffer"""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        buffered_poly1 = poly1.buffer(1e-6)
        buffered_poly2 = poly2.buffer(1e-6)
        return buffered_poly1.intersects(buffered_poly2)
    
    def calculate_outer_hex_side_length(self, inner_hex_data):
        """Calculate minimum outer hexagon side length that contains all inner hexagons"""
        if len(inner_hex_data) == 0:
            return 1000

        # Get all vertices of inner hexagons
        all_vertices = []
        for center_x, center_y, angle in inner_hex_data:
            vertices = self.create_hexagon_vertices(center_x, center_y, angle)
            all_vertices.extend(vertices)

        if not all_vertices:
            return 1000

        # Calculate tight bounding box
        xs = [v[0] for v in all_vertices]
        ys = [v[1] for v in all_vertices]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Calculate the diagonal of the bounding box
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        diagonal = math.sqrt(bbox_width**2 + bbox_height**2)
        
        # For a hexagon, the relationship is simpler
        # The outer hexagon needs to contain all points, so we use a conservative estimate
        side_length = diagonal / math.sqrt(3)
        side_length *= 1.1  # Add margin for safety
        
        return side_length
    
    def get_hexagon_centroid(self, vertices):
        """Calculate centroid of a hexagon"""
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        return (sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords))
    
    def calculate_force_repulsion(self, hex1_vertices, hex2_vertices, force_mag=10.0):
        """Calculate repulsive force between two hexagons when they overlap"""
        # This is a simplified repulsion model that pushes overlapping hexagons apart
        if not self.check_hexagon_overlap(hex1_vertices, hex2_vertices):
            return np.array([0.0, 0.0])
        
        # Get centroids
        cent1 = np.array(self.get_hexagon_centroid(hex1_vertices))
        cent2 = np.array(self.get_hexagon_centroid(hex2_vertices))
        
        # Vector from hex1 to hex2
        vec = cent2 - cent1
        dist = np.linalg.norm(vec)
        
        if dist < 1e-6:
            return np.array([0.0, 0.0])
        
        # Normalize direction vector
        dir_vec = vec / dist
        
        # Force magnitude inversely proportional to distance
        # When very close, force is large to push them apart
        force = force_mag / (dist + 0.1)
        
        return force * dir_vec
    
    def calculate_force_boundary(self, hex_vertices, outer_radius, force_mag=5.0):
        """Calculate attraction force from hexagon to outer boundary"""
        # Get centroid of the hexagon
        centroid = np.array(self.get_hexagon_centroid(hex_vertices))
        dist_from_center = np.linalg.norm(centroid)
        
        # If already within boundary, no force needed
        if dist_from_center < outer_radius * 0.95:
            return np.array([0.0, 0.0])
        
        # Direction toward center
        if dist_from_center > 1e-6:
            dir_vec = -centroid / dist_from_center
        else:
            dir_vec = np.array([0.0, 0.0])
        
        # Force magnitude based on how far from boundary
        excess = dist_from_center - outer_radius
        force = force_mag * max(0, excess) / (outer_radius + 1.0)
        
        return force * dir_vec
    
    def calculate_total_energy(self, hex_data, outer_radius):
        """Calculate total energy of the system"""
        total_energy = 0.0
        
        # Repulsion energy between overlapping hexagons
        for i in range(len(hex_data)):
            for j in range(i+1, len(hex_data)):
                center_x1, center_y1, angle1 = hex_data[i]
                center_x2, center_y2, angle2 = hex_data[j]
                
                hex1_vertices = self.create_hexagon_vertices(center_x1, center_y1, angle1)
                hex2_vertices = self.create_hexagon_vertices(center_x2, center_y2, angle2)
                
                if self.check_hexagon_overlap(hex1_vertices, hex2_vertices):
                    # Simplified energy proportional to overlap area (approximated)
                    total_energy += 1000.0
        
        return total_energy
    
    def simulate_physics_step(self, hex_data, outer_radius, steps=10):
        """Simulate physics step to improve hexagon arrangement"""
        # Create a copy to work with
        new_hex_data = [list(row) for row in hex_data]
        
        # Apply forces to each hexagon
        forces = []
        for i in range(len(new_hex_data)):
            center_x, center_y, angle = new_hex_data[i]
            hex_vertices = self.create_hexagon_vertices(center_x, center_y, angle)
            
            total_force = np.array([0.0, 0.0])
            
            # Force from boundary
            boundary_force = self.calculate_force_boundary(hex_vertices, outer_radius)
            total_force += boundary_force
            
            # Forces from other hexagons
            for j in range(len(new_hex_data)):
                if i != j:
                    center_x2, center_y2, angle2 = new_hex_data[j]
                    hex2_vertices = self.create_hexagon_vertices(center_x2, center_y2, angle2)
                    
                    repulsion = self.calculate_force_repulsion(hex_vertices, hex2_vertices)
                    total_force += repulsion
            
            forces.append(total_force)
        
        # Update positions (simple integration)
        for i in range(len(new_hex_data)):
            center_x, center_y, angle = new_hex_data[i]
            force = forces[i]
            
            # Update position based on force
            new_x = center_x + force[0] * 0.1
            new_y = center_y + force[1] * 0.1
            
            # Keep within reasonable bounds
            new_x = np.clip(new_x, -10, 10)
            new_y = np.clip(new_y, -10, 10)
            
            new_hex_data[i] = [new_x, new_y, angle]
        
        return np.array(new_hex_data)
    
    def optimize_with_newton(self, initial_hex_data, outer_radius):
        """Use Newton-based optimization for fine-tuning"""
        def objective(params):
            # Reshape parameters
            hex_data = params.reshape(-1, 3)
            
            # Calculate outer hexagon side length
            calculated_side_length = self.calculate_outer_hex_side_length(hex_data)
            
            # Check constraints and penalties
            outer_hex_vertices = self.create_hexagon_vertices(0, 0, 0, calculated_side_length)
            
            # Check containment
            total_penalty = 0
            for i in range(len(hex_data)):
                center_x, center_y, angle = hex_data[i]
                inner_hex_vertices = self.create_hexagon_vertices(center_x, center_y, angle)
                if not self.check_hexagon_containment(inner_hex_vertices, outer_hex_vertices):
                    total_penalty += 10000
            
            # Check overlaps
            for i in range(len(hex_data)):
                for j in range(i+1, len(hex_data)):
                    center_x1, center_y1, angle1 = hex_data[i]
                    center_x2, center_y2, angle2 = hex_data[j]
                    
                    hex1_vertices = self.create_hexagon_vertices(center_x1, center_y1, angle1)
                    hex2_vertices = self.create_hexagon_vertices(center_x2, center_y2, angle2)
                    
                    if self.check_hexagon_overlap(hex1_vertices, hex2_vertices):
                        total_penalty += 10000
            
            # Return fitness (inverse of outer hex side length + penalties)
            objective_value = 1.0 / calculated_side_length
            if total_penalty > 0:
                objective_value -= total_penalty
                
            return -objective_value  # Negative because we minimize
        
        def gradient(params):
            # Simple finite difference gradient approximation
            eps = 1e-6
            grad = np.zeros_like(params)
            
            for i in range(len(params)):
                params_plus = params.copy()
                params_minus = params.copy()
                params_plus[i] += eps
                params_minus[i] -= eps
                
                grad[i] = (objective(params_plus) - objective(params_minus)) / (2 * eps)
            
            return grad
        
        # Use scipy's minimize with BFGS method for optimization
        try:
            result = minimize(objective, initial_hex_data.flatten(), method='BFGS', 
                            jac=gradient, options={'maxiter': 50})
            return result.x.reshape(-1, 3)
        except:
            return initial_hex_data
    
    def generate_initial_configuration(self):
        """Generate an initial configuration with some heuristic"""
        # Start with a grid-like pattern with some randomness
        config = []
        # Center hexagon
        config.append([0.0, 0.0, 0.0])
        
        # Surrounding hexagons in a ring pattern
        angles = [0, 60, 120, 180, 240, 300]
        radius = 2.0
        
        for i, angle in enumerate(angles):
            rad = math.radians(angle)
            x = radius * math.cos(rad)
            y = radius * math.sin(rad)
            config.append([x, y, 0.0])
        
        # Additional hexagons
        config.append([-3.0, 0.0, 0.0])
        config.append([3.0, 0.0, 0.0])
        config.append([0.0, 3.0, 0.0])
        config.append([0.0, -3.0, 0.0])
        config.append([1.5, 2.6, 0.0])
        config.append([-1.5, 2.6, 0.0])
        config.append([1.5, -2.6, 0.0])
        config.append([-1.5, -2.6, 0.0])
        
        # Ensure we have exactly 11
        config = config[:11]
        
        # Add small random noise
        for i in range(len(config)):
            config[i][0] += random.uniform(-0.3, 0.3)
            config[i][1] += random.uniform(-0.3, 0.3)
            config[i][2] += random.uniform(-10, 10)
        
        return np.array(config)
    
    def optimize(self):
        """Main optimization routine"""
        # Generate initial configuration
        hex_data = self.generate_initial_configuration()
        
        # Iteratively improve using physics simulation and optimization
        best_fitness = -float('inf')
        best_hex_data = hex_data.copy()
        best_side_length = float('inf')
        
        # Simulate physics for several iterations
        for iteration in range(50):
            # Simulate physics
            hex_data = self.simulate_physics_step(hex_data, 100, steps=20)
            
            # Occasionally refine with optimization
            if iteration % 10 == 0:
                # Find outer hexagon size
                outer_side_length = self.calculate_outer_hex_side_length(hex_data)
                
                # Local optimization
                refined_hex_data = self.optimize_with_newton(hex_data, outer_side_length)
                refined_side_length = self.calculate_outer_hex_side_length(refined_hex_data)
                
                # Evaluate fitness
                fitness = 1.0 / refined_side_length
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_hex_data = refined_hex_data.copy()
                    best_side_length = refined_side_length
                    
                # Update hex_data for next iteration
                hex_data = refined_hex_data
            
            # Check if we're converging well
            if iteration > 20 and abs(best_fitness) > 0.2:
                # If we've been good for a while, let's add more refinement
                outer_side_length = self.calculate_outer_hex_side_length(hex_data)
                refined_hex_data = self.optimize_with_newton(hex_data, outer_side_length)
                refined_side_length = self.calculate_outer_hex_side_length(refined_hex_data)
                fitness = 1.0 / refined_side_length
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_hex_data = refined_hex_data.copy()
                    best_side_length = refined_side_length
        
        return best_hex_data, best_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    optimizer = HexagonPackingGravity()
    
    try:
        best_hex_data, best_side_length = optimizer.optimize()
        
        # Validate the result
        outer_hex_vertices = optimizer.create_hexagon_vertices(0, 0, 0, best_side_length)
        
        # Check containment
        for i in range(len(best_hex_data)):
            center_x, center_y, angle = best_hex_data[i]
            inner_hex_vertices = optimizer.create_hexagon_vertices(center_x, center_y, angle)
            if not optimizer.check_hexagon_containment(inner_hex_vertices, outer_hex_vertices):
                raise Exception("Containment check failed")
        
        # Check overlaps
        for i in range(len(best_hex_data)):
            for j in range(i+1, len(best_hex_data)):
                center_x1, center_y1, angle1 = best_hex_data[i]
                center_x2, center_y2, angle2 = best_hex_data[j]
                
                hex1_vertices = optimizer.create_hexagon_vertices(center_x1, center_y1, angle1)
                hex2_vertices = optimizer.create_hexagon_vertices(center_x2, center_y2, angle2)
                
                if optimizer.check_hexagon_overlap(hex1_vertices, hex2_vertices):
                    raise Exception("Overlap check failed")
        
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        return best_hex_data, outer_hex_data, best_side_length
        
    except Exception as e:
        # Fallback to simple grid if optimization fails
        print(f"Optimization failed: {e}")
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
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END