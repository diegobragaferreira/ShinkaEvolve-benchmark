# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
import time
import math
import random

class HexagonPacker:
    """Main class managing hexagon packing operations"""

    def __init__(self):
        self.hex_side_length = 1.0
        self.outer_center = np.array([0.0, 0.0])

    def hexagon_vertices(self, center_x, center_y, size=1, angle_deg=0):
        """Generate vertices of a regular hexagon given center, size, and rotation."""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            angle = angle_rad + i * np.pi / 3
            x = center_x + size * np.cos(angle)
            y = center_y + size * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)

    def get_outer_hexagon(self, outer_radius):
        """Get vertices of the outer hexagon with given radius."""
        return self.hexagon_vertices(self.outer_center[0], self.outer_center[1], outer_radius, 0)

    def validate_containment(self, hex_vertices, outer_radius):
        """Check if all vertices of a hexagon are inside the outer hexagon."""
        outer_vertices = self.get_outer_hexagon(outer_radius)
        outer_polygon = Polygon(outer_vertices)

        for vertex in hex_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False
        return True

    def validate_overlap(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons overlap using Shapely."""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)

    def calculate_max_distance_from_center(self, hex_data):
        """Calculate maximum distance from center to any hexagon vertex."""
        max_dist = 0
        for i in range(len(hex_data)):
            cx, cy, _ = hex_data[i]
            # Calculate distance to center plus hexagon radius
            dist = np.sqrt(cx**2 + cy**2) + self.hex_side_length
            max_dist = max(max_dist, dist)
        return max_dist

    def evaluate_configuration(self, hex_data, outer_radius, debug=False):
        """Evaluate current configuration: returns (validity, inv_radius)."""
        # Check for overlaps
        for i in range(len(hex_data)):
            hex1_vertices = self.hexagon_vertices(hex_data[i][0], hex_data[i][1],
                                                self.hex_side_length, hex_data[i][2])
            for j in range(i+1, len(hex_data)):
                hex2_vertices = self.hexagon_vertices(hex_data[j][0], hex_data[j][1],
                                                    self.hex_side_length, hex_data[j][2])
                if self.validate_overlap(hex1_vertices, hex2_vertices):
                    if debug:
                        print(f"Overlap detected between hexagons {i} and {j}")
                    return False, 0

        # Check containment
        for i in range(len(hex_data)):
            hex_vertices = self.hexagon_vertices(hex_data[i][0], hex_data[i][1],
                                               self.hex_side_length, hex_data[i][2])
            if not self.validate_containment(hex_vertices, outer_radius):
                if debug:
                    print(f"Containment failure for hexagon {i}")
                return False, 0

        # Return inverse of outer radius
        return True, 1.0 / outer_radius

class SymmetricInitializer:
    """Handles generation of symmetric initial configurations"""

    @staticmethod
    def create_hexagonal_tiling():
        """Create initial configuration using hexagonal tiling principle."""
        config = []

        # Central hexagon
        config.append([0, 0, 0])

        # First ring (6 hexagons)
        for i in range(6):
            angle = i * 60
            radius = 2.0  # Distance from origin
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            config.append([x, y, 0])

        # Second ring (6 hexagons) - arranged to fit optimally
        for i in range(6):
            angle = 30 + i * 60
            radius = 3.464  # sqrt(12) approximately
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            config.append([x, y, 0])

        return np.array(config)

    @staticmethod
    def create_better_initial_config():
        """Create a more refined initial configuration based on known good solutions."""
        # Based on previous work, this configuration should be close to optimal
        config = np.array([
            [0, 0, 0],           # center
            [-2.0, 0, 0],        # left
            [2.0, 0, 0],         # right
            [-1.0, 1.732, 0],    # top-left
            [1.0, 1.732, 0],     # top-right
            [-1.0, -1.732, 0],   # bottom-left
            [1.0, -1.732, 0],    # bottom-right
            [-3.0, 1.732, 0],    # far top-left
            [3.0, 1.732, 0],     # far top-right
            [-3.0, -1.732, 0],   # far bottom-left
            [3.0, -1.732, 0],    # far bottom-right
            [0, -3.464, 0]       # far bottom-center
        ])
        return config

    @staticmethod
    def create_kagome_lattice():
        """Create a configuration inspired by Kagome lattice structure."""
        # This creates a 2D hexagonal lattice arrangement
        config = [
            [0, 0, 0],           # center
            [-2.0, 0, 0],        # left
            [2.0, 0, 0],         # right
            [0, 2.0, 0],         # top
            [0, -2.0, 0],        # bottom
            [-1.0, 1.0, 0],      # top-left
            [1.0, 1.0, 0],       # top-right
            [-1.0, -1.0, 0],     # bottom-left
            [1.0, -1.0, 0],      # bottom-right
            [-2.0, 2.0, 0],      # top-left far
            [2.0, 2.0, 0],       # top-right far
            [-2.0, -2.0, 0],     # bottom-left far
            [2.0, -2.0, 0],      # bottom-right far
        ]

        # Adjust for the actual 12 hexagons needed
        return np.array(config[:12])

    @staticmethod
    def create_hcp_lattice():
        """Create a configuration inspired by Hexagonal Close-Packed structure."""
        # Create a layered HCP-like arrangement
        config = [
            [0, 0, 0],           # center
            [-1.732, 1.0, 0],    # top-left
            [1.732, 1.0, 0],     # top-right
            [0, -2.0, 0],        # bottom-center
            [-1.732, -1.0, 0],   # bottom-left
            [1.732, -1.0, 0],    # bottom-right
            [3.464, 0, 0],       # right far
            [-3.464, 0, 0],      # left far
            [0, 3.0, 0],         # top far
            [0, -3.0, 0],        # bottom far
            [1.732, 2.0, 0],     # top-right corner
            [-1.732, 2.0, 0],    # top-left corner
        ]

        # Adjust for the actual 12 hexagons needed
        return np.array(config[:12])

class Optimizer:
    """Handles optimization of hexagon positions and rotations"""

    def __init__(self, packer):
        self.packer = packer
        self.temperature_start = 0.8
        self.temperature_end = 0.01
        self.cooling_rate = 0.95

    def flatten_config(self, hex_data):
        """Convert 2D array to flat parameter list for optimization."""
        params = []
        for i in range(len(hex_data)):
            params.extend([hex_data[i][0], hex_data[i][1], hex_data[i][2]])  # positions + angles
        return np.array(params)

    def unflatten_config(self, params, original_config):
        """Reconstruct hex_data from flattened parameters."""
        config = original_config.copy()
        idx = 0
        for i in range(len(config)):
            config[i][0] = params[idx]
            config[i][1] = params[idx + 1]
            config[i][2] = params[idx + 2]
            idx += 3
        return config

    def objective_function(self, params, hex_data, outer_radius):
        """Objective function for optimization."""
        # Reconstruct configuration
        reconstructed_config = self.unflatten_config(params, hex_data)

        # Evaluate
        validity, inv_radius = self.packer.evaluate_configuration(reconstructed_config, outer_radius)

        if not validity:
            return 1e10  # Large penalty for invalid configurations
        return -inv_radius  # Negative because we maximize

    def simulated_annealing(self, initial_config, outer_radius, max_iterations=100):
        """Apply simulated annealing to escape local optima."""
        current_config = initial_config.copy()
        current_params = self.flatten_config(current_config)

        # Evaluate current solution
        _, current_score = self.packer.evaluate_configuration(current_config, outer_radius)

        best_config = current_config.copy()
        best_score = current_score

        # Temperature scheduling
        temperature = self.temperature_start

        for iteration in range(max_iterations):
            # Generate neighbor solution
            neighbor_params = current_params.copy()
            # Randomly perturb positions and angles
            for i in range(len(neighbor_params)):
                if i % 3 == 0 or i % 3 == 1:  # x, y coordinates
                    neighbor_params[i] += np.random.normal(0, 0.1)
                else:  # angle
                    neighbor_params[i] += np.random.normal(0, 5)
                    neighbor_params[i] = neighbor_params[i] % 360

            # Convert back to configuration
            neighbor_config = self.unflatten_config(neighbor_params, initial_config)

            # Evaluate neighbor
            validity, neighbor_score = self.packer.evaluate_configuration(neighbor_config, outer_radius)

            if validity:
                # Accept or reject based on simulated annealing criteria
                if neighbor_score > current_score:
                    current_config = neighbor_config.copy()
                    current_params = neighbor_params.copy()
                    current_score = neighbor_score
                else:
                    # Accept with probability based on temperature
                    delta = neighbor_score - current_score
                    if np.random.random() < np.exp(delta / temperature):
                        current_config = neighbor_config.copy()
                        current_params = neighbor_params.copy()
                        current_score = neighbor_score

                # Update best solution
                if current_score > best_score:
                    best_score = current_score
                    best_config = current_config.copy()

            # Cool down temperature
            temperature *= self.cooling_rate

        return best_config, best_score

    def optimize_positions_and_angles(self, initial_config, outer_radius):
        """Optimize positions and angles using constrained numerical optimization."""
        # First perform standard optimization
        # Flatten initial configuration
        initial_params = self.flatten_config(initial_config)

        # Define bounds for optimization: positions [-10,10], angles [0,360]
        bounds = []
        for i in range(len(initial_config)):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle

        # Perform optimization
        result = minimize(
            self.objective_function,
            initial_params,
            args=(initial_config, outer_radius),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500}
        )

        # Reconstruct optimized configuration
        optimized_config = self.unflatten_config(result.x, initial_config)

        # Apply simulated annealing for further improvement
        final_config, _ = self.simulated_annealing(optimized_config, outer_radius)

        return final_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize components
    packer = HexagonPacker()
    initializer = SymmetricInitializer()
    optimizer = Optimizer(packer)

    # Try different initial configurations to find a good starting point
    configs_to_try = [
        initializer.create_hexagonal_tiling(),
        initializer.create_better_initial_config(),
        initializer.create_kagome_lattice(),
        initializer.create_hcp_lattice()
    ]

    best_config = None
    best_inv_radius = 0
    best_outer_radius = float('inf')

    for initial_config in configs_to_try:
        # Set outer hexagon at center
        outer_center_x, outer_center_y = 0.0, 0.0
        packer.outer_center = np.array([outer_center_x, outer_center_y])

        # Estimate outer radius
        estimated_outer_radius = packer.calculate_max_distance_from_center(initial_config)

        # Optimize positions and angles
        optimized_config = optimizer.optimize_positions_and_angles(initial_config, estimated_outer_radius)

        # Validate the optimized configuration
        validity, inv_radius = packer.evaluate_configuration(optimized_config, estimated_outer_radius)

        if validity and inv_radius > best_inv_radius:
            best_inv_radius = inv_radius
            best_config = optimized_config.copy()
            best_outer_radius = 1.0 / inv_radius if inv_radius > 0 else float('inf')

        # If not valid, still try to refine it
        if not validity:
            # Try a few iterations of local refinement with noise
            for _ in range(3):
                # Add some noise to encourage exploration
                for i in range(len(optimized_config)):
                    optimized_config[i][0] += np.random.normal(0, 0.05)
                    optimized_config[i][1] += np.random.normal(0, 0.05)
                    # Keep angles within reasonable bounds
                    optimized_config[i][2] = optimized_config[i][2] % 360

                # Re-optimize locally
                optimized_config = optimizer.optimize_positions_and_angles(optimized_config, estimated_outer_radius)

                validity, inv_radius = packer.evaluate_configuration(optimized_config, estimated_outer_radius)
                if validity and inv_radius > best_inv_radius:
                    best_inv_radius = inv_radius
                    best_config = optimized_config.copy()
                    best_outer_radius = 1.0 / inv_radius if inv_radius > 0 else float('inf')

    # If we still don't have a good solution, fall back to a conservative approach
    if best_config is None:
        # Use the simplest configuration that works
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
            [0, -4, 0]
        ])
        best_inv_radius = 1.0 / 8.0  # Conservative estimate
        best_outer_radius = 8.0

    # Prepare return values
    inner_hex_data = np.array(best_config)
    outer_hex_data = np.array([outer_center_x, outer_center_y, 0])
    outer_hex_side_length = best_outer_radius * 2  # approximate side length

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END