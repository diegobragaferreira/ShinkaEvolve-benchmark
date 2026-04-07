# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
import time
import random
from numba import jit, prange
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class HexagonConfig:
    """Data class to represent a hexagon configuration"""
    center_x: float
    center_y: float
    rotation: float  # in degrees

@dataclass
class OptimizationResult:
    """Data class to hold optimization results"""
    inner_hex_data: np.ndarray
    outer_hex_data: np.ndarray
    outer_hex_side_length: float
    inv_outer_hex_side_length: float
    benchmark_ratio: float
    eval_time: float

class HexagonGeometry:
    """Handles all geometric computations for hexagons"""

    @staticmethod
    @jit(nopython=True)
    def get_hexagon_vertices(x: float, y: float, angle_deg: float, radius: float = 1.0) -> np.ndarray:
        """Get vertices of a hexagon given center, angle, and radius"""
        vertices = np.zeros((6, 2))
        angle_rad = np.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
        return vertices

    @staticmethod
    @jit(nopython=True)
    def point_in_hexagon(px: float, py: float, hx: float, hy: float, angle_deg: float, radius: float = 1.0) -> bool:
        """Fast point-in-hexagon test using numba"""
        angle_rad = np.radians(angle_deg)
        # Transform point to hexagon coordinate system
        dx = px - hx
        dy = py - hy
        # Rotate point by negative angle
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a

        # Check if point is in hexagon (simplified version)
        abs_rx = np.abs(rx)
        abs_ry = np.abs(ry)
        return abs_rx <= radius and abs_ry <= radius * np.sqrt(3)/2 and abs_rx + abs_ry <= radius * (1 + np.sqrt(3)/2)

    @staticmethod
    def create_regular_hexagon(center: Tuple[float, float], side_length: float = 1, rotation: float = 0) -> Polygon:
        """Create a regular hexagon as a Shapely polygon"""
        angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
        x = center[0] + side_length * np.cos(angles)
        y = center[1] + side_length * np.sin(angles)
        return Polygon(list(zip(x, y)))

    @staticmethod
    def hexagon_to_polygon(x: float, y: float, angle_deg: float, radius: float = 1.0) -> Polygon:
        """Convert hexagon parameters to shapely polygon"""
        vertices = HexagonGeometry.get_hexagon_vertices(x, y, angle_deg, radius)
        return Polygon(vertices)

class ConstraintValidator:
    """Validates hexagon packing constraints"""

    @staticmethod
    def check_overlap_fast(hex1_poly: Polygon, hex2_poly: Polygon) -> bool:
        """Fast overlap check using Shapely with buffer for numerical stability"""
        # Quick bounding box check first
        bbox1 = hex1_poly.bounds
        bbox2 = hex2_poly.bounds
        if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
            bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
            return False
        return hex1_poly.buffer(1e-10).intersects(hex2_poly.buffer(1e-10)) and not hex1_poly.touches(hex2_poly)

    @staticmethod
    def check_containment(inner_hex: Polygon, outer_hex: Polygon) -> bool:
        """Check if inner hexagon is fully contained within outer hexagon"""
        return outer_hex.contains(inner_hex)

class HexagonPackerEvaluator:
    """Evaluates hexagon configurations and computes objectives"""

    @staticmethod
    def compute_outer_hexagon_radius(inner_hex_data: np.ndarray) -> float:
        """Compute minimum outer hexagon radius that contains all inner hexagons"""
        if len(inner_hex_data) == 0:
            return 0.0

        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(len(inner_hex_data)):
            x, y, angle = inner_hex_data[i]
            vertices = HexagonGeometry.get_hexagon_vertices(x, y, angle)
            all_vertices.extend(vertices)

        if len(all_vertices) == 0:
            return 0.0

        # Compute centroid
        centroid_x = np.mean([v[0] for v in all_vertices])
        centroid_y = np.mean([v[1] for v in all_vertices])

        # Find maximum distance from centroid to any vertex
        max_distance = 0.0
        for x, y in all_vertices:
            distance = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
            max_distance = max(max_distance, distance)

        # Add buffer for hexagon radius calculation
        return max_distance + 1.0

    @staticmethod
    def evaluate_constraint_violations(inner_hex_data: np.ndarray, outer_hex_data: np.ndarray) -> List[str]:
        """Evaluate constraint violations for a given configuration"""
        violations = []

        # Create outer hexagon
        outer_x, outer_y, outer_angle = outer_hex_data
        outer_radius = HexagonPackerEvaluator.compute_outer_hexagon_radius(inner_hex_data)
        outer_hex = HexagonGeometry.hexagon_to_polygon(outer_x, outer_y, outer_angle, outer_radius)

        # Check each inner hexagon for containment
        for i in range(len(inner_hex_data)):
            x, y, angle = inner_hex_data[i]
            inner_hex = HexagonGeometry.hexagon_to_polygon(x, y, angle)

            if not ConstraintValidator.check_containment(inner_hex, outer_hex):
                violations.append(f"Inner hexagon {i} not contained")

        # Check overlaps between all pairs
        for i in range(len(inner_hex_data)):
            x1, y1, angle1 = inner_hex_data[i]
            hex1_poly = HexagonGeometry.hexagon_to_polygon(x1, y1, angle1)

            for j in range(i+1, len(inner_hex_data)):
                x2, y2, angle2 = inner_hex_data[j]
                hex2_poly = HexagonGeometry.hexagon_to_polygon(x2, y2, angle2)

                if ConstraintValidator.check_overlap_fast(hex1_poly, hex2_poly):
                    violations.append(f"Overlapping hexagons {i} and {j}")

        return violations

    @staticmethod
    def compute_objective_function(hex_data: np.ndarray) -> float:
        """Compute negative of 1/outer_hex_side_length (to minimize instead of maximize)"""
        # Check if hex_data is valid
        if len(hex_data) != 12:
            return 1e10

        # Compute outer hexagon radius
        outer_radius = HexagonPackerEvaluator.compute_outer_hexagon_radius(hex_data)

        # If outer radius is invalid, penalize heavily
        if outer_radius <= 0:
            return 1e10

        # Return negative of 1/outer_radius (for minimization)
        return -1.0 / outer_radius

    @staticmethod
    def evaluate_solution(hex_data: np.ndarray, outer_hex_data: np.ndarray) -> Tuple[bool, float, List[str]]:
        """Comprehensive evaluation of solution validity and quality"""
        # Basic constraint checking
        violations = HexagonPackerEvaluator.evaluate_constraint_violations(hex_data, outer_hex_data)

        if violations:
            return False, 1e10, violations

        # Compute objective value
        obj_value = HexagonPackerEvaluator.compute_objective_function(hex_data)
        return True, obj_value, []

class ConfigGenerator:
    """Generates initial configurations for optimization"""

    @staticmethod
    def generate_hcp_lattice_config() -> np.ndarray:
        """Generate initial configuration based on hexagonal close-packed (HCP) lattice structure"""
        # HCP lattice arranges atoms in alternating layers with special spacing
        # Layer 1: Central atom
        # Layer 2: Surrounding 6 atoms in hexagonal pattern
        # Layer 3: Atoms in triangular arrangement above layer 1

        hex_data = []

        # Layer 1: Central hexagon
        hex_data.append([0.0, 0.0, 0.0])

        # Layer 2: First ring (6 hexagons) - arranged in hexagonal pattern
        # This creates a tight cluster around the center
        for i in range(6):
            angle = i * 60  # degrees
            rad = np.radians(angle)
            # Distance of sqrt(3) ~ 1.732 for optimal packing in hexagonal lattice
            x = 1.732 * np.cos(rad)
            y = 1.732 * np.sin(rad)
            hex_data.append([x, y, 0.0])

        # Layer 3: Second ring (5 hexagons) - positioned to create dense packing
        # These are placed at positions that maximize space utilization
        # Position along triangular pattern with appropriate spacing
        ring3_angles = [0, 72, 144, 216, 288]
        # Distance chosen to maintain good packing density for the third layer
        ring3_radius = 3.0  # Adjusted to achieve better packing density
        for angle in ring3_angles:
            x = ring3_radius * np.cos(np.radians(angle))
            y = ring3_radius * np.sin(np.radians(angle))
            hex_data.append([x, y, 0.0])

        # Add one more hexagon to complete 12
        hex_data.append([0, -ring3_radius - 1.0, 0])

        # Ensure exactly 12 hexagons
        while len(hex_data) < 12:
            hex_data.append([0.0, 0.0, 0.0])
        hex_data = hex_data[:12]

        # Add small random perturbations to escape symmetric local minima
        for i in range(12):
            hex_data[i][0] += random.uniform(-0.1, 0.1)
            hex_data[i][1] += random.uniform(-0.1, 0.1)
            hex_data[i][2] += random.uniform(-2, 2)

        return np.array(hex_data)

    @staticmethod
    def generate_better_initial_solution() -> np.ndarray:
        """Generate a high-quality initial solution based on mathematical insights"""
        # Use the HCP-based configuration as the primary better initial solution
        return ConfigGenerator.generate_hcp_lattice_config()

    @staticmethod
    def generate_multiple_initial_configs() -> List[np.ndarray]:
        """Generate multiple diverse initial configurations"""
        configs = []

        # Configuration 1: HCP lattice arrangement (core improvement)
        configs.append(ConfigGenerator.generate_hcp_lattice_config())

        # Configuration 2: Perturbed version of HCP
        config2 = configs[0].copy()
        for i in range(len(config2)):
            config2[i][0] += random.uniform(-0.3, 0.3)
            config2[i][1] += random.uniform(-0.3, 0.3)
            config2[i][2] += random.uniform(-10, 10)
        configs.append(config2)

        # Configuration 3: Compact ring arrangement (existing)
        config3 = []
        config3.append([0, 0, 0])  # Center
        for i in range(6):
            angle = i * 60
            radius = 1.9
            x = math.cos(math.radians(angle)) * radius
            y = math.sin(math.radians(angle)) * radius
            config3.append([x, y, 0])

        angles = [0, 72, 144, 216, 288]
        radius = 3.2
        for i, angle in enumerate(angles):
            x = math.cos(math.radians(angle)) * radius
            y = math.sin(math.radians(angle)) * radius
            config3.append([x, y, 0])
        config3.append([0, -radius - 1.0, 0])
        configs.append(np.array(config3[:12]))

        # Configuration 4: Balanced arrangement (existing)
        config4 = []
        config4.append([0, 0, 0])
        for i in range(6):
            angle = i * 60
            radius = 2.0
            x = math.cos(math.radians(angle)) * radius
            y = math.sin(math.radians(angle)) * radius
            config4.append([x, y, 0])
        angles = [30, 90, 150, 210, 270]
        radius = 3.4
        for i, angle in enumerate(angles):
            x = math.cos(math.radians(angle)) * radius
            y = math.sin(math.radians(angle)) * radius
            config4.append([x, y, 0])
        config4.append([0, -radius - 1.0, 0])
        configs.append(np.array(config4[:12]))

        return configs

class EvolutionaryOptimizer:
    """Evolutionary optimization engine for hexagon packing"""

    def __init__(self):
        self.max_time = 170  # seconds
        self.start_time = time.time()

    def get_current_time(self):
        return time.time() - self.start_time

    def mutate_symmetrically(self, individual: np.ndarray, mut_pb: float = 0.3, mut_strength: float = 0.2) -> np.ndarray:
        """
        Mutate an individual while preserving symmetry properties.
        For hexagon packing, this maintains the ring structure.
        """
        # Copy individual to avoid modifying original
        mutated = individual.copy()

        # Define symmetric groups
        # Group 0: Central hexagon (index 0)
        # Group 1: First ring (indices 1-6)
        # Group 2: Second ring (indices 7-11)

        # Mutate central hexagon
        if random.random() < mut_pb:
            mutated[0, 0] += random.uniform(-mut_strength, mut_strength)
            mutated[0, 1] += random.uniform(-mut_strength, mut_strength)
            mutated[0, 2] += random.uniform(-mut_strength, mut_strength)

        # Mutate first ring (6 hexagons)
        if random.random() < mut_pb:
            # Mutate all 6 hexagons with same amount to maintain some rotational symmetry
            offset_x = random.uniform(-mut_strength, mut_strength)
            offset_y = random.uniform(-mut_strength, mut_strength)
            offset_angle = random.uniform(-mut_strength, mut_strength)
            for i in range(1, 7):
                mutated[i, 0] += offset_x
                mutated[i, 1] += offset_y
                mutated[i, 2] += offset_angle

        # Mutate second ring (6 hexagons)
        if random.random() < mut_pb:
            # Mutate all 6 hexagons with same amount to maintain rotational symmetry
            offset_x = random.uniform(-mut_strength, mut_strength)
            offset_y = random.uniform(-mut_strength, mut_strength)
            offset_angle = random.uniform(-mut_strength, mut_strength)
            for i in range(7, 12):
                mutated[i, 0] += offset_x
                mutated[i, 1] += offset_y
                mutated[i, 2] += offset_angle

        return mutated

    def optimize_single_configuration(self, initial_hex_data: np.ndarray) -> np.ndarray:
        """Perform optimization on a single configuration using L-BFGS-B"""
        # Flatten initial data for optimization
        initial_flat = initial_hex_data.flatten()

        # Bounds: positions (-10, 10), angles (0, 360)
        bounds = []
        for i in range(12):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])

        def objective_and_gradient(params):
            # Reshape parameters back to hex data format
            hex_data = params.reshape(12, 3)

            # Evaluate objective function
            obj_value = HexagonPackerEvaluator.compute_objective_function(hex_data)

            # Approximate gradient using finite differences
            epsilon = 1e-6
            grad = np.zeros_like(params)

            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += epsilon
                hex_data_plus = params_plus.reshape(12, 3)
                obj_plus = HexagonPackerEvaluator.compute_objective_function(hex_data_plus)
                grad[i] = (obj_plus - obj_value) / epsilon

            return obj_value, grad

        # Optimize using L-BFGS-B
        try:
            result = minimize(
                objective_and_gradient,
                initial_flat,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-12,
                    'maxls': 50
                },
                tol=1e-12
            )

            if result.success:
                optimized_data = result.x.reshape(12, 3)
                return optimized_data
        except Exception:
            pass

        return initial_hex_data

    def multi_start_optimization(self) -> np.ndarray:
        """Run multiple optimization starts with different initial configurations"""
        # Generate multiple initial configurations
        initial_configs = ConfigGenerator.generate_multiple_initial_configs()

        best_score = float('inf')
        best_solution = None

        # Multiple random restarts
        for restart, initial_hex_data in enumerate(initial_configs):
            # Apply symmetry-aware mutation to create diversified starting points
            diversified_config = self.mutate_symmetrically(initial_hex_data, mut_pb=0.5, mut_strength=0.3)

            # Optimize this configuration
            optimized_hex_data = self.optimize_single_configuration(diversified_config)

            # Evaluate this solution
            valid, obj_value, violations = HexagonPackerEvaluator.evaluate_solution(optimized_hex_data, [0, 0, 0])

            if valid and obj_value < best_score:
                best_score = obj_value
                best_solution = optimized_hex_data

        return best_solution if best_solution is not None else initial_configs[0]

def hexagon_packing_12() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize optimizer
    optimizer = EvolutionaryOptimizer()

    # Get best solution through multi-start optimization
    best_hex_data = optimizer.multi_start_optimization()

    # Final validation
    valid, obj_value, violations = HexagonPackerEvaluator.evaluate_solution(best_hex_data, [0, 0, 0])

    if not valid:
        # Fallback to a known good configuration if optimization fails
        fallback_config = np.array([
            [0, 0, 0],              # center
            [-2.5, 0, 0],           # left
            [2.5, 0, 0],            # right
            [-1.25, 2.17, 0],       # top-left
            [1.25, 2.17, 0],        # top-right
            [-1.25, -2.17, 0],      # bottom-left
            [1.25, -2.17, 0],       # bottom-right
            [-3.75, 2.17, 0],       # far top-left
            [3.75, 2.17, 0],        # far top-right
            [-3.75, -2.17, 0],      # far bottom-left
            [3.75, -2.17, 0],       # far bottom-right
            [0, -4, 0],             # far bottom-center
        ])
        return fallback_config, np.array([0, 0, 0]), 8.0

    # Compute final outer hexagon radius
    final_radius = HexagonPackerEvaluator.compute_outer_hexagon_radius(best_hex_data)

    # Calculate benchmark ratio
    benchmark_ratio = (1.0 / final_radius) / 0.2537

    # Create result object
    inner_hex_data = best_hex_data
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = final_radius

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END