# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import time
from typing import Tuple, List, Optional, NamedTuple
import warnings

# Data structure definitions
class HexagonConfig(NamedTuple):
    center_x: float
    center_y: float
    angle_deg: float

class PackingResult(NamedTuple):
    inner_hex_data: np.ndarray
    outer_hex_data: np.ndarray
    outer_hex_side_length: float

class GeometryHandler:
    """Handles all geometric computations for hexagon operations."""

    def __init__(self):
        self.hex_radius = 1.0
        self.hex_apothem = np.sqrt(3) / 2
        self.hex_height = 2 * self.hex_apothem
        self.hex_width = 2 * self.hex_radius

    def generate_hexagon_vertices(self, center_x: float, center_y: float, angle_deg: float) -> np.ndarray:
        """Generate vertices of a regular hexagon given center and rotation."""
        angle_rad = np.deg2rad(angle_deg)
        # Vertices of a unit hexagon centered at origin
        base_vertices = np.array([
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1, 0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        # Rotate and translate
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_vertices = base_vertices @ rotation_matrix.T
        return rotated_vertices + np.array([center_x, center_y])

    def create_hexagon_polygon(self, center_x: float, center_y: float, angle_deg: float) -> Polygon:
        """Create Shapely polygon representation of a hexagon."""
        vertices = self.generate_hexagon_vertices(center_x, center_y, angle_deg)
        return Polygon(vertices)

class ConstraintChecker:
    """Handles all constraint validation logic."""

    def __init__(self, geometry_handler: GeometryHandler):
        self.geo = geometry_handler

    def check_containment(self, hexagon: Polygon, outer_hex: Polygon) -> bool:
        """Check if hexagon is fully contained within outer hexagon."""
        return outer_hex.contains(hexagon) or outer_hex.touches(hexagon)

    def check_overlap(self, hex1: Polygon, hex2: Polygon) -> bool:
        """Check if two hexagons overlap."""
        return hex1.intersects(hex2) and not hex1.touches(hex2)

    def compute_constraint_violations(self, hex_configs: List[HexagonConfig],
                                    outer_radius: float) -> Tuple[bool, float]:
        """Compute constraint violations for a configuration."""
        # Create outer hexagon (scaled appropriately)
        outer_vertices = self.geo.generate_hexagon_vertices(0, 0, 0)
        outer_vertices *= outer_radius / self.geo.hex_radius
        outer_hex = Polygon(outer_vertices)

        # Check containment and overlap efficiently
        total_violation = 0.0
        inner_polygons = []

        for i, config in enumerate(hex_configs):
            inner_hex = self.geo.create_hexagon_polygon(config.center_x, config.center_y, config.angle_deg)
            inner_polygons.append(inner_hex)

            # Check containment - more robust way
            if not self.check_containment(inner_hex, outer_hex):
                # For containment violations, we measure the extent of violation
                try:
                    intersection = outer_hex.intersection(inner_hex)
                    if intersection.is_empty:
                        # Complete violation
                        total_violation += 10000.0
                    else:
                        # Partial violation
                        contained_area = intersection.area
                        total_violation += (inner_hex.area - contained_area) * 100.0
                except:
                    total_violation += 10000.0

            # Check overlaps with previously processed hexagons only
            # (We're iterating in order so we avoid redundant checks)
            for j in range(i):
                if self.check_overlap(inner_hex, inner_polygons[j]):
                    try:
                        intersection_area = inner_hex.intersection(inner_polygons[j]).area
                        # Penalty based on the amount of overlapping area
                        overlap_penalty = (inner_hex.area + inner_polygons[j].area - 2 * intersection_area) * 500.0
                        total_violation += overlap_penalty
                    except:
                        total_violation += 50000.0

        return total_violation == 0, total_violation

class Optimizer:
    """Handles the optimization process with configurable parameters."""

    def __init__(self, geometry_handler: GeometryHandler, constraint_checker: ConstraintChecker):
        self.geo = geometry_handler
        self.constraint_checker = constraint_checker
        self.max_iterations = 200
        self.population_size = 15
        self.mutation_factor = 0.5
        self.recombination_rate = 0.7

    def evaluate_fitness(self, positions_and_radius: np.ndarray) -> float:
        """Evaluate the fitness of a configuration."""
        # Extract parameters
        positions = positions_and_radius[:-1].reshape(-1, 3)
        outer_radius = positions_and_radius[-1]

        # Convert to list of configs
        hex_configs = [HexagonConfig(pos[0], pos[1], pos[2]) for pos in positions]

        # Check constraints
        valid, violation = self.constraint_checker.compute_constraint_violations(hex_configs, outer_radius)

        if not valid:
            # Return very poor fitness for invalid configurations
            return 1e12 + violation

        # Return negative inverse of outer radius (we want to maximize 1/R)
        # This means minimizing -1/R, which is equivalent to maximizing 1/R
        return -1.0 / outer_radius

    def get_initial_guess(self) -> np.ndarray:
        """Generate a highly symmetric initial configuration."""
        # Start with a 2D hexagonal lattice pattern with 12 positions
        # This exploits known good symmetric arrangements

        # Base positions in a 2D hexagonal lattice pattern
        base_positions = [
            [0, 0],           # Center
            [-2, 0],          # Left
            [2, 0],           # Right
            [0, 2],           # Top
            [0, -2],          # Bottom
            [-1, 1],          # Top-left
            [1, 1],           # Top-right
            [-1, -1],         # Bottom-left
            [1, -1],          # Bottom-right
            [-2.5, 1.5],      # Far top-left
            [2.5, 1.5],       # Far top-right
            [-2.5, -1.5],     # Far bottom-left
            [2.5, -1.5],      # Far bottom-right
        ]

        # Adjust positions to ensure they fit nicely in a hexagonal packing
        # Remove one position and adjust to make room for 12 hexes
        adjusted_positions = [
            [0, 0],           # Center
            [-2.5, 0],        # Left
            [2.5, 0],         # Right
            [-1.25, 2.17],    # Top-left
            [1.25, 2.17],     # Top-right
            [-1.25, -2.17],   # Bottom-left
            [1.25, -2.17],    # Bottom-right
            [-3.75, 2.17],    # Far top-left
            [3.75, 2.17],     # Far top-right
            [-3.75, -2.17],   # Far bottom-left
            [3.75, -2.17],    # Far bottom-right
            [0, -4],          # Far bottom
        ]

        # Flatten and add outer radius
        initial_positions = np.array(adjusted_positions)
        # Add rotation angles (all zero for now, we'll optimize them)
        positions_with_angles = np.column_stack([initial_positions, np.zeros(initial_positions.shape[0])])

        # Initial outer radius estimate (based on the arrangement)
        outer_radius_guess = 4.0

        return np.concatenate([positions_with_angles.flatten(), [outer_radius_guess]])

    def optimize(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Execute the optimization process."""
        # Get initial configuration
        initial_config = self.get_initial_guess()

        bounds = []
        # Position bounds
        for _ in range(12):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])
        # Outer radius bound
        bounds.append((1.0, 15.0))

        # First, do a global optimization step
        try:
            de_result = differential_evolution(
                self.evaluate_fitness,
                bounds,
                maxiter=self.max_iterations,
                popsize=self.population_size,
                mutation=(self.mutation_factor, 1),
                recombination=self.recombination_rate,
                seed=42,
                disp=False,
                tol=1e-6
            )
        except Exception as e:
            warnings.warn(f"Differential evolution failed: {e}")
            return initial_config[:-1].reshape(-1, 3), np.array([0, 0, 0]), initial_config[-1]

        # Then, local refinement
        try:
            refined_result = minimize(
                self.evaluate_fitness,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if refined_result.success:
                final_config = refined_result.x
            else:
                final_config = de_result.x
        except Exception as e:
            warnings.warn(f"Local optimization failed: {e}")
            final_config = de_result.x

        # Extract positions and radius
        positions = final_config[:-1].reshape(-1, 3)
        outer_radius = final_config[-1]

        # Convert back to requested format
        inner_hex_data = positions.copy()
        outer_hex_data = np.array([0, 0, 0])  # Centered at origin

        return inner_hex_data, outer_hex_data, outer_radius

def hexagon_packing_12() -> PackingResult:
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize components
    geo_handler = GeometryHandler()
    constraint_checker = ConstraintChecker(geo_handler)
    optimizer = Optimizer(geo_handler, constraint_checker)

    # Execute optimization
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimizer.optimize()

    return PackingResult(inner_hex_data, outer_hex_data, outer_hex_side_length)

# EVOLVE-BLOCK-END