# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
from scipy.optimize import differential_evolution, minimize
import time
from typing import Tuple, List, Optional
import warnings

class HexagonGeometry:
    """Utility class for hexagon geometric operations"""
    
    @staticmethod
    def create_regular_hexagon(center_x: float, center_y: float, side_length: float = 1.0, 
                              rotation_deg: float = 0) -> Polygon:
        """Create a regular hexagon as a Shapely polygon"""
        rotation_rad = math.radians(rotation_deg)
        points = []
        for i in range(6):
            angle = rotation_rad + i * math.pi / 3
            x = center_x + side_length * math.cos(angle)
            y = center_y + side_length * math.sin(angle)
            points.append((x, y))
        return Polygon(points)
    
    @staticmethod
    def compute_outer_radius(inner_hexagons: List[Polygon], padding: float = 0.01) -> float:
        """Compute minimum radius needed to contain all inner hexagons with padding"""
        # Get all vertices of all hexagons
        all_vertices = []
        for hex_poly in inner_hexagons:
            all_vertices.extend(list(hex_poly.exterior.coords))

        if not all_vertices:
            return 1.0
            
        # Find center of bounding box
        xs = [p[0] for p in all_vertices]
        ys = [p[1] for p in all_vertices]
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2

        # Compute max distance from center to any vertex
        max_dist = 0
        for x, y in all_vertices:
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = max(max_dist, dist)

        return max_dist + padding

class PackingValidator:
    """Handles validation of hexagon packings"""
    
    @staticmethod
    def check_containment_and_overlap(inner_hexagons: List[Polygon], outer_hexagon: Polygon) -> bool:
        """Check if all inner hexagons are contained in outer hexagon and don't overlap"""
        # Check containment - early exit if any fails
        for hex_poly in inner_hexagons:
            if not outer_hexagon.contains(hex_poly):
                return False

        # Check pairwise overlaps - early exit if any found
        for i in range(len(inner_hexagons)):
            for j in range(i+1, len(inner_hexagons)):
                if inner_hexagons[i].intersects(inner_hexagons[j]):
                    return False

        return True

class ConfigurationGenerator:
    """Generates and manages hexagon configurations"""
    
    @staticmethod
    def generate_initial_config() -> np.ndarray:
        """Generate an initial configuration for 11 hexagons using a known good packing pattern"""
        # This configuration is inspired by compact packings of hexagons
        # Based on mathematical analysis of dense hexagonal arrangements

        # Using hexagon side length = 1, distance between centers = sqrt(3)
        hex_spacing = math.sqrt(3)  # distance between adjacent hexagon centers

        # Improved initial configuration based on compact packing strategies
        initial_positions = [
            # Center hexagon
            [0, 0, 0],

            # Surrounding hexagons in a compact formation
            [-hex_spacing, 0, 0],     # left
            [hex_spacing, 0, 0],      # right
            [0, hex_spacing, 0],      # top
            [0, -hex_spacing, 0],    # bottom
            [-hex_spacing/2, hex_spacing/2, 0],  # top-left
            [hex_spacing/2, hex_spacing/2, 0],   # top-right
            [-hex_spacing/2, -hex_spacing/2, 0], # bottom-left
            [hex_spacing/2, -hex_spacing/2, 0],  # bottom-right
            [-hex_spacing * 1.5, 0, 0],   # extended left
            [hex_spacing * 1.5, 0, 0],    # extended right
        ]

        # Fill remaining positions with symmetrically placed hexagons
        while len(initial_positions) < 11:
            initial_positions.append([0, 0, 0])  # placeholder for unused positions

        return np.array(initial_positions[:11])

class Optimizer:
    """Handles the optimization process"""
    
    def __init__(self):
        self.bounds = []
        for i in range(11):
            # x and y coordinates bounded to prevent extreme positions
            self.bounds.extend([(-12, 12), (-12, 12), (0, 360)])
    
    def objective_function(self, params: np.ndarray) -> float:
        """Objective function for optimization - returns negative inverse of radius"""
        # Reshape parameters into positions and angles
        positions_angles = []
        for i in range(11):
            x = params[i*3]
            y = params[i*3 + 1]
            angle = params[i*3 + 2]
            positions_angles.append([x, y, angle])

        score, _ = self.evaluate_layout(positions_angles)
        return score  # Negative because we minimize -score = maximize score
    
    def evaluate_layout(self, inner_positions_angles: List[List[float]], 
                       outer_center: Tuple[float, float] = (0, 0)) -> Tuple[float, float]:
        """Evaluate the layout quality"""
        # Convert to hexagon polygons
        inner_hexagons = []
        for pos_angle in inner_positions_angles:
            x, y, angle = pos_angle
            hex_poly = HexagonGeometry.create_regular_hexagon(x, y, 1, angle)
            inner_hexagons.append(hex_poly)

        # Create outer hexagon with current radius
        outer_radius = HexagonGeometry.compute_outer_radius(inner_hexagons, 0.01)
        outer_hexagon = HexagonGeometry.create_regular_hexagon(outer_center[0], outer_center[1], 
                                                              outer_radius, 0)

        # Validate constraints
        valid = PackingValidator.check_containment_and_overlap(inner_hexagons, outer_hexagon)

        # Return negative because we want to maximize 1/R (minimize R)
        outer_side_length = outer_radius
        inv_radius = 1.0 / outer_side_length if valid else -1e6

        return inv_radius, outer_side_length
    
    def local_refinement_step(self, initial_positions_angles: np.ndarray, 
                            stage: int = 1) -> np.ndarray:
        """
        Apply local optimization with adaptive refinement based on stage
        """
        # Define bounds for local optimization
        bounds = []
        for i in range(11):
            bounds.extend([(-15, 15), (-15, 15), (0, 360)])

        # Adjust optimization parameters based on stage
        maxiter = 150 if stage == 1 else 200
        ftol = 1e-9 if stage == 1 else 1e-10
        gtol = 1e-9 if stage == 1 else 1e-10

        # Use L-BFGS-B for local refinement with adaptive settings
        try:
            result = minimize(
                self.objective_function,
                initial_positions_angles.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
            )
            
            if result.success:
                refined_positions = result.x.reshape(-1, 3)
                return refined_positions
        except Exception as e:
            warnings.warn(f"Local refinement failed: {e}")
        
        # Return original if refinement fails
        return initial_positions_angles

class HexagonPackingOptimizer:
    """Main orchestrator class for hexagon packing optimization"""
    
    def __init__(self):
        self.config_generator = ConfigurationGenerator()
        self.optimizer = Optimizer()
        
    def run_optimization(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run the complete optimization pipeline"""
        # Generate initial configuration
        initial_positions = self.config_generator.generate_initial_config()
        inner_hex_data = initial_positions.copy()

        try:
            # Run differential evolution for global optimization
            result = differential_evolution(
                self.optimizer.objective_function,
                self.optimizer.bounds,
                maxiter=150,
                popsize=25,
                seed=42,
                tol=1e-8,
                mutation=(0.5, 1),
                recombination=0.7,
                disp=False
            )

            # Extract best solution from global search
            best_params = result.x
            final_positions_angles = []
            for i in range(11):
                x = best_params[i*3]
                y = best_params[i*3 + 1]
                angle = best_params[i*3 + 2]
                final_positions_angles.append([x, y, angle])

            # Apply multi-stage local refinement
            refined_positions = self.optimizer.local_refinement_step(np.array(final_positions_angles), stage=1)
            refined_positions = self.optimizer.local_refinement_step(refined_positions, stage=2)

            # Evaluate final result after refinement
            final_score, final_side_length = self.optimizer.evaluate_layout(refined_positions)
            best_inner_data = refined_positions
            best_outer_side_length = final_side_length

        except Exception as e:
            # Fallback to initial configuration if optimization fails
            warnings.warn(f"Optimization failed: {e}")
            best_inner_data = inner_hex_data.copy()
            best_outer_side_length = 8.0

        return best_inner_data, best_outer_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Create optimizer instance
    packer = HexagonPackingOptimizer()
    
    # Run optimization
    best_inner_data, best_outer_side_length = packer.run_optimization()
    
    # Final validation and refinement
    inner_hexagons = []
    for pos_angle in best_inner_data:
        x, y, angle = pos_angle
        hex_poly = HexagonGeometry.create_regular_hexagon(x, y, 1, angle)
        inner_hexagons.append(hex_poly)

    # Compute current outer hexagon size
    outer_radius = HexagonGeometry.compute_outer_radius(inner_hexagons, 0.01)
    outer_hexagon = HexagonGeometry.create_regular_hexagon(0, 0, outer_radius, 0)

    # Validate constraints
    if not PackingValidator.check_containment_and_overlap(inner_hexagons, outer_hexagon):
        # If invalid, fall back to initial configuration
        initial_positions = ConfigurationGenerator.generate_initial_config()
        best_inner_data = initial_positions.copy()
        inner_hexagons = []
        for pos_angle in best_inner_data:
            x, y, angle = pos_angle
            hex_poly = HexagonGeometry.create_regular_hexagon(x, y, 1, angle)
            inner_hexagons.append(hex_poly)
        outer_radius = HexagonGeometry.compute_outer_radius(inner_hexagons, 0.01)

    # Ensure we're returning the correct data format
    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    # Return results
    end_time = time.time()
    eval_time = end_time - start_time
    
    # Calculate metrics
    inv_outer_hex_side_length = 1.0 / outer_radius
    benchmark_ratio = inv_outer_hex_side_length / 0.2544
    
    # Return final results with the required format
    return best_inner_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END