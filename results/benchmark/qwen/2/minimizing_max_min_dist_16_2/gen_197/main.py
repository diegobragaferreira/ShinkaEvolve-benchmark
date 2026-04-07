# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
from typing import Tuple, List, Optional
from scipy.spatial import SphericalVoronoi
import warnings

class SphericalTilingEvolver:
    """Evolver that transforms the 2D point dispersion problem to spherical geometry."""

    def __init__(self, num_points: int = 16, dimension: int = 2):
        self.num_points = num_points
        self.dimension = dimension
        self.benchmark_ratio = 0.2786
        self.max_iterations = 1000

    def project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project 2D points onto unit sphere using stereographic projection."""
        # Map [0,1] x [0,1] to sphere using stereographic projection
        # Stereographic projection from plane to sphere
        points_2d = points.copy()

        # Normalize to [-1,1] range for easier spherical mapping
        points_norm = (points_2d - 0.5) * 2

        # Convert to 3D points on unit sphere using stereographic projection
        # From plane (x,y) to sphere (x,y,z) where z = (x²+y²-1)/(x²+y²+1)
        x2 = points_norm[:, 0]**2
        y2 = points_norm[:, 1]**2
        denom = x2 + y2 + 1.0

        # Ensure we don't divide by zero (though shouldn't happen with valid inputs)
        denom = np.maximum(denom, 1e-10)

        # Project to sphere
        sphere_points = np.zeros((len(points_norm), 3))
        sphere_points[:, 0] = 2 * points_norm[:, 0] / denom
        sphere_points[:, 1] = 2 * points_norm[:, 1] / denom
        sphere_points[:, 2] = (x2 + y2 - 1.0) / denom

        return sphere_points

    def project_from_sphere(self, sphere_points: np.ndarray) -> np.ndarray:
        """Project 3D spherical points back to 2D plane using inverse stereographic projection."""
        # Inverse stereographic projection from sphere to plane
        # From sphere (x,y,z) to plane (x,y) where x = x/(1-z), y = y/(1-z)
        projected_points = np.zeros((len(sphere_points), 2))

        # Handle special case when z=1 (north pole) - map to infinity
        safe_z = np.clip(sphere_points[:, 2], -0.9999, 0.9999)

        # Inverse projection
        denom = 1.0 - safe_z
        # Avoid division by zero
        denom = np.maximum(denom, 1e-10)

        projected_points[:, 0] = sphere_points[:, 0] / denom
        projected_points[:, 1] = sphere_points[:, 1] / denom

        # Convert back from [-1,1] to [0,1] range
        projected_points = projected_points / 2.0 + 0.5

        # Clip to valid bounds
        projected_points = np.clip(projected_points, 0.001, 0.999)

        return projected_points

    def calculate_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Calculate min/max distance ratio along with actual values."""
        if len(points) < 2:
            return 0.0, 0.0, 0.0

        distances = pdist(points)
        if len(distances) == 0:
            return 0.0, 0.0, 0.0

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return 0.0, min_dist, max_dist

        ratio = min_dist / max_dist
        return ratio, min_dist, max_dist

    def generate_spherical_initial(self) -> np.ndarray:
        """Generate initial configuration on sphere using spherical tiling principles."""
        # Generate points using a known good spherical arrangement
        # We'll use vertices of a regular icosahedron as starting point
        # and then perturb them for optimization
        phi = (1 + math.sqrt(5)) / 2  # golden ratio

        # Generate icosahedral vertices (approximate spherical distribution)
        vertices = []

        # Regular icosahedron vertices
        # These are known to give good spherical point distributions
        t = 0.525731112119133606  # sqrt(5)/4
        s = 0.850650808352039932  # sqrt(5)/2 * (1/sqrt(5) + 1)

        # Add vertices of icosahedron
        vertices.extend([
            [0, s, t], [0, s, -t], [0, -s, t], [0, -s, -t],
            [t, 0, s], [-t, 0, s], [t, 0, -s], [-t, 0, -s],
            [s, t, 0], [-s, t, 0], [s, -t, 0], [-s, -t, 0]
        ])

        # Normalize vertices to unit sphere
        vertices = np.array(vertices)
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / norms

        # Take subset of vertices for 16 points
        if len(vertices) >= self.num_points:
            selected_vertices = vertices[:self.num_points]
        else:
            # If we have fewer vertices, duplicate and perturb
            selected_vertices = vertices[:self.num_points]

        # Add small random perturbations to break symmetry
        np.random.seed(42)
        perturbed = selected_vertices + np.random.normal(0, 0.05, selected_vertices.shape)

        # Renormalize to sphere
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / norms

        # Project back to 2D
        points_2d = self.project_from_sphere(perturbed)

        return points_2d

    def generate_grid_pattern(self) -> np.ndarray:
        """Generate structured grid pattern with improved distribution."""
        points = []
        rows = cols = 4

        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0

        # Hexagonal offset to improve distribution
        for i in range(rows):
            for j in range(cols):
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y

                # Ensure within bounds with margin
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))

                points.append([x, y])

        return np.array(points)

    def generate_fibonacci_spiral(self) -> np.ndarray:
        """Generate points using Fibonacci spiral."""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio

        for i in range(self.num_points):
            theta = math.acos(-1 + (2 * i) / (self.num_points - 1))
            phi_angle = (i * 2 * math.pi) / (phi * phi)

            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)

            # Map to [0.05, 0.95] range
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2

            points.append([x, y])

        return np.array(points)

    def generate_diverse_initial_configs(self) -> List[np.ndarray]:
        """Generate diverse set of initial configurations."""
        configs = []

        # Different strategies
        configs.append(self.generate_spherical_initial())
        configs.append(self.generate_grid_pattern())
        configs.append(self.generate_fibonacci_spiral())

        # Add variants with different perturbations
        np.random.seed(42)
        for config in configs[:2]:  # Only first two for variations
            # Use adaptive perturbation sizing based on configuration quality
            current_ratio, _, _ = self.calculate_ratio(config)
            # Larger perturbations for poorer configurations
            base_magnitude = max(0.01, 0.05 * (1.0 - current_ratio * 2))
            for perturbation_magnitude in [base_magnitude, base_magnitude * 1.5]:
                perturbed = config + np.random.normal(0, perturbation_magnitude, config.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                configs.append(perturbed)

        return configs

    def spherical_objective(self, x: np.ndarray) -> float:
        """Objective function working in spherical space."""
        # Reshape and project to sphere
        points_2d = x.reshape(-1, 2)
        sphere_points = self.project_to_sphere(points_2d)

        # Calculate distances on sphere (great circle distances)
        # Since points are on unit sphere, we can use dot products
        n_points = len(sphere_points)

        if n_points < 2:
            return 0.0

        # Compute dot products for spherical distances
        distances = []
        for i in range(n_points):
            for j in range(i+1, n_points):
                dot_product = np.dot(sphere_points[i], sphere_points[j])
                # Clamp to [-1,1] to handle numerical errors
                dot_product = np.clip(dot_product, -1.0, 1.0)
                # Great circle distance
                angle = np.arccos(dot_product)
                distances.append(angle)

        if len(distances) == 0:
            return 0.0

        distances = np.array(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return 0.0

        # Minimize negative ratio (maximize ratio)
        return -min_dist / max_dist

    def objective_function(self, x: np.ndarray) -> float:
        """Standard 2D objective function."""
        points = x.reshape(-1, 2)
        ratio, _, _ = self.calculate_ratio(points)
        return -ratio  # Negative for minimization

    def multi_scale_optimization(self, initial_points: np.ndarray) -> np.ndarray:
        """Multi-scale optimization combining global and local approaches."""
        # Start with global optimization in spherical space
        best_points = initial_points.copy()
        best_ratio, _, _ = self.calculate_ratio(best_points)

        # Parameters for optimization
        bounds = [(0.001, 0.999) for _ in range(self.num_points * self.dimension)]

        # Phase 1: Global search with various methods
        try:
            # Try L-BFGS-B for initial coarse optimization
            result1 = minimize(
                self.objective_function,
                initial_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4}
            )

            if result1.success:
                phase1_points = result1.x.reshape(-1, 2)
                ratio1, _, _ = self.calculate_ratio(phase1_points)
                if ratio1 > best_ratio:
                    best_points = phase1_points.copy()
                    best_ratio = ratio1

        except Exception:
            pass

        # Phase 2: Local refinement with SLSQP
        try:
            result2 = minimize(
                self.objective_function,
                best_points.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 150}
            )

            if result2.success:
                phase2_points = result2.x.reshape(-1, 2)
                ratio2, _, _ = self.calculate_ratio(phase2_points)
                if ratio2 > best_ratio:
                    best_points = phase2_points.copy()
                    best_ratio = ratio2

        except Exception:
            pass

        # Phase 3: Additional refinement with boundary-constrained optimization
        try:
            # Try with TNC for robustness
            result3 = minimize(
                self.objective_function,
                best_points.flatten(),
                method='TNC',
                bounds=bounds,
                options={'maxiter': 100}
            )

            if result3.success:
                phase3_points = result3.x.reshape(-1, 2)
                ratio3, _, _ = self.calculate_ratio(phase3_points)
                if ratio3 > best_ratio:
                    best_points = phase3_points.copy()
                    best_ratio = ratio3

        except Exception:
            pass

        return best_points

    def get_best_solution(self, configs: List[np.ndarray]) -> np.ndarray:
        """Find best solution among all starting configurations."""
        best_ratio = -np.inf
        best_points = None

        # Try each initial configuration with multi-scale optimization
        for i, config in enumerate(configs):
            # Apply multi-scale optimization
            optimized_points = self.multi_scale_optimization(config)

            ratio, _, _ = self.calculate_ratio(optimized_points)

            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()

        return best_points if best_points is not None else configs[0]

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    # Initialize evolver
    evolver = SphericalTilingEvolver(16, 2)

    # Generate diverse initial configurations
    initial_configs = evolver.generate_diverse_initial_configs()

    # Find best solution using multi-scale optimization
    best_points = evolver.get_best_solution(initial_configs)

    # Final refinement with additional optimization
    try:
        ratio, _, _ = evolver.calculate_ratio(best_points)
        if ratio < 0.25:  # If not very good, do another round of optimization
            best_points = evolver.multi_scale_optimization(best_points)
    except Exception:
        pass

    return best_points

# EVOLVE-BLOCK-END