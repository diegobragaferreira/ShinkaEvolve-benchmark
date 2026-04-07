# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, SphericalVoronoi
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
import random
from typing import Tuple, List

class SphereVoronoiOptimizer:
    def __init__(self):
        self.best_points = None
        self.best_ratio = 0.0
        self.dimension = 3
        self.num_points = 14

    def compute_min_max_ratio(self, points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum distances."""
        distances = pdist(points)
        return np.min(distances) / np.max(distances) if np.max(distances) > 0 else 0

    def icosahedron_vertices(self) -> np.ndarray:
        """Generate vertices of a regular icosahedron."""
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = []
        # Add vertices at (±1, ±φ, 0), (0, ±1, ±φ), (±φ, 0, ±1)
        for i in [1, -1]:
            for j in [1, -1]:
                vertices.append([i, j * phi, 0])
                vertices.append([0, i, j * phi])
                vertices.append([i * phi, 0, j])
        return np.array(vertices)

    def project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1)
        norms = np.where(norms == 0, 1, norms)
        return points / norms[:, np.newaxis]

    def compute_voronoi_entropy(self, points: np.ndarray) -> float:
        """Compute entropy of Voronoi cell areas to measure uniformity."""
        try:
            voronoi = Voronoi(points)
            # For small point sets, we compute cell areas manually
            # This is a simplified approximation for our case
            if len(points) <= 20:
                # Estimate area using cross products for triangles
                areas = []
                for i in range(len(points)):
                    # Get neighbors in a way that approximates cell boundaries
                    # Simple estimation via sampling
                    pass
                return 0.0  # Placeholder - real implementation would compute actual areas
            return 0.0
        except:
            return 0.0

    def generate_initial_configurations(self, num_configs: int = 5) -> List[np.ndarray]:
        """Generate multiple diverse initial configurations."""
        configs = []
        
        # Base icosahedron configuration
        ico_vertices = self.icosahedron_vertices()
        if len(ico_vertices) >= 14:
            ico_config = ico_vertices[:14].copy()
        else:
            ico_config = np.vstack([ico_vertices, ico_vertices[:(14-len(ico_vertices))]]).copy()
        ico_config = self.project_to_sphere(ico_config)
        configs.append(ico_config)

        # Alternative configurations: modified icosahedron and random perturbations
        for i in range(num_configs):
            np.random.seed(i * 1000)
            # Start with icosahedron and add noise
            base_config = ico_config.copy()
            noise_magnitude = 0.05
            noise = np.random.normal(0, noise_magnitude, base_config.shape)
            noisy_config = base_config + noise
            noisy_config = self.project_to_sphere(noisy_config)
            configs.append(noisy_config)

        return configs

    def compute_force_field(self, points: np.ndarray) -> np.ndarray:
        """Compute net force field on each point based on Voronoi geometry and distances."""
        n = len(points)
        forces = np.zeros_like(points)
        
        # Compute approximate Voronoi-based forces
        # For simplicity in this implementation, we'll use a hybrid approach:
        # 1. Repulsion based on distance squared (inverse square law)
        # 2. A global uniformity term based on pairwise differences
        # 3. Spatial weighting based on neighborhood density
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = points[i] - points[j]
                    dist_sq = np.dot(diff, diff)
                    
                    # Avoid division by zero
                    if dist_sq > 1e-12:
                        # Repulsive force inversely proportional to square of distance
                        # This encourages points to spread out
                        force_magnitude = 1.0 / (dist_sq * np.sqrt(dist_sq))
                        force_direction = diff / np.sqrt(dist_sq)
                        forces[i] += force_magnitude * force_direction
        
        return forces

    def voronoi_based_optimization(self, initial_points: np.ndarray, max_iter: int = 5000) -> Tuple[np.ndarray, float]:
        """Optimize points using Voronoi-based particle dynamics."""
        current_points = initial_points.copy()
        current_points = self.project_to_sphere(current_points)

        best_points = current_points.copy()
        best_ratio = self.compute_min_max_ratio(current_points)

        # Parameters for the optimization
        damping_factor = 0.95
        learning_rate = 0.01
        min_temperature = 1e-6
        max_stagnation = 1000
        stagnation_counter = 0

        for iteration in range(max_iter):
            # Compute forces
            forces = self.compute_force_field(current_points)
            
            # Apply forces with momentum-like damping
            velocities = forces * learning_rate
            current_points += velocities
            
            # Project back to sphere
            current_points = self.project_to_sphere(current_points)
            
            # Compute new ratio
            new_ratio = self.compute_min_max_ratio(current_points)
            
            # Update best if improved
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = current_points.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Cooling schedule based on iteration progress
            temperature = max(min_temperature, 0.1 * (1 - iteration / max_iter))
            
            # Add some thermal noise to prevent sticking to local minima
            if stagnation_counter > 0 and stagnation_counter % 100 == 0:
                noise = np.random.normal(0, temperature, current_points.shape)
                current_points += noise
                current_points = self.project_to_sphere(current_points)
            
            # Early stopping if no significant improvement
            if stagnation_counter > max_stagnation:
                break

        return best_points, best_ratio

    def local_search_refinement(self, points: np.ndarray, iterations: int = 100) -> np.ndarray:
        """Perform local search refinement."""
        current_points = points.copy()
        current_ratio = self.compute_min_max_ratio(current_points)

        for _ in range(iterations):
            # Try small adjustments to each coordinate
            for i in range(len(current_points)):
                for j in range(3):
                    old_val = current_points[i, j]
                    for delta in [-0.0005, 0.0005]:
                        current_points[i, j] = old_val + delta
                        current_points[i] = self.project_to_sphere(current_points[i:i+1])[0]
                        new_ratio = self.compute_min_max_ratio(current_points)
                        if new_ratio > current_ratio:
                            current_ratio = new_ratio
                        else:
                            current_points[i, j] = old_val  # Revert

        return current_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = SphereVoronoiOptimizer()

    # Generate diverse initial configurations
    initial_configs = optimizer.generate_initial_configurations(5)

    best_overall_points = None
    best_overall_ratio = 0.0

    # Try each initial configuration
    for i, config in enumerate(initial_configs):
        # Apply Voronoi-based optimization
        optimized_points, final_ratio = optimizer.voronoi_based_optimization(config, 5000)

        if final_ratio > best_overall_ratio:
            best_overall_ratio = final_ratio
            best_overall_points = optimized_points.copy()

    # Final refinement with local search
    if best_overall_points is not None:
        best_overall_points = optimizer.local_search_refinement(best_overall_points, 200)
        final_ratio = optimizer.compute_min_max_ratio(best_overall_points)
        if final_ratio > best_overall_ratio:
            best_overall_ratio = final_ratio

    # L-BFGS refinement for final polish
    def objective_function(x_flat: np.ndarray) -> float:
        points = x_flat.reshape(-1, 3)
        return -optimizer.compute_min_max_ratio(points)

    try:
        x0 = best_overall_points.flatten()
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
        )

        refined_points = result.x.reshape(-1, 3)
        refined_points = optimizer.project_to_sphere(refined_points)
        ratio = optimizer.compute_min_max_ratio(refined_points)

        if ratio > best_overall_ratio:
            best_overall_points = refined_points
    except:
        pass

    return best_overall_points

# EVOLVE-BLOCK-END
