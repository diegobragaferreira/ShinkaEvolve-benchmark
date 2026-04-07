# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import math
import time
import warnings

class HybridPointOptimizer:
    """Hybrid optimizer combining geometric principles and evolutionary algorithms."""

    def __init__(self, n_points=16, dimensions=2, seed=42, max_time_seconds=180):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        self.max_time_seconds = max_time_seconds
        np.random.seed(seed)

    def compute_distance_matrix(self, points):
        """Compute pairwise distance matrix for given points."""
        return squareform(pdist(points))

    def calculate_min_max_ratio(self, distance_matrix):
        """Calculate the ratio of minimum to maximum distances."""
        off_diagonal = distance_matrix[distance_matrix > 0]
        if len(off_diagonal) == 0:
            return 0.0
        d_min = np.min(off_diagonal)
        d_max = np.max(off_diagonal)
        return d_min / d_max if d_max > 0 else 0.0

    def _generate_hexagonal_lattice(self):
        """Generate precise hexagonal lattice with optimal spacing."""
        sqrt3 = math.sqrt(3)
        row_spacing = sqrt3 / 2
        col_spacing = 1.0
        
        points = []
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.n_points:
                    break
                x = j * col_spacing + (i % 2) * col_spacing / 2
                y = i * row_spacing
                points.append([x, y])
            if len(points) >= self.n_points:
                break

        points = np.array(points[:self.n_points])

        # Normalize to [0,1] with proper scaling and centering
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])

            if x_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
            if y_range > 0:
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

            # Scale and center properly
            scale_factor = 0.9
            center_x = np.mean(points[:, 0])
            center_y = np.mean(points[:, 1])

            points[:, 0] = 0.05 + scale_factor * (points[:, 0] - center_x) + 0.5
            points[:, 1] = 0.05 + scale_factor * (points[:, 1] - center_y) + 0.5

        return points

    def _generate_grid_points(self):
        """Generate points in a structured grid pattern."""
        n_per_side = int(np.ceil(np.sqrt(self.n_points)))
        x = np.linspace(0.1, 0.9, n_per_side)
        y = np.linspace(0.1, 0.9, n_per_side)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])[:self.n_points]
        return points

    def _generate_random_points(self):
        """Generate random points."""
        return np.random.rand(self.n_points, self.dimensions)

    def _generate_perturbed_points(self, base_points, perturbation_magnitude=0.02):
        """Add controlled perturbations to existing points."""
        perturbed = base_points + np.random.normal(0, perturbation_magnitude, base_points.shape)
        return np.clip(perturbed, 0, 1)

    def _generate_symmetry_breaking_points(self, base_points):
        """Apply sophisticated symmetry breaking to base points."""
        points = base_points.copy()
        
        # Apply different perturbations based on position
        np.random.seed(self.seed)
        
        for i in range(len(points)):
            # Use position-based perturbation patterns
            angle = i * 0.785398  # pi/4 increments
            noise_intensity = 0.008 + 0.003 * math.sin(angle)
            noise_x = np.random.normal(0, noise_intensity, 1)[0]
            noise_y = np.random.normal(0, noise_intensity, 1)[0]
            points[i] += [noise_x, noise_y]
        
        # Additional corner-specific perturbations
        corner_indices = [0, 3, 12, 15]  # Four corners of 4x4 grid
        for idx in corner_indices:
            if idx < len(points):
                points[idx] += np.random.normal(0, 0.015, 2)
        
        return np.clip(points, 0, 1)

    def _adaptive_local_optimization(self, initial_points, max_iter=500):
        """Apply adaptive local optimization with geometric constraints."""
        current_points = initial_points.copy()
        
        for iteration in range(max_iter):
            try:
                dist_matrix = self.compute_distance_matrix(current_points)
                ratio = self.calculate_min_max_ratio(dist_matrix)
                
                if ratio < 1e-10:
                    break
                    
            except Exception:
                break
            
            # For each point, compute optimal adjustment direction
            new_points = current_points.copy()
            updated = False
            
            for i in range(len(current_points)):
                original_point = current_points[i].copy()
                
                # Calculate gradient using finite differences
                best_direction = None
                best_improvement = 0
                
                # Try several directions for small perturbations
                directions = [
                    [0.001, 0], [0, 0.001], [-0.001, 0], [0, -0.001],
                    [0.000707, 0.000707], [-0.000707, 0.000707],
                    [0.000707, -0.000707], [-0.000707, -0.000707]
                ]
                
                for dx, dy in directions:
                    test_point = original_point + [dx, dy]
                    test_point = np.clip(test_point, 0, 1)
                    
                    # Create test configuration
                    test_points = current_points.copy()
                    test_points[i] = test_point
                    
                    try:
                        test_dist_matrix = self.compute_distance_matrix(test_points)
                        test_ratio = self.calculate_min_max_ratio(test_dist_matrix)
                        
                        if test_ratio > ratio + best_improvement:
                            best_improvement = test_ratio - ratio
                            best_direction = [dx, dy]
                            
                    except Exception:
                        continue
                
                # Apply best direction if beneficial
                if best_direction is not None and best_improvement > 1e-12:
                    new_points[i] = original_point + best_direction
                    updated = True
            
            # Update points if any improvements were made
            if updated:
                current_points = new_points.copy()
            else:
                break
        
        return current_points

    def _multi_start_optimization(self, max_iter=1000):
        """Perform multi-start optimization from various initial configurations."""
        best_points = None
        best_ratio = -np.inf
        start_time = time.time()

        # Generate diverse initial configurations
        initial_configs = []
        
        # Hexagonal lattice with symmetry breaking
        hex_points = self._generate_hexagonal_lattice()
        hex_points = self._generate_symmetry_breaking_points(hex_points)
        initial_configs.append(hex_points)
        
        # Grid-based with perturbations
        grid_points = self._generate_grid_points()
        grid_points = self._generate_perturbed_points(grid_points, 0.03)
        initial_configs.append(grid_points)
        
        # Random with geometric constraints
        rand_points = self._generate_random_points()
        rand_points = self._generate_perturbed_points(rand_points, 0.05)
        initial_configs.append(rand_points)
        
        # Additional hexagonal variations
        hex2_points = self._generate_hexagonal_lattice()
        hex2_points = self._generate_perturbed_points(hex2_points, 0.02)
        initial_configs.append(hex2_points)

        # Try multiple initial configurations with differential evolution
        for i, initial_config in enumerate(initial_configs):
            if time.time() - start_time > self.max_time_seconds - 10:
                break
                
            try:
                # First try differential evolution for global search
                def objective_function(x_flat):
                    points = x_flat.reshape(-1, self.dimensions)
                    points = np.clip(points, 0, 1)
                    
                    try:
                        distances = pdist(points)
                        d_min = np.min(distances)
                        d_max = np.max(distances)
                        
                        if d_max == 0:
                            return float('inf')
                            
                        return -(d_min / d_max)
                    except Exception:
                        return 1e6
                
                bounds = [(0, 1)] * (self.n_points * self.dimensions)
                
                try:
                    result = differential_evolution(
                        objective_function,
                        bounds,
                        maxiter=200,
                        popsize=15,
                        tol=1e-8,
                        seed=self.seed + i,
                        disp=False
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, self.dimensions)
                        dist_matrix = self.compute_distance_matrix(final_points)
                        ratio = self.calculate_min_max_ratio(dist_matrix)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                except Exception:
                    pass
                
                # If DE failed, use local refinement
                if best_points is None or time.time() - start_time > self.max_time_seconds - 10:
                    try:
                        # Local refinement with L-BFGS-B
                        def objective(x_flat):
                            points = x_flat.reshape(-1, self.dimensions)
                            points = np.clip(points, 0, 1)
                            
                            try:
                                dist_matrix = self.compute_distance_matrix(points)
                                ratio = self.calculate_min_max_ratio(dist_matrix)
                                return -ratio
                            except Exception:
                                return 1e6
                        
                        x0 = initial_config.flatten()
                        bounds = [(0, 1) for _ in range(len(x0))]
                        
                        result = minimize(
                            objective,
                            x0,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': max_iter//2, 'ftol': 1e-10, 'gtol': 1e-10}
                        )
                        
                        if result.success:
                            final_points = result.x.reshape(-1, self.dimensions)
                            dist_matrix = self.compute_distance_matrix(final_points)
                            ratio = self.calculate_min_max_ratio(dist_matrix)
                            
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = final_points.copy()
                    except Exception:
                        continue
                        
            except Exception as e:
                warnings.warn(f"Error in optimization round {i}: {str(e)}")
                continue

        # Final geometric refinement if we have a solution
        if best_points is not None and time.time() - start_time < self.max_time_seconds - 5:
            try:
                refined_points = self._adaptive_local_optimization(best_points, max_iter=200)
                dist_matrix = self.compute_distance_matrix(refined_points)
                final_ratio = self.calculate_min_max_ratio(dist_matrix)
                
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = refined_points
            except Exception:
                pass

        return best_points if best_points is not None else initial_configs[0]

    def optimize(self):
        """Main optimization routine."""
        try:
            # Perform multi-start optimization
            best_points = self._multi_start_optimization(max_iter=800)
            
            # Final validation and cleanup
            if best_points is not None:
                # Ensure all points are within bounds
                best_points = np.clip(best_points, 0, 1)
                
                # Final refinement
                final_points = self._adaptive_local_optimization(best_points, max_iter=100)
                final_points = np.clip(final_points, 0, 1)
                
                return final_points
            else:
                # Fallback to random initialization
                return self._generate_random_points()
                
        except Exception as e:
            warnings.warn(f"Optimization failed completely: {str(e)}")
            # Final fallback
            return self._generate_random_points()

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = HybridPointOptimizer(n_points=16, dimensions=2, seed=42, max_time_seconds=180)
    points = optimizer.optimize()
    return points

# EVOLVE-BLOCK-END