# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import warnings
import time
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Calculate pairwise distances using squareform for numerical stability
        distances = pdist(points)

        # Handle edge cases
        if len(distances) == 0 or np.allclose(distances, 0):
            return 1e10  # Large penalty for invalid configurations

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero - return large penalty if no distance
        if max_dist <= 0:
            return 1e10

        # Return negative ratio (since we want to maximize ratio, we minimize its negative)
        return -min_dist / max_dist

    def constraint_func(x):
        # Ensure points are within [0+eps,1-eps] x [0+eps,1-eps] for numerical stability
        eps = 1e-8
        points = x.reshape(-1, 2)
        constraints = []

        # x coordinates in [eps, 1-eps]
        constraints.append(points[:, 0].min() - eps)  # x_min >= eps
        constraints.append(1 - eps - points[:, 0].max())  # x_max <= 1-eps

        # y coordinates in [eps, 1-eps]
        constraints.append(points[:, 1].min() - eps)  # y_min >= eps
        constraints.append(1 - eps - points[:, 1].max())  # y_max <= 1-eps

        return np.array(constraints)

    def bounded_objective(x):
        # Boundary checking with clamping to safe bounds
        eps = 1e-8
        points = np.clip(x.reshape(-1, 2), eps, 1-eps).flatten()
        return objective(points)

    def generate_hexagonal_voronoi_initial_config():
        """Generate an initial configuration inspired by hexagonal Voronoi tiling"""
        np.random.seed(42)
        points = []
        
        # Create a hexagonal pattern that approximates Voronoi optimality
        # Using 4x4 grid but with precise geometric spacing
        
        # Base spacing for hexagonal packing
        spacing_x = 0.8 / 3.0  # Spacing in x direction
        spacing_y = 0.8 * np.sqrt(3) / 3.0  # Spacing in y direction for hexagonal
        
        # Generate points in hexagonal pattern
        for i in range(4):
            for j in range(4):
                if len(points) >= 16:
                    break
                # Hexagonal offset for odd rows
                x_offset = 0.1 + j * spacing_x
                y_offset = 0.1 + i * spacing_y
                if i % 2 == 1:
                    x_offset += spacing_x / 2.0
                
                points.append([x_offset, y_offset])
        
        # Convert to numpy array and add small random jitter for symmetry breaking
        points = np.array(points[:16])
        points += np.random.normal(0, 0.005, points.shape)
        
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        
        return points

    def adaptive_voronoi_refinement(points, max_iter=100):
        """Refine points using Voronoi-based geometric insights"""
        refined_points = points.copy()
        
        # Use a simple gradient-like approach based on Voronoi cells
        for iteration in range(max_iter):
            try:
                # Calculate forces between points based on distance relationships
                new_positions = refined_points.copy()
                total_force = np.zeros_like(refined_points)
                
                # Simple repulsion model based on inverse square distance
                for i in range(len(refined_points)):
                    for j in range(len(refined_points)):
                        if i != j:
                            dist_vec = refined_points[i] - refined_points[j]
                            dist = np.linalg.norm(dist_vec)
                            if dist > 0.001:  # Avoid division by zero
                                force_magnitude = 1.0 / (dist * dist)
                                force_direction = dist_vec / dist
                                total_force[i] += force_magnitude * force_direction
                
                # Apply forces with damping
                learning_rate = 0.01
                new_positions += learning_rate * total_force
                
                # Project back to feasible region
                new_positions = np.clip(new_positions, 1e-8, 1-1e-8)
                
                # Check convergence
                if np.linalg.norm(new_positions - refined_points) < 1e-6:
                    break
                    
                refined_points = new_positions
                
            except Exception:
                # If calculation fails, fall back to simple optimization
                break
                
        return refined_points

    def multi_phase_optimization(initial_points):
        """Perform multi-phase optimization with Voronoi insights"""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = float('inf')
        
        # Phase 1: Global optimization with Differential Evolution
        try:
            x0 = current_points.flatten()
            bounds = [(1e-8, 1-1e-8) for _ in range(32)]
            
            de_result = differential_evolution(
                bounded_objective,
                bounds,
                seed=42,
                maxiter=30,  # Reduced for faster execution
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )
            
            if de_result.success:
                current_points = de_result.x.reshape(-1, 2)
                # Calculate ratio for this configuration
                distances = pdist(current_points)
                if len(distances) > 0 and np.max(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    ratio = min_dist / max_dist
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_points = current_points.copy()
                        
        except Exception as e:
            pass
            
        # Phase 2: Voronoi-based geometric refinement
        try:
            voronoi_refined = adaptive_voronoi_refinement(current_points, max_iter=50)
            # Calculate ratio for this configuration
            distances = pdist(voronoi_refined)
            if len(distances) > 0 and np.max(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                ratio = min_dist / max_dist
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_points = voronoi_refined.copy()
        except Exception as e:
            pass
            
        # Phase 3: Fine-tuning with local optimization
        try:
            # Use SLSQP for final refinement
            x0 = best_points.flatten()
            bounds = [(1e-8, 1-1e-8) for _ in range(32)]
            
            result = minimize(
                bounded_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-9},
                callback=None
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                # Calculate ratio for this configuration
                distances = pdist(final_points)
                if len(distances) > 0 and np.max(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    ratio = min_dist / max_dist
                    if ratio < best_ratio:
                        best_points = final_points.copy()
                        
        except Exception as e:
            pass
            
        return best_points

    # Generate sophisticated initial configuration based on hexagonal Voronoi principles
    initial_config = generate_hexagonal_voronoi_initial_config()

    # Apply multi-phase optimization with Voronoi insights
    optimized_points = multi_phase_optimization(initial_config)

    # Final safety check
    optimized_points = np.clip(optimized_points, 1e-8, 1-1e-8)

    return optimized_points


# EVOLVE-BLOCK-END