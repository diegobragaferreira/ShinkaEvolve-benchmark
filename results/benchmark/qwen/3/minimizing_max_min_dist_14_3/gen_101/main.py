# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import time
from numba import jit
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_distances_numba(points):
    """Fast distance computation using numba for better performance"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(3):
                diff = points[i,k] - points[j,k]
                dist += diff * diff
            dist = np.sqrt(dist)
            distances[i,j] = dist
            distances[j,i] = dist
    return distances

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 3)
        
        # Calculate distance matrix efficiently
        distances = cdist(points, points)
        
        # Set diagonal to large value to ignore self-distances
        np.fill_diagonal(distances, np.inf)

        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio to maximize (since we minimize in scipy)
        # Handle case where max_dist is 0 (should never happen with distinct points)
        if max_dist > 1e-10:
            return -min_dist / max_dist
        else:
            return 0

    def fibonacci_sphere_points(n):
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(n):
            # Latitude
            phi = np.arccos(1 - 2*i/(n-1))
            # Longitude
            theta = 2 * np.pi * i / golden_ratio

            # Convert to Cartesian coordinates
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            points.append([x, y, z])

        return np.array(points)

    def spherical_pack_initialization():
        """Initialize using principles from spherical packing theory"""
        # Start with Fibonacci point distribution on sphere
        fib_points = fibonacci_sphere_points(14)
        
        # Scale and position to be in [0,1]^3
        # Normalize to unit sphere
        norms = np.linalg.norm(fib_points, axis=1, keepdims=True)
        fib_points = fib_points / norms
        
        # Scale to approximately unit diameter and center in [0,1]^3
        fib_points *= 0.45  # Scale to make sure they fit nicely
        fib_points = fib_points + 0.5  # Center around origin
        
        # Add some symmetry breaking noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, fib_points.shape)
        fib_points += noise
        
        # Ensure all points are within bounds
        fib_points = np.clip(fib_points, 0, 1)
        
        return fib_points

    def constrained_evolution_search(initial_points):
        """Use evolutionary algorithm with geometric constraints"""
        # Define bounds for each coordinate [0, 1]
        bounds = [(0, 1)] * 14 * 3
        
        # Custom fitness function that also penalizes bad configurations
        def fitness_with_penalty(x):
            points = x.reshape(-1, 3)
            
            # Calculate distances
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)
            
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist <= 1e-10:
                return 1e10  # Penalty for invalid configuration
                
            ratio = min_dist / max_dist
            
            # Add geometric penalty for points that are too close together
            # This encourages more evenly spread points
            penalty = 0
            for i in range(14):
                for j in range(i+1, 14):
                    dist = distances[i,j]
                    if dist < 0.1:  # Very close points penalize heavily
                        penalty += 1000 * (0.1 - dist)**2
                        
            return -(ratio - penalty * 1e-6)  # Negative because we minimize
        
        # Use differential evolution with good parameter choices for this problem
        try:
            result = differential_evolution(
                fitness_with_penalty,
                bounds,
                seed=42,
                maxiter=200,
                popsize=25,
                mutation=(0.5, 1.0),
                recombination=0.7,
                tol=1e-12,
                callback=None
            )
            
            # Return optimized points
            return result.x.reshape(-1, 3)
            
        except Exception as e:
            return initial_points

    def local_improvement_refinement(points, max_iter=100):
        """Apply sophisticated local refinement to polish the solution"""
        current_points = points.copy()
        
        # Try multiple optimization approaches
        for attempt in range(3):
            try:
                # Slight random perturbation for diversity
                np.random.seed(42 + attempt)
                perturbed = current_points + np.random.normal(0, 0.005, current_points.shape)
                perturbed = np.clip(perturbed, 0, 1)
                
                # Use L-BFGS-B with very tight tolerances
                x0_refine = perturbed.flatten()

                def obj_for_lbfgs(x):
                    points_refined = x.reshape(-1, 3)
                    distances = cdist(points_refined, points_refined)
                    np.fill_diagonal(distances, np.inf)

                    min_dist = np.min(distances)
                    max_dist = np.max(distances)

                    if max_dist > 1e-10:
                        return -min_dist / max_dist
                    else:
                        return 0

                result_refine = minimize(
                    obj_for_lbfgs,
                    x0_refine,
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * 42,
                    options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000},
                    tol=1e-12
                )

                refined_points = result_refine.x.reshape(-1, 3)
                refined_points = np.clip(refined_points, 0, 1)

                # Calculate ratio for refined points
                distances = cdist(refined_points, refined_points)
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)

                if max_dist > 1e-10:
                    ratio = min_dist / max_dist
                    current_ratio = np.min(cdist(current_points, current_points)[np.triu_indices(14,1)]) / \
                                   np.max(cdist(current_points, current_points)[np.triu_indices(14,1)])
                    
                    if ratio > current_ratio:
                        current_points = refined_points
                        
            except:
                continue
                
        return current_points

    # Phase 1: Initialize with principled spherical packing approach
    initial_points = spherical_pack_initialization()
    
    # Phase 2: Global optimization with evolutionary algorithm
    evolved_points = constrained_evolution_search(initial_points)
    
    # Phase 3: Local refinement to polish
    refined_points = local_improvement_refinement(evolved_points)
    
    # Phase 4: Final validation with bounds checking and small additional local search
    final_points = np.clip(refined_points, 0, 1)
    
    # Do one final quick local optimization
    try:
        x0_final = final_points.flatten()
        
        def final_obj(x):
            points_final = x.reshape(-1, 3)
            distances = cdist(points_final, points_final)
            np.fill_diagonal(distances, np.inf)
            
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist > 1e-10:
                return -min_dist / max_dist
            else:
                return 0
                
        final_result = minimize(
            final_obj,
            x0_final,
            method='L-BFGS-B',
            bounds=[(0, 1)] * 42,
            options={'ftol': 1e-15, 'gtol': 1e-15, 'maxiter': 500},
            tol=1e-15
        )
        
        final_points = final_result.x.reshape(-1, 3)
        final_points = np.clip(final_points, 0, 1)
        
    except:
        pass

    return final_points

# EVOLVE-BLOCK-END
