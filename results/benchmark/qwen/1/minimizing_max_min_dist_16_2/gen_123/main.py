# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Voronoi
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective(x):
        # Reshape into points
        points = x.reshape(-1, 2)

        # Calculate pairwise distances using squareform for numerical stability
        distances = squareform(pdist(points))

        # Set diagonal to large value to ignore self-distances
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return -1e10

        # Return negative ratio to minimize (we want to maximize ratio)
        return -d_min / d_max

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distance with explicit numerical stability."""
        if len(points) < 2:
            return 0.0

        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def initialize_points():
        """Initialize points using a hybrid structured approach for better starting configuration."""
        np.random.seed(42)
        
        # Create multiple candidate initializations
        candidates = []
        
        # 1. Hexagonal grid pattern with jitter
        grid_size = 4
        x = np.linspace(0.1, 0.9, grid_size)
        y = np.linspace(0.1, 0.9, grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        # Add small random perturbations
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        points = np.clip(points, 0, 1)
        candidates.append(points[:16])
        
        # 2. Fibonacci spiral pattern for better distribution
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(16):
            theta = 2 * np.pi * i / phi
            r = np.sqrt(i / 15) if i > 0 else 0.5
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            # Add noise for diversity
            x += np.random.normal(0, 0.005)
            y += np.random.normal(0, 0.005)
            points.append([x, y])
        candidates.append(np.array(points))
        
        # 3. Random initialization
        candidates.append(np.random.rand(16, 2))
        
        # Select the best candidate based on initial ratio
        best_candidate = candidates[0]
        best_ratio = calculate_min_max_ratio(best_candidate)
        
        for candidate in candidates[1:]:
            ratio = calculate_min_max_ratio(candidate)
            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate = candidate
                
        return best_candidate

    def voronoi_relaxation(points, max_iter=100, tolerance=1e-6):
        """Perform Voronoi relaxation to improve point distribution."""
        current_points = points.copy()
        
        for iteration in range(max_iter):
            try:
                # Compute Voronoi diagram
                vor = Voronoi(current_points)
                
                # Calculate new positions as centroids of Voronoi cells
                new_points = np.zeros_like(current_points)
                converged = True
                
                # Process each point
                for i in range(len(current_points)):
                    # Get vertices of Voronoi cell for point i
                    region = vor.regions[vor.point_region[i]]
                    
                    if -1 in region or len(region) < 3:
                        # Handle unbounded regions (use current position with slight adjustment)
                        new_points[i] = current_points[i] + np.random.normal(0, 0.001, 2)
                        continue
                        
                    # Extract vertices of the Voronoi cell
                    vertices = np.array([vor.vertices[j] for j in region if j >= 0])
                    
                    if len(vertices) < 3:
                        # Not enough vertices, use current position
                        new_points[i] = current_points[i]
                        continue
                        
                    # Compute centroid of polygon (Voronoi cell)
                    centroid = np.mean(vertices, axis=0)
                    
                    # Apply boundary constraints with epsilon padding
                    centroid = np.clip(centroid, 1e-8, 1-1e-8)
                    
                    # Update point position
                    new_points[i] = centroid
                    
                    # Check for convergence
                    if np.linalg.norm(new_points[i] - current_points[i]) > tolerance:
                        converged = False
                        
                # Apply damping factor for stable convergence
                damping = 0.9
                current_points = current_points + damping * (new_points - current_points)
                
                # Ensure points stay within bounds
                current_points = np.clip(current_points, 0, 1)
                
                # Early stopping if converged
                if converged:
                    break
                    
            except Exception:
                # If Voronoi computation fails, use simple perturbation
                current_points = current_points + np.random.normal(0, 0.001, current_points.shape)
                current_points = np.clip(current_points, 0, 1)
                
        return current_points

    # Set up bounds (0 to 1 for each coordinate)
    bounds = [(0, 1)] * 32

    # Phase 1: Initialize with hybrid approach
    best_points = initialize_points()
    best_ratio = calculate_min_max_ratio(best_points)

    # Phase 2: Voronoi relaxation for global improvement
    relaxed_points = voronoi_relaxation(best_points, max_iter=50, tolerance=1e-5)
    relaxed_ratio = calculate_min_max_ratio(relaxed_points)
    
    if relaxed_ratio > best_ratio:
        best_points = relaxed_points
        best_ratio = relaxed_ratio

    # Phase 3: Adaptive Differential Evolution with progressive tightening
    try:
        # Start with coarser parameters and progressively tighten
        de_params_history = [
            {'maxiter': 50, 'popsize': 20, 'atol': 1e-6, 'rtol': 1e-6, 'mutation': (0.8, 1.0), 'recombination': 0.8},
            {'maxiter': 100, 'popsize': 25, 'atol': 1e-8, 'rtol': 1e-8, 'mutation': (0.9, 1.0), 'recombination': 0.9},
            {'maxiter': 150, 'popsize': 30, 'atol': 1e-10, 'rtol': 1e-10, 'mutation': (0.95, 1.0), 'recombination': 0.95}
        ]
        
        for params in de_params_history:
            result_de = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=params['maxiter'],
                popsize=params['popsize'],
                atol=params['atol'],
                rtol=params['rtol'],
                mutation=params['mutation'],
                recombination=params['recombination'],
                disp=False
            )

            if result_de.success:
                # Phase 4: Progressive local refinement with varying tolerances
                refinement_strategies = [
                    {'method': 'L-BFGS-B', 'options': {'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}},
                    {'method': 'L-BFGS-B', 'options': {'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-14}},
                    {'method': 'SLSQP', 'options': {'maxiter': 100, 'ftol': 1e-13}}
                ]

                for strategy in refinement_strategies:
                    try:
                        refined = minimize(
                            objective,
                            result_de.x,
                            method=strategy['method'],
                            bounds=bounds,
                            options=strategy['options']
                        )

                        if refined.success:
                            final_points = refined.x.reshape(-1, 2)
                            ratio = calculate_min_max_ratio(final_points)

                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = final_points.copy()

                    except Exception:
                        continue

    except Exception:
        pass

    # Phase 5: Final fallback optimization if needed
    if best_ratio < 1e-8:  # If we still have a very poor solution
        try:
            # Try another round of differential evolution with different settings
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=30,
                atol=1e-12,
                rtol=1e-12,
                mutation=(0.7, 1.0),
                recombination=0.9,
                disp=False
            )

            if result.success:
                # Final local refinement
                refined = minimize(
                    objective,
                    result.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15}
                )

                if refined.success:
                    final_points = refined.x.reshape(-1, 2)
                    ratio = calculate_min_max_ratio(final_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()

        except Exception:
            pass

    # Ensure final points are within valid bounds
    best_points = np.clip(best_points, 0, 1)

    return best_points

# EVOLVE-BLOCK-END