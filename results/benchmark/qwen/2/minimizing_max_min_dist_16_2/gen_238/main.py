# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given point configuration."""
        # Ensure points are within unit square
        points = np.clip(points, 0, 1)

        # Compute pairwise distances
        distances = squareform(pdist(points))

        # Set diagonal to large value so it doesn't affect min
        np.fill_diagonal(distances, np.inf)

        # Get minimum and maximum distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return ratio
        if max_dist > 0:
            return min_dist / max_dist
        else:
            return 0.0

    def objective_function(points_flat):
        # Reshape flat array back to 16x2 points
        points = points_flat.reshape(-1, 2)

        # Ensure points are within unit square
        points = np.clip(points, 0, 1)

        # Compute pairwise distances
        distances = squareform(pdist(points))

        # Set diagonal to large value so it doesn't affect min
        np.fill_diagonal(distances, np.inf)

        # Get minimum and maximum distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio since we want to maximize
        if max_dist > 0:
            return -min_dist / max_dist
        else:
            return -np.inf

    # Initialize with enhanced spherical-inspired arrangement
    np.random.seed(42)

    # Create a more sophisticated initial configuration inspired by spherical point distributions
    # Use golden spiral pattern for more uniform coverage
    points = []
    n_points = 16

    # Golden spiral approach for better point distribution
    phi = (1 + np.sqrt(5)) / 2  # golden ratio

    for i in range(n_points):
        # Map to 2D square using golden spiral projection
        # This gives a more uniform distribution than simple grid
        theta = 2 * np.pi * i / phi
        radius = np.sqrt(i / (n_points - 1)) if i < n_points - 1 else 1.0

        # Convert to Cartesian coordinates in [0,1] range
        x = 0.5 + radius * np.cos(theta) * 0.4
        y = 0.5 + radius * np.sin(theta) * 0.4

        # Add small random perturbation to break symmetry
        x += (np.random.random() - 0.5) * 0.08
        y += (np.random.random() - 0.5) * 0.08

        points.append([x, y])

    points = np.array(points)

    # Clip to unit square
    points = np.clip(points, 0, 1)

    # Fix corner points to break symmetry and avoid degenerate solutions
    points[0] = [0.0, 0.0]  # Bottom-left corner
    points[3] = [1.0, 0.0]  # Bottom-right corner
    points[12] = [0.0, 1.0] # Top-left corner
    points[15] = [1.0, 1.0] # Top-right corner

    # Multi-start optimization with differential evolution
    best_points = points.copy()
    best_ratio = compute_min_max_ratio(points)

    # Try multiple differential evolution restarts with varying perturbations
    bounds = [(0, 1) for _ in range(32)]

    # Run multiple optimization restarts with different seeds
    for restart in range(5):
        # Create slightly different initial configuration for each restart
        restart_points = points.copy()

        # Add small random perturbations to all points
        for i in range(16):
            restart_points[i, 0] += (np.random.random() - 0.5) * 0.02
            restart_points[i, 1] += (np.random.random() - 0.5) * 0.02

        # Clip to unit square
        restart_points = np.clip(restart_points, 0, 1)

        # Fix corners again after perturbations
        restart_points[0] = [0.0, 0.0]
        restart_points[3] = [1.0, 0.0]
        restart_points[12] = [0.0, 1.0]
        restart_points[15] = [1.0, 1.0]

        # Run differential evolution
        try:
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42 + restart,
                disp=False
            )

            # Evaluate DE result
            de_ratio = -objective_function(de_result.x)
            if de_ratio > best_ratio:
                best_ratio = de_ratio
                best_points = de_result.x.reshape(-1, 2)
                best_points = np.clip(best_points, 0, 1)

                # Fix corners in the best solution
                best_points[0] = [0.0, 0.0]
                best_points[3] = [1.0, 0.0]
                best_points[12] = [0.0, 1.0]
                best_points[15] = [1.0, 1.0]
        except:
            continue

    # Refine with local optimization using L-BFGS-B
    try:
        refined_result = minimize(
            objective_function,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
        )

        final_points = refined_result.x.reshape(-1, 2)
        final_points = np.clip(final_points, 0, 1)

        # Fix corners in final solution
        final_points[0] = [0.0, 0.0]
        final_points[3] = [1.0, 0.0]
        final_points[12] = [0.0, 1.0]
        final_points[15] = [1.0, 1.0]

        final_ratio = compute_min_max_ratio(final_points)
        if final_ratio > best_ratio:
            best_points = final_points
    except:
        pass

    # Apply a final progressive refinement step to further improve the solution
    current_points = best_points.copy()
    current_ratio = best_ratio

    # Progressive refinement using local search with fine-grained movements
    improvement_threshold = 1e-8
    iteration_limit = 50

    for iteration in range(iteration_limit):
        improved = False
        # Try moving each point to see if we can improve the ratio
        for point_idx in range(16):
            original_point = current_points[point_idx].copy()
            best_move_ratio = current_ratio
            best_move_position = original_point.copy()

            # Sample potential moves more densely around current position
            for dx in np.linspace(-0.03, 0.03, 7):
                for dy in np.linspace(-0.03, 0.03, 7):
                    new_x = original_point[0] + dx
                    new_y = original_point[1] + dy

                    new_x = np.clip(new_x, 0, 1)
                    new_y = np.clip(new_y, 0, 1)

                    # Temporarily move the point
                    current_points[point_idx] = [new_x, new_y]

                    # Check the resulting ratio
                    new_ratio = compute_min_max_ratio(current_points)

                    if new_ratio > best_move_ratio:
                        best_move_ratio = new_ratio
                        best_move_position = [new_x, new_y]

                    # Restore original position
                    current_points[point_idx] = original_point

            # Apply the best move found
            if best_move_ratio > current_ratio:
                current_points[point_idx] = best_move_position
                current_ratio = best_move_ratio
                improved = True

                # Update our best solution if this is better
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()

        # If no improvement was made, stop early
        if not improved:
            break

    return best_points

# EVOLVE-BLOCK-END