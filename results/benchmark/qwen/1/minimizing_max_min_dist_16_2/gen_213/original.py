# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import time
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape x into points
        points = x.reshape(-1, 2)

        # Add small padding to avoid boundary issues
        points = np.clip(points, 1e-8, 1-1e-8)

        # Compute full pairwise distance matrix for better numerical stability
        distance_matrix = squareform(pdist(points))

        # Zero out diagonal to avoid considering distance to self
        np.fill_diagonal(distance_matrix, np.inf)

        # Compute min and max distances from the full matrix
        d_min = np.min(distance_matrix)
        d_max = np.max(distance_matrix)

        # Avoid division by zero or invalid cases
        if d_max <= 1e-12:
            return -np.inf

        # Return negative ratio to maximize (since we're minimizing the negative)
        return -d_min / d_max

    def simulated_annealing(x0, objective, bounds, max_iter=1000, initial_temp=1.0):
        """Simple simulated annealing implementation"""
        current_x = x0.copy()
        current_energy = objective(current_x)
        best_x = current_x.copy()
        best_energy = current_energy

        temp = initial_temp
        cooling_rate = 0.995

        for i in range(max_iter):
            # Generate neighbor by adding small random perturbation
            neighbor_x = current_x + np.random.normal(0, 0.001, len(current_x))
            # Apply bounds
            neighbor_x = np.clip(neighbor_x, [b[0] for b in bounds], [b[1] for b in bounds])

            neighbor_energy = objective(neighbor_x)

            # Accept or reject the neighbor
            if neighbor_energy < current_energy or \
               np.random.rand() < np.exp(-(neighbor_energy - current_energy) / temp):
                current_x = neighbor_x
                current_energy = neighbor_energy

                if current_energy < best_energy:
                    best_x = current_x.copy()
                    best_energy = current_energy

            # Cool down temperature
            temp *= cooling_rate

        return best_x, best_energy

    def create_initial_guesses():
        """Create multiple diverse initial guesses"""
        initial_guesses = []

        # 1. Random initialization (already used)
        np.random.seed(42)
        random_points = np.random.rand(16, 2)
        initial_guesses.append(random_points.flatten())

        # 2. Golden spiral initialization for better distribution
        golden_angle = 2.399963229728653  # ~4π/(3+√5)
        points_spiral = []
        for i in range(16):
            radius = np.sqrt(i/15.0)  # Normalize to [0,1]
            angle = i * golden_angle
            x = 0.5 + radius * np.cos(angle) * 0.45
            y = 0.5 + radius * np.sin(angle) * 0.45
            points_spiral.append([x, y])
        initial_guesses.append(np.array(points_spiral).flatten())

        # 3. Hexagonal grid initialization (approximate)
        hex_points = []
        rows = 4
        cols = 4
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0

        for i in range(rows):
            for j in range(cols):
                if len(hex_points) >= 16:
                    break
                x = j * spacing_x
                y = i * spacing_y
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                hex_points.append([x, y])

        # Trim or pad to 16 points
        if len(hex_points) < 16:
            extra_points = np.random.rand(16 - len(hex_points), 2)
            hex_points.extend(extra_points.tolist())
        elif len(hex_points) > 16:
            hex_points = hex_points[:16]

        initial_guesses.append(np.array(hex_points).flatten())

        # 4. Perturbed grid initialization
        np.random.seed(123)
        grid_points = np.array([[i/3, j/3] for i in range(4) for j in range(4) if i*4+j < 16]).reshape(-1, 2)
        perturbed_grid = grid_points + np.random.normal(0, 0.05, (16, 2))
        perturbed_grid = np.clip(perturbed_grid, 0, 1)
        initial_guesses.append(perturbed_grid.flatten())

        # 5. Another random initialization with different seed
        np.random.seed(246)
        random_points_2 = np.random.rand(16, 2)
        initial_guesses.append(random_points_2.flatten())

        return initial_guesses

    # Define bounds for each coordinate (0 to 1 for both x and y) with small padding
    bounds = [(1e-8, 1-1e-8) for _ in range(32)]  # 16 points * 2 coordinates each

    # Create diverse initial guesses
    initial_guesses = create_initial_guesses()

    best_result = None
    best_ratio = -np.inf

    # Multi-start approach with different initializations
    for i, initial_guess in enumerate(initial_guesses):
        try:
            # Use differential evolution for global search
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42 + i,
                maxiter=250,      # Increased iterations
                popsize=40,       # Larger population size
                tol=1e-9,         # Tighter tolerance
                recombination=0.9, # Higher recombination rate
                mutation=(0.8, 1.0), # Different mutation strategy
                disp=False
            )

            # Local refinement with adaptive stopping criteria
            local_result = minimize(
                objective,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000}
            )

            # If L-BFGS-B fails, try SLSQP as backup
            if not local_result.success:
                local_result = minimize(
                    objective,
                    de_result.x,
                    method='SLSQP',
                    bounds=bounds,
                    options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000}
                )

            # If both fail, try simulated annealing as final fallback
            if not local_result.success:
                try:
                    sa_result, sa_energy = simulated_annealing(
                        de_result.x,
                        objective,
                        bounds,
                        max_iter=500,
                        initial_temp=0.1
                    )
                    local_result = type('obj', (object,), {'x': sa_result, 'fun': sa_energy, 'success': True})()
                except:
                    pass

            # Keep track of the best solution found
            if -local_result.fun > best_ratio:
                best_ratio = -local_result.fun
                best_result = local_result.x

        except Exception as e:
            # Skip failed optimizations and continue with others
            continue

    # Final refinement with enhanced settings if we found a good candidate
    if best_result is not None:
        final_result = minimize(
            objective,
            best_result,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-13, 'gtol': 1e-13, 'maxiter': 1500}
        )

        # If L-BFGS fails, try SLSQP as final fallback
        if not final_result.success:
            final_result = minimize(
                objective,
                best_result,
                method='SLSQP',
                bounds=bounds,
                options={'ftol': 1e-13, 'gtol': 1e-13, 'maxiter': 1500}
            )

        # If both still fail, try simulated annealing as ultimate fallback
        if not final_result.success:
            try:
                sa_result, sa_energy = simulated_annealing(
                    best_result,
                    objective,
                    bounds,
                    max_iter=1000,
                    initial_temp=0.1
                )
                final_result = type('obj', (object,), {'x': sa_result, 'fun': sa_energy, 'success': True})()
            except:
                pass

        points = final_result.x.reshape(-1, 2)
    else:
        # Fallback to best initial guess if everything else failed
        points = initial_guesses[0].reshape(-1, 2)

    # Ensure all points are within [0,1]^2 bounds (final safeguard)
    points = np.clip(points, 0, 1)

    return points


# EVOLVE-BLOCK-END