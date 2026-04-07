# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective(x):
        # Reshape x into 16 points
        points = x.reshape(-1, 2)
        # Calculate pairwise distances
        distances = pdist(points)
        # Avoid division by zero
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        # Minimize negative of min/max ratio (equivalent to maximizing min/max ratio)
        return -min_dist / max_dist

    def generate_structured_grid_initial():
        """Generate a structured 4x4 grid with adaptive perturbation based on distance distribution"""
        # Create a regular 4x4 grid
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                points.append([x, y])

        points = np.array(points)

        # Calculate current distance distribution
        distances = pdist(points)
        if len(distances) > 0:
            current_ratio = np.min(distances) / np.max(distances) if np.max(distances) > 0 else 0

            # Adjust perturbation magnitude based on how balanced the distance distribution is
            # If distances are already well-balanced, perturb less; otherwise perturb more
            perturbation_magnitude = max(0.01, 0.05 * (1.0 - current_ratio * 10)) if current_ratio > 0 else 0.03

            # Add adaptive perturbation
            np.random.seed(42)
            perturbation = np.random.normal(0, perturbation_magnitude, points.shape)
            points += perturbation

            # Clip to valid range
            points = np.clip(points, 0.001, 0.999)

        # Apply symmetry-breaking constraints by fixing corner points
        # Fix bottom-left corner
        points[0] = [0.1, 0.1]
        # Fix bottom-right corner
        points[3] = [0.9, 0.1]
        # Fix top-left corner
        points[12] = [0.1, 0.9]
        # Fix top-right corner
        points[15] = [0.9, 0.9]

        return points

    def generate_fibonacci_spiral():
        """Generate points using Fibonacci spiral for good distribution"""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        for i in range(16):
            theta = math.acos(-1 + (2 * i) / 15)  # elevation angle
            phi_angle = (i * 2 * math.pi) / (phi * phi)  # azimuthal angle

            # Convert to cartesian coordinates
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)

            # Map to [0.05, 0.95] range to avoid boundaries
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2

            points.append([x, y])

        return np.array(points)

    def generate_perturbed_grid_initial():
        """Generate a perturbed regular grid initial configuration"""
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                points.append([x, y])

        points = np.array(points)

        # Add controlled random perturbation
        np.random.seed(42)
        perturbation = np.random.normal(0, 0.02, points.shape)
        points += perturbation

        # Clip to valid range
        points = np.clip(points, 0.001, 0.999)

        # Apply symmetry-breaking constraints by fixing corner points
        # Fix bottom-left corner
        points[0] = [0.1, 0.1]
        # Fix bottom-right corner
        points[3] = [0.9, 0.1]
        # Fix top-left corner
        points[12] = [0.1, 0.9]
        # Fix top-right corner
        points[15] = [0.9, 0.9]

        return points

    def generate_ring_initial():
        """Generate points in concentric rings for better coverage"""
        points = []
        # Two concentric rings
        radii = [0.3, 0.7]
        angles_per_ring = [8, 8]  # 8 points per ring

        for r_idx, (radius, num_angles) in enumerate(zip(radii, angles_per_ring)):
            for i in range(num_angles):
                angle = 2 * math.pi * i / num_angles
                x = 0.5 + radius * math.cos(angle) * 0.4
                y = 0.5 + radius * math.sin(angle) * 0.4
                # Ensure within bounds
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))
                points.append([x, y])

        points = np.array(points)

        # Apply symmetry-breaking constraints by fixing some key points
        # Fix first point (could be any point to break rotation symmetry)
        points[0] = [0.1, 0.1]
        # Fix a point near the middle of first ring
        points[4] = [0.5, 0.1]  # Point on the left side of first ring

        return points

    def hierarchical_optimization(bounds, maxiter_global=30, maxiter_local=100):
        """
        Hierarchical optimization: first coarse search, then fine refinement
        """
        # Phase 1: Coarse optimization with 2x2 grid
        coarse_points = []
        for i in range(2):
            for j in range(2):
                # Sample points in a coarse grid pattern
                x = 0.25 + i * 0.5
                y = 0.25 + j * 0.5
                coarse_points.append([x, y])

        # Add 4 more points around the edges to explore boundaries
        edge_points = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        coarse_points.extend(edge_points)

        coarse_points = np.array(coarse_points[:8])  # Use first 8 points for 2x2 grid

        # Add perturbations for diversity
        np.random.seed(42)
        perturbation = np.random.normal(0, 0.05, coarse_points.shape)
        coarse_points += perturbation
        coarse_points = np.clip(coarse_points, 0.001, 0.999)

        # Create 16-point configuration from coarse points
        final_coarse_config = np.tile(coarse_points, (2, 1))[:16]

        # Phase 2: Fine grid search around promising areas
        fine_configs = []
        # Generate 4 different fine grid configurations
        for offset in [0, 0.05, 0.1, 0.15]:
            fine_point = np.array([
                [offset, offset],
                [offset, 0.5-offset],
                [0.5-offset, offset],
                [0.5-offset, 0.5-offset]
            ])
            fine_point = np.tile(fine_point, (4, 1))[:16]
            fine_configs.append(fine_point)

        # Combine with more diverse initial configurations
        initial_configs = [generate_structured_grid_initial(),
                          generate_fibonacci_spiral(),
                          generate_perturbed_grid_initial(),
                          generate_ring_initial()]

        # Add fine grid configurations
        initial_configs.extend(fine_configs)

        # Add aggressive perturbations
        np.random.seed(42)
        aggressive_perturbed = generate_perturbed_grid_initial() + np.random.normal(0, 0.05, (16, 2))
        aggressive_perturbed = np.clip(aggressive_perturbed, 0.001, 0.999)
        initial_configs.append(aggressive_perturbed)

        # Try optimization from different starting points with improved strategy
        best_ratio = -np.inf
        best_points = None

        # Multi-stage optimization approach with enhanced strategy
        for i, initial_config in enumerate(initial_configs):
            # Early exit if we've already achieved a very good solution
            if best_ratio > 0.25:  # Early stopping threshold
                break

            try:
                # Stage 1: Global optimization with differential evolution
                de_result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=maxiter_global,  # Reduced iterations for speed
                    popsize=8,   # Smaller population for faster execution
                    seed=42+i,   # Different seed for each config
                    tol=1e-6,
                    mutation=(0.5, 1),
                    recombination=0.7
                )

                # Stage 2: Hybrid local refinement
                refined_x = hybrid_local_search(de_result.x, bounds, maxiter=maxiter_local)

                # Check the refined result
                final_points = refined_x.reshape(-1, 2)
                distances = pdist(final_points)

                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)

                    if max_dist > 0:
                        ratio = min_dist / max_dist

                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()

            except Exception as e:
                # If optimization fails, continue to next initial config
                continue

        return best_points if best_points is not None else initial_configs[-1]

    def hybrid_local_search(x0, bounds, maxiter=50):
        """
        Hybrid local search combining simulated annealing and gradient-based refinement
        """
        # First try L-BFGS-B with symmetry constraints
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-9, 'gtol': 1e-9}
            )
            if result.success:
                return result.x
        except:
            pass

        # If that fails, try the original x0 as fallback
        return x0

    def apply_symmetry_constraints(points):
        """
        Apply fixed positions to break symmetry in the final solution
        """
        # Fix the same corner points as in initialization to maintain consistency
        points[0] = [0.1, 0.1]      # bottom-left
        points[3] = [0.9, 0.1]      # bottom-right
        points[12] = [0.1, 0.9]     # top-left
        points[15] = [0.9, 0.9]     # top-right
        return points

    # Define bounds for coordinates
    bounds = [(0.001, 0.999) for _ in range(32)]

    # Use hierarchical optimization instead of direct multi-start
    best_points = hierarchical_optimization(bounds, maxiter_global=20, maxiter_local=80)

    # Apply final symmetry constraints to ensure consistent symmetry breaking
    if best_points is not None:
        best_points = apply_symmetry_constraints(best_points)

    # If no good solution was found, return the last attempted configuration
    if best_points is None:
        # Fallback to the best performing initial configuration
        fallback_points = generate_structured_grid_initial()
        # Add small random noise to break symmetry
        np.random.seed(42)
        fallback_points += np.random.normal(0, 0.01, fallback_points.shape)
        fallback_points = np.clip(fallback_points, 0.001, 0.999)
        best_points = fallback_points

    return best_points

# EVOLVE-BLOCK-END