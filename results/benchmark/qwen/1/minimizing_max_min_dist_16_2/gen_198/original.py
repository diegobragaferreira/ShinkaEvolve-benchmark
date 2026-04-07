# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    np.random.seed(42)

    def compute_ratio(points):
        """Compute min/max distance ratio for given point configuration."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = squareform(pdist(points))

        # Mask diagonal elements (distance to self is 0)
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Handle case where all points might be coincident
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def initialize_geometric_distribution():
        """Create initial point distribution using a more sophisticated geometric approach."""
        # Start with a 4x4 grid pattern
        points = np.zeros((16, 2))

        # Create roughly hexagonal-like structure with specific spacing
        row_indices = []
        col_indices = []

        # Generate pattern that avoids regular grid clustering
        for i in range(4):
            for j in range(4):
                row_indices.append(i)
                col_indices.append(j)

        # Apply offset pattern for better distribution
        for k in range(16):
            i, j = row_indices[k], col_indices[k]
            # Offset every other row for better hexagonal packing
            x = j * 0.25 + (i % 2) * 0.125
            y = i * 0.25

            # Add small perturbation to avoid perfect grid formation
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)

            points[k] = [x, y]

        # Normalize to [0.1, 0.9] range
        points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.8 + 0.1
        points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.8 + 0.1

        # Apply slight random perturbations to avoid degeneracy
        noise = np.random.normal(0, 0.005, points.shape)
        points += noise

        # Clamp to bounds
        points = np.clip(points, 0.01, 0.99)

        return points

    def geometric_particle_swarm_optimization(points, max_iter=5000):
        """Custom geometric PSO that uses distance-based guidance."""
        n_points = len(points)
        dim = 2

        # Initialize particles
        particles = [points.copy()]
        velocities = [np.zeros_like(points)]

        # Best positions
        personal_best = [points.copy()]
        personal_best_scores = [compute_ratio(points)]

        # Global best
        global_best = points.copy()
        global_best_score = compute_ratio(points)

        # Parameters
        c1, c2 = 2.0, 2.0  # Cognitive and social coefficients
        w_start, w_end = 0.9, 0.4  # Inertia weight
        max_velocity = 0.1

        for iter_num in range(max_iter):
            # Adaptive parameters
            w = w_start - (w_start - w_end) * iter_num / max_iter

            # Evaluate all particles
            for i, particle in enumerate(particles):
                score = compute_ratio(particle)

                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best[i] = particle.copy()
                    personal_best_scores[i] = score

                    # Update global best
                    if score > global_best_score:
                        global_best = particle.copy()
                        global_best_score = score

                # Velocity update using geometric reasoning
                for j in range(n_points):
                    # Random component
                    r1, r2 = np.random.rand(2)

                    # Cognitive component (toward personal best)
                    cognitive = c1 * r1 * (personal_best[i][j] - particle[j])

                    # Social component (toward global best)
                    social = c2 * r2 * (global_best[j] - particle[j])

                    # Inertia component
                    inertia = w * velocities[i][j]

                    # Geometric distance-aware velocity
                    velocity = inertia + cognitive + social

                    # Apply distance-aware limiting
                    velocity_mag = np.linalg.norm(velocity)
                    if velocity_mag > max_velocity:
                        velocity = velocity / velocity_mag * max_velocity

                    velocities[i][j] = velocity

            # Position update
            for i, particle in enumerate(particles):
                # Update position with bounded checking
                new_pos = particle + velocities[i]

                # Boundary handling with epsilon padding
                epsilon = 1e-6
                new_pos[:, 0] = np.clip(new_pos[:, 0], epsilon, 1 - epsilon)
                new_pos[:, 1] = np.clip(new_pos[:, 1], epsilon, 1 - epsilon)

                particles[i] = new_pos

            # Early stopping if improvement is minimal
            if iter_num > 100 and abs(global_best_score - personal_best_scores[0]) < 1e-8:
                break

            if iter_num % 500 == 0:
                pass  # Progress tracking

        return global_best

    def local_refinement(points, iterations=300):
        """Refine solution using local search with adaptive steps."""
        current_points = points.copy()

        for _ in range(iterations):
            current_ratio = compute_ratio(current_points)

            # Gradient estimation via finite differences
            eps = 1e-4
            best_points = current_points.copy()
            best_ratio = current_ratio

            # Try moving each point in small increments
            for i in range(len(current_points)):
                for dim in range(2):
                    # Try positive and negative step
                    for step_sign in [-1, 1]:
                        test_points = current_points.copy()
                        test_points[i, dim] += step_sign * eps

                        # Clamp to bounds
                        test_points[i, 0] = np.clip(test_points[i, 0], 0.001, 0.999)
                        test_points[i, 1] = np.clip(test_points[i, 1], 0.001, 0.999)

                        test_ratio = compute_ratio(test_points)
                        if test_ratio > best_ratio:
                            best_ratio = test_ratio
                            best_points = test_points.copy()

            # If we found an improvement, use it
            if best_ratio > current_ratio:
                current_points = best_points
            else:
                # Reduce step size and try again
                eps *= 0.5

                # Early stopping if changes become negligible
                if eps < 1e-8:
                    break

        return current_points

    def multi_resolution_optimization(initial_points):
        """Perform optimization at multiple resolution levels."""
        # Level 1: Coarse optimization
        coarse_points = initial_points.copy()
        coarse_points = geometric_particle_swarm_optimization(coarse_points, 1000)

        # Level 2: Medium resolution
        medium_points = local_refinement(coarse_points, 150)

        # Level 3: Fine tuning
        fine_points = local_refinement(medium_points, 200)

        return fine_points

    # Main optimization process
    # Step 1: Initialize with geometric distribution
    initial_points = initialize_geometric_distribution()

    # Step 2: Multi-resolution optimization
    optimized_points = multi_resolution_optimization(initial_points)

    # Step 3: Final local refinement
    final_points = local_refinement(optimized_points, 100)

    return final_points

# EVOLVE-BLOCK-END