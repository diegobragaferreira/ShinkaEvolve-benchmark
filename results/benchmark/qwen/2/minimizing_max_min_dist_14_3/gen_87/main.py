# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        distances = pdist(points)
        return np.min(distances) / np.max(distances)

    def initialize_points():
        """Initialize points using a more sophisticated mathematical approach for 14-point distribution."""
        # Use a configuration based on known optimal or near-optimal arrangements
        # For 14 points on sphere, we can start with a modified icosahedral structure

        # Pre-computed approximate optimal configuration based on mathematical optimization
        # These coordinates are chosen to provide better initial distribution
        points = np.array([
            [0.000000, 0.000000, 1.000000],
            [0.894427, 0.000000, 0.447214],
            [0.276393, 0.850651, 0.447214],
            [-0.723607, 0.525731, 0.447214],
            [-0.723607, -0.525731, 0.447214],
            [0.276393, -0.850651, 0.447214],
            [0.000000, 0.000000, -1.000000],
            [-0.894427, 0.000000, -0.447214],
            [-0.276393, 0.850651, -0.447214],
            [0.723607, 0.525731, -0.447214],
            [0.723607, -0.525731, -0.447214],
            [-0.276393, -0.850651, -0.447214],
            [0.525731, 0.850651, 0.000000],
            [-0.525731, -0.850651, 0.000000]
        ])

        # Apply small random perturbations to break symmetries
        np.random.seed(42)  # For reproducibility
        perturbation_magnitude = 0.02  # Even smaller to preserve structure
        points += np.random.normal(0, perturbation_magnitude, points.shape)

        # Normalize to ensure points are on unit sphere
        norms = np.linalg.norm(points, axis=1)
        safe_norms = np.where(norms == 0, 1.0, norms)
        points = points / safe_norms[:, np.newaxis]

        # Apply slight rotation to further break symmetries
        angle = np.pi / 12
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        # Rotate around y-axis
        for i in range(14):
            x, y, z = points[i]
            points[i] = [x * cos_a + z * sin_a, y, -x * sin_a + z * cos_a]

        return points

    def optimize_points(initial_points, max_iterations=10000, temp_start=1.0, cooling_rate=0.9995):
        """Optimize point placement using simulated annealing with adaptive perturbation."""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)

        # Add checkpointing to save best solution found so far
        best_checkpoint = best_points.copy()
        best_checkpoint_ratio = best_ratio

        temp = temp_start
        checkpoint_interval = 1000  # Save checkpoint every N iterations
        stagnation_counter = 0
        max_stagnation = 1000  # Max iterations without improvement before cooling

        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_window = 500
        # Track absolute best for stagnation detection
        abs_best_ratio = best_ratio
        abs_best_iteration = 0

        def project_to_unit_sphere(points):
            """Strictly project points to unit sphere maintaining exact constraint."""
            norms = np.linalg.norm(points, axis=1)
            # Avoid division by zero
            norms = np.where(norms == 0, 1.0, norms)
            return points / norms[:, np.newaxis]

        for iteration in range(max_iterations):
            # Adaptive cooling schedule based on recent performance
            # Enhanced stagnation detection with relative improvement tracking
            if stagnation_counter > max_stagnation:
                # Check if we've actually lost significant ground
                if iteration - abs_best_iteration > 2000 and abs_best_ratio > 0:
                    rel_improvement = (best_ratio - abs_best_ratio) / abs_best_ratio
                    if rel_improvement < -0.01:  # Significant degradation
                        temp *= 0.95  # More aggressive cooling
                    else:
                        temp *= 0.98  # Moderate cooling
                else:
                    temp *= 0.95  # More aggressive cooling when stuck for long periods
                stagnation_counter = 0
            else:
                # Dynamic cooling rate that adapts to convergence behavior
                if len(recent_improvements) >= 10:
                    avg_improvement = np.mean(recent_improvements[-10:])
                    if avg_improvement < 1e-6:  # Very slow improvement
                        temp *= 0.98  # Cool faster
                    elif avg_improvement < 1e-5:  # Slow improvement
                        temp *= 0.995  # Moderate cooling
                    else:  # Rapid improvement
                        temp *= cooling_rate  # Normal cooling
                else:
                    temp *= cooling_rate  # Normal cooling initially

            # Stop if temperature gets too low
            if temp < 1e-10:
                break

            # Adaptive perturbation sizing based on current temperature and progress
            # Start with larger perturbations, decrease as optimization progresses
            base_perturbation_scale = temp * 0.02

            # Intelligent point selection based on distance characteristics
            # Calculate all pairwise distances to understand current distribution
            distances = pdist(current_points)

            # Select point that would most benefit from adjustment
            # Focus on extremes: points that are involved in small distances or large distances
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                mean_dist = np.mean(distances)

                # Determine if we're dealing with mostly close or far points
                # If many points are close to each other, we want to expand
                # If many points are far apart, we want to contract
                if min_dist < mean_dist * 0.3:
                    # We have many small distances, prefer to move points that are too close
                    # Find indices of points involved in small distances
                    small_distance_indices = set()
                    for i in range(len(current_points)):
                        for j in range(i+1, len(current_points)):
                            if pdist([current_points[i], current_points[j]])[0] < mean_dist * 0.5:
                                small_distance_indices.add(i)
                                small_distance_indices.add(j)

                    # If we found points involved in small distances, bias selection towards them
                    if small_distance_indices:
                        point_idx = np.random.choice(list(small_distance_indices))
                    else:
                        point_idx = np.random.randint(0, len(current_points))
                elif max_dist > mean_dist * 1.5:
                    # We have many large distances, prefer to move points that are too far
                    # Find indices of points involved in large distances
                    large_distance_indices = set()
                    for i in range(len(current_points)):
                        for j in range(i+1, len(current_points)):
                            if pdist([current_points[i], current_points[j]])[0] > mean_dist * 1.5:
                                large_distance_indices.add(i)
                                large_distance_indices.add(j)

                    # Bias selection towards points involved in large distances
                    if large_distance_indices:
                        point_idx = np.random.choice(list(large_distance_indices))
                    else:
                        point_idx = np.random.randint(0, len(current_points))
                else:
                    # Default to random selection
                    point_idx = np.random.randint(0, len(current_points))
            else:
                point_idx = np.random.randint(0, len(current_points))

            # Create perturbation vector with adaptive scale
            perturbation = np.random.normal(0, base_perturbation_scale, 3)
            # Apply perturbation
            new_points = current_points.copy()
            new_points[point_idx] += perturbation

            # Strictly enforce unit sphere constraint
            new_points = project_to_unit_sphere(new_points)

            # Compute new ratio
            new_ratio = compute_min_max_ratio(new_points)

            # Accept or reject based on Metropolis criterion
            if new_ratio > best_ratio or np.random.rand() < math.exp((new_ratio - best_ratio) / temp):
                current_points = new_points
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = new_points.copy()
                    # Update absolute best tracking
                    if new_ratio > abs_best_ratio:
                        abs_best_ratio = new_ratio
                        abs_best_iteration = iteration
                    stagnation_counter = 0  # Reset stagnation counter on significant improvement
                    # Update checkpoint
                    if new_ratio > best_checkpoint_ratio:
                        best_checkpoint = new_points.copy()
                        best_checkpoint_ratio = new_ratio
                else:
                    # Only increment stagnation counter for minor improvements (below threshold)
                    if new_ratio > best_ratio + 1e-8:  # Significant improvement
                        stagnation_counter = 0  # Reset if substantial improvement
                    else:
                        stagnation_counter += 1  # Increment if only tiny improvement
            else:
                stagnation_counter += 1  # Increment if rejected

            # Track recent improvements for adaptive cooling
            recent_improvements.append(new_ratio - best_ratio)
            if len(recent_improvements) > improvement_window:
                recent_improvements.pop(0)

            # Occasionally print progress
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}, Best ratio: {best_ratio:.6f}, Temp: {temp:.6f}")

        # Return the best solution found
        return best_checkpoint, best_checkpoint_ratio

    # Initialize with a good configuration
    initial_points = initialize_points()

    # Optimize using simulated annealing
    optimized_points, final_ratio = optimize_points(initial_points)

    # Ensure we're returning the best solution found
    return optimized_points


# EVOLVE-BLOCK-END