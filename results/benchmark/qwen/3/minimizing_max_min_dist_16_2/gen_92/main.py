# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    np.random.seed(42)
    n = 16
    d = 2

    # Create an enhanced hexagonal grid configuration
    def create_enhanced_hexagonal_grid():
        # Arrange points in a 4x4 grid with improved spacing and symmetry breaking
        rows = 4
        cols = 4

        points = []

        # Create hexagonal arrangement with better distribution
        for i in range(rows):
            for j in range(cols):
                # Offset every other row for hexagonal packing
                x = j + 0.5 * (i % 2)
                y = i * np.sqrt(3)/2
                points.append([x, y])

        # Convert to numpy array
        points = np.array(points)

        # Normalize to [0,1] x [0,1] with better scaling
        max_x = cols - 0.5
        max_y = (rows - 1) * np.sqrt(3)/2

        points[:, 0] = points[:, 0] / max_x
        points[:, 1] = points[:, 1] / max_y

        # Apply strategic perturbation to break symmetry
        noise_scale = 0.025
        points += np.random.normal(0, noise_scale, points.shape)

        # Ensure points stay within bounds
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)

        # Apply additional strategic displacement
        for i in range(0, len(points), 3):
            if i < len(points):
                points[i] += np.array([0.01, -0.01])
                points[i] = np.clip(points[i], 0, 1)

        return points

    # Calculate min/max distance ratio efficiently
    def calculate_ratio(points):
        if len(points) < 2:
            return 0

        # Compute pairwise distances using scipy
        distances = pdist(points)

        if len(distances) == 0:
            return 0

        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max <= 0:
            return 0

        return d_min / d_max

    # Enhanced simulated annealing with adaptive strategies
    def optimize_points(initial_points, max_iter=15000):
        current_points = initial_points.copy()
        current_ratio = calculate_ratio(current_points)

        # Advanced cooling schedule with adaptive parameters
        T = 0.3  # Higher initial temperature for extensive exploration
        cooling_rate = 0.9993  # Moderate cooling rate for balanced exploration/exploitation
        min_temp = 1e-6

        best_points = current_points.copy()
        best_ratio = current_ratio

        # Track statistics for adaptive cooling
        recent_improvements = []
        improvement_window = 50
        stagnation_count = 0
        max_stagnation = 200

        # Track the best solution seen so far for early stopping
        best_seen_ratio = current_ratio
        last_best_update = 0

        for iteration in range(max_iter):
            # Adaptive cooling based on recent performance
            if len(recent_improvements) >= improvement_window:
                avg_improvement = np.mean(recent_improvements[-improvement_window:])
                if avg_improvement < 1e-5:  # Stagnation detected
                    stagnation_count += 1
                    if stagnation_count > max_stagnation:
                        # Cool faster if stagnating
                        T *= 0.96
                        stagnation_count = 0
                else:
                    stagnation_count = 0

                # Occasionally warm up if there has been significant improvement
                if avg_improvement > 1e-4 and iteration > 1000 and np.random.rand() < 0.05:
                    T = min(T * 1.1, 1.0)  # Warm up occasionally

            # Standard cooling
            T *= cooling_rate

            if T < min_temp:
                break

            # Choose perturbation type based on current state and performance
            perturbation_type = 'neighborhood' if np.random.rand() < 0.3 else 'single'

            new_points = current_points.copy()
            accepted = False

            if perturbation_type == 'single':
                # Single point perturbation with boundary awareness
                idx = np.random.randint(len(current_points))
                perturbation_magnitude = T * 0.05

                # Adaptive perturbation magnitude based on point density in neighborhood
                nearby_count = 0
                for i in range(len(current_points)):
                    if i != idx:
                        dist = np.sqrt(np.sum((current_points[idx] - current_points[i])**2))
                        if dist < 0.15:  # Within some local radius
                            nearby_count += 1

                # Adjust perturbation size based on local density
                if nearby_count > 3:
                    perturbation_magnitude *= 0.5  # Reduce if dense
                elif nearby_count < 2:
                    perturbation_magnitude *= 1.5  # Increase if sparse

                new_points[idx, 0] += np.random.normal(0, perturbation_magnitude)
                new_points[idx, 1] += np.random.normal(0, perturbation_magnitude)

                # Boundary enforcement with reflection for better handling
                new_points[idx, 0] = np.clip(new_points[idx, 0], 0, 1)
                new_points[idx, 1] = np.clip(new_points[idx, 1], 0, 1)

                # Additional boundary handling: if too close to edge, push away
                if new_points[idx, 0] < 0.01 or new_points[idx, 0] > 0.99:
                    new_points[idx, 0] = np.clip(new_points[idx, 0], 0.01, 0.99)
                if new_points[idx, 1] < 0.01 or new_points[idx, 1] > 0.99:
                    new_points[idx, 1] = np.clip(new_points[idx, 1], 0.01, 0.99)

                accepted = True

            else:
                # Neighborhood-based perturbation for coordinated moves
                # Select two points that are relatively close
                candidates = list(range(len(current_points)))
                np.random.shuffle(candidates)

                # Find a pair of points that are reasonably close
                selected_pair = None
                for i in range(len(candidates)-1):
                    idx1 = candidates[i]
                    for j in range(i+1, len(candidates)):
                        idx2 = candidates[j]
                        dist = np.sqrt(np.sum((current_points[idx1] - current_points[idx2])**2))
                        if dist < 0.25:  # Only consider nearby pairs
                            selected_pair = (idx1, idx2)
                            break
                    if selected_pair:
                        break

                if selected_pair is not None:
                    idx1, idx2 = selected_pair
                    perturbation_magnitude = T * 0.04

                    # Coordinate the movement of both points
                    delta1 = np.random.normal(0, perturbation_magnitude, 2)
                    delta2 = np.random.normal(0, perturbation_magnitude, 2)

                    # Add some correlation to make movements more meaningful
                    correlation_factor = 0.3
                    delta1 = delta1 * (1 - correlation_factor) + delta2 * correlation_factor
                    delta2 = delta2 * (1 - correlation_factor) + delta1 * correlation_factor

                    new_points[idx1, :] += delta1
                    new_points[idx2, :] += delta2

                    # Enforce boundaries
                    new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
                    new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)

                    # Boundary correction for both points
                    for k in [idx1, idx2]:
                        if new_points[k, 0] < 0.01:
                            new_points[k, 0] = 0.01
                        elif new_points[k, 0] > 0.99:
                            new_points[k, 0] = 0.99
                        if new_points[k, 1] < 0.01:
                            new_points[k, 1] = 0.01
                        elif new_points[k, 1] > 0.99:
                            new_points[k, 1] = 0.99

                    accepted = True
                else:
                    # If no suitable pair, fall back to single point
                    idx = np.random.randint(len(current_points))
                    perturbation_magnitude = T * 0.05
                    new_points[idx, 0] += np.random.normal(0, perturbation_magnitude)
                    new_points[idx, 1] += np.random.normal(0, perturbation_magnitude)
                    new_points[idx, 0] = np.clip(new_points[idx, 0], 0, 1)
                    new_points[idx, 1] = np.clip(new_points[idx, 1], 0, 1)
                    accepted = True

            if accepted:
                # Accept or reject the new solution
                new_ratio = calculate_ratio(new_points)

                # Track recent improvements
                if new_ratio > current_ratio:
                    recent_improvements.append(new_ratio - current_ratio)
                    if len(recent_improvements) > improvement_window * 2:
                        recent_improvements.pop(0)

                # Metropolis criterion
                if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / T):
                    current_points = new_points
                    current_ratio = new_ratio

                    if current_ratio > best_ratio:
                        best_ratio = current_ratio
                        best_points = current_points.copy()

                        # Update best seen ratio and reset counter
                        if current_ratio > best_seen_ratio:
                            best_seen_ratio = current_ratio
                            last_best_update = iteration

            # Early stopping condition - if we haven't improved in a long time
            if iteration - last_best_update > 3000:
                break

        return best_points, best_ratio

    # Multi-scale optimization approach
    def multi_scale_optimization(initial_points):
        best_points = initial_points.copy()
        best_ratio = calculate_ratio(best_points)

        # Scale 1: Coarse optimization
        points = initial_points.copy()
        for iter_coarse in range(300):
            new_points = points.copy()
            # Large step size for broad search
            idx = np.random.randint(len(points))
            new_points[idx, 0] += np.random.normal(0, 0.05, 1)
            new_points[idx, 1] += np.random.normal(0, 0.05, 1)
            new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
            new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)

            new_ratio = calculate_ratio(new_points)
            if new_ratio > best_ratio or np.random.rand() < np.exp((new_ratio - best_ratio) * 10):
                points = new_points.copy()
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = points.copy()

        # Scale 2: Medium optimization
        for iter_medium in range(700):
            new_points = points.copy()
            # Medium step size
            idx = np.random.randint(len(points))
            new_points[idx, 0] += np.random.normal(0, 0.01, 1)
            new_points[idx, 1] += np.random.normal(0, 0.01, 1)
            new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
            new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)

            new_ratio = calculate_ratio(new_points)
            if new_ratio > best_ratio or np.random.rand() < np.exp((new_ratio - best_ratio) * 50):
                points = new_points.copy()
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = points.copy()

        # Scale 3: Fine optimization
        for iter_fine in range(1000):
            new_points = points.copy()
            # Small step size
            idx = np.random.randint(len(points))
            new_points[idx, 0] += np.random.normal(0, 0.002, 1)
            new_points[idx, 1] += np.random.normal(0, 0.002, 1)
            new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
            new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)

            new_ratio = calculate_ratio(new_points)
            if new_ratio > best_ratio or np.random.rand() < np.exp((new_ratio - best_ratio) * 100):
                points = new_points.copy()
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = points.copy()

        return best_points

    # Generate initial configuration
    initial_points = create_enhanced_hexagonal_grid()

    # Try multiple restarts with diverse initialization strategies to escape local optima
    best_solution = initial_points.copy()
    best_ratio = calculate_ratio(initial_points)

    # Multiple restarts with different initialization strategies
    for restart in range(5):
        # Strategy 1: Perturbed hexagonal grid (base case)
        if restart == 0:
            points = create_enhanced_hexagonal_grid()
        # Strategy 2: Random points with fixed seed
        elif restart == 1:
            np.random.seed(42 + restart)
            points = np.random.rand(16, 2)
        # Strategy 3: Hexagonal grid with different noise level
        elif restart == 2:
            np.random.seed(42 + restart)
            points = create_enhanced_hexagonal_grid()
            points += np.random.normal(0, 0.01, points.shape)
            points[:, 0] = np.clip(points[:, 0], 0, 1)
            points[:, 1] = np.clip(points[:, 1], 0, 1)
        # Strategy 4: Triangular lattice pattern
        elif restart == 3:
            points = create_triangular_lattice()
        # Strategy 5: Another perturbed hexagonal grid with different seed
        else:
            np.random.seed(100 + restart)
            points = create_enhanced_hexagonal_grid()
            points += np.random.normal(0, 0.03, points.shape)
            points[:, 0] = np.clip(points[:, 0], 0, 1)
            points[:, 1] = np.clip(points[:, 1], 0, 1)

        # Apply multi-scale optimization
        optimized_points = multi_scale_optimization(points)
        ratio = calculate_ratio(optimized_points)

        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = optimized_points.copy()

    # Final optimization using enhanced simulated annealing
    optimized_points, final_ratio = optimize_points(best_solution)

    return optimized_points

def create_triangular_lattice():
    """Create a triangular lattice pattern for alternative initialization"""
    # Create points arranged in a triangular (honeycomb) pattern
    points = []
    rows = 4
    cols = 4

    for i in range(rows):
        for j in range(cols):
            # Triangular offset pattern
            x = j + (i % 2) * 0.5
            y = i * np.sqrt(3)/2
            points.append([x, y])

    points = np.array(points)

    # Normalize to fit in [0,1] x [0,1]
    max_x = cols - 0.5
    max_y = (rows - 1) * np.sqrt(3)/2

    points[:, 0] = points[:, 0] / max_x
    points[:, 1] = points[:, 1] / max_y

    # Add small random perturbation
    np.random.seed(42)
    points += np.random.normal(0, 0.015, points.shape)
    points[:, 0] = np.clip(points[:, 0], 0, 1)
    points[:, 1] = np.clip(points[:, 1], 0, 1)

    return points

# EVOLVE-BLOCK-END