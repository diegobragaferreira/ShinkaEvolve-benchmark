# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import SphericalVoronoi
import time
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)

    # Initialize points using Fibonacci sphere for good spread
    points = fibonacci_sphere(14)

    # Optimize using advanced spherical Voronoi-based approach
    best_points, best_ratio = optimize_with_spherical_voronoi(points)

    return best_points

def fibonacci_sphere(n):
    """Generate points on sphere using Fibonacci spiral method"""
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def compute_min_max_ratio(points):
    """Compute the ratio of minimum to maximum pairwise distances"""
    if len(points) < 2:
        return 0.0, 0.0

    distances = pdist(points)
    d_min = np.min(distances)
    d_max = np.max(distances)

    if d_max == 0:
        return 0.0, 0.0

    return d_min / d_max, d_min

def compute_spherical_voronoi_constraints(points):
    """Analyze current configuration using enhanced spherical Voronoi to identify constraints"""
    try:
        # Create spherical Voronoi diagram
        sv = SphericalVoronoi(points)

        # Compute Voronoi cell areas and identify small cells (indicating dense regions)
        cell_areas = sv.volume
        mean_area = np.mean(cell_areas)

        # Find points in cells that are below 20% of average area (dense regions)
        dense_point_indices = np.where(cell_areas < 0.2 * mean_area)[0]

        # Also identify points that have many close neighbors using more sophisticated thresholds
        distances = pdist(points)
        if len(distances) == 0:
            return np.ones(len(points)) * 0.5

        distance_matrix = squareform(distances)
        # Use multiple percentile thresholds to detect different types of constraints
        percentile_10 = np.percentile(distances, 10)
        percentile_25 = np.percentile(distances, 25)

        # Count neighbors within different thresholds
        close_neighbors_10 = np.sum(distance_matrix < percentile_10, axis=1)
        close_neighbors_25 = np.sum(distance_matrix < percentile_25, axis=1)

        # Combine criteria for constraint identification
        constraint_scores = np.zeros(len(points))
        for i in range(len(points)):
            # Score based on number of very close neighbors (more critical)
            constraint_scores[i] = close_neighbors_10[i] * 2.0
            # Add score for moderately close neighbors
            constraint_scores[i] += close_neighbors_25[i] * 0.5
            # Bonus for being in a dense region
            if i in dense_point_indices:
                constraint_scores[i] += 15.0

        return constraint_scores
    except:
        # Fallback if Voronoi computation fails
        return np.ones(len(points)) * 0.5

def select_point_for_perturbation(points, constraint_scores=None):
    """Select point to perturb based on enhanced constraint analysis and geometric properties"""
    if constraint_scores is not None:
        # Enhanced point selection strategy
        # Create a composite score combining constraint scores with geometric awareness

        # Get geometric properties
        distances = pdist(points)
        if len(distances) == 0:
            return np.random.randint(0, len(points))

        avg_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Calculate point-wise metrics for better selection
        point_metrics = np.zeros(len(points))

        for i in range(len(points)):
            # Get distances to all neighbors
            neighbor_distances = cdist([points[i]], points)[0]
            neighbor_distances = neighbor_distances[neighbor_distances > 0]

            if len(neighbor_distances) > 0:
                min_dist = np.min(neighbor_distances)
                max_dist = np.max(neighbor_distances)
                avg_dist = np.mean(neighbor_distances)
                std_dist = np.std(neighbor_distances)

                # Create a metric that captures how "problematic" this point is:
                # High constraint score + low average distance = highly problematic
                problem_metric = constraint_scores[i] * (1.0 + min_dist / avg_distance if avg_distance > 0 else 1.0)
                point_metrics[i] = problem_metric

        # Stratified sampling approach:
        # 1. 60% chance - select based on high composite metric
        # 2. 25% chance - select random from top 3 most problematic points
        # 3. 15% chance - global random selection (explore more broadly)

        if np.random.random() < 0.6:
            # Select based on weighted probability from high-scoring points
            # But make it more sophisticated - use softmax for better probability distribution
            weights = np.exp(point_metrics / np.max(point_metrics + 1e-8))  # Avoid division by zero
            weights = weights / np.sum(weights)  # Normalize
            selected_idx = np.random.choice(len(points), p=weights)
        elif np.random.random() < 0.85:  # 25% chance
            # Select from top 3 most problematic points
            top_indices = np.argsort(point_metrics)[-3:]
            selected_idx = np.random.choice(top_indices)
        else:
            # 15% chance - global random for broad exploration
            selected_idx = np.random.randint(0, len(points))
    else:
        # Fallback to random selection with some bias toward points in denser regions
        # This helps in case Voronoi analysis fails
        selected_idx = np.random.randint(0, len(points))

    return selected_idx

def compute_targeted_perturbation(points, point_idx, constraint_scores, prev_delta=None, iteration=0):
    """Compute a targeted perturbation that should improve the configuration with enhanced guidance"""
    # Determine if we should expand or contract based on local geometry
    distances = cdist([points[point_idx]], points)[0]
    # Remove self-distance
    distances = distances[distances > 0]

    if len(distances) == 0:
        # No neighbors, random perturb
        perturbation = np.random.normal(0, 0.01, 3)
        return perturbation

    avg_distance = np.mean(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    std_distance = np.std(distances)

    # Calculate constraint severity for this point
    constraint_severity = constraint_scores[point_idx]

    # Enhanced geometric analysis
    # Identify if this point is in a problematic region
    close_neighbors = np.sum(distances < avg_distance * 0.5)
    far_neighbors = np.sum(distances > avg_distance * 1.5)

    # If point is too close to others, push away strongly with gradient-based approach
    if min_distance < avg_distance * 0.35:
        # Gradient-based repulsion
        repulsion = np.zeros(3)
        total_weight = 0.0

        # Analyze all close neighbors with different weights based on distance
        for i in range(len(points)):
            if i != point_idx:
                diff = points[point_idx] - points[i]
                dist = np.linalg.norm(diff)
                if dist > 0 and dist < avg_distance * 0.7:  # Close enough to matter
                    # Use inverse square law with distance-dependent weighting
                    weight = 1.0 / (dist * dist + 1e-8)
                    repulsion += diff / dist * weight
                    total_weight += weight

        if total_weight > 0:
            repulsion = repulsion / total_weight
            repulsion = repulsion / np.linalg.norm(repulsion) if np.linalg.norm(repulsion) > 0 else repulsion

            # Enhanced magnitude calculation incorporating multiple factors
            proximity_factor = min_distance / avg_distance
            variability_factor = std_distance / avg_distance if avg_distance > 0 else 1.0
            constraint_factor = 1.0 + constraint_severity * 0.15

            # Dynamic magnitude based on how constrained the point is
            magnitude = 0.025 * constraint_factor * (1.0 + proximity_factor * 1.5) * (1.0 + variability_factor * 0.5)

            # Add momentum from previous perturbation if available
            if prev_delta is not None and iteration > 0:
                momentum_factor = 0.3
                perturbation = repulsion * magnitude + momentum_factor * prev_delta
            else:
                perturbation = repulsion * magnitude

            return perturbation
        else:
            # Fallback to random perturbation with enhanced magnitude
            return np.random.normal(0, 0.02, 3)

    # If point is isolated, try to attract it to the cluster
    elif far_neighbors > 2 and max_distance > avg_distance * 1.8:
        # Attraction-based perturbation toward cluster center
        attraction = np.zeros(3)
        total_weight = 0.0

        # Attract to points that are reasonably close
        for i in range(len(points)):
            if i != point_idx and distances[i-1] < avg_distance * 1.3:
                diff = points[i] - points[point_idx]
                dist = np.linalg.norm(diff)
                if dist > 0:
                    weight = 1.0 / (dist * dist + 1e-8)
                    attraction += diff / dist * weight
                    total_weight += weight

        if total_weight > 0:
            attraction = attraction / total_weight
            attraction = attraction / np.linalg.norm(attraction) if np.linalg.norm(attraction) > 0 else attraction

            # Magnitude based on how isolated the point is
            isolation_score = far_neighbors / len(points)  # Fraction of far neighbors
            magnitude = 0.01 * (1.0 + isolation_score * 2.0) * (1.0 + constraint_severity * 0.1)

            return attraction * magnitude
        else:
            # Fallback to random perturbation
            return np.random.normal(0, 0.01, 3)
    else:
        # Medium spacing situation with enhanced perturbation
        # Analyze the distribution around this point
        median_distance = np.median(distances)

        # Create a more sophisticated perturbation that considers:
        # 1. Local density (more dense = more repulsion)
        # 2. Global distribution characteristics
        # 3. Center positioning

        # Local density factor
        density_factor = np.sum(distances < avg_distance * 0.6) / len(distances)

        # Create gradient-based perturbation from nearby points
        gradient = np.zeros(3)
        total_gradient_weight = 0.0

        for i in range(len(points)):
            if i != point_idx:
                diff = points[point_idx] - points[i]
                dist = np.linalg.norm(diff)
                if dist > 0:
                    # Weight by inverse distance and local density influence
                    weight = 1.0 / (dist * dist + 1e-8) * (1.0 + density_factor * 0.5)
                    gradient += diff / dist * weight
                    total_gradient_weight += weight

        # Normalize gradient if exists
        if total_gradient_weight > 0:
            gradient = gradient / total_gradient_weight
            gradient = gradient / np.linalg.norm(gradient) if np.linalg.norm(gradient) > 0 else gradient
        else:
            gradient = np.zeros(3)

        # Center positioning component
        center_dist = np.linalg.norm(points[point_idx])
        center_push = np.zeros(3)
        if center_dist > 0:
            # Push toward a preferred radial distribution (around 0.5)
            radial_adjust = 0.5 - center_dist
            center_push = points[point_idx] * radial_adjust / center_dist

        # Random component for exploration
        random_perturb = np.random.normal(0, 0.005, 3)

        # Blend components with dynamic weights based on constraint severity and context
        if constraint_severity > 10.0:
            # High constraint - emphasize gradient and repulsion
            gradient_weight = 0.6
            center_weight = 0.2
            random_weight = 0.2
        elif constraint_severity > 5.0:
            # Medium constraint - balanced
            gradient_weight = 0.4
            center_weight = 0.3
            random_weight = 0.3
        else:
            # Low constraint - emphasize exploration
            gradient_weight = 0.2
            center_weight = 0.2
            random_weight = 0.6

        # Apply momentum if available
        if prev_delta is not None and iteration > 0:
            momentum_factor = 0.25
            perturbation = (
                gradient * gradient_weight * 0.03 +
                center_push * center_weight * 0.02 +
                random_perturb * random_weight * 0.01 +
                momentum_factor * prev_delta
            )
        else:
            perturbation = (
                gradient * gradient_weight * 0.03 +
                center_push * center_weight * 0.02 +
                random_perturb * random_weight * 0.01
            )

        return perturbation

def optimize_with_spherical_voronoi(initial_points, max_time=360):
    """Optimize using spherical Voronoi analysis combined with adaptive simulated annealing"""
    points = initial_points.copy()
    current_ratio, min_dist = compute_min_max_ratio(points)

    # Parameters for adaptive simulated annealing
    temp = 1.0
    min_temp = 1e-10
    base_cooling_rate = 0.9995
    max_iter = 500000

    # Track best solution
    best_points = points.copy()
    best_ratio = current_ratio
    best_min_dist = min_dist

    start_time = time.time()
    iter_count = 0

    # Track convergence for adaptive cooling
    recent_ratios = []
    max_recent = 50

    # Track improvement steps for early stopping
    last_improvement = 0
    patience = 500  # How many iterations without improvement before cooling faster

    # Track previous perturbation for momentum
    prev_delta = None

    # Phase detection variables
    exploration_phase = True
    exploitation_phase = False
    fine_tuning_phase = False
    phase_transition_count = 0

    while temp > min_temp and iter_count < max_iter and time.time() - start_time < max_time:
        # Compute constraint scores for current configuration
        constraint_scores = compute_spherical_voronoi_constraints(points)

        # Select point to perturb
        point_to_move = select_point_for_perturbation(points, constraint_scores)

        # Create new candidate point with targeted perturbation
        new_points = points.copy()
        perturbation = compute_targeted_perturbation(points, point_to_move, constraint_scores, prev_delta, iter_count)
        new_points[point_to_move] += perturbation

        # Store this perturbation for momentum
        prev_delta = perturbation.copy()

        # Project back onto sphere
        norm = np.linalg.norm(new_points[point_to_move])
        if norm > 0:
            new_points[point_to_move] = new_points[point_to_move] / norm

        # Compute new ratio
        new_ratio, new_min_dist = compute_min_max_ratio(new_points)

        # Accept or reject based on Metropolis criterion
        if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
            points = new_points
            current_ratio = new_ratio
            min_dist = new_min_dist

            # Update best solution if improved
            if new_ratio > best_ratio:
                best_points = new_points.copy()
                best_ratio = new_ratio
                best_min_dist = new_min_dist
                last_improvement = iter_count

        # Adaptive cooling with phase-aware adjustments
        recent_ratios.append(current_ratio)
        if len(recent_ratios) > max_recent:
            recent_ratios.pop(0)

        # Phase detection based on performance
        if len(recent_ratios) >= 20:
            recent_change = recent_ratios[-1] - recent_ratios[0]
            # Determine current optimization phase
            if temp > 0.3 and recent_change > 1e-5:
                exploration_phase = True
                exploitation_phase = False
                fine_tuning_phase = False
                # Reset patience for exploration phase
                patience = 500
            elif temp <= 0.3 and temp > 0.05 and recent_change > 1e-7:
                exploration_phase = False
                exploitation_phase = True
                fine_tuning_phase = False
                # Reduce patience for exploitation
                patience = 250
            else:
                exploration_phase = False
                exploitation_phase = False
                fine_tuning_phase = True
                # Further reduce patience for fine-tuning
                patience = 100

        # Adjust cooling rate based on phase and performance
        if len(recent_ratios) >= 10:
            recent_improvement = recent_ratios[-1] - recent_ratios[0]
            if recent_improvement < 1e-8:
                # Slow progress, cool faster
                cooling_rate = base_cooling_rate * 1.15
            elif recent_improvement > 1e-4:
                # Fast progress, cool slower
                cooling_rate = base_cooling_rate * 0.98
            else:
                cooling_rate = base_cooling_rate
        else:
            cooling_rate = base_cooling_rate

        # Phase-specific cooling adjustments
        if exploration_phase:
            # During exploration, be more aggressive with cooling
            cooling_rate *= 1.05
        elif exploitation_phase:
            # During exploitation, moderate cooling
            cooling_rate *= 1.0
        elif fine_tuning_phase:
            # During fine-tuning, be very conservative
            cooling_rate *= 0.9999

        # Apply cooling
        temp *= cooling_rate
        iter_count += 1

        # Early stopping if no significant improvement
        if iter_count - last_improvement > patience and temp < 1e-6:
            # Reduce temperature more aggressively
            temp *= 0.95

        # Periodic refinement - every 1000 iterations with adaptive frequency
        if iter_count % max(500, 1000 - iter_count // 200) == 0 and iter_count > 0:
            # Do a more intensive local search for current best
            refined_points = points.copy()
            for _ in range(50):
                point_idx = np.random.randint(0, len(points))
                # Small perturbation to see if we can improve
                new_point = refined_points[point_idx] + np.random.normal(0, 0.001, 3)
                norm = np.linalg.norm(new_point)
                if norm > 0:
                    new_point = new_point / norm
                refined_points[point_idx] = new_point

                new_ratio, _ = compute_min_max_ratio(refined_points)
                if new_ratio > current_ratio:
                    points = refined_points.copy()
                    current_ratio = new_ratio
            # Reset temp to allow for further exploration after refinement
            temp = min(temp * 1.2, 1.0)

    return best_points, best_ratio

# EVOLVE-BLOCK-END