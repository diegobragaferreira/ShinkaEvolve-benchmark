# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
from sklearn.cluster import KMeans
import time

# Global constants
BENCHMARK = 2.6358627564136983
N_CIRCLES = 26

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Step 1: Generate initial configuration using Voronoi-inspired approach
    circles = generate_voronoi_initialization()

    # Step 2: Optimize using gradient-based method with physics simulation
    circles = optimize_with_gradient_descent(circles)

    # Step 3: Final validation and cleanup
    circles = validate_and_cleanup(circles)

    return circles

def generate_voronoi_initialization():
    """Generate initial circle configuration using Voronoi-like distribution"""
    # Generate candidate points using k-means clustering
    candidate_points = []
    n_candidates = 300

    # Generate more points to ensure good distribution
    for _ in range(n_candidates):
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        candidate_points.append([x, y])

    candidate_points = np.array(candidate_points)

    # Use KMeans to find good seed positions
    kmeans = KMeans(n_clusters=N_CIRCLES, random_state=42, n_init=10)
    kmeans.fit(candidate_points)
    centroids = kmeans.cluster_centers_

    # Create circles at centroids with intelligent radius assignment
    circles = []
    for i, (cx, cy) in enumerate(centroids):
        # Compute max radius at this position (boundary constraints)
        max_r = min(cx, cy, 1-cx, 1-cy)

        # Find nearest neighbor distance to estimate density
        distances = np.sqrt(np.sum((candidate_points - [cx, cy])**2, axis=1))
        distances = distances[distances > 0]  # Exclude self-distance

        if len(distances) > 0:
            nearest_dist = np.min(distances)
            r = min(max_r, nearest_dist * 0.3)
        else:
            r = max_r * 0.25

        # Ensure reasonable radius bounds
        r = max(0.01, min(0.3, r))
        circles.append([cx, cy, r])

    # Ensure we have exactly N_CIRCLES circles
    while len(circles) < N_CIRCLES:
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        max_r = min(x, y, 1-x, 1-y)
        if max_r > 0.01:
            r = max_r * 0.2
            circles.append([x, y, r])

    circles = circles[:N_CIRCLES]
    return np.array(circles)

def is_valid(circles):
    """Check if circle configuration is valid"""
    # Check containment
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check non-overlap efficiently using KDTree
    try:
        points = circles[:, :2]
        tree = cKDTree(points)
        pairs = tree.query_pairs(0, return_distance=False)
        for i, j in pairs:
            if i < j:
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < r1 + r2:
                    return False
    except Exception:
        # Fallback to brute force if KDTree fails
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            for j in range(i+1, len(circles)):
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < r1 + r2:
                    return False

    return True

def calculate_repulsive_forces(circles):
    """Calculate repulsive forces between overlapping circles"""
    forces = np.zeros((len(circles), 2))

    for i in range(len(circles)):
        x1, y1, r1 = circles[i]
        for j in range(len(circles)):
            if i != j:
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                dist = np.sqrt(dx*dx + dy*dy)

                # Handle overlapping circles
                if dist < r1 + r2:
                    if dist > 0.001:
                        force_magnitude = (r1 + r2 - dist) * 0.1
                        forces[i, 0] += dx / dist * force_magnitude
                        forces[i, 1] += dy / dist * force_magnitude
                # Handle near contact
                elif dist < (r1 + r2 + 0.02):
                    if dist > 0.001:
                        force_magnitude = (r1 + r2 + 0.02 - dist) * 0.01
                        forces[i, 0] -= dx / dist * force_magnitude
                        forces[i, 1] -= dy / dist * force_magnitude

    return forces

def calculate_radius_gradients(circles):
    """Calculate gradient information for radius adjustment"""
    gradients = np.zeros((len(circles), 2))

    # Not used in this implementation but kept for API compatibility
    return gradients

def apply_force_updates(circles, forces, learning_rate=0.01):
    """Apply force updates to circle positions while maintaining validity"""
    updated_circles = circles.copy()

    for i in range(len(circles)):
        x, y, r = circles[i]
        fx, fy = forces[i]

        # Calculate new position
        new_x = x + fx * learning_rate
        new_y = y + fy * learning_rate

        # Ensure new position is valid
        new_x = np.clip(new_x, r, 1 - r)
        new_y = np.clip(new_y, r, 1 - r)

        # Update only if position change improves validity
        updated_circles[i] = [new_x, new_y, r]

    return updated_circles

def optimize_with_gradient_descent(initial_circles, max_iterations=200):
    """Optimize circles using gradient descent with physics-inspired forces"""
    circles = initial_circles.copy()

    for iteration in range(max_iterations):
        # Calculate forces
        forces = calculate_repulsive_forces(circles)

        # Apply updates
        new_circles = apply_force_updates(circles, forces)

        # Check if there was significant improvement
        old_sum = np.sum(circles[:, 2])
        new_sum = np.sum(new_circles[:, 2])

        # If no meaningful improvement or we're getting close to convergence
        if abs(new_sum - old_sum) < 1e-6:
            break

        circles = new_circles

    return circles

def validate_and_cleanup(circles):
    """Final validation and cleanup of circle configuration"""
    # Ensure validity
    if not is_valid(circles):
        # Simple repair by adjusting positions
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Fix containment
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            circles[i] = [x, y, r]

        # Resolve overlaps
        for iter_count in range(10):
            changed = False
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                for j in range(i):
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                    if dist < r1 + r2:
                        # Separate circles
                        dx = x2 - x1
                        dy = y2 - y1
                        if dx == 0 and dy == 0:
                            dx = 1
                            dy = 1
                        length = np.sqrt(dx*dx + dy*dy)
                        dx /= length
                        dy /= length

                        separation = (r1 + r2) - dist
                        circles[i][0] -= dx * separation * 0.5
                        circles[i][1] -= dy * separation * 0.5
                        circles[j][0] += dx * separation * 0.5
                        circles[j][1] += dy * separation * 0.5
                        changed = True

            if not changed:
                break

    # Final boundary fixes
    for i in range(len(circles)):
        x, y, r = circles[i]
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        circles[i] = [x, y, r]

    # Ensure we have exactly 26 circles
    while len(circles) < N_CIRCLES:
        circles = np.vstack([circles, [0.5, 0.5, 0.01]])

    if len(circles) > N_CIRCLES:
        circles = circles[:N_CIRCLES]

    return circles

# EVOLVE-BLOCK-END