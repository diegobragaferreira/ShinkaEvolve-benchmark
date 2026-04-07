# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial import cKDTree

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def check_collision(circle1, circle2):
    """Check if two circles collide"""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    distance_squared = (x1 - x2)**2 + (y1 - y2)**2
    return distance_squared < (r1 + r2)**2

def is_valid_position(circle, circles):
    """Check if a circle position is valid (within bounds and no collisions)"""
    x, y, r = circle

    # Check boundary constraints
    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
        return False

    # Check collision with existing circles
    for existing_circle in circles:
        if check_collision(circle, existing_circle):
            return False

    return True

def compute_max_radius(x, y, circles):
    """Compute the maximum radius for a circle at position (x,y) without overlapping existing circles"""
    if len(circles) == 0:
        return min(x, 1-x, y, 1-y)

    # Find minimum distance to any existing circle center
    min_distance = float('inf')
    for cx, cy, cr in circles:
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        min_distance = min(min_distance, distance)

    # Maximum radius is limited by boundaries and distance to other circles
    boundary_radius = min(x, 1-x, y, 1-y)
    collision_radius = min_distance - 0.0001  # Small epsilon to avoid numerical issues

    return min(boundary_radius, collision_radius) if collision_radius > 0 else 0

def place_circle_greedy(circles, max_circles):
    """Place circles greedily with maximum radius"""
    new_circles = circles.copy()
    placed = 0

    # Predefined strategic positions for initial placement
    strategic_positions = [
        (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),  # corners
        (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5),  # edges
        (0.5, 0.5),  # center
    ]

    # Place initial strategic circles
    for i, (x, y) in enumerate(strategic_positions[:min(9, max_circles)]):
        if placed >= max_circles:
            break
        # Try to place with maximum possible radius
        max_radius = compute_max_radius(x, y, new_circles[:placed])
        if max_radius > 0:
            new_circle = (x, y, max_radius)
            if is_valid_position(new_circle, new_circles[:placed]):
                new_circles[placed] = new_circle
                placed += 1

    # Fill remaining spots with greedy approach with better candidate selection
    while placed < max_circles:
        best_circle = None
        best_radius = 0

        # Generate candidates strategically
        candidates = []

        # Add some random candidates around existing circles (neighborhood sampling)
        if placed > 0:
            for i in range(placed):
                cx, cy, cr = new_circles[i]
                for _ in range(10):  # 10 candidates near each existing circle
                    angle = random.uniform(0, 2*np.pi)
                    distance = random.uniform(0.05, 0.2)
                    x = cx + distance * np.cos(angle)
                    y = cy + distance * np.sin(angle)
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        candidates.append((x, y))

        # Add random samples in the entire space
        for _ in range(500):
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)
            candidates.append((x, y))

        # Find the best valid circle among candidates
        for x, y in candidates:
            # Compute maximum possible radius for this position
            max_radius = compute_max_radius(x, y, new_circles[:placed])
            if max_radius <= best_radius:
                continue
            test_circle = (x, y, max_radius)
            if is_valid_position(test_circle, new_circles[:placed]):
                best_circle = test_circle
                best_radius = max_radius

        if best_circle is None:
            # If we can't find a valid circle, try placing a very small circle
            # This shouldn't happen often with good initialization
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)
            test_circle = (x, y, 0.001)
            if is_valid_position(test_circle, new_circles[:placed]):
                new_circles[placed] = test_circle
                placed += 1
            else:
                break  # Can't place more circles
        else:
            new_circles[placed] = best_circle
            placed += 1

    return new_circles

def optimize_positions(circles, iterations=50):
    """Improve circle positions through local optimization"""
    circles = circles.copy()

    for _ in range(iterations):
        improved = False

        # Try to move each circle slightly to improve the total sum
        for i in range(len(circles)):
            old_x, old_y, old_r = circles[i]

            # Store original values
            orig_circle = circles[i].copy()

            # Try small random moves
            best_circle = orig_circle.copy()
            best_sum = sum(circle[2] for circle in circles)

            for _ in range(20):  # Number of random attempts per circle
                # Slightly perturb the position
                new_x = old_x + random.uniform(-0.02, 0.02)
                new_y = old_y + random.uniform(-0.02, 0.02)

                # Ensure new position is within bounds
                new_x = max(0.01, min(0.99, new_x))
                new_y = max(0.01, min(0.99, new_y))

                # Compute new radius
                new_r = compute_max_radius(new_x, new_y, np.concatenate([circles[:i], circles[i+1:]]))

                if new_r > 0:
                    # Create a temporary circle array to test this change
                    temp_circles = circles.copy()
                    temp_circles[i] = (new_x, new_y, new_r)

                    # Check if valid (no overlap)
                    valid = True
                    for j in range(len(temp_circles)):
                        if j != i and not is_valid_position(temp_circles[j], np.concatenate([temp_circles[:j], temp_circles[j+1:]])):
                            valid = False
                            break

                    if valid:
                        # Calculate new sum
                        new_sum = sum(circle[2] for circle in temp_circles)

                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_circle = (new_x, new_y, new_r)

            # Update if improvement found
            if not np.allclose(best_circle, orig_circle):
                circles[i] = best_circle
                improved = True

        # Break early if no improvement was made
        if not improved:
            break

    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))

    # Use greedy initialization to get a good starting point
    circles = place_circle_greedy(circles, n)

    # Apply local optimization to refine positions
    circles = optimize_positions(circles)

    return circles

# EVOLVE-BLOCK-END