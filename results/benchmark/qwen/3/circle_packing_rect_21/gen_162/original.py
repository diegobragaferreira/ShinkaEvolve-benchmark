# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import random
import math
from typing import Tuple

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions - since perimeter = 4, width + height = 2
    # Optimal rectangle is 1x1 (square) for maximum area utilization
    rect_width = 1.0
    rect_height = 1.0

    # Phase 1: Enhanced multi-scale initialization with geometric insights
    circles = np.zeros((21, 3))

    # Enhanced initialization using Voronoi-based approach with strategic seeding
    # Start with a regular hexagonal grid pattern for good space filling
    rows = 4
    cols = 6

    # Calculate base spacing
    x_spacing = rect_width / (cols + 1)
    y_spacing = rect_height / (rows + 1)

    # Place circles with alternating row offset for hexagonal packing
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= 21:
                break
            x = (j + 1) * x_spacing
            y = (i + 1) * y_spacing
            if i % 2 == 1:
                x += x_spacing * 0.5
            # Add randomization to escape local optima but keep structure
            x += random.uniform(-0.005, 0.005)
            y += random.uniform(-0.005, 0.005)
            circles[idx] = [x, y, 0.015]  # Slightly larger initial radius
            idx += 1

    # Add strategic boundary seeding for better utilization of container edges
    # Corner points
    corner_seeds = [
        [0.05, 0.05, 0.02],
        [rect_width - 0.05, 0.05, 0.02],
        [0.05, rect_height - 0.05, 0.02],
        [rect_width - 0.05, rect_height - 0.05, 0.02]
    ]

    # Edge midpoints
    edge_seeds = [
        [rect_width/2, 0.05, 0.02],
        [rect_width/2, rect_height - 0.05, 0.02],
        [0.05, rect_height/2, 0.02],
        [rect_width - 0.05, rect_height/2, 0.02]
    ]

    # Additional boundary seeds to encourage edge utilization
    boundary_seeds = [
        [0.15, 0.15, 0.015],
        [rect_width - 0.15, 0.15, 0.015],
        [0.15, rect_height - 0.15, 0.015],
        [rect_width - 0.15, rect_height - 0.15, 0.015]
    ]

    # Fill remaining slots with boundary seeds
    for i in range(len(corner_seeds)):
        if idx < 21:
            circles[idx] = corner_seeds[i]
            idx += 1

    for i in range(len(edge_seeds)):
        if idx < 21:
            circles[idx] = edge_seeds[i]
            idx += 1

    for i in range(len(boundary_seeds)):
        if idx < 21:
            circles[idx] = boundary_seeds[i]
            idx += 1

    # Phase 2: Multi-scale optimization with enhanced convergence strategies
    best_radius_sum = 0
    best_circles = None

    # Multi-scale approach with different resolution levels
    scale_levels = [0.1, 0.05, 0.02, 0.01]  # From coarse to fine resolution

    for scale_idx, scale in enumerate(scale_levels):
        # Multi-start approach with adaptive parameters based on scale level
        for start_iter in range(8):  # Reduced from 10 to 8 for faster execution
            # Reset current circles for this iteration
            current_circles = circles.copy()

            # Scale-specific parameters
            if scale_idx == 0:  # Coarse scale - heavy exploration
                max_iterations = 200
                step_size = 0.1 * scale
                temp = 2.0  # High temperature for exploration
                decay_factor = 0.96
                perturb_freq = 10
            elif scale_idx == 1:  # Medium coarse scale
                max_iterations = 150
                step_size = 0.07 * scale
                temp = 1.2
                decay_factor = 0.97
                perturb_freq = 15
            elif scale_idx == 2:  # Fine scale
                max_iterations = 100
                step_size = 0.03 * scale
                temp = 0.6
                decay_factor = 0.98
                perturb_freq = 20
            else:  # Finest scale
                max_iterations = 150
                step_size = 0.01 * scale
                temp = 0.3
                decay_factor = 0.99
                perturb_freq = 25

            # Progressive optimization with learning rate adaptation
            for iteration in range(max_iterations):
                improved = False

                # Shuffle circle indices for diverse search
                circle_indices = list(range(21))
                random.shuffle(circle_indices)

                # For better adaptation, compute gradient-like information
                # Track how much each circle's radius changed in recent iterations
                prev_radii = current_circles[:, 2].copy()

                for i in circle_indices:
                    # Compute maximum possible radius at this location
                    max_radius = _compute_max_radius_adaptive(current_circles, i, rect_width, rect_height)

                    # Adaptive acceptance probability based on improvement rate
                    old_r = current_circles[i, 2]
                    radius_difference = max_radius - old_r

                    if radius_difference > 1e-6:  # Significant improvement
                        # Accept with probability based on temperature and improvement
                        prob = min(1.0, math.exp(radius_difference / (temp + 1e-8)))
                        if random.random() < prob:
                            current_circles[i, 2] = max_radius
                            improved = True
                    elif radius_difference > -1e-6 and old_r < max_radius:  # Very slight improvement
                        current_circles[i, 2] = max_radius
                        improved = True

                # Update temperature for cooling schedule
                temp *= decay_factor

                # Strategic perturbation to escape local minima with frequency adjusted for scale
                if iteration % perturb_freq == 0 and iteration > 0:
                    # Perturb more aggressively at coarser scales
                    perturb_count = int(3 + scale_idx * 2)  # More perturbations at coarser scales
                    for _ in range(perturb_count):
                        i = random.randint(0, 20)
                        # Apply strategic perturbations towards better regions
                        # Add bias towards boundary exploitation at early phases
                        if scale_idx < 2:
                            # Encourage boundary movement at early scales
                            boundary_bias = 0.3  # 30% chance of boundary push
                            if random.random() < boundary_bias:
                                # Push towards boundary
                                x, y, r = current_circles[i]
                                if x < 0.1:
                                    current_circles[i, 0] = min(x + random.uniform(0.01, 0.03), 0.15)
                                elif x > rect_width - 0.1:
                                    current_circles[i, 0] = max(x - random.uniform(0.01, 0.03), rect_width - 0.15)
                                if y < 0.1:
                                    current_circles[i, 1] = min(y + random.uniform(0.01, 0.03), 0.15)
                                elif y > rect_height - 0.1:
                                    current_circles[i, 1] = max(y - random.uniform(0.01, 0.03), rect_height - 0.15)
                        else:
                            # Normal random perturbation
                            current_circles[i, 0] += random.uniform(-step_size, step_size)
                            current_circles[i, 1] += random.uniform(-step_size, step_size)

                        # Clamp to bounds
                        current_circles[i, 0] = np.clip(current_circles[i, 0], 0.01, rect_width - 0.01)
                        current_circles[i, 1] = np.clip(current_circles[i, 1], 0.01, rect_height - 0.01)

                        # Recompute max radius after perturbation
                        max_radius = _compute_max_radius_adaptive(current_circles, i, rect_width, rect_height)
                        current_circles[i, 2] = max_radius

                # Adaptive early stopping based on convergence rate
                if not improved:
                    if iteration > 50:
                        break
                else:
                    # If we're making progress, continue for a bit longer
                    pass

            # Final validation and scoring
            if _validate_configuration(current_circles, rect_width, rect_height):
                radius_sum = np.sum(current_circles[:, 2])
                if radius_sum > best_radius_sum:
                    best_radius_sum = radius_sum
                    best_circles = current_circles.copy()

    # Ensure we have a valid solution even if optimizations failed
    if best_circles is None:
        # Fallback to improved simple grid placement with more iterations
        circles = np.zeros((21, 3))
        rows, cols = 3, 7
        x_spacing = rect_width / (cols + 1)
        y_spacing = rect_height / (rows + 1)

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= 21:
                    break
                x = (j + 1) * x_spacing + random.uniform(-0.015, 0.015)
                y = (i + 1) * y_spacing + random.uniform(-0.015, 0.015)
                r = 0.025
                circles[idx] = [x, y, r]
                idx += 1

        # Local optimization on fallback with more iterations
        for _ in range(300):  # More iterations for fallback
            improved = False
            for i in range(21):
                max_radius = _compute_max_radius_adaptive(circles, i, rect_width, rect_height)
                if max_radius > circles[i, 2]:
                    circles[i, 2] = max_radius
                    improved = True
            if not improved:
                break

        best_circles = circles

    # Final normalization and boundary checks with more aggressive clamping
    for i in range(21):
        x, y, r = best_circles[i]
        # Ensure circles are within bounds and radius is reasonable
        r = min(r, x, rect_width - x, y, rect_height - y)
        if r <= 0.001:
            r = 0.015
        best_circles[i] = [x, y, r]

    return best_circles

def _compute_max_radius_adaptive(circles, index, width, height):
    """Compute the maximum possible radius using enhanced collision detection."""
    x, y, _ = circles[index]

    # Minimum distance to boundaries
    min_dist_to_boundaries = min(x, width - x, y, height - y)

    # Check collisions with other circles using optimized approach
    min_dist_to_others = float('inf')

    # Vectorized approach for better performance
    other_circles = np.delete(circles, index, axis=0)

    if len(other_circles) > 0:
        # Extract coordinates of other circles
        x2_vals = other_circles[:, 0]
        y2_vals = other_circles[:, 1]
        r2_vals = other_circles[:, 2]

        # Compute distances efficiently
        dx = x - x2_vals
        dy = y - y2_vals
        distances = np.sqrt(dx*dx + dy*dy)

        # Calculate possible radius constraints
        radius_constraints = distances - r2_vals

        # Find minimum constraint (but only for positive constraints)
        positive_constraints = radius_constraints[radius_constraints > 0]
        if len(positive_constraints) > 0:
            min_dist_to_others = np.min(positive_constraints)

    # Return the minimum of all constraints, with safety margin
    max_radius = min(min_dist_to_boundaries, min_dist_to_others)

    # Ensure minimum positive radius
    return max(0.001, max_radius)

def _validate_configuration(circles, width, height):
    """Validate that all circles are within bounds and non-overlapping."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Check boundary conditions
        if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
            return False

        # Check overlap with other circles with more precise tolerance
        for j in range(i + 1, len(circles)):
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
            # Allow very small overlap (numerical precision) but not actual overlap
            if distance < r + r2 - 1e-10:
                return False

    return True

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")