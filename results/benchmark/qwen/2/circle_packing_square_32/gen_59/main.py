# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import math
from scipy.optimize import minimize_scalar, minimize
from scipy.spatial import cKDTree

def generate_initial_hex_grid(n_circles):
    """Generate initial circle positions using hexagonal grid"""
    sqrt_n = math.ceil(math.sqrt(n_circles))
    rows = math.ceil(n_circles / sqrt_n)
    cols = math.ceil(n_circles / rows)

    positions = []
    spacing = 0.12

    for i in range(rows):
        for j in range(cols):
            x_offset = 0.5 * (i % 2)
            x = (j + x_offset) * spacing
            y = i * spacing * math.sqrt(3) / 2

            if x <= 1 - spacing/2 and y <= 1 - spacing/2:
                positions.append([x, y])

            if len(positions) >= n_circles:
                break
        if len(positions) >= n_circles:
            break

    positions = positions[:n_circles]

    # Ensure positions are within bounds
    for pos in positions:
        pos[0] = max(spacing/2, min(1 - spacing/2, pos[0]))
        pos[1] = max(spacing/2, min(1 - spacing/2, pos[1]))

    return np.array(positions)

def compute_voronoi_areas(positions):
    """Compute approximate Voronoi cell areas for density estimation"""
    try:
        vor = Voronoi(positions)
        areas = []
        for i in range(len(positions)):
            # Approximate area based on Voronoi region vertices
            # This is a simplified approach since Voronoi regions can be complex
            areas.append(1.0)  # Placeholder for actual area calculation
        return np.array(areas)
    except:
        return np.ones(len(positions))

def adaptive_radius_initialization(positions, n_circles):
    """Initialize radii adaptively based on local density"""
    # Use spatial indexing for efficient neighbor search
    tree = cKDTree(positions)

    # Find nearest neighbors for each point
    distances, indices = tree.query(positions, k=min(6, n_circles),
                                   return_distance=True, workers=-1)

    # Compute local density (inverse of average distance to neighbors)
    avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self-distance
    # Avoid division by zero
    avg_distances = np.where(avg_distances == 0, 1e-8, avg_distances)
    densities = 1.0 / avg_distances

    # Normalize densities
    normalized_densities = densities / np.max(densities)

    # Initialize radii inversely proportional to density (lower density = larger radius)
    base_radius = 0.05
    initial_radii = base_radius * (1.0 - 0.7 * normalized_densities)

    # Ensure minimum radius and reasonable upper bounds
    initial_radii = np.clip(initial_radii, 0.01, 0.2)

    return initial_radii

def calculate_smooth_constraints(circles, penalty_weight=1000.0):
    """Calculate smooth exponential penalties for constraints with adaptive weights"""
    n = len(circles)
    penalty = 0.0

    # For better control, compute violation magnitudes
    boundary_violations = []
    overlap_violations = []

    # Boundary penalties using smooth exponential function
    for i in range(n):
        x, y, r = circles[i]

        # Compute violations for all four boundaries
        left_violation = max(0, r - x)
        right_violation = max(0, x + r - 1)
        bottom_violation = max(0, r - y)
        top_violation = max(0, y + r - 1)

        # Store violations for adaptive penalty calculation
        if left_violation > 0:
            boundary_violations.append(left_violation)
        if right_violation > 0:
            boundary_violations.append(right_violation)
        if bottom_violation > 0:
            boundary_violations.append(bottom_violation)
        if top_violation > 0:
            boundary_violations.append(top_violation)

        # Exponential penalty for boundary violations (adaptive weight)
        if left_violation > 0:
            penalty += penalty_weight * np.exp(10 * left_violation)
        if right_violation > 0:
            penalty += penalty_weight * np.exp(10 * right_violation)
        if bottom_violation > 0:
            penalty += penalty_weight * np.exp(10 * bottom_violation)
        if top_violation > 0:
            penalty += penalty_weight * np.exp(10 * top_violation)

    # Overlap penalties using smooth exponential function with spatial indexing
    positions = circles[:, :2]
    tree = cKDTree(positions)

    # Find nearby pairs efficiently
    max_radius = np.max(circles[:, 2])
    pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

    # Check actual overlaps for candidate pairs
    for i, j in pairs:
        if i < j:  # Avoid duplicate checks
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            if dist < r1 + r2:
                overlap = (r1 + r2 - dist)
                overlap_violations.append(overlap)
                # Adaptive penalty based on violation magnitude
                penalty += penalty_weight * np.exp(10 * overlap)

    return penalty, boundary_violations, overlap_violations

def improved_circle_packing():
    """Improved circle packing algorithm with adaptive penalties and better initialization"""
    n = 32

    # Phase 1: Initial placement with hexagonal grid
    initial_positions = generate_initial_hex_grid(n)

    # Phase 1.5: Adaptive radius initialization based on local density
    initial_radii = adaptive_radius_initialization(initial_positions, n)

    circles = np.column_stack([initial_positions, initial_radii])

    # Phase 2: Multi-stage optimization with adaptive penalty weights
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])

    # Use adaptive penalty weights
    penalty_weights = [1000.0, 5000.0, 10000.0]  # Different weights for different stages

    # Multiple refinement rounds with varying strategies
    for stage in range(3):
        current_penalty_weight = penalty_weights[stage]

        # Stage 1: Coarse optimization
        if stage == 0:
            iterations = 5
        else:
            iterations = 10

        for round_num in range(iterations):
            # Local optimization for each circle considering constraints
            for i in range(n):
                # Define optimization function for single circle
                def optimize_single_circle(params):
                    x, y, r = params
                    # Boundary constraints
                    if x < r or x > 1-r or y < r or y > 1-r:
                        return 1e10

                    # Overlap penalties with updated penalty weight
                    penalty = 0
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = circles[j]
                            dist = np.sqrt((x-x2)**2 + (y-y2)**2)
                            if dist < r + r2:
                                penalty += current_penalty_weight * (r + r2 - dist)

                    # Objective: maximize radius (minimize negative radius)
                    return -r + penalty

                # Optimize just this circle
                try:
                    # Start with current values
                    current = circles[i].copy()
                    x0, y0, r0 = current

                    # Optimize radius first
                    def radius_obj(r):
                        return optimize_single_circle([x0, y0, r])

                    # Find optimal radius with bounds
                    lower_bound = 0.001
                    upper_bound = min(0.5, 1-x0, x0, 1-y0, y0)

                    if upper_bound > lower_bound:
                        res = minimize_scalar(radius_obj, bounds=(lower_bound, upper_bound), method='bounded')
                        new_r = max(lower_bound, min(upper_bound, res.x))

                        # Update position and radius
                        circles[i] = [x0, y0, new_r]
                except:
                    pass  # Skip if optimization fails

            # Evaluate current solution
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()

    # Final cleanup with constraint enforcement
    final_circles = best_circles.copy()
    for i in range(n):
        x, y, r = final_circles[i]
        # Ensure validity with stricter bounds
        r = max(0.001, min(0.45, r))
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        final_circles[i] = [x, y, r]

    return final_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    return improved_circle_packing()

# EVOLVE-BLOCK-END