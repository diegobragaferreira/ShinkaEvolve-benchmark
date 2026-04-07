# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Uses an enhanced physics-based simulation approach with spatial indexing and improved optimization strategies.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Container parameters (perimeter = 4 => width + height = 2)
    container_width = 1.0
    container_height = 1.0

    # Physics simulation parameters
    n_circles = 21
    max_iterations = 15000
    dt = 0.005
    repulsion_strength = 150.0
    boundary_strength = 100.0
    radius_adjustment_factor = 0.005
    min_radius = 0.001
    max_radius = 0.25

    # Initialize circles with random positions and small radii
    circles = np.zeros((n_circles, 3))

    # Random initialization within container bounds
    np.random.seed(42)  # For reproducibility
    circles[:, 0] = np.random.uniform(0.01, container_width - 0.01, n_circles)  # x coordinates
    circles[:, 1] = np.random.uniform(0.01, container_height - 0.01, n_circles)  # y coordinates
    circles[:, 2] = np.random.uniform(0.01, 0.1, n_circles)  # Initial small radii

    # Normalize radii to get approximately right total area
    total_radius = np.sum(circles[:, 2])
    target_sum = 1.5  # Slightly increased target for better utilization
    scaling_factor = target_sum / total_radius if total_radius > 0 else 1.0
    circles[:, 2] *= scaling_factor

    # Store previous positions and radii for convergence check
    prev_positions = circles[:, :2].copy()
    prev_radii = circles[:, 2].copy()

    # Precompute spatial structure for efficient overlap detection
    tree = None

    # Physics simulation loop
    for iteration in range(max_iterations):
        # Recompute spatial tree every 50 iterations for efficiency
        if iteration % 50 == 0:
            positions = circles[:, :2]
            tree = cKDTree(positions)

        # Get current state
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Initialize forces
        forces = np.zeros_like(positions)

        # Use spatial tree for efficient neighbor search
        if tree is not None:
            # Find neighbors within 2*(max_radius) distance
            pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')
            for i, j in pairs:
                if i < j:  # Avoid double counting
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)

                    # If circles overlap or nearly overlap
                    if distance < (radii[i] + radii[j]):
                        # Repulsive force magnitude
                        force_magnitude = repulsion_strength * (1.0 - distance/(radii[i] + radii[j]))

                        # Direction of force (from j to i)
                        if distance > 1e-8:
                            fx = force_magnitude * dx / distance
                            fy = force_magnitude * dy / distance
                        else:
                            # Random direction if too close
                            angle = np.random.uniform(0, 2*np.pi)
                            fx = force_magnitude * np.cos(angle)
                            fy = force_magnitude * np.sin(angle)

                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
        else:
            # Fallback to full pairwise calculation
            for i in range(n_circles):
                for j in range(i+1, n_circles):
                    if i != j:
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        distance = np.sqrt(dx*dx + dy*dy)

                        # If circles overlap
                        if distance < (radii[i] + radii[j]):
                            # Repulsive force magnitude
                            force_magnitude = repulsion_strength * (1.0 - distance/(radii[i] + radii[j]))

                            # Direction of force (from j to i)
                            if distance > 1e-8:
                                fx = force_magnitude * dx / distance
                                fy = force_magnitude * dy / distance
                            else:
                                # Random direction if too close
                                angle = np.random.uniform(0, 2*np.pi)
                                fx = force_magnitude * np.cos(angle)
                                fy = force_magnitude * np.sin(angle)

                            forces[i, 0] += fx
                            forces[i, 1] += fy
                            forces[j, 0] -= fx
                            forces[j, 1] -= fy

        # Boundary forces (attract circles back into container)
        for i in range(n_circles):
            x, y = positions[i]
            r = radii[i]

            # Left boundary
            if x - r < 0:
                forces[i, 0] += boundary_strength * (0 - (x - r))
            # Right boundary
            if x + r > container_width:
                forces[i, 0] += boundary_strength * (container_width - (x + r))
            # Bottom boundary
            if y - r < 0:
                forces[i, 1] += boundary_strength * (0 - (y - r))
            # Top boundary
            if y + r > container_height:
                forces[i, 1] += boundary_strength * (container_height - (y + r))

        # Update positions
        for i in range(n_circles):
            # Apply forces to position
            positions[i, 0] += forces[i, 0] * dt
            positions[i, 1] += forces[i, 1] * dt

            # Keep within bounds
            positions[i, 0] = np.clip(positions[i, 0], radii[i], container_width - radii[i])
            positions[i, 1] = np.clip(positions[i, 1], radii[i], container_height - radii[i])

        # Radius adjustment with intelligent strategy
        if iteration % 100 == 0:
            # Try to increase radii where possible
            for i in range(n_circles):
                current_radius = radii[i]

                # Check if we can safely increase radius
                can_increase = True

                # Use spatial tree for efficient neighbor checking
                if tree is not None:
                    # Find nearby circles
                    nearby_indices = tree.query_ball_point(positions[i], 2 * (current_radius + max_radius))
                    nearby_indices = [idx for idx in nearby_indices if idx != i]

                    for j in nearby_indices:
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        distance = np.sqrt(dx*dx + dy*dy)

                        if distance < (current_radius + radii[j] + 0.001):
                            can_increase = False
                            break
                else:
                    # Fallback to slower check
                    for j in range(n_circles):
                        if i != j:
                            dx = positions[i, 0] - positions[j, 0]
                            dy = positions[i, 1] - positions[j, 1]
                            distance = np.sqrt(dx*dx + dy*dy)

                            if distance < (current_radius + radii[j] + 0.001):
                                can_increase = False
                                break

                # Increase radius if safe and beneficial
                if can_increase and current_radius < max_radius:
                    new_radius = current_radius + radius_adjustment_factor
                    # Don't let it exceed bounds
                    new_radius = min(new_radius, container_width/4, container_height/4)
                    # But also don't increase too much if we're making progress
                    if new_radius > current_radius:
                        radii[i] = new_radius

            # Reset tree for next iteration
            tree = None

        # Check for convergence
        if iteration % 1000 == 0 and iteration > 0:
            pos_change = np.mean(np.linalg.norm(positions - prev_positions, axis=1))
            rad_change = np.mean(np.abs(radii - prev_radii))

            if pos_change < 0.0001 and rad_change < 0.0001:
                # Try to refine further with smaller steps
                dt *= 0.5
                if dt < 0.001:
                    break

        prev_positions = positions.copy()
        prev_radii = radii.copy()

    # Final cleanup - ensure all circles are properly contained and have valid radii
    for i in range(n_circles):
        # Keep circle within container bounds
        circles[i, 0] = np.clip(positions[i, 0], circles[i, 2], container_width - circles[i, 2])
        circles[i, 1] = np.clip(positions[i, 1], circles[i, 2], container_height - circles[i, 2])
        circles[i, 2] = np.clip(circles[i, 2], min_radius, max_radius)

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")