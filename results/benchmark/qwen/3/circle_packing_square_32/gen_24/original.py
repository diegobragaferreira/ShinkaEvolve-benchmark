# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    n = 32
    circles = np.zeros((n, 3))

    # Improved initialization using a more strategic grid placement
    def initialize_better():
        # Create a rectangular grid that fills the unit square
        sqrt_n = int(np.ceil(np.sqrt(n)))
        grid_size = sqrt_n
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)

        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                positions.append([x, y])

        # Take first n positions and add small random offsets
        positions = positions[:n]
        for i, pos in enumerate(positions):
            # Add small random offset to avoid perfect regularity
            pos[0] += np.random.uniform(-0.005, 0.005)
            pos[1] += np.random.uniform(-0.005, 0.005)
            # Clamp to valid range
            pos[0] = np.clip(pos[0], 0.01, 0.99)
            pos[1] = np.clip(pos[1], 0.01, 0.99)
            positions[i] = pos

        return positions

    # Better initialization with improved edge handling
    def initialize_strategic():
        # Use a more strategic approach placing circles near corners and edges
        positions = []

        # Place circles in a pattern that tries to use space efficiently
        # Start with a few circles in corners and edges
        corner_positions = [
            [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9],
            [0.05, 0.5], [0.5, 0.05], [0.5, 0.95], [0.95, 0.5]
        ]

        # Add more positions in a grid pattern
        grid_positions = []
        grid_size = 4
        spacing = 0.25
        for i in range(grid_size):
            for j in range(grid_size):
                x = 0.1 + j * spacing
                y = 0.1 + i * spacing
                if x <= 0.9 and y <= 0.9:
                    grid_positions.append([x, y])

        # Combine and take first n positions
        all_positions = corner_positions + grid_positions
        positions = all_positions[:n]

        # Add small random perturbations
        for i, pos in enumerate(positions):
            pos[0] += np.random.uniform(-0.01, 0.01)
            pos[1] += np.random.uniform(-0.01, 0.01)
            # Clamp to valid range
            pos[0] = np.clip(pos[0], 0.01, 0.99)
            pos[1] = np.clip(pos[1], 0.01, 0.99)

        return positions

    # Initialize positions with better strategy
    positions = initialize_strategic()

    # Set initial radii - start with values that can grow significantly for better packing
    radii = np.full(n, 0.05)  # Increased initial radius

    # Optimized constraint checking with vectorized operations
    def check_constraints_vectorized(circles_array):
        """Vectorized constraint checking for better performance"""
        # Check containment
        x, y, r = circles_array[:, 0], circles_array[:, 1], circles_array[:, 2]
        containment_ok = (r <= x) & (x <= 1-r) & (r <= y) & (y <= 1-r)
        if not np.all(containment_ok):
            return False

        # Check overlaps using vectorized distance computation
        # Compute pairwise distances between all circles
        distances = cdist(circles_array[:, :2], circles_array[:, :2])
        # Create mask for pairs that should not overlap (distance < sum of radii)
        radii_matrix = np.add.outer(r, r)  # Sum of radii for all pairs
        # Mask out the diagonal (same circle)
        np.fill_diagonal(distances, np.inf)
        # Check if any pair violates overlap constraint
        overlaps = distances < radii_matrix
        if np.any(overlaps):
            return False

        return True

    # Calculate total sum of radii
    def total_radius_sum(circles_array):
        return np.sum(circles_array[:, 2])

    # Enhanced local search with multiple perturbation strategies
    def enhanced_local_search(initial_circles, max_iterations=50000):
        # Start with initial configuration
        current_circles = initial_circles.copy()
        current_score = total_radius_sum(current_circles)

        # Parameters for SA
        temperature = 0.05
        min_temp = 1e-6
        cooling_rate = 0.9999
        iteration = 0

        best_circles = current_circles.copy()
        best_score = current_score

        # Keep track of recent improvements for early stopping
        recent_improvements = []
        patience = 500
        patience_counter = 0

        # Store previously accepted solutions to avoid cycling
        accepted_solutions = set()

        while temperature > min_temp and iteration < max_iterations:
            # Choose perturbation type with probability
            perturbation_type = np.random.choice(['position', 'radius', 'both'], p=[0.4, 0.3, 0.3])

            # Generate neighbor solution by perturbing one circle
            idx = np.random.randint(0, n)

            # Save current state
            old_x, old_y, old_r = current_circles[idx]

            # Different perturbation strategies
            new_x, new_y, new_r = old_x, old_y, old_r
            if perturbation_type == 'position':
                new_x = old_x + np.random.uniform(-0.01, 0.01)
                new_y = old_y + np.random.uniform(-0.01, 0.01)
                new_r = old_r
            elif perturbation_type == 'radius':
                new_r = old_r + np.random.uniform(-0.008, 0.008)
                new_x = old_x
                new_y = old_y
            else:  # both
                new_x = old_x + np.random.uniform(-0.01, 0.01)
                new_y = old_y + np.random.uniform(-0.01, 0.01)
                new_r = old_r + np.random.uniform(-0.005, 0.005)

            # Ensure new_r is positive and within reasonable bounds
            new_r = np.clip(new_r, 0.001, 0.2)

            # Ensure new positions are within bounds
            new_x = np.clip(new_x, new_r, 1 - new_r)
            new_y = np.clip(new_y, new_r, 1 - new_r)

            # Create new circles array
            new_circles = current_circles.copy()
            new_circles[idx] = [new_x, new_y, new_r]

            # Check if new configuration is valid
            if check_constraints_vectorized(new_circles):
                new_score = total_radius_sum(new_circles)

                # Accept or reject based on simulated annealing criteria
                delta = new_score - current_score

                # Accept if better or with probability based on temperature
                if delta > 0 or np.exp(delta / temperature) > np.random.random():
                    current_circles = new_circles
                    current_score = new_score

                    # Update best solution
                    if current_score > best_score:
                        best_circles = current_circles.copy()
                        best_score = current_score
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Early stopping if no improvement for a while
                    if patience_counter > patience:
                        break

            iteration += 1
            # Cool down
            temperature *= cooling_rate

            # Track recent improvements for early stopping
            recent_improvements.append(current_score)
            if len(recent_improvements) > 100:
                recent_improvements.pop(0)

        return best_circles

    # Run optimization
    try:
        final_circles = enhanced_local_search(np.column_stack([positions, radii]), max_iterations=50000)

        # Final validation with vectorized check
        if not check_constraints_vectorized(final_circles):
            # If invalid, fallback to a simpler approach
            positions = initialize_better()
            final_circles = np.column_stack([positions, [0.03]*n])

        circles = final_circles
    except Exception as e:
        # Fallback to a simpler uniform distribution if optimization fails
        positions = [[i*0.15 + 0.075, j*0.15 + 0.075] for i in range(6) for j in range(6)][:n]
        circles = np.column_stack([positions, [0.02]*n])

    return circles

# EVOLVE-BLOCK-END