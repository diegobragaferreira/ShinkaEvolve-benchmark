# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Problem parameters
    n_circles = 32
    square_size = 1.0

    # Enhanced initialization using strategic hexagonal pattern with better spacing
    def initialize_strategic_hexagonal():
        # Create a more efficient hexagonal grid pattern
        rows = int(math.ceil(math.sqrt(n_circles * 1.2)))  # Slightly more rows for better coverage
        cols = int(math.ceil(n_circles / rows))

        # Optimize hexagon parameters for 32 circles in unit square
        side_length = 0.13  # Fine-tuned for 32 circles
        height = side_length * math.sqrt(3) / 2

        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n_circles:
                    break
                # Offset every other row for hexagonal packing
                x = (j + (i % 2) * 0.5) * side_length * 2
                y = i * height
                # Only add if within bounds and close enough to center
                if x <= 1 and y <= 1:
                    positions.append([x, y])

        # Fill remaining positions strategically
        if len(positions) < n_circles:
            # Add strategic corner placements
            corner_positions = [
                [0.08, 0.08], [0.08, 0.92], [0.92, 0.08], [0.92, 0.92],
                [0.15, 0.5], [0.5, 0.15], [0.5, 0.85], [0.85, 0.5]
            ]
            for pos in corner_positions:
                if len(positions) >= n_circles:
                    break
                # Add small random offset to avoid perfect regularity
                pos[0] += np.random.uniform(-0.008, 0.008)
                pos[1] += np.random.uniform(-0.008, 0.008)
                # Clamp to valid range
                pos[0] = np.clip(pos[0], side_length, 1 - side_length)
                pos[1] = np.clip(pos[1], side_length, 1 - side_length)
                positions.append(pos)

        # Fill remaining positions with random placements near center
        while len(positions) < n_circles:
            x = random.uniform(side_length, 1 - side_length)
            y = random.uniform(side_length, 1 - side_length)
            positions.append([x, y])

        positions = positions[:n_circles]
        return positions

    # Vectorized constraint checking for better performance
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

    # Fitness function with vectorized constraint checking
    def evaluate(individual):
        # Convert individual to circles array (x, y, r)
        circles = np.array(individual).reshape(-1, 3)
        
        # Check constraints
        total_radius = np.sum(circles[:, 2])
        
        # Penalty for invalid positions
        penalty = 0
        
        # Check containment constraints efficiently
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]

        # Vectorized containment check
        containment_ok = (radii <= x_coords) & (x_coords <= 1 - radii) & \
                         (radii <= y_coords) & (y_coords <= 1 - radii)
        if not np.all(containment_ok):
            penalty += 1000000

        # Efficient overlap checking using vectorized operations
        if len(circles) > 1:
            # Use cdist for pairwise distance calculation
            points = circles[:, :2]
            distances = cdist(points, points)

            # Create matrix of sums of radii
            radii_matrix = np.add.outer(radii, radii)

            # Zero out diagonal (same circle comparison)
            np.fill_diagonal(distances, np.inf)

            # Check if any distances are less than sum of radii
            overlaps = distances < radii_matrix
            if np.any(overlaps):
                penalty += 1000000

        # Return fitness (negative because we want to maximize)
        return (-total_radius - penalty,),
    
    # Initialize positions with better strategic hexagonal packing
    def initialize_population_strategy(pop_size):
        population = []
        for _ in range(pop_size):
            # Get initial strategic hexagonal positions
            positions = initialize_strategic_hexagonal()

            # Set initial radii based on proximity and positioning
            circles = []
            for i, (x, y) in enumerate(positions):
                # Start with a reasonable initial radius that can be optimized
                r = min(0.06, x, 1-x, y, 1-y)
                # Make some variations for diversity with strategic bias
                r = max(0.001, r * random.uniform(0.7, 1.3))
                circles.append((x, y, r))

            population.append([item for circle in circles for item in circle])
        return population

    # Enhanced local optimization with adaptive perturbations
    def enhanced_local_optimize(circles_array):
        # Start with the current best solution
        current_circles = circles_array.copy()
        current_total_radius = np.sum(current_circles[:, 2])

        # Parameters for SA - fine-tuned for better convergence
        temperature = 0.08
        min_temp = 1e-6
        cooling_rate = 0.9998
        max_iterations = 70000
        iteration = 0

        best_circles = current_circles.copy()
        best_score = current_total_radius

        # Keep track of recent improvements for early stopping
        recent_improvements = []
        patience = 400
        patience_counter = 0

        while temperature > min_temp and iteration < max_iterations:
            # Choose perturbation type with probability
            perturbation_type = np.random.choice(['position', 'radius', 'both'], p=[0.4, 0.3, 0.3])

            # Generate neighbor solution by perturbing one circle
            idx = np.random.randint(0, n_circles)

            # Save current state
            old_x, old_y, old_r = current_circles[idx]

            # Different perturbation strategies with adaptive step sizes
            new_x, new_y, new_r = old_x, old_y, old_r
            if perturbation_type == 'position':
                # Adaptive step size based on radius but bounded
                step_size = min(0.025, old_r * 0.6) 
                new_x = old_x + np.random.uniform(-step_size, step_size)
                new_y = old_y + np.random.uniform(-step_size, step_size)
                new_r = old_r
            elif perturbation_type == 'radius':
                # Adaptive step size based on current radius
                step_size = min(0.015, old_r * 0.4)
                new_r = old_r + np.random.uniform(-step_size, step_size)
                new_x = old_x
                new_y = old_y
            else:  # both
                # Adaptive step size for combined move
                step_size_pos = min(0.025, old_r * 0.6)
                step_size_rad = min(0.015, old_r * 0.4)
                new_x = old_x + np.random.uniform(-step_size_pos, step_size_pos)
                new_y = old_y + np.random.uniform(-step_size_pos, step_size_pos)
                new_r = old_r + np.random.uniform(-step_size_rad, step_size_rad)

            # Ensure new_r is positive and within reasonable bounds
            new_r = np.clip(new_r, 0.001, 0.2)

            # Ensure new positions are within bounds
            new_x = np.clip(new_x, new_r, 1 - new_r)
            new_y = np.clip(new_y, new_r, 1 - new_r)

            # Create new circles array
            new_circles = current_circles.copy()
            new_circles[idx] = [new_x, new_y, new_r]

            # Check if new configuration is valid using vectorized method
            if check_constraints_vectorized(new_circles):
                new_score = np.sum(new_circles[:, 2])

                # Accept or reject based on simulated annealing criteria
                delta = new_score - current_total_radius

                # Accept if better or with probability based on temperature
                if delta > 0 or np.exp(delta / temperature) > np.random.random():
                    current_circles = new_circles
                    current_total_radius = new_score

                    # Update best solution
                    if current_total_radius > best_score:
                        best_circles = current_circles.copy()
                        best_score = current_total_radius
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
            recent_improvements.append(current_total_radius)
            if len(recent_improvements) > 100:
                recent_improvements.pop(0)

        return best_circles

    # Initial optimization with strategic placement
    try:
        positions = initialize_strategic_hexagonal()
        initial_radii = np.full(n_circles, 0.05)
        initial_circles = np.column_stack([positions, initial_radii])
        
        # Apply enhanced local optimization
        optimized_circles = enhanced_local_optimize(initial_circles)
        
        # Final validation
        if not check_constraints_vectorized(optimized_circles):
            # Fallback to simple initialization if still invalid
            positions = initialize_strategic_hexagonal()
            optimized_circles = np.column_stack([positions, [0.03]*n_circles])

    except Exception as e:
        # Fallback to basic solution if anything fails
        print(f"Optimization failed with error: {e}")
        # Return simple configuration
        positions = [[i*0.15 + 0.075, j*0.15 + 0.075] for i in range(6) for j in range(6)][:n_circles]
        result = np.column_stack([positions, [0.02]*n_circles])
        return result

    return optimized_circles

# EVOLVE-BLOCK-END
