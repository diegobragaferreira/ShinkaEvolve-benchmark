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

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max <= 0:
            return 0
        return d_min / d_max

    def create_enhanced_hexagonal_grid():
        """Create enhanced hexagonal grid with better spacing and symmetry breaking"""
        # Arrange 16 points in a 4x4 grid with proper hexagonal spacing
        rows = 4
        cols = 4
        points = []

        spacing_x = 1.0
        spacing_y = np.sqrt(3) / 2

        for i in range(rows):
            for j in range(cols):
                # Offset every other row for hexagonal packing
                x = j * spacing_x + (i % 2) * spacing_x * 0.5
                y = i * spacing_y
                points.append([x, y])

        # Convert to numpy array
        points = np.array(points)

        # Normalize to [0,1] x [0,1]
        max_x = (cols - 1) + 0.5  # Account for offset in last row
        max_y = (rows - 1) * spacing_y

        points[:, 0] = points[:, 0] / max_x
        points[:, 1] = points[:, 1] / max_y

        # Add strategic perturbations to break symmetry
        np.random.seed(42)
        noise = np.random.normal(0, 0.015, points.shape)

        # Emphasize perturbation on corner points
        corner_indices = [0, 3, 12, 15]  # Four corners of 4x4 grid
        noise[corner_indices] *= 2.0

        points += noise
        points = np.clip(points, 0, 1)

        return points

    def create_alternative_configurations():
        """Generate multiple alternative initial configurations with more diversity"""
        configs = []

        # Configuration 1: Enhanced hexagonal grid (base case)
        configs.append(create_enhanced_hexagonal_grid())

        # Configuration 2: Random but constrained with fixed seed
        np.random.seed(42)
        configs.append(np.random.rand(16, 2))

        # Configuration 3: Grid with low perturbations (0.005 noise)
        grid_points = create_enhanced_hexagonal_grid()
        np.random.seed(43)
        perturbations = np.random.normal(0, 0.005, (16, 2))
        configs.append(np.clip(grid_points + perturbations, 0, 1))

        # Configuration 4: Grid with moderate perturbations (0.01 noise)
        grid_points = create_enhanced_hexagonal_grid()
        np.random.seed(44)
        perturbations = np.random.normal(0, 0.01, (16, 2))
        configs.append(np.clip(grid_points + perturbations, 0, 1))

        # Configuration 5: Grid with high perturbations (0.02 noise)
        grid_points = create_enhanced_hexagonal_grid()
        np.random.seed(45)
        perturbations = np.random.normal(0, 0.02, (16, 2))
        configs.append(np.clip(grid_points + perturbations, 0, 1))

        # Configuration 6: Triangular lattice pattern
        triangular_points = []
        rows = 4
        cols = 4
        spacing_x = 1.0
        spacing_y = np.sqrt(3)/2

        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x * 0.5
                y = i * spacing_y
                triangular_points.append([x, y])

        triangular_points = np.array(triangular_points)
        # Normalize triangular lattice
        max_x = (cols - 1) + 0.5
        max_y = (rows - 1) * spacing_y
        triangular_points[:, 0] = triangular_points[:, 0] / max_x
        triangular_points[:, 1] = triangular_points[:, 1] / max_y
        configs.append(np.clip(triangular_points[:16], 0, 1))

        # Configuration 7: Perturbed triangular lattice
        np.random.seed(46)
        triangular_perturbed = triangular_points[:16] + np.random.normal(0, 0.01, (16, 2))
        configs.append(np.clip(triangular_perturbed, 0, 1))

        # Configuration 8: Completely random with fixed seed for reproducibility
        np.random.seed(123)
        configs.append(np.random.rand(16, 2))

        # Configuration 9: Another hexagonal variation with different spacing
        hex_points = []
        rows = 4
        cols = 4
        spacing_x = 0.8
        spacing_y = np.sqrt(3)/2 * 0.8

        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x * 0.5
                y = i * spacing_y
                hex_points.append([x, y])

        hex_points = np.array(hex_points)
        max_x = (cols - 1) + 0.5
        max_y = (rows - 1) * spacing_y
        hex_points[:, 0] = hex_points[:, 0] / max_x
        hex_points[:, 1] = hex_points[:, 1] / max_y
        configs.append(np.clip(hex_points[:16], 0, 1))

        # Configuration 10: Uniform grid with different perturbation
        uniform_grid = []
        for i in range(4):
            for j in range(4):
                uniform_grid.append([i/3, j/3])
        np.random.seed(47)
        perturbations = np.random.normal(0, 0.015, (16, 2))
        configs.append(np.clip(np.array(uniform_grid[:16]) + perturbations, 0, 1))

        return configs

    def enhanced_simulated_annealing(initial_points, max_iter=3000):
        """Enhanced simulated annealing with adaptive cooling and mixed perturbations"""
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)

        # Better cooling schedule with adaptive parameters
        T = 0.5  # Higher initial temperature for extensive exploration
        cooling_rate = 0.9992  # Moderate cooling rate
        min_temp = 1e-6

        best_points = current_points.copy()
        best_ratio = current_ratio

        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_window = 50

        for iteration in range(max_iter):
            # Adaptive cooling based on recent performance
            if len(recent_improvements) >= improvement_window:
                avg_improvement = np.mean(recent_improvements[-improvement_window:])
                if avg_improvement < 1e-5:  # Stagnation detected
                    T *= 0.95  # Cool faster if stagnating
                elif avg_improvement > 1e-4:  # Good progress
                    T *= 1.01  # Warm up occasionally to escape local minima

            T *= cooling_rate

            if T < min_temp:
                break

            # Try different types of perturbations for diversity
            perturbation_type = np.random.choice(['single', 'neighborhood'], p=[0.7, 0.3])

            new_points = current_points.copy()
            accepted = False

            if perturbation_type == 'single':
                # Single point perturbation with adaptive magnitude
                idx = np.random.randint(len(current_points))
                perturbation_magnitude = T * 0.1

                new_points[idx, 0] += np.random.normal(0, perturbation_magnitude)
                new_points[idx, 1] += np.random.normal(0, perturbation_magnitude)

                # Enforce boundaries with better reflection and clamping
                for i in range(len(new_points)):
                    for j in range(2):
                        if new_points[i, j] < 0:
                            # More aggressive reflection to avoid edge traps
                            new_points[i, j] = abs(new_points[i, j])
                            # Also add a small random displacement to avoid getting stuck
                            new_points[i, j] += np.random.uniform(-0.01, 0.01)
                        elif new_points[i, j] > 1:
                            # More aggressive reflection to avoid edge traps
                            new_points[i, j] = 2 - new_points[i, j]
                            # Also add a small random displacement to avoid getting stuck
                            new_points[i, j] += np.random.uniform(-0.01, 0.01)

                        # Final clamping to ensure within bounds
                        new_points[i, j] = np.clip(new_points[i, j], 0, 1)

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
                    perturbation_magnitude = T * 0.05

                    # Calculate distance between the points to inform the movement
                    dist = np.sqrt(np.sum((current_points[idx1] - current_points[idx2])**2))

                    # If points are very close, make them move apart (repulsion)
                    # If they're far apart, make them move together (attraction)
                    if dist < 0.10:
                        # Repulsion - move points further apart
                        direction = current_points[idx1] - current_points[idx2]
                        direction = direction / (np.linalg.norm(direction) + 1e-8)  # Avoid division by zero
                        delta1 = direction * perturbation_magnitude * 2.0
                        delta2 = -direction * perturbation_magnitude * 2.0
                    else:
                        # Attraction - move points closer together
                        direction = current_points[idx2] - current_points[idx1]
                        direction = direction / (np.linalg.norm(direction) + 1e-8)
                        delta1 = direction * perturbation_magnitude * 0.5
                        delta2 = -direction * perturbation_magnitude * 0.5

                    # Add some randomness to make it more exploratory
                    delta1 += np.random.normal(0, perturbation_magnitude * 0.3, 2)
                    delta2 += np.random.normal(0, perturbation_magnitude * 0.3, 2)

                    new_points[idx1, :] += delta1
                    new_points[idx2, :] += delta2

                    # Enforce boundaries with better handling
                    for i in range(len(new_points)):
                        for j in range(2):
                            if new_points[i, j] < 0:
                                new_points[i, j] = abs(new_points[i, j])
                                new_points[i, j] += np.random.uniform(-0.005, 0.005)
                            elif new_points[i, j] > 1:
                                new_points[i, j] = 2 - new_points[i, j]
                                new_points[i, j] += np.random.uniform(-0.005, 0.005)

                            # Final clamping
                            new_points[i, j] = np.clip(new_points[i, j], 0, 1)

                    accepted = True
                else:
                    # Fall back to single point
                    idx = np.random.randint(len(current_points))
                    perturbation_magnitude = T * 0.1
                    new_points[idx, 0] += np.random.normal(0, perturbation_magnitude)
                    new_points[idx, 1] += np.random.normal(0, perturbation_magnitude)

                    # Enforce boundaries with reflection
                    for i in range(len(new_points)):
                        for j in range(2):
                            if new_points[i, j] < 0:
                                new_points[i, j] = -new_points[i, j]  # Reflect
                            elif new_points[i, j] > 1:
                                new_points[i, j] = 2 - new_points[i, j]  # Reflect

                    accepted = True

            if accepted:
                # Accept or reject the new solution
                new_ratio = compute_min_max_ratio(new_points)

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

        return best_points, best_ratio

    # Generate multiple initial configurations
    initial_configs = create_alternative_configurations()

    best_ratio = -np.inf
    best_points = None
    all_results = []

    # Try each initial configuration with optimization
    for i, initial_points in enumerate(initial_configs):
        # Clip initial points to valid bounds
        initial_points = np.clip(initial_points, 0, 1)

        # Try enhanced simulated annealing
        try:
            sa_points, sa_ratio = enhanced_simulated_annealing(initial_points, max_iter=2000)
            ratio_sa = sa_ratio
        except Exception:
            ratio_sa = 0  # fallback to 0 if SA fails

        # Try more aggressive SA
        try:
            sa_points2, sa_ratio2 = enhanced_simulated_annealing(initial_points, max_iter=4000)
            ratio_sa2 = sa_ratio2
        except Exception:
            ratio_sa2 = 0  # fallback to 0 if SA fails

        # Keep track of all results
        all_results.append((sa_points, ratio_sa, "simulated_annealing"))
        all_results.append((sa_points2, ratio_sa2, "simulated_annealing_aggressive"))

        # Select best among these strategies
        max_ratio = max(ratio_sa, ratio_sa2)
        if max_ratio > best_ratio:
            best_ratio = max_ratio
            # Find which solution was best
            if max_ratio == ratio_sa:
                best_points = sa_points.copy()
            else:
                best_points = sa_points2.copy()

    # Perform additional optimization rounds with the best solution found so far
    if best_points is not None:
        # Run several more optimization rounds from the best solution
        for _ in range(3):  # 3 more optimization rounds
            try:
                # Run enhanced simulated annealing from current best
                new_points, new_ratio = enhanced_simulated_annealing(best_points.copy(), max_iter=1000)
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = new_points.copy()
            except Exception:
                pass  # Continue if optimization fails

    # If no optimization succeeded, return the best initial configuration
    if best_points is None:
        # Fallback to simple enhanced hexagonal configuration
        best_points = create_enhanced_hexagonal_grid()

    return best_points

# EVOLVE-BLOCK-END