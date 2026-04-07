# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
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

    # Create structured hexagonal initial lattice configuration with enhanced asymmetry
    def create_hexagonal_initialization():
        # Create a 4x4 hexagonal grid pattern
        points = []
        sqrt3 = np.sqrt(3)

        # Hexagonal grid parameters
        spacing_x = 1.0 / 3.0
        spacing_y = sqrt3 / 4.0

        for i in range(4):
            for j in range(4):
                x = j * spacing_x
                y = i * spacing_y

                # Offset odd rows for hexagonal pattern
                if i % 2 == 1:
                    x += spacing_x / 2

                # Add systematic asymmetry based on position to break symmetry
                # This creates more effective symmetry breaking than simple random noise
                position_factor = (i * 7 + j * 3) % 10
                noise_scale = 0.015 + position_factor * 0.003

                # Use more structured noise patterns with directional bias
                x += np.random.normal(0, noise_scale * 0.7)
                y += np.random.normal(0, noise_scale * 0.7)

                # Add slight directional bias to encourage better distribution
                if i % 3 == 0:
                    x += np.random.normal(0, noise_scale * 0.2)
                if j % 3 == 0:
                    y += np.random.normal(0, noise_scale * 0.2)

                points.append([x, y])

        points = np.array(points)

        # Normalize to [0,1] x [0,1] properly
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

        if x_max > x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

        # Ensure all points are within bounds
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)

        return points

    # Calculate min/max distance ratio efficiently
    def calculate_ratio(points):
        if len(points) < 2:
            return 0

        # Compute pairwise distances
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances

        if distances.size == 0:
            return 0

        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max <= 0:
            return 0

        return d_min / d_max

    # Enhanced optimization using improved simulated annealing with adaptive cooling
    def optimize_points(initial_points, max_iter=15000):
        current_points = initial_points.copy()
        current_ratio = calculate_ratio(current_points)

        # Improved parameters for this approach
        T = 0.4  # Higher initial temperature for better exploration
        cooling_rate = 0.9997  # Slightly faster cooling rate
        min_temp = 1e-6
        best_points = current_points.copy()
        best_ratio = current_ratio

        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_window = 75

        # Voronoi-based neighborhood perturbation strategy with enhanced force guidance
        def get_voronoi_guided_perturbation(points, temperature, iteration):
            # Create a copy of the point array
            new_points = points.copy()

            # Compute Voronoi diagram for current configuration
            try:
                vor = Voronoi(points)
            except:
                # Fallback to random perturbation if Voronoi computation fails
                perturb_idx = np.random.randint(len(points))
                base_magnitude = temperature * 0.12
                adaptive_magnitude = base_magnitude * (1.0 - iteration / max_iter * 0.4)

                dx = np.random.normal(0, adaptive_magnitude)
                dy = np.random.normal(0, adaptive_magnitude)

                candidate_point = points[perturb_idx].copy()
                candidate_point[0] += dx
                candidate_point[1] += dy

                # Boundary handling
                candidate_point[0] = np.clip(candidate_point[0], 0, 1)
                candidate_point[1] = np.clip(candidate_point[1], 0, 1)

                new_points[perturb_idx] = candidate_point
                return new_points, current_ratio

            # Enhanced point selection strategy using Voronoi analysis
            # Calculate Voronoi-based forces for each point
            forces = np.zeros((len(points), 2))
            cell_areas = []

            for i in range(len(points)):
                try:
                    region = vor.regions[vor.point_region[i]]
                    if -1 not in region and len(region) > 0:
                        vertices = np.array([vor.vertices[j] for j in region if j >= 0])
                        if len(vertices) >= 3:
                            # Calculate area using shoelace formula
                            x = vertices[:, 0]
                            y = vertices[:, 1]
                            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                            cell_areas.append(area)
                        else:
                            cell_areas.append(0)
                    else:
                        cell_areas.append(0)
                except:
                    cell_areas.append(0)

            # Use cell area and Voronoi geometry to select perturbation point
            # Points with very small cells are likely in dense regions
            if len(cell_areas) > 0:
                # Select point with smallest cell area OR at random with some probability
                # This balances between exploring dense regions and maintaining diversity
                if np.random.random() < 0.7:  # 70% chance to pick dense region
                    perturb_idx = np.argmin(cell_areas)
                else:  # 30% chance to pick randomly for exploration
                    perturb_idx = np.random.randint(len(points))
            else:
                perturb_idx = np.random.randint(len(points))

            # Calculate force-based perturbation direction using Voronoi neighbors
            # Compute repulsion forces from neighbors and attraction to Voronoi cell centroid
            base_magnitude = temperature * 0.15
            adaptive_magnitude = base_magnitude * (1.0 - iteration / max_iter * 0.4)

            # Try several candidates guided by Voronoi structure and forces
            best_candidate_ratio = current_ratio
            best_candidate_pos = points[perturb_idx].copy()

            # Sample many candidates for better selection
            num_samples = 35  # Slightly increased sampling

            # Create perturbations with more intelligent strategies
            for _ in range(num_samples):
                # Strategy 1: Force-guided perturbation based on Voronoi geometry
                candidate_point = points[perturb_idx].copy()

                # Add force-directed component
                force_dir = np.random.normal(0, 1, 2)
                force_magnitude = np.linalg.norm(force_dir)
                if force_magnitude > 0:
                    force_dir = force_dir / force_magnitude * adaptive_magnitude * 0.7

                # Add noise component
                noise_dir = np.random.normal(0, 1, 2)
                noise_magnitude = np.linalg.norm(noise_dir)
                if noise_magnitude > 0:
                    noise_dir = noise_dir / noise_magnitude * adaptive_magnitude * 0.3

                # Combine force and noise
                total_perturbation = force_dir + noise_dir

                candidate_point[0] += total_perturbation[0]
                candidate_point[1] += total_perturbation[1]

                # Apply enhanced boundary handling that respects Voronoi structure
                # If too close to boundary, move away from it while preserving structure
                margin = 0.015  # Tighter margin for better boundary handling
                if candidate_point[0] < margin:
                    candidate_point[0] = margin + np.random.random() * 0.01
                elif candidate_point[0] > 1 - margin:
                    candidate_point[0] = 1 - margin - np.random.random() * 0.01
                if candidate_point[1] < margin:
                    candidate_point[1] = margin + np.random.random() * 0.01
                elif candidate_point[1] > 1 - margin:
                    candidate_point[1] = 1 - margin - np.random.random() * 0.01

                # Ensure within bounds
                candidate_point[0] = np.clip(candidate_point[0], 0, 1)
                candidate_point[1] = np.clip(candidate_point[1], 0, 1)

                # Test this move
                test_points = new_points.copy()
                test_points[perturb_idx] = candidate_point

                test_ratio = calculate_ratio(test_points)
                if test_ratio > best_candidate_ratio:
                    best_candidate_ratio = test_ratio
                    best_candidate_pos = candidate_point.copy()

            # Update the point with the best candidate
            new_points[perturb_idx] = best_candidate_pos

            return new_points, best_candidate_ratio

        # Main optimization loop
        for iteration in range(max_iter):
            # Adaptive cooling based on recent improvements
            if len(recent_improvements) > improvement_window:
                recent_improvements.pop(0)

            # Cooling schedule with more aggressive adaptation
            if len(recent_improvements) > 0 and sum(recent_improvements[-20:]) == 0:
                # If no improvements recently, cool slower to allow exploration
                T *= 0.9999
            else:
                T *= cooling_rate

            if T < min_temp:
                break

            # Get Voronoi-guided perturbation
            new_points, new_ratio = get_voronoi_guided_perturbation(current_points, T, iteration)

            # Accept or reject the new solution using Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / T):
                current_points = new_points
                current_ratio = new_ratio

                # Track improvement
                recent_improvements.append(1 if new_ratio > current_ratio else 0)

                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
            else:
                recent_improvements.append(0)

        return best_points, best_ratio

    # Multi-start optimization with diverse initial configurations
    def create_multiple_initializations():
        initializations = []

        # 1. Enhanced hexagonal grid with systematic asymmetry
        initializations.append(create_hexagonal_initialization())

        # 2. Random initialization
        rand_points = np.random.rand(16, 2)
        initializations.append(rand_points)

        # 3. Grid initialization with different spacing
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = i * 0.25 + np.random.normal(0, 0.015)
                y = j * 0.25 + np.random.normal(0, 0.015)
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                grid_points.append([x, y])
        initializations.append(np.array(grid_points))

        # 4. Perturbed hexagonal grid
        hex_points = create_hexagonal_initialization()
        hex_points += np.random.normal(0, 0.025, hex_points.shape)
        hex_points[:, 0] = np.clip(hex_points[:, 0], 0, 1)
        hex_points[:, 1] = np.clip(hex_points[:, 1], 0, 1)
        initializations.append(hex_points)

        # 5. Triangular lattice pattern with different spacing
        tri_points = []
        sqrt3 = np.sqrt(3)
        spacing_x = 1.0 / 3.0
        spacing_y = sqrt3 / 4.0

        for i in range(4):
            for j in range(4):
                x = j * spacing_x
                y = i * spacing_y

                if i % 2 == 1:
                    x += spacing_x / 2

                # Add noise with systematic variations
                x += (np.random.random() - 0.5) * 0.018
                y += (np.random.random() - 0.5) * 0.018

                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                tri_points.append([x, y])

        initializations.append(np.array(tri_points))

        return initializations

    # Run multiple optimizations from different starting points
    initial_configs = create_multiple_initializations()

    best_final_points = None
    best_ratio = -np.inf

    # Run optimization from each initial configuration with increased iterations
    for i, initial_config in enumerate(initial_configs):
        print(f"Starting optimization run {i+1}...")
        final_points, ratio = optimize_points(initial_config, max_iter=12000)

        if ratio > best_ratio:
            best_ratio = ratio
            best_final_points = final_points

    # Final refinement with the best configuration using a slightly different approach
    if best_final_points is not None:
        # Try one more optimization run with the best configuration but with a different cooling schedule
        final_points, final_ratio = optimize_points(best_final_points, max_iter=5000)
        return final_points
    else:
        # Fallback to hexagonal initialization
        return create_hexagonal_initialization()

# EVOLVE-BLOCK-END