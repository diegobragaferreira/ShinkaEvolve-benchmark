# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape x into 16 points
        points = x.reshape(-1, 2)
        # Calculate pairwise distances
        distances = pdist(points)
        # Minimize negative of min/max ratio (equivalent to maximizing min/max ratio)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0  # Avoid division by zero
        return -min_dist / max_dist

    def evaluate_solution(points):
        """Efficiently evaluate a solution's quality"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist

    def generate_hexagonal_initial():
        """Generate initial configuration based on hexagonal lattice for better spacing"""
        # Create a 4x4 hexagonal pattern
        points = []
        rows, cols = 4, 4
        spacing_x = 1.0 / (cols - 1)
        spacing_y = 1.0 / (rows - 1)

        for i in range(rows):
            for j in range(cols):
                # Offset every other row for hexagonal packing
                x_offset = 0.0 if i % 2 == 0 else spacing_x * 0.5
                x = (j * spacing_x) + x_offset
                y = i * spacing_y

                # Ensure points are within bounds
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))

                points.append([x, y])

        return np.array(points)

    def generate_fibonacci_spiral():
        """Generate points using Fibonacci spiral for good distribution"""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        for i in range(16):
            theta = math.acos(-1 + (2 * i) / 15)  # elevation angle
            phi_angle = (i * 2 * math.pi) / (phi * phi)  # azimuthal angle

            # Convert to cartesian coordinates
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)

            # Map to [0.05, 0.95] range to avoid boundaries
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2

            points.append([x, y])

        return np.array(points)

    def generate_regular_grid():
        """Generate regular grid initial configuration with symmetry breaking"""
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                points.append([x, y])

        # Break symmetry by slightly perturbing corner points differently
        points[0] = [0.05, 0.05]  # Bottom-left corner
        points[15] = [0.95, 0.95]  # Top-right corner
        points[3] = [0.95, 0.05]   # Bottom-right corner
        points[12] = [0.05, 0.95]  # Top-left corner

        return np.array(points)

    def generate_adaptive_grid():
        """Generate structured grid with adaptive perturbation based on current distribution"""
        # Create a regular 4x4 grid
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                points.append([x, y])

        points = np.array(points)
        
        # Adjust perturbation magnitude based on current solution quality
        # Calculate distance distribution for initial configuration
        distances = pdist(points)
        if len(distances) > 0:
            # If already well-distributed, use smaller perturbations
            current_ratio = np.min(distances) / np.max(distances) if np.max(distances) > 0 else 0
            perturbation_magnitude = max(0.005, 0.03 * (1.0 - current_ratio * 5))
        else:
            perturbation_magnitude = 0.02

        # Add adaptive perturbation
        np.random.seed(42)
        perturbation = np.random.normal(0, perturbation_magnitude, points.shape)
        points += perturbation

        # Clip to valid range
        points = np.clip(points, 0.001, 0.999)
        return points

    def generate_structured_distribution():
        """Generate a smart structured distribution with anchors and radial spread"""
        points = []
        # Anchor points at key locations
        anchors = [
            [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],  # corners
            [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5],  # midpoints
        ]
        
        # Add radial points around center
        for i in range(8):
            angle = 2 * math.pi * i / 8
            radius = 0.25
            x = 0.5 + radius * math.cos(angle) * 0.4
            y = 0.5 + radius * math.sin(angle) * 0.4
            anchors.append([x, y])
            
        # Create points from anchors with varied perturbations
        np.random.seed(42)
        for anchor in anchors[:16]:
            # Add different perturbation magnitudes to each anchor
            perturbation_magnitude = 0.01 + np.random.random() * 0.02
            x = anchor[0] + np.random.normal(0, perturbation_magnitude)
            y = anchor[1] + np.random.normal(0, perturbation_magnitude)
            x = max(0.001, min(0.999, x))
            y = max(0.001, min(0.999, y))
            points.append([x, y])
            
        return np.array(points)

    def generate_multi_scale_grid():
        """Generate grid with multiple scales for better exploration"""
        # Different grid sizes and configurations
        configs = []
        
        # Regular grid
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                grid_points.append([x, y])
        configs.append(np.array(grid_points))
        
        # Perturbed grid
        np.random.seed(42)
        perturbed = np.array(grid_points) + np.random.normal(0, 0.02, (16, 2))
        perturbed = np.clip(perturbed, 0.001, 0.999)
        configs.append(perturbed)
        
        # Corner-perturbed grid
        corner_perturbed = np.array(grid_points)
        corner_perturbed[0] = [0.1, 0.1]      # Bottom-left
        corner_perturbed[15] = [0.9, 0.9]     # Top-right
        corner_perturbed[3] = [0.9, 0.1]      # Bottom-right
        corner_perturbed[12] = [0.1, 0.9]     # Top-left
        configs.append(corner_perturbed)
        
        return configs

    def generate_adaptive_perturbation(base_points, iteration=0):
        """Generate perturbed points with adaptive magnitude based on iteration"""
        # Base perturbation magnitude decreases with iterations to focus on refinement
        base_magnitude = 0.05 * (1.0 - iteration * 0.1)
        base_magnitude = max(0.005, base_magnitude)

        # Add random perturbations
        perturbation = np.random.normal(0, base_magnitude, base_points.shape)
        perturbed_points = base_points + perturbation

        # Clip to valid range
        perturbed_points = np.clip(perturbed_points, 0.001, 0.999)
        return perturbed_points

    # Generate diverse initial configurations with better structure
    np.random.seed(42)
    initial_configs = [
        generate_hexagonal_initial(),
        generate_fibonacci_spiral(),
        generate_regular_grid(),
        generate_adaptive_grid(),
        generate_structured_distribution()
    ]

    # Add multiple perturbed versions of structured configurations
    structured_configs = generate_multi_scale_grid()
    for config in structured_configs:
        # Multiple perturbed variants
        for i in range(3):
            np.random.seed(42 + i)
            perturbed = config + np.random.normal(0, 0.01 + i * 0.005, config.shape)
            perturbed = np.clip(perturbed, 0.001, 0.999)
            initial_configs.append(perturbed)

    # Add strategies from the second version for enhanced diversity
    # Generate multiple diverse initial configurations from second version
    additional_configs = [
        generate_hexagonal_initial(),
        generate_fibonacci_spiral(),
        generate_regular_grid()
    ]

    # Add perturbed versions with enhanced diversity patterns
    for i, config in enumerate(additional_configs):
        for j in range(4):  # Increase from 3 to 4 perturbed versions per base config
            # Add more strategic perturbations
            np.random.seed(42 + i * 10 + j)
            perturbed = config.copy()

            # Apply different perturbation patterns for variety
            if j == 0:
                # Small uniform perturbation
                perturbation = np.random.normal(0, 0.01, config.shape)
            elif j == 1:
                # Larger perturbation to explore more distant regions
                perturbation = np.random.normal(0, 0.03, config.shape)
            elif j == 2:
                # Perturbation with some directional bias
                perturbation = np.random.normal(0, 0.015, config.shape)
                # Add slight bias towards center
                center = np.array([0.5, 0.5])
                for k in range(len(perturbed)):
                    perturbed[k] += (center - perturbed[k]) * 0.02
            else:
                # Very aggressive perturbation
                perturbation = np.random.normal(0, 0.04, config.shape)

            perturbed += perturbation
            perturbed = np.clip(perturbed, 0.001, 0.999)
            initial_configs.append(perturbed)

    # Define bounds for each coordinate (between 0 and 1)
    bounds = [(0.001, 0.999) for _ in range(32)]

    best_ratio = -np.inf
    best_points = None

    # Stage 1: Global search with differential evolution for rough exploration
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=15,  # Reduced iterations for speed
            popsize=6,   # Smaller population for faster execution
            seed=42,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7
        )

        # Refine with local optimization using tight tolerances
        local_result = minimize(
            objective,
            de_result.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-12}
        )

        if local_result.success:
            final_points = local_result.x.reshape(-1, 2)
            ratio = evaluate_solution(final_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = final_points.copy()
    except Exception:
        pass

    # Stage 2: Multi-start local optimization from diverse initial points
    # Only try the most promising initial configurations to save time
    initial_configs_subset = initial_configs[:10]  # Limit to first 10 configs
    
    for i, initial_config in enumerate(initial_configs_subset):
        try:
            # Apply different optimization parameters based on configuration type
            if i < 5:  # First group - use tighter tolerances
                result = minimize(
                    objective,
                    initial_config.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 75, 'ftol': 1e-10, 'gtol': 1e-10}
                )
            else:  # Later groups - use moderate tolerances
                result = minimize(
                    objective,
                    initial_config.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8}
                )

            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = evaluate_solution(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()

        except Exception:
            continue

    # Stage 3: If solution is still weak, try focused grid search
    if best_points is None or best_ratio < 0.20:  # Only if solution is poor
        # Try a more focused search on promising regions
        test_grid = np.linspace(0.15, 0.85, 5)  # Focus on interior region
        for i in range(len(test_grid)):
            for j in range(len(test_grid)):
                base_x = test_grid[i]
                base_y = test_grid[j]

                # Create structured perturbation based on position
                np.random.seed(42 + i * 5 + j)
                base_points = np.array([[base_x, base_y]] * 16)
                # Add position-dependent perturbation
                perturbation = np.random.normal(0, 0.02, (16, 2))
                perturbed_points = base_points + perturbation
                perturbed_points = np.clip(perturbed_points, 0.001, 0.999)

                try:
                    result = minimize(
                        objective,
                        perturbed_points.flatten(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 40, 'ftol': 1e-10}
                    )

                    if result.success:
                        final_points = result.x.reshape(-1, 2)
                        ratio = evaluate_solution(final_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()

                except Exception:
                    continue

    # Stage 4: Final refinement with improved optimization settings
    if best_points is not None:
        try:
            # Try a final high-precision optimization
            result = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15}
            )

            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = evaluate_solution(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
        except Exception:
            pass

    # Return the best solution found
    if best_points is None:
        # Fallback to the best structured initial configuration
        fallback_config = generate_adaptive_grid()
        # Add small random noise to break any remaining symmetries
        np.random.seed(42)
        fallback_points = fallback_config + np.random.normal(0, 0.005, fallback_config.shape)
        fallback_points = np.clip(fallback_points, 0.001, 0.999)
        best_points = fallback_points

    return best_points

# EVOLVE-BLOCK-END