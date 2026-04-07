# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    np.random.seed(42)

    # Enhanced hexagonal initialization with mathematical precision
    def initialize_enhanced_hexagonal():
        # Create a mathematically precise hexagonal arrangement using golden ratio for better packing
        golden_ratio = (1 + np.sqrt(5)) / 2
        row_spacing = np.sqrt(3) / 2
        col_spacing = 1.0

        # Arrange 16 points in a hexagonal pattern (4x4 grid with proper offsets)
        points = []
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                # Offset every other row for true hexagonal packing
                x = j * col_spacing + (i % 2) * col_spacing * 0.5
                y = i * row_spacing
                points.append([x, y])

        points = np.array(points)

        # Normalize to fit precisely in [0,1] x [0,1] with proper scaling
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

        if x_max > x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

        # Apply sophisticated deterministic symmetry breaking
        # Use Fibonacci sequence and prime-based patterns for asymmetry
        fibonacci_indices = []
        a, b = 1, 1
        while len(fibonacci_indices) < 16:
            fibonacci_indices.append(a)
            a, b = b, a + b

        for i in range(len(points)):
            # Pattern based on Fibonacci for large perturbations
            if i in fibonacci_indices[:len(fibonacci_indices)//2]:
                scale = 0.025
            elif i % 5 == 0:
                scale = 0.015
            else:
                scale = 0.01

            # Apply directional perturbations based on golden ratio phase
            phase = (i * golden_ratio) % 1
            theta = phase * 2 * np.pi
            dx = scale * np.cos(theta) * (0.8 + np.random.random() * 0.4)
            dy = scale * np.sin(theta) * (0.8 + np.random.random() * 0.4)

            points[i, 0] += dx
            points[i, 1] += dy

        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)

        return points

    # Triangular lattice initialization
    def initialize_triangular_lattice():
        points = []
        rows = 4
        cols = 4
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Triangular offset pattern
                x = j + (i % 2) * 0.5
                y = i * np.sqrt(3) / 2
                points.append([x, y])

        points = np.array(points[:16])
        
        # Normalize to [0,1] x [0,1]
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

        if x_max > x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

        # Add small noise to break symmetry
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        
        return points

    # Random initialization with boundary awareness
    def initialize_random_aware():
        points = np.random.rand(16, 2)
        
        # Apply boundary-aware adjustments to prevent clustering
        boundary_threshold = 0.02
        for i in range(len(points)):
            if points[i, 0] < boundary_threshold:
                points[i, 0] = boundary_threshold + np.random.uniform(0, boundary_threshold/2)
            elif points[i, 0] > 1 - boundary_threshold:
                points[i, 0] = 1 - boundary_threshold - np.random.uniform(0, boundary_threshold/2)
                
            if points[i, 1] < boundary_threshold:
                points[i, 1] = boundary_threshold + np.random.uniform(0, boundary_threshold/2)
            elif points[i, 1] > 1 - boundary_threshold:
                points[i, 1] = 1 - boundary_threshold - np.random.uniform(0, boundary_threshold/2)
        
        return points

    # Uniform grid with perturbations
    def initialize_perturbed_uniform():
        points = []
        for i in range(4):
            for j in range(4):
                points.append([i/3, j/3])
        
        points = np.array(points)
        
        # Add substantial perturbations
        for i in range(len(points)):
            noise = np.random.normal(0, 0.03, 2)
            points[i] += noise
            
        points = np.clip(points, 0, 1)
        return points

    # Compute min/max distance ratio efficiently
    def compute_ratio(points):
        if len(points) < 2:
            return 0

        # Use cKDTree for efficient nearest neighbor search
        tree = cKDTree(points)

        # Find minimum distance (excluding self-distance)
        distances, indices = tree.query(points, k=2)
        d_min = np.min(distances[:, 1])

        # Find maximum distance efficiently
        distances = cdist(points, points)
        np.fill_diagonal(distances, 0)
        d_max = np.max(distances)

        if d_max == 0:
            return 0

        return d_min / d_max

    # Generate neighbor solution with intelligent moves
    def generate_neighbor_adaptive(points, step_size=0.02):
        new_points = points.copy()

        # Choose move type with preference for more effective strategies
        move_type = np.random.choice(['single', 'pair', 'cluster'], p=[0.5, 0.3, 0.2])

        if move_type == 'single':
            # Single point move with adaptive step size
            idx = np.random.randint(len(points))
            adaptive_step = step_size * np.random.uniform(0.5, 1.5)
            movement = np.random.normal(0, adaptive_step, 2)
            new_points[idx] += movement

        elif move_type == 'pair':
            # Move two nearby points together with better selection
            # Sample and find actual closest points among sampled
            sample_size = min(10, len(points))
            sample_indices = np.random.choice(len(points), sample_size, replace=False)
            sample_points = points[sample_indices]

            # Efficient distance calculation for sample
            distances = cdist(sample_points, sample_points)
            np.fill_diagonal(distances, np.inf)

            # Get indices of closest points among sample
            min_indices = np.unravel_index(np.argmin(distances), distances.shape)
            actual_idx1 = sample_indices[min_indices[0]]
            actual_idx2 = sample_indices[min_indices[1]]

            # Move them together
            adaptive_step = step_size * np.random.uniform(0.8, 1.2)
            movement = np.random.normal(0, adaptive_step, 2)
            new_points[actual_idx1] += movement
            new_points[actual_idx2] += movement

        else:  # cluster
            # Move a small cluster of points together
            num_cluster = min(3, len(points) // 3)
            cluster_indices = np.random.choice(len(points), num_cluster, replace=False)

            # Move them towards centroid
            adaptive_step = step_size * np.random.uniform(0.6, 1.0)
            movement = np.random.normal(0, adaptive_step, 2)

            for idx in cluster_indices:
                new_points[idx] += movement

        # Apply boundary constraints
        new_points = np.clip(new_points, 0, 1)

        return new_points

    # Advanced adaptive optimization with better cooling and early stopping
    def adaptive_optimization(initial_points):
        points = initial_points.copy()
        current_ratio = compute_ratio(points)
        best_points = points.copy()
        best_ratio = current_ratio

        # Enhanced adaptive cooling schedule
        temp = 0.1
        cooling_rate = 0.9995
        min_temp = 1e-6

        # Tracking for adaptive cooling
        improvement_window = 50
        recent_improvements = []
        stagnation_count = 0
        max_stagnation = 200

        max_iterations = 3000
        for iteration in range(max_iterations):
            # Dynamic temperature adjustment based on recent performance
            if len(recent_improvements) >= improvement_window:
                avg_improvement = np.mean(recent_improvements[-improvement_window:])
                if avg_improvement < 1e-6:
                    temp *= 0.98
                elif avg_improvement > 1e-3:
                    temp *= 1.02
                else:
                    temp *= 0.999

            temp = max(temp * cooling_rate, min_temp)

            # Early stopping for stagnation
            if stagnation_count > max_stagnation:
                break

            # Generate neighbor solution
            new_points = generate_neighbor_adaptive(points, step_size=temp)
            new_ratio = compute_ratio(new_points)

            # Track improvements
            if new_ratio > current_ratio:
                recent_improvements.append(new_ratio - current_ratio)
                if len(recent_improvements) > improvement_window * 2:
                    recent_improvements.pop(0)
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
                points = new_points.copy()
                current_ratio = new_ratio

                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = points.copy()

        return best_points, best_ratio

    # Multi-start optimization with diverse initializations
    initializers = [
        initialize_enhanced_hexagonal,
        initialize_triangular_lattice,
        initialize_random_aware,
        initialize_perturbed_uniform
    ]

    best_solution = None
    best_ratio = 0
    
    # Run optimization from multiple starting points
    for i, initializer in enumerate(initializers):
        # Set different seed for each initialization
        np.random.seed(42 + i)
        
        # Initialize points
        points = initializer()
        
        # Optimize
        optimized_points, ratio = adaptive_optimization(points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = optimized_points.copy()

    # Return the best solution found
    return best_solution if best_solution is not None else initialize_enhanced_hexagonal()

# EVOLVE-BLOCK-END