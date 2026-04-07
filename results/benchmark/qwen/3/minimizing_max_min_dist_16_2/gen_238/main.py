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
    
    class PointOptimizer:
        def __init__(self):
            self.best_points = None
            self.best_ratio = 0.0
            
        def initialize_hexagonal(self):
            """Create enhanced hexagonal pattern with mathematical precision."""
            golden_ratio = (1 + np.sqrt(5)) / 2
            row_spacing = np.sqrt(3) / 2
            col_spacing = 1.0

            points = []
            rows = 4
            cols = 4

            for i in range(rows):
                for j in range(cols):
                    x = j * col_spacing + (i % 2) * col_spacing * 0.5
                    y = i * row_spacing
                    points.append([x, y])

            points = np.array(points)

            # Normalize properly
            x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
            y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

            if x_max > x_min:
                points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
            if y_max > y_min:
                points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

            # Apply deterministic symmetry breaking
            fibonacci_indices = []
            a, b = 1, 1
            while len(fibonacci_indices) < 16:
                fibonacci_indices.append(a)
                a, b = b, a + b

            for i in range(len(points)):
                if i in fibonacci_indices[:len(fibonacci_indices)//2]:
                    scale = 0.025
                elif i % 5 == 0:
                    scale = 0.015
                else:
                    scale = 0.01

                phase = (i * golden_ratio) % 1
                theta = phase * 2 * np.pi
                dx = scale * np.cos(theta) * (0.8 + np.random.random() * 0.4)
                dy = scale * np.sin(theta) * (0.8 + np.random.random() * 0.4)

                points[i, 0] += dx
                points[i, 1] += dy

            points = np.clip(points, 0, 1)
            return points

        def initialize_triangular(self):
            """Create triangular lattice pattern."""
            points = []
            rows = 4
            cols = 4
            
            for i in range(rows):
                for j in range(cols):
                    if len(points) >= 16:
                        break
                    x = j + (i % 2) * 0.5
                    y = i * np.sqrt(3) / 2
                    points.append([x, y])

            points = np.array(points[:16])
            
            # Normalize
            x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
            y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

            if x_max > x_min:
                points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
            if y_max > y_min:
                points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

            # Add noise
            points += np.random.normal(0, 0.01, points.shape)
            points = np.clip(points, 0, 1)
            
            return points

        def initialize_random_aware(self):
            """Create random points with boundary awareness."""
            points = np.random.rand(16, 2)
            
            # Boundary handling
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

        def initialize_perturbed_uniform(self):
            """Create perturbed uniform grid."""
            points = []
            for i in range(4):
                for j in range(4):
                    points.append([i/3, j/3])
            
            points = np.array(points)
            
            # Add perturbations
            for i in range(len(points)):
                noise = np.random.normal(0, 0.03, 2)
                points[i] += noise
                
            points = np.clip(points, 0, 1)
            return points

        def compute_ratio(self, points):
            """Efficiently compute min/max distance ratio."""
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

        def generate_neighbor(self, points, step_size=0.02):
            """Generate neighbor solution with intelligent moves."""
            new_points = points.copy()

            # Choose move type
            move_type = np.random.choice(['single', 'pair', 'cluster'], p=[0.5, 0.3, 0.2])

            if move_type == 'single':
                # Single point move
                idx = np.random.randint(len(points))
                adaptive_step = step_size * np.random.uniform(0.5, 1.5)
                movement = np.random.normal(0, adaptive_step, 2)
                new_points[idx] += movement

            elif move_type == 'pair':
                # Pair move with better selection
                sample_size = min(10, len(points))
                sample_indices = np.random.choice(len(points), sample_size, replace=False)
                sample_points = points[sample_indices]

                distances = cdist(sample_points, sample_points)
                np.fill_diagonal(distances, np.inf)

                min_indices = np.unravel_index(np.argmin(distances), distances.shape)
                actual_idx1 = sample_indices[min_indices[0]]
                actual_idx2 = sample_indices[min_indices[1]]

                adaptive_step = step_size * np.random.uniform(0.8, 1.2)
                movement = np.random.normal(0, adaptive_step, 2)
                new_points[actual_idx1] += movement
                new_points[actual_idx2] += movement

            else:  # cluster
                # Cluster move
                num_cluster = min(3, len(points) // 3)
                cluster_indices = np.random.choice(len(points), num_cluster, replace=False)

                adaptive_step = step_size * np.random.uniform(0.6, 1.0)
                movement = np.random.normal(0, adaptive_step, 2)

                for idx in cluster_indices:
                    new_points[idx] += movement

            # Apply boundary constraints
            new_points = np.clip(new_points, 0, 1)
            return new_points

        def optimize_single(self, initial_points):
            """Single optimization run with adaptive cooling."""
            points = initial_points.copy()
            current_ratio = self.compute_ratio(points)
            best_points = points.copy()
            best_ratio = current_ratio

            temp = 0.1
            cooling_rate = 0.9995
            min_temp = 1e-6

            improvement_window = 50
            recent_improvements = []
            stagnation_count = 0
            max_stagnation = 200

            max_iterations = 3000
            for iteration in range(max_iterations):
                if len(recent_improvements) >= improvement_window:
                    avg_improvement = np.mean(recent_improvements[-improvement_window:])
                    if avg_improvement < 1e-6:
                        temp *= 0.98
                    elif avg_improvement > 1e-3:
                        temp *= 1.02
                    else:
                        temp *= 0.999

                temp = max(temp * cooling_rate, min_temp)

                if stagnation_count > max_stagnation:
                    break

                new_points = self.generate_neighbor(points, step_size=temp)
                new_ratio = self.compute_ratio(new_points)

                if new_ratio > current_ratio:
                    recent_improvements.append(new_ratio - current_ratio)
                    if len(recent_improvements) > improvement_window * 2:
                        recent_improvements.pop(0)
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
                    points = new_points.copy()
                    current_ratio = new_ratio

                    if current_ratio > best_ratio:
                        best_ratio = current_ratio
                        best_points = points.copy()

            return best_points, best_ratio

    # Main execution
    np.random.seed(42)
    
    optimizer = PointOptimizer()
    
    # Multiple initializations
    initializers = [
        optimizer.initialize_hexagonal,
        optimizer.initialize_triangular,
        optimizer.initialize_random_aware,
        optimizer.initialize_perturbed_uniform
    ]

    best_solution = None
    best_ratio = 0
    
    # Run optimization from multiple starting points
    for i, initializer in enumerate(initializers):
        np.random.seed(42 + i)
        points = initializer()
        optimized_points, ratio = optimizer.optimize_single(points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = optimized_points.copy()

    # Return best solution
    return best_solution if best_solution is not None else optimizer.initialize_hexagonal()

# EVOLVE-BLOCK-END