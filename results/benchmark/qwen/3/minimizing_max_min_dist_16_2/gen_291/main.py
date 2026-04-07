# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hexagonal lattice foundation with adaptive simulated annealing optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    
    def create_hexagonal_lattice():
        """Create a hexagonal lattice pattern optimized for 16 points"""
        # For 16 points, we'll use a 4x4 hexagonal arrangement
        # Hexagonal lattice spacing
        spacing = 1.0 / 3.0  # To fit well in unit square
        points = []
        
        # Generate hexagonal grid with proper offsets
        for i in range(4):
            for j in range(4):
                # Offset every other row
                x_offset = 0.5 * (i % 2)
                x = (j * spacing) + x_offset
                y = i * spacing * np.sqrt(3) / 2
                
                # Add small random perturbations to break perfect symmetry
                x += np.random.normal(0, 0.01) * spacing
                y += np.random.normal(0, 0.01) * spacing
                
                points.append([x, y])
        
        points = np.array(points[:16])
        # Normalize to [0,1] square
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)
        return points
    
    def create_fibonacci_spiral():
        """Create points using Fibonacci spiral for good distribution"""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(16):
            y = 1 - (i / float(16 - 1)) * 2  # y from 1 to -1
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            # Map to 2D unit square
            x_mapped = (x + 1) / 2
            y_mapped = (z + 1) / 2
            
            # Add slight boundary bias
            x_mapped = np.clip(x_mapped, 0.05, 0.95)
            y_mapped = np.clip(y_mapped, 0.05, 0.95)
            
            points.append([x_mapped, y_mapped])
        
        return np.array(points)
    
    def create_triangular_lattice():
        """Create triangular lattice pattern"""
        points = []
        # Regular triangular lattice with 4x4 grid
        spacing = 1.0 / 3.0
        sqrt3 = np.sqrt(3)
        
        for i in range(4):
            for j in range(4):
                x = j * spacing
                y = i * spacing * sqrt3 / 2
                # Offset every other row
                if i % 2 == 1:
                    x += spacing / 2
                
                # Add noise to break symmetry
                x += np.random.normal(0, 0.01) * spacing
                y += np.random.normal(0, 0.01) * spacing
                
                points.append([x, y])
        
        points = np.array(points[:16])
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)
        return points
    
    def calculate_ratio(points):
        """Calculate the min/max distance ratio"""
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
    
    def boundary_penalty(points, penalty_factor=10.0):
        """Apply boundary penalty to discourage points near edges"""
        penalty = 0
        boundary_threshold = 0.02
        
        for point in points:
            if point[0] < boundary_threshold or point[0] > 1 - boundary_threshold:
                penalty += penalty_factor * (boundary_threshold - min(point[0], 1 - point[0]))
            if point[1] < boundary_threshold or point[1] > 1 - boundary_threshold:
                penalty += penalty_factor * (boundary_threshold - min(point[1], 1 - point[1]))
        
        return penalty
    
    def adaptive_simulated_annealing(initial_points, max_iterations=5000):
        """Enhanced simulated annealing with adaptive cooling and multi-scale moves"""
        current_points = initial_points.copy()
        current_ratio = calculate_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Initial temperature and adaptive parameters
        temp = 0.1
        min_temp = 1e-6
        cooling_rate = 0.9995
        
        # Track performance for adaptive cooling
        recent_improvements = []
        improvement_window = 50
        stagnation_count = 0
        max_stagnation = 100
        
        for iteration in range(max_iterations):
            # Adaptive temperature adjustment
            if len(recent_improvements) >= improvement_window:
                avg_improvement = np.mean(recent_improvements[-improvement_window:])
                if avg_improvement < 1e-6:
                    # Stagnation detected - increase temperature slightly
                    temp = min(temp * 1.1, 0.5)
                    stagnation_count += 1
                    if stagnation_count > max_stagnation:
                        temp = 0.05  # Reset to medium temperature
                else:
                    stagnation_count = 0
            
            # Dynamic move selection based on iteration stage
            if iteration < max_iterations * 0.3:
                # Early stage: aggressive exploration
                move_type = np.random.choice(['single', 'pair', 'cluster'], 
                                           p=[0.6, 0.3, 0.1])
                step_size = 0.03
            elif iteration < max_iterations * 0.7:
                # Middle stage: balanced exploration/exploitation
                move_type = np.random.choice(['single', 'pair', 'cluster'], 
                                           p=[0.5, 0.4, 0.1])
                step_size = 0.015
            else:
                # Late stage: fine tuning
                move_type = np.random.choice(['single', 'pair'], 
                                           p=[0.7, 0.3])
                step_size = 0.005
            
            # Create neighbor solution
            neighbor_points = current_points.copy()
            
            if move_type == 'single':
                # Move single point
                idx = np.random.randint(len(neighbor_points))
                neighbor_points[idx, 0] += np.random.normal(0, step_size)
                neighbor_points[idx, 1] += np.random.normal(0, step_size)
                
            elif move_type == 'pair':
                # Move two nearby points together
                distances = pdist(neighbor_points)
                if len(distances) > 0:
                    min_indices = np.unravel_index(np.argmin(distances), 
                                                 (len(neighbor_points), len(neighbor_points)))
                    idx1, idx2 = min_indices
                    movement = np.random.normal(0, step_size * 0.8, 2)
                    neighbor_points[idx1] += movement
                    neighbor_points[idx2] += movement
                    
            else:  # cluster
                # Move cluster of points
                cluster_size = min(3, len(neighbor_points) // 3 + 1)
                cluster_indices = np.random.choice(len(neighbor_points), 
                                                 cluster_size, replace=False)
                movement = np.random.normal(0, step_size * 0.7, 2)
                for idx in cluster_indices:
                    neighbor_points[idx] += movement
            
            # Apply boundary constraints
            neighbor_points = np.clip(neighbor_points, 0, 1)
            
            # Evaluate neighbor
            neighbor_ratio = calculate_ratio(neighbor_points)
            
            # Metropolis criterion with temperature
            if neighbor_ratio > current_ratio:
                current_points = neighbor_points
                current_ratio = neighbor_ratio
                if neighbor_ratio > best_ratio:
                    best_points = neighbor_points.copy()
                    best_ratio = neighbor_ratio
                    # Reset stagnation counter on improvement
                    stagnation_count = 0
            else:
                # Accept worse solutions with probability
                delta = neighbor_ratio - current_ratio
                if delta < 0:
                    acceptance_prob = math.exp(delta / temp)
                    if np.random.random() < acceptance_prob:
                        current_points = neighbor_points
                        current_ratio = neighbor_ratio
            
            # Update temperature
            temp *= cooling_rate
            temp = max(temp, min_temp)
            
            # Track improvements
            if neighbor_ratio > current_ratio:
                recent_improvements.append(neighbor_ratio - current_ratio)
                if len(recent_improvements) > improvement_window * 2:
                    recent_improvements.pop(0)
            
            # Early exit condition
            if temp < min_temp and stagnation_count > 50:
                break
                
        return best_points, best_ratio
    
    # Generate multiple initial configurations
    initial_configs = [
        create_hexagonal_lattice(),
        create_fibonacci_spiral(),
        create_triangular_lattice(),
        np.random.rand(16, 2),  # Random configuration
    ]
    
    # Also try perturbed hexagonal lattice
    hex_lattice = create_hexagonal_lattice()
    perturbed_hex = hex_lattice + np.random.normal(0, 0.02, hex_lattice.shape)
    perturbed_hex = np.clip(perturbed_hex, 0, 1)
    initial_configs.append(perturbed_hex)
    
    best_solution = None
    best_ratio = -np.inf
    
    # Try each initial configuration
    for i, initial_config in enumerate(initial_configs):
        try:
            # Run adaptive simulated annealing
            optimized_points, ratio = adaptive_simulated_annealing(initial_config)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_solution = optimized_points.copy()
                
        except Exception as e:
            warnings.warn(f"Error with initial config {i}: {str(e)}")
            continue
    
    # If no success, fall back to the hexagonal lattice
    if best_solution is None:
        best_solution = create_hexagonal_lattice()
    
    return best_solution

# EVOLVE-BLOCK-END