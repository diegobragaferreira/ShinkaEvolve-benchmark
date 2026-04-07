# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import time
from scipy.optimize import minimize
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances
        distances = pdist(points)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def compute_potential_energy(points, alpha=1.0):
        """Compute potential energy based on inverse distance relationships."""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        # Avoid division by zero for very close points
        distances = np.maximum(distances, 1e-10)
        # Potential energy is sum of inverse distances raised to power alpha
        potential = np.sum(1.0 / (distances ** alpha))
        return potential

    def compute_geometric_diversity(points):
        """Measure geometric diversity using spread and uniformity metrics."""
        if len(points) < 2:
            return 0.0
            
        # Compute mean distance from center
        center = np.mean(points, axis=0)
        distances_from_center = np.linalg.norm(points - center, axis=1)
        mean_distance = np.mean(distances_from_center)
        
        # Compute variance of distances from center (lower variance = more uniform)
        variance_from_center = np.var(distances_from_center)
        
        # Normalize by max possible spread
        if mean_distance > 0:
            uniformity = 1.0 / (1.0 + variance_from_center / (mean_distance ** 2))
        else:
            uniformity = 0.0
            
        return uniformity

    def compute_distance_ratio_landscape(points):
        """Analyze the distance landscape to understand distribution characteristics."""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        distances = distances[distances > 0]  # Remove zero distances
        
        if len(distances) == 0:
            return 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Ratio of min to mean as indicator of clustering
        if mean_dist > 0:
            clustering_indicator = min_dist / mean_dist
        else:
            clustering_indicator = 0.0
            
        # Normalized standard deviation as measure of spread consistency
        if mean_dist > 0:
            spread_consistency = std_dist / mean_dist
        else:
            spread_consistency = 1.0
            
        return clustering_indicator, spread_consistency

    def generate_golden_ratio_initialization():
        """Generate initial configuration using golden ratio distribution."""
        points = []
        
        # Golden ratio constant
        phi = (1 + np.sqrt(5)) / 2
        psi = 1 / phi
        
        # Generate 16 points using golden ratio spiral
        for i in range(16):
            # Radial component
            r = i / 15.0  # Scale from 0 to 1
            
            # Angular component using golden angle
            theta = i * 2 * np.pi * psi
            
            # Convert to cartesian coordinates
            x = 0.5 + r * np.cos(theta) * 0.4
            y = 0.5 + r * np.sin(theta) * 0.4
            
            # Add small deterministic perturbation to break symmetry
            perturbation = np.sin(i * 0.7) * 0.005 + np.cos(i * 0.3) * 0.003
            x += perturbation * np.sin(i * 0.5)
            y += perturbation * np.cos(i * 0.5)
            
            points.append([x, y])
            
        points = np.array(points)
        # Add small random noise for further symmetry breaking
        points += np.random.normal(0, 0.002, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def generate_structured_grid():
        """Generate structured grid with enhanced asymmetry."""
        points = []
        
        # Create a 4x4 grid with systematic perturbations
        for i in range(4):
            for j in range(4):
                x_base = j * 0.25 + (i % 2) * 0.125
                y_base = i * 0.25
                
                # Enhanced asymmetry factor based on position
                asym_x = np.sin(i * 1.7 + j * 0.3) * 0.004
                asym_y = np.cos(i * 0.5 + j * 1.1) * 0.004
                
                x = x_base + asym_x
                y = y_base + asym_y
                
                points.append([x, y])
                
        points = np.array(points)
        points = np.clip(points, 0, 1)
        # Add controlled noise
        points += np.random.normal(0, 0.003, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def generate_random_initialization():
        """Generate random uniform distribution."""
        return np.random.rand(16, 2)

    def generate_diverse_initial_configs():
        """Generate multiple diverse initial configurations."""
        configs = []
        
        # Golden ratio spiral initialization
        configs.append(generate_golden_ratio_initialization())
        
        # Structured grid with asymmetry
        configs.append(generate_structured_grid())
        
        # Random initialization
        configs.append(generate_random_initialization())
        
        # Additional configurations with different patterns
        # Pattern 1: Fibonacci-like spiral
        points1 = []
        for i in range(16):
            angle = i * 2.4  # Modified golden angle
            radius = i * 0.05
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4
            points1.append([x, y])
        configs.append(np.clip(np.array(points1) + np.random.normal(0, 0.005, (16, 2)), 0, 1))
        
        # Pattern 2: Checkerboard with offsets
        points2 = []
        for i in range(4):
            for j in range(4):
                x_base = j * 0.25 + (i % 2) * 0.125 + (i + j) * 0.002
                y_base = i * 0.25 + (i + j) * 0.002
                points2.append([x_base, y_base])
        configs.append(np.clip(np.array(points2) + np.random.normal(0, 0.003, (16, 2)), 0, 1))
        
        # Pattern 3: Concentric rings
        points3 = []
        for i in range(16):
            # Place points in concentric rings
            ring_radius = 0.3 * (i % 4 + 1) / 4.0
            angle = i * 2 * np.pi / 16.0
            x = 0.5 + ring_radius * np.cos(angle)
            y = 0.5 + ring_radius * np.sin(angle)
            points3.append([x, y])
        configs.append(np.clip(np.array(points3) + np.random.normal(0, 0.004, (16, 2)), 0, 1))
        
        return configs

    def adaptive_clustering_move(points, cluster_size=3, max_attempts=50):
        """Perform adaptive cluster moves for better exploration."""
        candidate_points = points.copy()
        
        # Try multiple times to get a good cluster move
        for attempt in range(max_attempts):
            # Select cluster size adaptively based on optimization state
            try:
                if len(points) < 6:
                    num_points = 2
                else:
                    num_points = min(cluster_size, len(points))
                    
                # Select random subset of points
                indices = random.sample(range(len(points)), num_points)
                
                # Calculate centroid of cluster
                cluster_centroid = np.mean(candidate_points[indices], axis=0)
                
                # Generate movement
                move_vector = np.random.normal(0, 0.015, 2)
                new_centroid = np.clip(cluster_centroid + move_vector, 0, 1)
                delta = new_centroid - cluster_centroid
                
                # Apply movement to all points in cluster
                for idx in indices:
                    candidate_points[idx] += delta
                    
                # Check if this improves quality
                return candidate_points
                
            except Exception:
                continue
                
        # Fallback to single point move if cluster move fails
        idx = np.random.randint(0, len(points))
        candidate_points[idx] += np.random.normal(0, 0.02, 2)
        return np.clip(candidate_points, 0, 1)

    def potential_field_optimization(initial_points, max_iter=1000):
        """Optimize using potential field approach with distance-based forces."""
        points = initial_points.copy()
        current_ratio = compute_min_max_ratio(points)
        
        best_points = points.copy()
        best_ratio = current_ratio
        
        # Dynamic parameters
        learning_rate = 0.05
        temperature = 1.0
        
        for iteration in range(max_iter):
            # Compute force vectors based on distance relationships
            forces = np.zeros_like(points)
            n = len(points)
            
            # Calculate forces between all pairs (simplified for efficiency)
            for i in range(n):
                for j in range(i + 1, n):
                    diff = points[i] - points[j]
                    distance = np.linalg.norm(diff) + 1e-10
                    
                    # Repulsive force (inverse square law)
                    force_magnitude = 1.0 / (distance ** 2)
                    force_direction = diff / distance
                    
                    forces[i] += force_magnitude * force_direction
                    forces[j] -= force_magnitude * force_direction
            
            # Apply forces with learning rate and temperature
            points += learning_rate * forces * temperature
            
            # Keep within bounds
            points = np.clip(points, 0, 1)
            
            # Update best solution
            new_ratio = compute_min_max_ratio(points)
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = points.copy()
            
            # Adaptive cooling
            if iteration % 100 == 0 and iteration > 0:
                temperature *= 0.95
                
            # Early stopping criteria
            if iteration > 200 and abs(new_ratio - current_ratio) < 1e-6:
                break
                
            current_ratio = new_ratio
            
        return best_points

    def geometric_diversity_guided_optimization(initial_points, max_iter=500):
        """Optimize using geometric diversity as guidance."""
        points = initial_points.copy()
        current_ratio = compute_min_max_ratio(points)
        
        best_points = points.copy()
        best_ratio = current_ratio
        
        # Track diversity metrics
        previous_diversity = compute_geometric_diversity(points)
        diversity_improvements = []
        
        for iteration in range(max_iter):
            # Generate candidate solution
            candidate_points = points.copy()
            
            # Choose move type based on iteration
            move_type = random.random()
            
            if move_type < 0.7:
                # Standard single point move
                idx = np.random.randint(0, len(points))
                candidate_points[idx] += np.random.normal(0, 0.01, 2)
            elif move_type < 0.95:
                # Cluster move
                candidate_points = adaptive_clustering_move(points)
            else:
                # Global perturbation
                candidate_points += np.random.normal(0, 0.02, candidate_points.shape)
            
            # Clip to bounds
            candidate_points = np.clip(candidate_points, 0, 1)
            
            # Evaluate candidate
            candidate_ratio = compute_min_max_ratio(candidate_points)
            
            # Accept or reject with simulated annealing logic
            if candidate_ratio > current_ratio:
                points = candidate_points
                current_ratio = candidate_ratio
            else:
                # Accept with probability based on temperature
                temperature = max(0.1, 1.0 - iteration / max_iter)
                if np.random.rand() < np.exp((candidate_ratio - current_ratio) / temperature):
                    points = candidate_points
                    current_ratio = candidate_ratio
            
            # Update best solution
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = points.copy()
            
            # Diversity tracking for early stopping
            current_diversity = compute_geometric_diversity(points)
            diversity_improvements.append(current_diversity - previous_diversity)
            previous_diversity = current_diversity
            
            # Early stopping if convergence detected
            if len(diversity_improvements) > 20:
                recent_changes = diversity_improvements[-10:]
                if all(abs(change) < 1e-6 for change in recent_changes):
                    break
                    
        return best_points

    def hybrid_optimization(initial_points, max_time=175):
        """Execute hybrid optimization approach."""
        start_time = time.time()
        
        # Phase 1: Global exploration with potential field
        points = initial_points.copy()
        phase1_time = max_time * 0.4
        phase1_start = time.time()
        
        # Use potential field optimization for initial exploration
        points = potential_field_optimization(points, max_iter=500)
        
        # Phase 2: Local refinement with geometric diversity
        phase2_time = max_time * 0.5
        phase2_start = time.time()
        
        points = geometric_diversity_guided_optimization(points, max_iter=800)
        
        # Phase 3: Final optimization with simple gradient approach (if time permits)
        if time.time() - start_time < max_time * 0.9:
            try:
                # Simple gradient-based refinement using finite differences
                def objective_func(params):
                    p = params.reshape(-1, 2)
                    return -compute_min_max_ratio(p)
                
                # Flatten and optimize
                flat_points = points.flatten()
                bounds = [(0, 1) for _ in range(32)]
                
                # Simple optimization attempt
                result = minimize(objective_func, flat_points, method='L-BFGS-B', bounds=bounds, 
                                options={'maxiter': 200, 'ftol': 1e-8})
                
                if result.success:
                    points = result.x.reshape(-1, 2)
                    points = np.clip(points, 0, 1)
            except:
                pass  # Continue with current points if optimization fails
                
        return points

    # Main optimization procedure
    np.random.seed(42)
    
    # Generate diverse initial configurations
    initial_configs = generate_diverse_initial_configs()
    
    # Optimization parameters
    best_final_points = None
    best_final_ratio = -np.inf
    
    # Run optimization from each configuration
    for i, initial_config in enumerate(initial_configs):
        try:
            config_points = hybrid_optimization(initial_config, max_time=175 / len(initial_configs))
            config_ratio = compute_min_max_ratio(config_points)
            
            if config_ratio > best_final_ratio:
                best_final_ratio = config_ratio
                best_final_points = config_points.copy()
                
        except Exception as e:
            continue  # Skip failed configurations
    
    # Fallback to default if nothing worked
    if best_final_points is None:
        # Use golden ratio initialization as fallback
        best_final_points = generate_golden_ratio_initialization()
        
    return best_final_points

# EVOLVE-BLOCK-END