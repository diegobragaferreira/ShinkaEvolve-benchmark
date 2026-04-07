# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
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

    def create_hexagonal_grid():
        """Create an optimized hexagonal grid pattern for 16 points"""
        # Generate points in a hexagonal lattice pattern
        points = []
        
        # 4 rows of hexagonal packing
        rows = 4
        cols = 4
        
        # Hexagonal spacing constants
        sqrt3 = math.sqrt(3)
        row_spacing = 1.0 / (rows - 1) if rows > 1 else 1.0
        col_spacing = 1.0 / (cols - 1) if cols > 1 else 1.0
        
        for i in range(rows):
            for j in range(cols):
                # Offset every other row for hexagonal packing
                x_offset = 0.0 if i % 2 == 0 else 0.5 * col_spacing
                x = (j * col_spacing) + x_offset
                y = i * row_spacing
                
                # Ensure within bounds
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))
                
                points.append([x, y])
        
        return np.array(points)

    def adaptive_perturbation(points, current_ratio):
        """Apply adaptive perturbations based on current solution quality"""
        # Calculate current distance statistics
        distances = pdist(points)
        if len(distances) == 0:
            return points
        
        # Determine perturbation magnitude based on solution quality
        # If solution is poor (low ratio), use larger perturbations
        # If solution is good (high ratio), use smaller perturbations
        if current_ratio < 0.1:
            # Very poor solution - aggressive perturbation
            perturbation_magnitude = 0.05
        elif current_ratio < 0.2:
            # Poor solution - moderate perturbation
            perturbation_magnitude = 0.03
        else:
            # Good solution - fine perturbation
            perturbation_magnitude = 0.015
            
        # Apply perturbations
        np.random.seed(42)
        perturbed = points + np.random.normal(0, perturbation_magnitude, points.shape)
        
        # Clip to valid bounds
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0.001, 0.999)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0.001, 0.999)
        
        return perturbed

    def multi_stage_optimization(initial_points):
        """Perform multi-stage optimization with adaptive refinement"""
        current_points = initial_points.copy()
        current_ratio = evaluate_solution(current_points)
        
        # Stage 1: Coarse global search (if solution is poor)
        if current_ratio < 0.15:
            # Use a more aggressive optimization approach
            bounds = [(0.001, 0.999) for _ in range(32)]
            
            # Try multiple optimization runs with different parameters
            best_points = current_points.copy()
            best_ratio = current_ratio
            
            # Try with different tolerance levels
            for ftol, gtol in [(1e-8, 1e-8), (1e-10, 1e-10)]:
                try:
                    result = minimize(
                        objective,
                        current_points.flatten(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50, 'ftol': ftol, 'gtol': gtol}
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 2)
                        ratio = evaluate_solution(final_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            
                except:
                    continue
                    
            current_points = best_points
            current_ratio = best_ratio

        # Stage 2: Medium refinement - local optimization
        if current_ratio < 0.25:
            bounds = [(0.001, 0.999) for _ in range(32)]
            
            try:
                result = minimize(
                    objective,
                    current_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = evaluate_solution(final_points)
                    if ratio > current_ratio:
                        current_points = final_points
                        current_ratio = ratio
                        
            except:
                pass

        # Stage 3: Fine tuning - highest precision
        if current_ratio < 0.30:
            bounds = [(0.001, 0.999) for _ in range(32)]
            
            try:
                result = minimize(
                    objective,
                    current_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = evaluate_solution(final_points)
                    if ratio > current_ratio:
                        current_points = final_points
                        current_ratio = ratio
                        
            except:
                pass

        return current_points

    def create_better_initial():
        """Create improved initial configuration using hexagonal pattern with adaptive perturbation"""
        # Start with hexagonal grid
        initial_points = create_hexagonal_grid()
        
        # Apply adaptive perturbation based on initial solution quality
        initial_ratio = evaluate_solution(initial_points)
        if initial_ratio < 0.2:  # If initial solution isn't great
            initial_points = adaptive_perturbation(initial_points, initial_ratio)
            
        return initial_points

    # Main optimization loop with enhanced strategy
    best_ratio = -np.inf
    best_points = None
    
    # Strategy 1: Hexagonal grid + adaptive optimization
    try:
        initial_points = create_better_initial()
        refined_points = multi_stage_optimization(initial_points)
        ratio = evaluate_solution(refined_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = refined_points.copy()
            
    except Exception as e:
        pass

    # Strategy 2: Multiple hexagonal-based starting points with different perturbations
    try:
        # Generate perturbed versions
        for i in range(5):
            np.random.seed(42 + i * 10)
            
            # Start with hexagonal grid
            base_points = create_hexagonal_grid()
            
            # Add variation through different perturbations
            if i == 0:
                # Small random perturbation
                perturbation = np.random.normal(0, 0.01, base_points.shape)
            elif i == 1:
                # Medium perturbation
                perturbation = np.random.normal(0, 0.02, base_points.shape)
            elif i == 2:
                # Large perturbation
                perturbation = np.random.normal(0, 0.03, base_points.shape)
            elif i == 3:
                # Aggressive perturbation
                perturbation = np.random.normal(0, 0.04, base_points.shape)
            else:
                # Very aggressive perturbation with directional bias
                perturbation = np.random.normal(0, 0.05, base_points.shape)
                # Add slight bias towards center
                center = np.array([0.5, 0.5])
                for k in range(len(base_points)):
                    direction = center - base_points[k]
                    perturbation[k] += direction * 0.01
            
            perturbed_points = base_points + perturbation
            perturbed_points[:, 0] = np.clip(perturbed_points[:, 0], 0.001, 0.999)
            perturbed_points[:, 1] = np.clip(perturbed_points[:, 1], 0.001, 0.999)
            
            # Optimize this version
            refined_points = multi_stage_optimization(perturbed_points)
            ratio = evaluate_solution(refined_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
                
    except Exception as e:
        pass

    # Strategy 3: Focused grid search around promising regions
    try:
        # Test specific grid patterns
        test_patterns = [
            # Centered pattern
            [(0.4, 0.4), (0.6, 0.4), (0.4, 0.6), (0.6, 0.6)],
            # Diagonal pattern
            [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8)],
            # Cross pattern
            [(0.5, 0.2), (0.5, 0.8), (0.2, 0.5), (0.8, 0.5)]
        ]
        
        for pattern in test_patterns:
            base_points = np.array(pattern)
            if len(base_points) < 16:
                # Expand to 16 points by adding nearby points
                expanded_points = base_points.copy()
                for i in range(16 - len(base_points)):
                    base_point = base_points[i % len(base_points)]
                    # Add nearby point with small random offset
                    offset = np.random.normal(0, 0.05, 2)
                    new_point = np.clip(base_point + offset, 0.001, 0.999)
                    expanded_points = np.vstack([expanded_points, new_point])
                base_points = expanded_points[:16]
            else:
                base_points = base_points[:16]
            
            # Perturb around the base points
            np.random.seed(42)
            perturbed_points = base_points + np.random.normal(0, 0.02, base_points.shape)
            perturbed_points[:, 0] = np.clip(perturbed_points[:, 0], 0.001, 0.999)
            perturbed_points[:, 1] = np.clip(perturbed_points[:, 1], 0.001, 0.999)
            
            # Optimize this version
            refined_points = multi_stage_optimization(perturbed_points)
            ratio = evaluate_solution(refined_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
                
    except Exception as e:
        pass

    # Final refinement with high precision
    if best_points is not None:
        try:
            bounds = [(0.001, 0.999) for _ in range(32)]
            
            result = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = evaluate_solution(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
                    
        except Exception as e:
            pass

    # Fallback strategy if nothing worked well
    if best_points is None:
        # Use the hexagonal grid as fallback with small random noise
        fallback_points = create_hexagonal_grid()
        np.random.seed(42)
        fallback_points += np.random.normal(0, 0.005, fallback_points.shape)
        fallback_points[:, 0] = np.clip(fallback_points[:, 0], 0.001, 0.999)
        fallback_points[:, 1] = np.clip(fallback_points[:, 1], 0.001, 0.999)
        best_points = fallback_points

    return best_points

# EVOLVE-BLOCK-END