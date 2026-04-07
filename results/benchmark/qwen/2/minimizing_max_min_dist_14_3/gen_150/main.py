# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import math
import random
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def compute_min_max_ratio(points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum distances."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0.0
        return d_min / d_max

    def fibonacci_sphere(n):
        """Generate n points distributed approximately uniformly on a sphere."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            # Distribute points more evenly
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)

            # Better distribution using Fibonacci sequence
            theta = np.arctan2(np.sin(i * 2 * np.pi / phi), np.cos(i * 2 * np.pi / phi))
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def project_to_sphere(points):
        """Project points onto unit sphere."""
        if points.ndim == 1:
            points = points.reshape(1, -1)
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms[:, np.newaxis]

    def compute_targeted_perturbation(points, idx, temp):
        """Compute a targeted perturbation based on local geometry analysis."""
        try:
            # Get distances to all other points
            distances = cdist([points[idx]], points)[0]
            distances = distances[distances > 0]  # Remove self-distance
            
            if len(distances) == 0:
                # No neighbors, random perturbation
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                return direction * 0.01 * temp
            
            avg_distance = np.mean(distances)
            min_distance = np.min(distances)
            max_distance = np.max(distances)
            
            # Analyze the point's local geometry
            # If point is too close to others, push it away
            if min_distance < avg_distance * 0.4:
                # Compute repulsion force from close neighbors
                repulsion = np.zeros(3)
                for i in range(len(points)):
                    if i != idx:
                        diff = points[idx] - points[i]
                        dist = np.linalg.norm(diff)
                        if dist > 0 and dist < avg_distance * 0.7:
                            # Inverse distance weighted repulsion
                            repulsion += diff / dist * (1.0 / dist**2)
                
                # If there's repulsion, normalize and apply
                if np.linalg.norm(repulsion) > 0:
                    repulsion = repulsion / np.linalg.norm(repulsion)
                    # Magnitude depends on how close it is
                    magnitude = 0.05 * (1.0 - min_distance / avg_distance) * temp
                    return repulsion * magnitude
                else:
                    # Fallback to random perturbation
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    return direction * 0.02 * temp
                    
            elif max_distance > avg_distance * 1.5:
                # If point is far from others, maybe pull it closer to balance
                # Compute attraction toward average position
                attraction = np.mean(points, axis=0) - points[idx]
                attraction_norm = np.linalg.norm(attraction)
                if attraction_norm > 0:
                    attraction = attraction / attraction_norm
                    # Magnitude inversely related to distance from center
                    center_distance = np.linalg.norm(points[idx])
                    magnitude = 0.01 * (1.0 - center_distance * 0.5) * temp
                    return -attraction * magnitude
                else:
                    # Random perturbation
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    return direction * 0.01 * temp
                    
            else:
                # Moderate distances - small adjustment
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                # Magnitude based on how balanced the distances are
                balance_score = abs(min_distance - avg_distance) / avg_distance + \
                               abs(max_distance - avg_distance) / avg_distance
                magnitude = 0.01 * (1.0 - balance_score * 0.5) * temp
                return direction * magnitude
                
        except Exception:
            # Fallback to simple random perturbation
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            return direction * 0.01 * temp

    def simulated_annealing_optimization(initial_points, max_iter=15000, temp_schedule=None):
        """Optimize points using simulated annealing with adaptive cooling."""
        current_points = initial_points.copy()
        current_points = project_to_sphere(current_points)

        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)

        # Adaptive cooling schedule with dynamic adjustments
        temperature = 0.1
        cooling_rate = 0.9995
        min_temp = 1e-8
        patience_counter = 0
        patience_limit = 3000
        
        # Track recent improvements for adaptive cooling
        recent_improvements = []
        max_recent = 20

        for iteration in range(max_iter):
            # Store current configuration
            old_points = current_points.copy()
            old_ratio = compute_min_max_ratio(current_points)

            # Select random point to perturb
            idx = np.random.randint(len(current_points))

            # Apply targeted perturbation
            perturbation = compute_targeted_perturbation(current_points, idx, temperature)
            
            # Apply perturbation
            current_points[idx] += perturbation

            # Project back to sphere
            current_points[idx] = project_to_sphere(current_points[idx:idx+1])[0]

            # Compute new ratio
            new_ratio = compute_min_max_ratio(current_points)

            # Accept or reject based on Metropolis criterion
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = current_points.copy()
                patience_counter = 0  # Reset patience when improvement found
            elif np.random.random() < np.exp((new_ratio - old_ratio) / temperature):
                # Accept worse solution with probability
                pass  # Keep the new configuration
            else:
                # Revert to previous configuration
                current_points = old_points

            # Adaptive cooling based on recent performance
            recent_improvements.append(new_ratio)
            if len(recent_improvements) > max_recent:
                recent_improvements.pop(0)
                
            # Adjust cooling rate based on recent progress
            if len(recent_improvements) >= 5:
                recent_improvement = recent_improvements[-1] - recent_improvements[0]
                if recent_improvement < 1e-8:
                    # Very slow progress, cool faster
                    cooling_rate = min(0.9999, cooling_rate * 1.05)  # Speed up cooling
                elif recent_improvement > 1e-5:
                    # Fast progress, cool slower
                    cooling_rate = max(0.999, cooling_rate * 0.98)  # Slow down cooling
                    
            # Apply temperature cooling
            temperature = max(min_temp, temperature * cooling_rate)
            
            # Early stopping if no improvement for a long time
            patience_counter += 1
            if patience_counter > patience_limit:
                break

            # Periodic restart to escape local optima
            if iteration % 5000 == 0 and iteration > 0:
                current_ratio = compute_min_max_ratio(current_points)
                if current_ratio < best_ratio * 0.99:
                    # Restart with better configuration
                    current_points = best_points.copy()
                    temperature = min(temperature * 1.5, 0.5)  # Increase temperature
                    patience_counter = 0

        return best_points, best_ratio

    # Generate multiple diverse initial configurations
    initial_points_set = []
    
    # Phase 1: Multiple Fibonacci sphere initializations with different offsets
    for seed in range(5):
        np.random.seed(seed * 100)
        points = fibonacci_sphere(14)
        # Add slight perturbations to break symmetries
        points += np.random.normal(0, 0.01, (14, 3))
        points = project_to_sphere(points)
        initial_points_set.append(points)
    
    # Phase 2: Icosahedral-inspired initialization
    np.random.seed(42)
    ico_points = generate_icosahedral_points()
    ico_points = project_to_sphere(ico_points)
    initial_points_set.append(ico_points)
    
    # Phase 3: Random perturbed spherical initialization
    random_points = np.random.randn(14, 3)
    random_points = project_to_sphere(random_points)
    initial_points_set.append(random_points)

    # Optimize each initialization and keep the best
    best_points = None
    best_ratio = 0

    for i, initial_points in enumerate(initial_points_set):
        optimized_points, final_ratio = simulated_annealing_optimization(initial_points, 10000)
        
        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_points = optimized_points.copy()
    
    # Refine the best solution with additional optimization
    if best_points is not None:
        final_points, _ = simulated_annealing_optimization(best_points, 5000)
        return final_points

    # Fallback to single optimization if nothing worked
    initial_points = fibonacci_sphere(14)
    initial_points = project_to_sphere(initial_points)
    optimized_points, _ = simulated_annealing_optimization(initial_points, 15000)
    return optimized_points

def generate_icosahedral_points():
    """Generate points arranged like an icosahedron for better symmetry."""
    # Icosahedron vertices scaled to unit sphere
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    points = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    
    # Convert to numpy array and normalize
    points = np.array(points)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    # We need 14 points, so add two more from edge midpoints
    # For simplicity, we'll take the first two points and create additional ones
    # by averaging pairs and normalizing
    additional_points = []
    for i in range(0, len(points), 2):
        if len(additional_points) < 2:
            mid = (points[i] + points[i+1]) / 2
            norm = np.linalg.norm(mid)
            if norm > 0:
                additional_points.append(mid / norm)
    
    # Combine all points
    all_points = np.vstack([points[:12], additional_points])
    
    # Ensure exactly 14 points by padding or truncating
    if len(all_points) < 14:
        # Fill with random points on sphere
        extra_points = fibonacci_sphere(14 - len(all_points))
        all_points = np.vstack([all_points, extra_points])
    elif len(all_points) > 14:
        all_points = all_points[:14]
        
    return all_points

def fibonacci_sphere(n):
    """Generate n points distributed approximately uniformly on a sphere."""
    points = []
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    for i in range(n):
        # Distribute points more evenly
        z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
        radius = np.sqrt(1 - z*z)

        # Better distribution using Fibonacci sequence
        theta = np.arctan2(np.sin(i * 2 * np.pi / phi), np.cos(i * 2 * np.pi / phi))
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)

# EVOLVE-BLOCK-END