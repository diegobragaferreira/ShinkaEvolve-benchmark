# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def _validate_placement(circles: np.ndarray) -> bool:
    """Validate that circles are within bounds and don't overlap."""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check non-overlap constraints using KDTree for efficiency
    points = circles[:, :2]
    try:
        tree = cKDTree(points)
        # Query pairs within a reasonable distance to avoid O(n^2) check
        # This uses a small multiple of the smallest radius to find close pairs
        min_radius = min(circles[:, 2]) if len(circles[:, 2]) > 0 else 0.01
        pairs = tree.query_pairs(2 * min_radius, output_type='ndarray')
        
        for i, j in pairs:
            if i < j:  # Avoid double-checking
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2
                if distance_sq < min_distance_sq:
                    return False
    except Exception:
        # Fallback to brute force for validation
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2
                if distance_sq < min_distance_sq:
                    return False
                    
    return True

def _generate_hexagonal_initialization(n_circles: int) -> np.ndarray:
    """Generate initial circle positions using hexagonal lattice for better density."""
    circles = np.zeros((n_circles, 3))
    
    # Use hexagonal packing approach for better initial distribution
    # Calculate grid dimensions for roughly n circles
    rows = int(np.ceil(np.sqrt(n_circles / 0.866)))  # 0.866 is hexagonal packing density
    cols = int(np.ceil(n_circles / rows))

    # For hexagonal pattern
    hex_spacing_x = 0.95 / cols
    hex_spacing_y = hex_spacing_x * 0.866  # sqrt(3)/2

    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n_circles:
                break
            # Offset every other row
            offset = (i % 2) * (hex_spacing_x / 2)
            x = (j * hex_spacing_x + offset) + 0.025
            y = (i * hex_spacing_y) + 0.025
            
            # Initial radius based on spacing and some randomness
            r = min(hex_spacing_x, hex_spacing_y) * random.uniform(0.3, 0.5)
            r = min(r, x, 1-x, y, 1-y)
            
            circles[count] = [x, y, r]
            count += 1
            
    return circles

def _generate_grid_initialization(n_circles: int) -> np.ndarray:
    """Generate initial grid-based circle positions."""
    circles = np.zeros((n_circles, 3))
    
    # Create a regular grid pattern with some randomness
    sqrt_n = int(np.ceil(np.sqrt(n_circles)))
    rows = int(np.ceil(n_circles / sqrt_n))
    cols = int(np.ceil(n_circles / rows))
    
    spacing_x = 0.95 / (cols + 1)
    spacing_y = 0.95 / (rows + 1)
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n_circles:
                break
            # Add some randomness to avoid systematic patterns
            x = (j + 1) * spacing_x + random.uniform(-spacing_x/4, spacing_x/4)
            y = (i + 1) * spacing_y + random.uniform(-spacing_y/4, spacing_y/4)
            # Initial radius based on spacing and some randomness
            r = min(spacing_x, spacing_y) * random.uniform(0.3, 0.5)
            circles[count] = [x, y, r]
            count += 1
            
    return circles

def _generate_voronoi_initialization(n_circles: int) -> np.ndarray:
    """Generate initial circle positions using advanced Voronoi-inspired approach."""
    # Generate candidates in a grid pattern with jitter
    grid_size = max(4, int(np.ceil(np.sqrt(n_circles * 1.2))))
    x_coords = np.linspace(0.05, 0.95, grid_size)
    y_coords = np.linspace(0.05, 0.95, grid_size)
    
    # Generate all grid points with strategic jittering
    grid_points = []
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            # Apply jittering with grid-dependent pattern
            jitter_x = np.random.uniform(-0.02, 0.02) * (1.0 + 0.1 * np.sin(i * 0.5))
            jitter_y = np.random.uniform(-0.02, 0.02) * (1.0 + 0.1 * np.cos(j * 0.5))
            grid_points.append([x + jitter_x, y + jitter_y])

    # If we have more circles than grid points, add some random points
    if len(grid_points) < n_circles:
        extra_points = n_circles - len(grid_points)
        for _ in range(extra_points):
            grid_points.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])

    # Shuffle the points to avoid systematic bias
    random.shuffle(grid_points)

    # Take the first n_circles points
    points = np.array(grid_points[:n_circles])

    # Initialize circles with calculated initial radii based on expected density
    circles = np.zeros((n_circles, 3))
    circles[:, 0] = points[:, 0]  # x coordinates
    circles[:, 1] = points[:, 1]  # y coordinates
    
    # Estimate initial radii based on expected density
    avg_density = n_circles / (0.9 * 0.9)  # Expected density in unit square
    estimated_radius = min(0.1, 0.5 / np.sqrt(avg_density))
    circles[:, 2] = max(0.01, estimated_radius)  # Initial small radii

    return circles

def _local_search_improvement(circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """Apply constraint-aware local search to fine-tune the solution."""
    improved_circles = circles.copy()
    n = len(improved_circles)
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try to increase radii while maintaining constraints
        for i in range(n):
            x, y, r = improved_circles[i]
            
            # Calculate maximum possible radius at this position
            max_radius = min(x, 1-x, y, 1-y)
            
            # Binary search for maximum safe radius
            low, high = r, max_radius
            best_radius = r
            
            # Binary search iterations for efficiency
            for _ in range(15):
                if abs(high - low) < 1e-6:
                    break
                mid = (low + high) / 2
                valid = True
                
                # Check overlap constraints quickly
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = improved_circles[j]
                        dist_sq = (x - x2)**2 + (y - y2)**2
                        if dist_sq < (mid + r2)**2:
                            valid = False
                            break
                
                if valid:
                    best_radius = mid
                    low = mid
                else:
                    high = mid
            
            # Apply improvement if beneficial
            if best_radius > r + 1e-6:
                improved_circles[i, 2] = best_radius
                improved = True
        
        # If no improvements from radius increases, try position adjustments
        if not improved:
            # Process only a subset of circles for efficiency  
            circle_indices = list(range(n))  
            random.shuffle(circle_indices)
            
            for i in circle_indices[:max(n//4, 1)]:  # Process about 25% randomly
                x, y, r = improved_circles[i]
                
                # Try small movements in 4 directions for speed
                movements = [(-0.005, 0), (0.005, 0), (0, -0.005), (0, 0.005)]
                
                best_x, best_y = x, y
                best_score = -float('inf')
                best_radius = r
                
                for dx, dy in movements:
                    new_x, new_y = x + dx, y + dy
                    
                    # Check bounds
                    if new_x - r < 0 or new_x + r > 1 or new_y - r < 0 or new_y + r > 1:
                        continue
                        
                    # Quick overlap check with fewest neighbors
                    overlap_penalty = 0
                    valid = True
                    
                    # Only check with a few closest neighbors for efficiency
                    points = improved_circles[:, :2]
                    distances = np.sqrt(np.sum((points - [new_x, new_y])**2, axis=1))
                    closest_indices = np.argsort(distances)[:min(5, len(distances))]
                    
                    for j in closest_indices:
                        if i != j:
                            x2, y2, r2 = improved_circles[j]
                            dist_sq = (new_x - x2)**2 + (new_y - y2)**2
                            min_dist_sq = (r + r2)**2
                            if dist_sq < min_dist_sq:
                                overlap_penalty += (min_dist_sq - dist_sq) * 1000
                                valid = False
                    
                    if valid:
                        # Simple scoring based on radius
                        score = r - overlap_penalty * 0.0001
                        if score > best_score:
                            best_score = score
                            best_x, best_y = new_x, new_y
                
                # Apply the best movement if it helps
                if best_x != x or best_y != y:
                    improved_circles[i, 0] = best_x
                    improved_circles[i, 1] = best_y
                    improved = True
        
        # Exit early if no improvements
        if not improved:
            break
    
    return improved_circles

def _constraint_aware_local_search(circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """Apply constraint-aware local search to fine-tune the solution."""
    improved_circles = circles.copy()
    n = len(improved_circles)
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try to increase radii while maintaining constraints
        for i in range(n):
            x, y, r = improved_circles[i]
            
            # Calculate maximum possible radius at this position
            max_radius = min(x, 1-x, y, 1-y)
            
            if max_radius <= r + 0.001:
                continue
                
            # Find neighbors more efficiently
            neighbors = []
            for j in range(n):
                if i != j:
                    x2, y2, r2 = improved_circles[j]
                    dist_sq = (x - x2)**2 + (y - y2)**2
                    min_dist_sq = (r + r2)**2
                    if dist_sq < min_dist_sq * 1.1:  # Allow some buffer
                        neighbors.append((j, dist_sq, min_dist_sq))
            
            # Binary search for maximum safe radius
            low, high = r, max_radius
            best_radius = r
            
            # Limited binary search iterations for efficiency
            for _ in range(15):
                if abs(high - low) < 1e-6:
                    break
                mid = (low + high) / 2
                valid = True
                
                # Check overlap constraints quickly
                for j, dist_sq, min_dist_sq in neighbors:
                    x2, y2, r2 = improved_circles[j]
                    dist_sq = (x - x2)**2 + (y - y2)**2
                    if dist_sq < (mid + r2)**2:
                        valid = False
                        break
                
                if valid:
                    best_radius = mid
                    low = mid
                else:
                    high = mid
            
            # Apply improvement if beneficial
            if best_radius > r + 1e-6:
                improved_circles[i, 2] = best_radius
                improved = True
        
        # If no improvements from radius increases, try position adjustments
        if not improved:
            # Process only a subset of circles for efficiency
            circle_indices = list(range(n))  
            random.shuffle(circle_indices)
            
            for i in circle_indices[:max(n//4, 1)]:  # Process about 25% randomly
                x, y, r = improved_circles[i]
                
                # Try small movements in 4 directions for speed
                movements = [(-0.005, 0), (0.005, 0), (0, -0.005), (0, 0.005)]
                
                best_x, best_y = x, y
                best_score = -float('inf')
                best_radius = r
                
                for dx, dy in movements:
                    new_x, new_y = x + dx, y + dy
                    
                    # Check bounds
                    if new_x - r < 0 or new_x + r > 1 or new_y - r < 0 or new_y + r > 1:
                        continue
                        
                    # Quick overlap check with fewest neighbors
                    overlap_penalty = 0
                    valid = True
                    
                    # Only check with a few closest neighbors for efficiency
                    points = improved_circles[:, :2]
                    distances = np.sqrt(np.sum((points - [new_x, new_y])**2, axis=1))
                    closest_indices = np.argsort(distances)[:min(5, len(distances))]
                    
                    for j in closest_indices:
                        if i != j:
                            x2, y2, r2 = improved_circles[j]
                            dist_sq = (new_x - x2)**2 + (new_y - y2)**2
                            min_dist_sq = (r + r2)**2
                            if dist_sq < min_dist_sq:
                                overlap_penalty += (min_dist_sq - dist_sq) * 1000
                                valid = False
                    
                    if valid:
                        # Simple scoring based on radius
                        score = r - overlap_penalty * 0.0001
                        if score > best_score:
                            best_score = score
                            best_x, best_y = new_x, new_y
                
                # Apply the best movement if it helps
                if best_x != x or best_y != y:
                    improved_circles[i, 0] = best_x
                    improved_circles[i, 1] = best_y
                    improved = True
            
            # Exit early if no improvements
            if not improved:
                break
    
    return improved_circles

def _improve_with_gradient_descent(circles: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Apply gradient-based improvement with smarter radius adjustments."""
    improved_circles = circles.copy()
    n = len(improved_circles)
    
    for iteration in range(max_iter):
        updated = False
        
        # Process circles in batches for parallel-like efficiency
        batch_size = max(1, n // 4)
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices[:batch_size]:
            x, y, r = improved_circles[i]
            max_radius = min(x, 1-x, y, 1-y)
            
            # Quick check - only proceed if there's potential for improvement
            if max_radius <= r + 0.001:
                continue
            
            # Find closest neighbors for constraint checking
            points = improved_circles[:, :2]
            distances = np.sqrt(np.sum((points - [x, y])**2, axis=1))
            closest_indices = np.argsort(distances)[1:min(6, len(distances))]  # Up to 5 neighbors
            
            # Check overlap constraints efficiently
            valid = True
            current_max_radius = max_radius
            
            for j in closest_indices:
                x2, y2, r2 = improved_circles[j]
                dist_sq = (x - x2)**2 + (y - y2)**2
                if dist_sq < (r + r2)**2:
                    valid = False
                    break
            
            if valid:
                # Try to increase radius
                if r < max_radius - 0.001:
                    test_radius = min(max_radius, r + 0.005)
                    # Quick verification
                    valid_test = True
                    for j in closest_indices:
                        x2, y2, r2 = improved_circles[j]
                        dist_sq = (x - x2)**2 + (y - y2)**2
                        if dist_sq < (test_radius + r2)**2:
                            valid_test = False
                            break
                    
                    if valid_test:
                        improved_circles[i, 2] = test_radius
                        updated = True
            else:
                # Reduce radius to resolve overlap
                safe_radius = max(0.001, min(r, max_radius))
                if safe_radius < r - 1e-6:
                    improved_circles[i, 2] = safe_radius
                    updated = True
        
        if not updated:
            break
    
    return improved_circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid evolutionary-local search approach with optimized initialization.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    seed = 42
    
    # Generate initial configuration using Voronoi-inspired method
    circles = _generate_voronoi_initialization(n)
    
    # Apply initial local search to improve the starting point
    circles = _constraint_aware_local_search(circles, max_iterations=100)
    
    # Apply gradient-based improvement
    circles = _improve_with_gradient_descent(circles, max_iter=50)
    
    # Final validation and refinement
    if not _validate_placement(circles):
        # Use fallback if necessary
        circles = _generate_hexagonal_initialization(n)
        circles = _constraint_aware_local_search(circles, max_iterations=100)
        
    return circles

# EVOLVE-BLOCK-END