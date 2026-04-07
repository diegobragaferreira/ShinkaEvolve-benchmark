# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, KDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time
from collections import defaultdict

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a Voronoi-guided local search approach:
    1. Generate initial candidate positions via weighted Voronoi tessellation
    2. Apply a hybrid optimization algorithm combining evolutionary and local search techniques
    3. Use kd-tree for efficient collision detection
    4. Apply post-processing to expand boundary circles

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    def check_collision(circle1, circle2):
        """Check if two circles collide"""
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2
        dist_sq = (x1-x2)**2 + (y1-y2)**2
        return dist_sq < (r1 + r2)**2

    def check_containment(circle):
        """Check if circle is fully contained in unit square"""
        x, y, r = circle
        return r <= x <= 1-r and r <= y <= 1-r

    def get_total_radius(circles_array):
        """Calculate sum of all radii"""
        return np.sum(circles_array[:, 2])

    def build_kd_tree(circles_array):
        """Build KDTree for efficient spatial queries"""
        return KDTree(circles_array[:, :2])

    def get_neighbors(kdtree, circle, max_radius):
        """Get nearby circles using KDTree"""
        neighbors = kdtree.query_ball_point(circle[:2], 2 * max_radius)
        return neighbors

    def is_valid_configuration(circles_array, kdtree=None):
        """Check if configuration is valid (no overlaps, fully contained)"""
        n = len(circles_array)

        # Check containment
        for circle in circles_array:
            if not check_containment(circle):
                return False

        # Check collisions efficiently using KDTree if available
        if kdtree is not None:
            for i in range(n):
                circle = circles_array[i]
                neighbors = get_neighbors(kdtree, circle, circle[2])
                for j in neighbors:
                    if i != j and check_collision(circle, circles_array[j]):
                        return False
        else:
            # Fallback to brute force
            for i in range(n):
                for j in range(i+1, n):
                    if check_collision(circles_array[i], circles_array[j]):
                        return False

        return True

    def evaluate_fitness(circles_array):
        """Fitness function with adaptive penalty handling"""
        if not is_valid_configuration(circles_array):
            # Calculate penalty based on violation severity
            penalty = 0
            
            # Count containment violations
            containment_violations = 0
            for circle in circles_array:
                if not check_containment(circle):
                    containment_violations += 1
                    
            # Count overlap violations
            overlap_violations = 0
            for i in range(len(circles_array)):
                for j in range(i+1, len(circles_array)):
                    if check_collision(circles_array[i], circles_array[j]):
                        overlap_violations += 1
            
            # Apply penalty based on number of violations
            penalty = -1000 * (containment_violations + overlap_violations)
            return penalty
            
        return get_total_radius(circles_array)

    def initialize_with_weighted_voronoi():
        """Initialize circle positions using weighted Voronoi approach"""
        # Generate initial points using a grid-like pattern
        n_points = 128  # More points for better Voronoi coverage
        points = []

        # Create a systematic point distribution with some randomness
        for i in range(12):
            for j in range(12):
                x = 0.05 + i * 0.0833 + np.random.uniform(-0.01, 0.01)
                y = 0.05 + j * 0.0833 + np.random.uniform(-0.01, 0.01)
                if 0 <= x <= 1 and 0 <= y <= 1:
                    points.append([x, y])

        points = np.array(points)

        # Create Voronoi diagram
        try:
            vor = Voronoi(points)

            # Get Voronoi vertices as candidate centers
            candidates = []
            weights = []
            for vertex in vor.vertices:
                if 0 <= vertex[0] <= 1 and 0 <= vertex[1] <= 1:
                    candidates.append(vertex)
                    # Compute boundary weight - higher weight for vertices near edges/corners
                    x, y = vertex
                    dist_to_boundary = min(x, 1-x, y, 1-y)
                    # Higher weight for closer to boundary
                    weight = max(0.5, 1.0 - dist_to_boundary * 3.0)
                    weights.append(weight)

            # If we have enough candidates, sample with weights
            if len(candidates) >= 32:
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                selected_indices = np.random.choice(len(candidates), 32, replace=False, p=weights)
                centers = np.array([candidates[i] for i in selected_indices])
            else:
                # Fall back to sampling original points
                selected_indices = np.random.choice(len(points), 32, replace=False)
                centers = points[selected_indices]

        except:
            # Fallback to simple random initialization
            centers = np.random.rand(32, 2)

        # Initialize with small radii
        circles = np.zeros((32, 3))
        circles[:, 0] = centers[:, 0]
        circles[:, 1] = centers[:, 1]
        circles[:, 2] = 0.015  # Slightly smaller initial radius

        return circles

    def mutate(circles_array, generation, max_gen):
        """Mutate the circles array with adaptive mutation rate"""
        mutated = circles_array.copy()
        # Adaptive mutation rate - decrease over generations
        mutation_rate = 0.15 * (1 - generation/max_gen)
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Mutate center position with adaptive strength
                pos_mutation = np.random.normal(0, 0.005 + generation * 0.0001)
                mutated[i, 0] += pos_mutation
                mutated[i, 1] += pos_mutation
                
                # Clamp positions to valid range
                mutated[i, 0] = np.clip(mutated[i, 0], 0, 1)
                mutated[i, 1] = np.clip(mutated[i, 1], 0, 1)

                # Mutate radius
                rad_mutation = np.random.normal(0, 0.005)
                mutated[i, 2] += rad_mutation
                mutated[i, 2] = max(0.001, mutated[i, 2])  # Ensure positive radius

        return mutated

    def optimize_single_circle(circles_array, idx, kdtree):
        """Optimize a single circle using gradient ascent with constraint checking"""
        original = circles_array[idx].copy()
        best = circles_array.copy()
        best_fitness = evaluate_fitness(best)
        circle = circles_array[idx]
        
        # Try several directions for local search
        for _ in range(20):
            test = circles_array.copy()
            # Small random perturbation
            test[idx, 0] += np.random.normal(0, 0.002)
            test[idx, 1] += np.random.normal(0, 0.002)
            test[idx, 2] += np.random.normal(0, 0.001)
            
            # Clamp to bounds
            test[idx, 0] = np.clip(test[idx, 0], 0, 1)
            test[idx, 1] = np.clip(test[idx, 1], 0, 1)
            test[idx, 2] = max(0.001, test[idx, 2])
            
            # Check if it's valid
            if is_valid_configuration(test, kdtree):
                fitness = evaluate_fitness(test)
                if fitness > best_fitness:
                    best = test.copy()
                    best_fitness = fitness
                    
        return best

    def expand_boundary_circles(circles_array):
        """Expand circles that are near boundaries to their maximum possible size"""
        expanded = circles_array.copy()
        
        # For each circle near boundary, try to increase radius
        for i in range(len(expanded)):
            x, y, r = expanded[i]
            
            # Check if circle is near boundary
            if min(x, 1-x, y, 1-y) < 0.05:  # Within 5% of edge
                # Try to expand radius while maintaining validity
                max_radius = min(x, 1-x, y, 1-y)  # Maximum possible radius for this position
                
                # Binary search approach to find largest valid radius
                low = r
                high = max_radius
                best_radius = r
                
                # Check if we can safely increase
                temp_circles = expanded.copy()
                temp_circles[i, 2] = high
                if is_valid_configuration(temp_circles):
                    # Binary search for maximum radius
                    while low <= high:
                        mid = (low + high) / 2
                        temp_circles[i, 2] = mid
                        if is_valid_configuration(temp_circles):
                            best_radius = mid
                            low = mid + 0.0001
                        else:
                            high = mid - 0.0001
                        if high - low < 0.0001:
                            break
                    
                    expanded[i, 2] = best_radius
                    
        return expanded

    def local_search_step(circles_array, kdtree):
        """Perform a comprehensive local search step"""
        # Try to improve each circle individually
        improved = circles_array.copy()
        for i in range(len(improved)):
            improved = optimize_single_circle(improved, i, kdtree)
            
        return improved

    # Main optimization algorithm
    max_iterations = 200
    best_fitness = float('-inf')
    best_solution = None
    
    # Initialize population with high-quality samples
    initial_solutions = []
    for _ in range(5):
        initial = initialize_with_weighted_voronoi()
        initial_solutions.append(initial)
        fitness = evaluate_fitness(initial)
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = initial.copy()

    # Iterative improvement using local search
    for iteration in range(max_iterations):
        # Sample from existing good solutions
        if len(initial_solutions) > 0:
            current = initial_solutions[np.random.randint(len(initial_solutions))]
        else:
            current = initialize_with_weighted_voronoi()
            
        # Build KDTree for fast collision detection
        kdtree = build_kd_tree(current)
        
        # Local search
        current = local_search_step(current, kdtree)
        
        # Try boundary expansion
        current = expand_boundary_circles(current)
        
        # Recalculate fitness after improvements
        fitness = evaluate_fitness(current)
        
        # Update best if improved
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = current.copy()
            
        # Store this solution for future sampling
        if len(initial_solutions) < 10:
            initial_solutions.append(current)
        else:
            # Replace oldest solution
            initial_solutions[iteration % 10] = current
            
        # Print progress
        if iteration % 20 == 0:
            print(f"Iteration {iteration}: Best fitness = {best_fitness:.6f}")

    # Final refinement
    if best_solution is not None:
        # Final boundary expansion
        final_solution = expand_boundary_circles(best_solution)
        
        # Perform one final local search
        kdtree_final = build_kd_tree(final_solution)
        final_solution = local_search_step(final_solution, kdtree_final)
        
        # Ensure final validity
        if is_valid_configuration(final_solution):
            return final_solution
        else:
            # Return the last valid solution found
            return best_solution
    else:
        # Return the best initialization
        return initialize_with_weighted_voronoi()

# EVOLVE-BLOCK-END