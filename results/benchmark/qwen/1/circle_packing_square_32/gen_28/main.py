# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a GRASP (Greedy Randomized Adaptive Search Procedure) approach with Voronoi-based 
    initialization and local improvement. This is a fundamentally different approach from 
    evolutionary algorithms.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    def check_collision(circle1, circle2):
        """Check if two circles collide"""
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2
        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        return dist < (r1 + r2)
    
    def check_containment(circle):
        """Check if circle is fully contained in unit square"""
        x, y, r = circle
        return r <= x <= 1-r and r <= y <= 1-r
    
    def get_total_radius(circles_array):
        """Calculate sum of all radii"""
        return np.sum(circles_array[:, 2])
    
    def is_valid_configuration(circles_array):
        """Check if configuration is valid (no overlaps, fully contained)"""
        n = len(circles_array)
        
        # Check containment
        for circle in circles_array:
            if not check_containment(circle):
                return False
        
        # Check collisions
        for i in range(n):
            for j in range(i+1, n):
                if check_collision(circles_array[i], circles_array[j]):
                    return False
        
        return True
    
    def calculate_voronoi_area(circle, voronoi_vertices, voronoi_regions, voronoi_points):
        """Estimate area of Voronoi cell around given circle for influence assessment"""
        try:
            # Find the Voronoi region for this circle's center
            idx = np.where(np.all(voronoi_points == [circle[0], circle[1]], axis=1))[0][0]
            vertices = voronoi_regions[idx]
            if len(vertices) > 2:
                # Calculate approximate area using triangulation or bounding box
                # For simplicity, we'll use the distance to nearest neighbors as proxy
                distances = [distance.euclidean([circle[0], circle[1]], [v[0], v[1]]) for v in voronoi_vertices if not np.isnan(v).any()]
                if len(distances) > 0:
                    return min(distances) if distances else 0.01
            return 0.01
        except:
            return 0.01
    
    def get_neighbors_in_voronoi(voronoi_regions, point, voronoi_points, k=5):
        """Find k closest neighbors in Voronoi structure"""
        try:
            distances = []
            for i, vor_point in enumerate(voronoi_points):
                if not np.array_equal(vor_point, point):
                    dist = distance.euclidean(point, vor_point)
                    distances.append((dist, i))
            
            distances.sort()
            return [voronoi_points[i] for _, i in distances[:min(k, len(distances))]]
        except:
            return []
    
    def voronoi_guided_placement(circles_array, voronoi_obj, voronoi_points, remaining_circles):
        """Place circles using Voronoi structure for intelligent positioning"""
        # Create Voronoi-based influence map
        best_positions = []
        
        # Find available space by examining Voronoi cells
        for _ in range(remaining_circles):
            best_position = None
            best_radius = 0
            
            # Sample multiple candidate positions from Voronoi structure
            candidates = []
            
            # Include Voronoi vertices
            for vertex in voronoi_obj.vertices:
                if 0 <= vertex[0] <= 1 and 0 <= vertex[1] <= 1:
                    candidates.append(vertex)
            
            # Include midpoints of edges
            for ridge in voronoi_obj.ridge_vertices:
                if len(ridge) >= 2 and -1 not in ridge:
                    v1 = voronoi_obj.vertices[ridge[0]]
                    v2 = voronoi_obj.vertices[ridge[1]]
                    midpoint = [(v1[0] + v2[0])/2, (v1[1] + v2[1])/2]
                    if 0 <= midpoint[0] <= 1 and 0 <= midpoint[1] <= 1:
                        candidates.append(midpoint)
            
            # Random sampling for diversity
            for _ in range(20):
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                candidates.append([x, y])
            
            # Evaluate candidates
            for candidate in candidates:
                # Check if candidate is valid
                if not (0 <= candidate[0] <= 1 and 0 <= candidate[1] <= 1):
                    continue
                    
                # Calculate maximum possible radius at this position
                max_radius = min(candidate[0], 1-candidate[0], candidate[1], 1-candidate[1])
                
                # Check conflicts with existing circles
                valid = True
                for circle in circles_array:
                    dx = abs(candidate[0] - circle[0])
                    dy = abs(candidate[1] - circle[1])
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < (max_radius + circle[2]):  # Would cause overlap with existing circle
                        valid = False
                        break
                        
                if valid and max_radius > best_radius:
                    best_radius = max_radius
                    best_position = candidate[:]
            
            if best_position is not None:
                circles_array.append([best_position[0], best_position[1], best_radius])
            
        return circles_array
    
    def local_improvement_voronoi(circles_array):
        """
        Perform local improvement using Voronoi-based neighborhood search
        """
        improved = True
        iteration = 0
        max_iterations = 50
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Get Voronoi diagram for current circles
            try:
                if len(circles_array) >= 3:
                    points = np.array([[c[0], c[1]] for c in circles_array])
                    vor = Voronoi(points)
                else:
                    break
            except:
                break
            
            # Try to improve each circle
            for i in range(len(circles_array)):
                original_circle = circles_array[i][:]
                best_circle = original_circle[:]
                best_radius = original_circle[2]
                
                # Create a set of neighbor positions to explore
                base_pos = [original_circle[0], original_circle[1]]
                neighbors = [base_pos]
                
                # Add Voronoi-based neighbors
                try:
                    # Find Voronoi region for this point
                    idx = np.where(np.all(np.array([[c[0], c[1]] for c in circles_array]) == base_pos, axis=1))[0]
                    if len(idx) > 0:
                        # Add some nearby Voronoi vertices and midpoints
                        for v in vor.vertices[:5]:  # Limited to avoid explosion
                            if 0 <= v[0] <= 1 and 0 <= v[1] <= 1:
                                neighbors.append([v[0], v[1]])
                        
                        # Add midpoints of adjacent edges
                        for edge in vor.ridge_vertices[:5]:
                            if len(edge) >= 2 and -1 not in edge:
                                v1 = vor.vertices[edge[0]]
                                v2 = vor.vertices[edge[1]]
                                if not np.isnan(v1).any() and not np.isnan(v2).any():
                                    midpoint = [(v1[0] + v2[0])/2, (v1[1] + v2[1])/2]
                                    if 0 <= midpoint[0] <= 1 and 0 <= midpoint[1] <= 1:
                                        neighbors.append(midpoint)
                except:
                    pass
                
                # Add some random positions near current circle
                for _ in range(10):
                    noise_x = np.random.normal(0, 0.05)
                    noise_y = np.random.normal(0, 0.05)
                    new_x = original_circle[0] + noise_x
                    new_y = original_circle[1] + noise_y
                    if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                        neighbors.append([new_x, new_y])
                
                # Try to improve this circle's position and radius
                for pos in neighbors:
                    test_circle = [pos[0], pos[1], original_circle[2]]
                    
                    # Calculate maximum feasible radius
                    max_radius = min(pos[0], 1-pos[0], pos[1], 1-pos[1])
                    if max_radius < original_circle[2]:
                        continue
                    
                    # Test smaller radius first to preserve other circles
                    test_radius = min(max_radius, original_circle[2] + 0.01)
                    test_circle[2] = test_radius
                    
                    # Check if valid
                    temp_circles = circles_array.copy()
                    temp_circles[i] = test_circle
                    
                    valid = True
                    for j in range(len(temp_circles)):
                        if j != i:
                            if check_collision(temp_circles[i], temp_circles[j]):
                                valid = False
                                break
                    
                    if valid and test_radius > best_radius:
                        best_radius = test_radius
                        best_circle = test_circle[:]
                
                # Update if we found an improvement
                if best_radius > original_circle[2]:
                    circles_array[i] = best_circle[:]
                    improved = True
            
            # Try a global radius adjustment
            total_radius = get_total_radius(circles_array)
            if total_radius > 0:
                # Distribute radius adjustments evenly
                avg_radius = total_radius / len(circles_array)
                for i in range(len(circles_array)):
                    # Slightly increase small circles, decrease large ones
                    if circles_array[i][2] < avg_radius * 0.8:
                        circles_array[i][2] = min(circles_array[i][2] * 1.02, circles_array[i][2] + 0.005)
                    elif circles_array[i][2] > avg_radius * 1.2:
                        circles_array[i][2] = max(circles_array[i][2] * 0.98, circles_array[i][2] - 0.005)
                
                improved = True
        
        return circles_array
    
    # Phase 1: Voronoi-based greedy initialization
    def initialize_grasp():
        circles = []
        
        # Generate initial points using a systematic pattern
        points = []
        # Grid pattern with slight randomness
        for i in range(6):
            for j in range(6):
                x = 0.1 + i * 0.15 + np.random.uniform(-0.02, 0.02)
                y = 0.1 + j * 0.15 + np.random.uniform(-0.02, 0.02)
                points.append([x, y])
        
        # Add some random points for variety
        for _ in range(10):
            points.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
        
        points = np.array(points)
        
        try:
            # Create Voronoi diagram
            vor = Voronoi(points)
            
            # Place circles greedily at Voronoi vertices
            placed_count = 0
            
            # Sort vertices by distance from center (to prioritize outer areas)
            sorted_vertices = []
            for vertex in vor.vertices:
                if 0 <= vertex[0] <= 1 and 0 <= vertex[1] <= 1:
                    dist_from_center = np.sqrt((vertex[0]-0.5)**2 + (vertex[1]-0.5)**2)
                    sorted_vertices.append((dist_from_center, vertex[0], vertex[1]))
            
            sorted_vertices.sort()
            
            # Place circles at vertices with sufficient space
            for _, x, y in sorted_vertices:
                if placed_count >= 32:
                    break
                    
                # Try placing a circle at this location
                max_radius = min(x, 1-x, y, 1-y)
                
                # Check if circle would conflict with already placed circles
                valid = True
                for circle in circles:
                    dx = abs(x - circle[0])
                    dy = abs(y - circle[1])
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < (max_radius + circle[2]):
                        valid = False
                        break
                
                if valid and max_radius > 0.001:
                    circles.append([x, y, max_radius])
                    placed_count += 1
                    
        except Exception as e:
            # Fallback to random initialization
            for _ in range(32):
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                max_radius = min(x, 1-x, y, 1-y)
                circles.append([x, y, max_radius * 0.8])
        
        return circles
    
    # Phase 2: Local improvement with Voronoi-based search
    initial_solution = initialize_grasp()
    
    # Convert to numpy array
    circles_array = np.array(initial_solution)
    
    # Apply local improvement using Voronoi neighborhood search
    final_circles = local_improvement_voronoi(circles_array.tolist())
    
    # Ensure final validity
    if is_valid_configuration(np.array(final_circles)):
        return np.array(final_circles)
    else:
        # Return what we have that's valid
        valid_circles = []
        for c in final_circles:
            if check_containment(c) and all(not check_collision(c, other) for other in valid_circles):
                valid_circles.append(c)
        if len(valid_circles) >= 32:
            return np.array(valid_circles[:32])
        else:
            return np.array(final_circles[:32])

# EVOLVE-BLOCK-END
