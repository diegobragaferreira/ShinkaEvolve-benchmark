# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List, Optional
import time
from collections import defaultdict
import math

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_voronoi_cell_areas(points: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Compute Voronoi cell areas for given points, with bounded region constraints.
    """
    try:
        # Create Voronoi diagram
        vor = Voronoi(points)
        
        # For each point, find its Voronoi cell area
        areas = []
        for i in range(len(points)):
            # Get vertices of Voronoi cell for point i
            region = vor.regions[vor.point_region[i]]
            if -1 in region or not region:  # Infinite region or empty
                # Approximate area using bounding box and distance to neighbors
                neighbors = [j for j in range(len(points)) if j != i]
                if neighbors:
                    # Use bounding box approach for infinite regions
                    min_dist = min(distance.euclidean(points[i], points[j]) for j in neighbors)
                    area = min_dist * min_dist  # Rough estimate
                else:
                    area = 1.0
                areas.append(area)
            else:
                # Compute area of finite polygon
                vertices = [vor.vertices[j] for j in region if j >= 0]
                if len(vertices) >= 3:
                    # Clip vertices to rectangle bounds
                    clipped_vertices = []
                    for v in vertices:
                        if 0 <= v[0] <= rect_width and 0 <= v[1] <= rect_height:
                            clipped_vertices.append(v)
                    
                    if len(clipped_vertices) >= 3:
                        # Use shoelace formula for polygon area
                        x_coords = [v[0] for v in clipped_vertices]
                        y_coords = [v[1] for v in clipped_vertices]
                        area = 0.5 * abs(sum(x_coords[i] * y_coords[i+1] - x_coords[i+1] * y_coords[i] 
                                            for i in range(len(x_coords)-1)) + 
                                        x_coords[-1] * y_coords[0] - x_coords[0] * y_coords[-1])
                    else:
                        area = 1.0  # Default if not enough vertices
                else:
                    area = 1.0  # Default for insufficient vertices
                areas.append(area)
                
        return np.array(areas)
    except Exception as e:
        # Fallback - return equal areas
        return np.ones(len(points))

def is_valid_configuration(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
    """
    Check if all circles are valid (within bounds and non-overlapping).
    """
    n = len(circles)
    
    # Boundary check
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
            return False
    
    # Overlap check
    if n <= 1:
        return True
    
    coords = circles[:, :2]
    radii = circles[:, 2]
    
    # Use distance matrix for efficient overlap detection
    try:
        dist_matrix = cdist(coords, coords)
        for i in range(n):
            for j in range(i+1, n):
                distance = dist_matrix[i, j]
                min_distance = radii[i] + radii[j]
                if distance < min_distance:
                    return False
    except:
        # Fallback to direct computation
        for i in range(n):
            for j in range(i+1, n):
                distance = np.linalg.norm(coords[i] - coords[j])
                min_distance = radii[i] + radii[j]
                if distance < min_distance:
                    return False
    
    return True

class VoronoiCirclePacker:
    """Voronoi-based circle packing optimizer."""
    
    def __init__(self, n_circles: int = 21, rect_width: float = 1.0, rect_height: float = 1.0):
        self.n_circles = n_circles
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.max_iterations = 1000
        
    def initialize_voronoi_based(self) -> np.ndarray:
        """
        Initialize circles using Voronoi-based approach for better spatial distribution.
        """
        # Start with random points
        points = np.random.rand(self.n_circles, 2)
        points[:, 0] *= self.rect_width
        points[:, 1] *= self.rect_height
        
        # Refine using a simple Lloyd relaxation-like approach
        for _ in range(10):
            # Compute Voronoi diagram
            try:
                vor = Voronoi(points)
                # Compute centroids of Voronoi cells (Lloyd relaxation step)
                new_points = np.zeros_like(points)
                count = np.zeros(self.n_circles)
                
                for i in range(self.n_circles):
                    region = vor.regions[vor.point_region[i]]
                    if -1 not in region and len(region) > 2:
                        vertices = [vor.vertices[j] for j in region if j >= 0]
                        if len(vertices) >= 3:
                            # Compute centroid of polygon
                            x_coords = [v[0] for v in vertices]
                            y_coords = [v[1] for v in vertices]
                            centroid_x = sum(x_coords) / len(x_coords)
                            centroid_y = sum(y_coords) / len(y_coords)
                            new_points[i] = [centroid_x, centroid_y]
                            count[i] = 1
                        else:
                            new_points[i] = points[i]
                            count[i] = 1
                    else:
                        new_points[i] = points[i]
                        count[i] = 1
                
                # Average points
                mask = count > 0
                if np.any(mask):
                    points[mask] = new_points[mask] / count[mask][:, np.newaxis]
            except:
                pass  # If Voronoi fails, just use random points
        
        # Create initial circles with varying radii based on proximity
        circles = np.zeros((self.n_circles, 3))
        coords = points
        
        # Use minimum distance to neighbors to determine initial radii
        for i in range(self.n_circles):
            min_dist = float('inf')
            for j in range(self.n_circles):
                if i != j:
                    dist = np.linalg.norm(coords[i] - coords[j])
                    min_dist = min(min_dist, dist)
            
            # Set radius to avoid overlaps, but make it meaningful
            radius = min(0.1, min_dist / 3.0) if min_dist < float('inf') else 0.05
            radius = max(0.005, radius)  # Minimum radius
            
            circles[i] = [coords[i][0], coords[i][1], radius]
        
        return circles
    
    def compute_constraint_weights(self, circles: np.ndarray) -> np.ndarray:
        """
        Compute weights representing constraint density based on Voronoi diagram.
        Low weight = low constraint density = more aggressive mutation.
        High weight = high constraint density = conservative mutation.
        """
        n = len(circles)
        if n < 2:
            return np.ones(n) * 0.5
            
        # Get Voronoi cell areas
        coords = circles[:, :2]
        try:
            areas = compute_voronoi_cell_areas(coords, self.rect_width, self.rect_height)
            # Weight inversely proportional to area (smaller areas = more constrained)
            weights = 1.0 / (areas + 1e-8)  # Add small value to avoid division by zero
            # Normalize to [0.5, 1.5] range
            if np.max(weights) > 0:
                weights = 0.5 + 1.0 * (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
            else:
                weights = np.ones(n) * 1.0
        except:
            # Fallback to simple neighbor count
            weights = np.ones(n)
            for i in range(n):
                neighbor_count = 0
                for j in range(n):
                    if i != j:
                        dist = np.linalg.norm(coords[i] - coords[j])
                        if dist < 2 * circles[i, 2]:  # Within influence radius
                            neighbor_count += 1
                weights[i] = 1.0 + 0.2 * neighbor_count  # Normalize and bias toward higher density
                
        return weights
    
    def adaptive_mutate(self, circles: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Perform adaptive mutation based on constraint weights.
        """
        mutated = circles.copy()
        
        for i in range(len(mutated)):
            x, y, r = mutated[i]
            
            # Use weights to adjust mutation strength
            # Lower weight (more constrained area) = smaller mutation
            # Higher weight (less constrained area) = larger mutation
            weight = weights[i]
            
            # Mutation strengths based on constraint level
            pos_mutation_strength = 0.02 / weight
            rad_mutation_strength = 0.01 / weight
            
            # Mutate position
            if random.random() < 0.3:  # Lower mutation rate for stability
                x += np.random.normal(0, pos_mutation_strength)
                y += np.random.normal(0, pos_mutation_strength)
                
                # Ensure position stays within bounds
                x = np.clip(x, r, self.rect_width - r)
                y = np.clip(y, r, self.rect_height - r)
            
            # Mutate radius
            if random.random() < 0.3:
                r += np.random.normal(0, rad_mutation_strength)
                # Ensure radius remains positive
                r = max(0.001, r)
            
            mutated[i] = [x, y, r]
            
        return mutated
    
    def local_improve(self, circles: np.ndarray, max_steps: int = 100) -> np.ndarray:
        """
        Apply local improvement using a greedy search.
        """
        improved = circles.copy()
        current_fitness = np.sum(improved[:, 2])  # Sum of radii
        
        for step in range(max_steps):
            # Try to improve each circle individually
            for i in range(len(improved)):
                # Save current state
                old_x, old_y, old_r = improved[i]
                
                # Try several random moves and see if any improve fitness
                best_fitness = current_fitness
                best_move = (old_x, old_y, old_r)
                
                # Try multiple random moves
                for _ in range(5):
                    # Try position moves
                    new_x = old_x + np.random.normal(0, 0.01)
                    new_y = old_y + np.random.normal(0, 0.01)
                    new_r = old_r + np.random.normal(0, 0.002)
                    
                    # Clip to bounds
                    new_x = np.clip(new_x, new_r, self.rect_width - new_r)
                    new_y = np.clip(new_y, new_r, self.rect_height - new_r)
                    new_r = max(0.001, new_r)
                    
                    # Test this move
                    test_config = improved.copy()
                    test_config[i] = [new_x, new_y, new_r]
                    
                    # Quick validity check
                    if not is_valid_configuration(test_config, self.rect_width, self.rect_height):
                        continue
                        
                    # Calculate fitness
                    test_fitness = np.sum(test_config[:, 2])
                    
                    if test_fitness > best_fitness:
                        best_fitness = test_fitness
                        best_move = (new_x, new_y, new_r)
                
                # Apply best move if it's better
                if best_fitness > current_fitness:
                    improved[i] = best_move
                    current_fitness = best_fitness
                    
        return improved
    
    def optimize(self) -> np.ndarray:
        """
        Main optimization routine.
        """
        # Step 1: Initialize using Voronoi-based method
        circles = self.initialize_voronoi_based()
        
        # Step 2: Iteratively improve using constraint-aware mutations
        best_solution = circles.copy()
        best_fitness = np.sum(circles[:, 2])
        
        # Main optimization loop
        for iteration in range(100):
            # Compute constraint weights
            weights = self.compute_constraint_weights(circles)
            
            # Mutate
            mutated = self.adaptive_mutate(circles, weights)
            
            # Local improvement
            improved = self.local_improve(mutated, max_steps=20)
            
            # Check if this is better
            current_fitness = np.sum(improved[:, 2])
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = improved.copy()
            
            # Update current circles for next iteration
            circles = improved
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}, Current fitness: {current_fitness:.6f}")

        # Final local refinement
        refined = self.local_improve(best_solution, max_steps=100)
        final_fitness = np.sum(refined[:, 2])
        
        print(f"Final fitness: {final_fitness:.6f}")
        return refined

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: perimeter = 4 => width + height = 2
    # Optimize rectangle aspect ratio for better packing
    rect_width = 1.2
    rect_height = 0.8

    # Initialize optimizer
    optimizer = VoronoiCirclePacker(
        n_circles=21,
        rect_width=rect_width,
        rect_height=rect_height
    )
    
    # Run optimization
    best_solution = optimizer.optimize()

    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")