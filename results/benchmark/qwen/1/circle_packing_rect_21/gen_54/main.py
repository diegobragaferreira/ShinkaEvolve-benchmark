# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance_matrix
from scipy.spatial.distance import cdist
import time
from functools import lru_cache

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (width + height = 2)
    rect_width = 1.0
    rect_height = 1.0
    
    # Number of circles
    n = 21
    
    # Initialize circles with a structured pattern
    circles = np.zeros((n, 3))
    
    # Start with a hexagonal packing pattern
    rows = int(np.sqrt(n)) + 1
    cols = int(np.ceil(n / rows))
    
    # Calculate spacing to fit in rectangle
    margin = 0.05
    max_radius = min(rect_width, rect_height) * 0.08
    
    # Create hexagonal grid
    x_spacing = max_radius * 2.5
    y_spacing = max_radius * 2.165  # sqrt(3)/2 * 2
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = margin + j * x_spacing
            y = margin + i * y_spacing
            
            if i % 2 == 1:
                x += x_spacing / 2
                
            # Adjust for bounds
            x = max(max_radius, min(rect_width - max_radius, x))
            y = max(max_radius, min(rect_height - max_radius, y))
            
            circles[idx] = [x, y, max_radius]
            idx += 1
    
    # Physics-based optimization with Voronoi guidance
    def compute_voronoi_forces(circles, rect_width, rect_height):
        """Compute forces based on Voronoi diagram for conflict resolution."""
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Compute Voronoi diagram
        try:
            vor = Voronoi(positions)
        except:
            # Fallback to simple distance-based approach
            return np.zeros_like(positions)
        
        forces = np.zeros_like(positions)
        
        # For each circle, compute repulsion from Voronoi boundaries
        for i in range(len(circles)):
            # Calculate minimum distance to Voronoi edges and vertices
            # This represents how constrained this circle is
            region_idx = vor.point_region[i]
            if region_idx != -1 and region_idx < len(vor.regions):
                region = vor.regions[region_idx]
                if len(region) > 0 and -1 not in region:
                    # Collect vertices in this region
                    region_vertices = []
                    for vertex_idx in region:
                        if vertex_idx < len(vor.vertices):
                            region_vertices.append(vor.vertices[vertex_idx])
                    
                    if len(region_vertices) > 0:
                        region_vertices = np.array(region_vertices)
                        
                        # Compute distances to edges of Voronoi region
                        # Simple approach: push away from the nearest boundary
                        min_dist = float('inf')
                        closest_boundary = None
                        
                        # Check distance to rectangle boundaries
                        x, y = positions[i]
                        r = radii[i]
                        
                        # Distance to boundaries
                        dist_to_left = x - r
                        dist_to_right = rect_width - (x + r)
                        dist_to_bottom = y - r
                        dist_to_top = rect_height - (y + r)
                        
                        # Get the minimum boundary distance
                        boundary_dists = [dist_to_left, dist_to_right, dist_to_bottom, dist_to_top]
                        min_boundary_dist = min(boundary_dists)
                        
                        if min_boundary_dist < 0.01:  # Very close to boundary
                            # Apply boundary repulsion force
                            boundary_force = np.zeros(2)
                            if dist_to_left < 0.01:
                                boundary_force[0] = 0.01
                            elif dist_to_right < 0.01:
                                boundary_force[0] = -0.01
                            if dist_to_bottom < 0.01:
                                boundary_force[1] = 0.01
                            elif dist_to_top < 0.01:
                                boundary_force[1] = -0.01
                            
                            forces[i] += boundary_force * 50.0
        
        return forces
    
    def compute_overlap_forces(circles, rect_width, rect_height):
        """Compute forces to resolve overlaps between circles."""
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        forces = np.zeros_like(positions)
        
        # Compute all pairwise distances
        distances = cdist(positions, positions)
        
        # Process pairs to compute repulsive forces
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                if i != j:
                    dist = distances[i, j]
                    r1, r2 = radii[i], radii[j]
                    
                    # Check if overlapping
                    if dist < (r1 + r2):
                        # Compute repulsion force
                        direction = positions[i] - positions[j]
                        norm = np.linalg.norm(direction)
                        if norm > 1e-8:
                            direction = direction / norm
                            # Force magnitude inversely proportional to distance
                            force_magnitude = 1.0 / (dist + 1e-8)
                            forces[i] += direction * force_magnitude * 10.0
                            forces[j] -= direction * force_magnitude * 10.0
        
        return forces
    
    def compute_total_forces(circles, rect_width, rect_height):
        """Compute total forces acting on each circle."""
        overlap_forces = compute_overlap_forces(circles, rect_width, rect_height)
        voronoi_forces = compute_voronoi_forces(circles, rect_width, rect_height)
        
        # Combine forces
        total_forces = overlap_forces + voronoi_forces
        
        # Apply boundary constraints
        for i in range(len(circles)):
            x, y = circles[i, 0], circles[i, 1]
            r = circles[i, 2]
            
            # Boundary forces
            if x - r < 0:
                total_forces[i, 0] += 100.0 * (0 - (x - r))
            if x + r > rect_width:
                total_forces[i, 0] -= 100.0 * ((x + r) - rect_width)
            if y - r < 0:
                total_forces[i, 1] += 100.0 * (0 - (y - r))
            if y + r > rect_height:
                total_forces[i, 1] -= 100.0 * ((y + r) - rect_height)
        
        return total_forces
    
    def apply_forces(circles, forces, learning_rate=0.01):
        """Apply computed forces to update circle positions."""
        new_circles = circles.copy()
        
        for i in range(len(circles)):
            # Apply position updates
            new_circles[i, 0] += forces[i, 0] * learning_rate
            new_circles[i, 1] += forces[i, 1] * learning_rate
            
            # Clamp to boundaries
            r = new_circles[i, 2]
            new_circles[i, 0] = np.clip(new_circles[i, 0], r, 1.0 - r)
            new_circles[i, 1] = np.clip(new_circles[i, 1], r, 1.0 - r)
        
        return new_circles
    
    def compute_energy(circles, rect_width, rect_height):
        """Compute energy representing solution quality."""
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Energy from overlaps
        energy = 0
        distances = cdist(positions, positions)
        
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                if i != j:
                    dist = distances[i, j]
                    r1, r2 = radii[i], radii[j]
                    
                    if dist < (r1 + r2):
                        # Overlap energy
                        overlap = (r1 + r2) - dist
                        energy += overlap * overlap * 10000.0
        
        # Boundary energy
        for i in range(len(circles)):
            x, y = circles[i, 0], circles[i, 1]
            r = circles[i, 2]
            boundary_energy = 0
            
            if x - r < 0:
                boundary_energy += (0 - (x - r)) ** 2
            if x + r > rect_width:
                boundary_energy += ((x + r) - rect_width) ** 2
            if y - r < 0:
                boundary_energy += (0 - (y - r)) ** 2
            if y + r > rect_height:
                boundary_energy += ((y + r) - rect_height) ** 2
                
            energy += boundary_energy * 1000.0
        
        return energy
    
    def is_valid_solution(circles, rect_width, rect_height):
        """Check if solution is valid."""
        # Check boundary constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                return False
                
        # Check overlap constraints
        if len(circles) > 1:
            coords = circles[:, :2]
            radii = circles[:, 2]
            distances = cdist(coords, coords)
            # Create mask for upper triangle (avoid double counting)
            mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
            # Check overlaps
            overlap_distances = distances[mask]
            overlap_radii = (radii[:, None] + radii[None, :])[mask]
            if np.any(overlap_distances < overlap_radii):
                return False
                    
        return True
    
    def adjust_radii_based_on_voronoi(circles, rect_width, rect_height):
        """Adjust radii based on Voronoi-based spatial analysis."""
        positions = circles[:, :2]
        radii = circles[:, 2].copy()
        
        try:
            vor = Voronoi(positions)
        except:
            return circles
            
        # For each circle, determine how much it can grow
        for i in range(len(circles)):
            x, y, r = circles[i]
            
            # Find minimum distance to Voronoi vertices
            # This gives insight into how much space is available around the circle
            region_idx = vor.point_region[i]
            if region_idx != -1 and region_idx < len(vor.regions):
                region = vor.regions[region_idx]
                if len(region) > 0 and -1 not in region:
                    # Get all vertices in this region
                    region_vertices = []
                    for vertex_idx in region:
                        if vertex_idx < len(vor.vertices):
                            region_vertices.append(vor.vertices[vertex_idx])
                    
                    if len(region_vertices) > 0:
                        # Find the minimum distance to any vertex in this region
                        region_vertices = np.array(region_vertices)
                        distances = np.sqrt(np.sum((region_vertices - [x, y])**2, axis=1))
                        min_distance = np.min(distances)
                        
                        # Estimate how much we can increase the radius before hitting a boundary
                        # But avoid being too aggressive
                        max_increase = min_distance * 0.9 - r
                        
                        if max_increase > 0:
                            # Allow small increase with some randomness
                            r_increase = min(max_increase, np.random.uniform(0, max_increase * 0.3))
                            new_radius = min(r + r_increase, 0.3)  # Cap at reasonable value
                            circles[i, 2] = new_radius
        
        return circles
    
    # Main optimization loop
    start_time = time.time()
    current_energy = compute_energy(circles, rect_width, rect_height)
    
    max_iterations = 1000
    for iteration in range(max_iterations):
        # Periodically recompute Voronoi-based adjustments
        if iteration % 50 == 0:
            circles = adjust_radii_based_on_voronoi(circles, rect_width, rect_height)
            current_energy = compute_energy(circles, rect_width, rect_height)
        
        # Compute forces
        forces = compute_total_forces(circles, rect_width, rect_height)
        
        # Apply forces
        new_circles = apply_forces(circles, forces, learning_rate=0.001)
        
        # Check if we're making progress
        new_energy = compute_energy(new_circles, rect_width, rect_height)
        
        # Only accept improvement or minor deterioration
        if new_energy < current_energy or np.random.random() < 0.05:
            circles = new_circles
            current_energy = new_energy
            
            # Occasionally make larger jumps when stuck
            if iteration % 100 == 0 and iteration > 0:
                # Slightly randomize positions to escape local minima
                for i in range(len(circles)):
                    circles[i, 0] += np.random.normal(0, 0.005)
                    circles[i, 1] += np.random.normal(0, 0.005)
                    circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1.0 - circles[i, 2])
                    circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1.0 - circles[i, 2])
        
        # Early termination if energy is very low
        if current_energy < 1e-6:
            break
        
        # Periodically validate solution
        if iteration % 200 == 0:
            if not is_valid_solution(circles, rect_width, rect_height):
                # Reset to better configuration if invalid
                circles = generate_hexagonal_pattern(rect_width, rect_height, n)
    
    # Final refinement
    for _ in range(200):
        forces = compute_total_forces(circles, rect_width, rect_height)
        circles = apply_forces(circles, forces, learning_rate=0.0005)
    
    # Ensure final solution is valid
    if not is_valid_solution(circles, rect_width, rect_height):
        # Fall back to hexagonal pattern
        circles = generate_hexagonal_pattern(rect_width, rect_height, n)
    
    # Final radius adjustment using Voronoi
    circles = adjust_radii_based_on_voronoi(circles, rect_width, rect_height)
    
    return circles

def generate_hexagonal_pattern(width, height, n):
    """Helper function to generate hexagonal pattern."""
    circles = np.zeros((n, 3))
    
    # Determine grid parameters
    rows = int(np.sqrt(n)) + 1
    cols = int(np.ceil(n / rows))
    
    # Calculate spacing
    margin = 0.05
    max_radius = min(width, height) * 0.08
    
    # Create hexagonal grid
    x_spacing = max_radius * 2.5
    y_spacing = max_radius * 2.165  # sqrt(3)/2 * 2
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = margin + j * x_spacing
            y = margin + i * y_spacing
            
            if i % 2 == 1:
                x += x_spacing / 2
                
            # Adjust for bounds
            x = max(max_radius, min(width - max_radius, x))
            y = max(max_radius, min(height - max_radius, y))
            
            circles[idx] = [x, y, max_radius]
            idx += 1
            
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
