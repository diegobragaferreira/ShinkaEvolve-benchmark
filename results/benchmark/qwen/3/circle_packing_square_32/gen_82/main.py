# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a structured hexagonal pattern
    def generate_hexagonal_initial():
        # Use a 6x6 grid (36 positions) to comfortably fit 32 circles
        rows, cols = 6, 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Hexagonal packing: offset odd rows
        circles = []
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                    
                x_offset = (i % 2) * (spacing_x / 2)
                x = (j * spacing_x) + x_offset + spacing_x / 2
                y = (i * spacing_y) + spacing_y / 2
                
                # Ensure within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    circles.append([x, y, 0.02])  # Small initial radius
                    
        return np.array(circles[:n])
    
    # Generate initial configuration
    circles = generate_hexagonal_initial()
    
    # Compute Voronoi diagram and helper functions
    def compute_voronoi_regions(points):
        """Compute Voronoi regions for given points"""
        try:
            vor = Voronoi(points)
            return vor
        except:
            return None
    
    def get_voronoi_areas(vor):
        """Calculate area of each Voronoi cell"""
        if vor is None:
            return None
            
        areas = []
        for region in vor.regions:
            if len(region) > 0 and -1 not in region:
                # Calculate polygon area
                vertices = [vor.vertices[i] for i in region]
                if len(vertices) >= 3:
                    # Simple polygon area calculation
                    area = 0
                    for i in range(len(vertices)):
                        j = (i + 1) % len(vertices)
                        area += vertices[i][0] * vertices[j][1]
                        area -= vertices[j][0] * vertices[i][1]
                    areas.append(abs(area) / 2)
                else:
                    areas.append(0)
            else:
                areas.append(0)
        return areas
    
    def compute_voronoi_cell_areas(points, vor):
        """Compute areas of Voronoi cells for each point"""
        if vor is None:
            return [0] * len(points)
            
        areas = []
        for i, point in enumerate(points):
            # Find Voronoi cell for this point
            cell_vertices = []
            for j, region in enumerate(vor.regions):
                if len(region) > 0 and -1 not in region:
                    # Check if this region corresponds to our point
                    if i in vor.point_region:
                        region_idx = vor.point_region[i]
                        if j == region_idx:
                            cell_vertices = [vor.vertices[k] for k in region]
                            break
            
            # If we found vertices, compute area
            if len(cell_vertices) >= 3:
                area = 0
                for j in range(len(cell_vertices)):
                    k = (j + 1) % len(cell_vertices)
                    area += cell_vertices[j][0] * cell_vertices[k][1]
                    area -= cell_vertices[k][0] * cell_vertices[j][1]
                areas.append(abs(area) / 2)
            else:
                areas.append(0)
                
        return areas
    
    def get_safe_radius(position, radii, positions, idx):
        """Get the maximum safe radius for circle at idx"""
        x, y = position
        # Boundary constraints
        r_boundary = min(x, 1-x, y, 1-y)
        
        # Overlap constraints
        r_overlap = r_boundary
        for i in range(len(positions)):
            if i != idx:
                dx = x - positions[i][0]
                dy = y - positions[i][1]
                dist = np.sqrt(dx*dx + dy*dy)
                # Must be at least radius_sum apart
                max_r = dist - radii[i]
                if max_r > 0:
                    r_overlap = min(r_overlap, max_r)
        
        return min(r_boundary, r_overlap)
    
    def validate_configuration(circles):
        """Validate that all constraints are satisfied"""
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Check boundary constraints
        for i, (x, y, r) in enumerate(circles):
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlap constraints
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < (radii[i] + radii[j]):
                    return False
                    
        return True
    
    # Main optimization loop
    max_iterations = 1000
    improvement_threshold = 1e-6
    
    for iteration in range(max_iterations):
        # Get current positions and radii
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Compute Voronoi diagram of current positions
        vor = compute_voronoi_regions(positions)
        
        # Calculate weighted improvements based on Voronoi regions
        # Circles in larger Voronoi cells have more room to grow
        total_improvement = 0
        
        # Update each circle's radius in a smart way
        for i in range(n):
            # Current state
            x, y = positions[i]
            current_radius = radii[i]
            
            # Compute safe maximum radius
            safe_radius = get_safe_radius((x, y), radii, positions, i)
            
            # Prefer expansion in directions with larger Voronoi cells
            # This is a proxy for available space
            expansion_amount = min(0.005, safe_radius - current_radius)
            
            if expansion_amount > improvement_threshold:
                new_radius = min(safe_radius, current_radius + expansion_amount)
                circles[i, 2] = new_radius
                total_improvement += expansion_amount
        
        # If no meaningful improvement, break
        if total_improvement < improvement_threshold:
            break
    
    # Final validation and cleanup
    if not validate_configuration(circles):
        # Revert to best valid configuration
        circles = generate_hexagonal_initial()
        
        # Simple greedy expansion
        for _ in range(100):
            improved = False
            for i in range(n):
                x, y = circles[i, 0], circles[i, 1]
                current_radius = circles[i, 2]
                
                # Safety limit
                r_boundary = min(x, 1-x, y, 1-y)
                
                # Check overlaps with others
                r_overlap = r_boundary
                for j in range(n):
                    if i != j:
                        dx = x - circles[j, 0]
                        dy = y - circles[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        max_r = dist - circles[j, 2]
                        if max_r > 0:
                            r_overlap = min(r_overlap, max_r)
                
                # Expand if beneficial
                new_radius = min(r_overlap, current_radius + 0.002)
                if new_radius > current_radius + 1e-6:
                    circles[i, 2] = new_radius
                    improved = True
            
            if not improved:
                break
    
    return circles

# EVOLVE-BLOCK-END