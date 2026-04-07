# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial import Voronoi, KDTree
from scipy.spatial.distance import cdist
import time
from typing import List, Tuple

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

class SpatialGrid:
    """Efficient spatial grid for collision detection"""
    def __init__(self, cell_size: float = 0.1):
        self.cell_size = cell_size
        self.grid = {}

    def _hash(self, x: float, y: float) -> Tuple[int, int]:
        """Hash coordinates to grid cell"""
        return (int(x / self.cell_size), int(y / self.cell_size))

    def add_circle(self, circle: Tuple[float, float, float]):
        """Add a circle to the spatial grid"""
        x, y, r = circle
        cell = self._hash(x, y)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(circle)

    def remove_circle(self, circle: Tuple[float, float, float]):
        """Remove a circle from the spatial grid"""
        x, y, r = circle
        cell = self._hash(x, y)
        if cell in self.grid:
            try:
                self.grid[cell].remove(circle)
            except ValueError:
                pass  # Circle not found in cell

    def get_neighbors(self, circle: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Get neighboring circles in nearby cells"""
        x, y, r = circle
        neighbors = []
        cell = self._hash(x, y)

        # Check all 9 surrounding cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if neighbor_cell in self.grid:
                    neighbors.extend(self.grid[neighbor_cell])

        return neighbors

def check_collision(circle1: Tuple[float, float, float], circle2: Tuple[float, float, float]) -> bool:
    """Check if two circles collide"""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    distance_squared = (x1 - x2)**2 + (y1 - y2)**2
    return distance_squared < (r1 + r2)**2

def is_valid_position(circle: Tuple[float, float, float], 
                     circles: List[Tuple[float, float, float]], 
                     spatial_grid: SpatialGrid = None) -> bool:
    """Check if a circle position is valid (within bounds and no collisions)"""
    x, y, r = circle

    # Check boundary constraints
    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
        return False

    # Use spatial grid for collision checking if available
    if spatial_grid is not None:
        # Get potential neighbors from spatial grid
        neighbors = spatial_grid.get_neighbors(circle)
        for existing_circle in neighbors:
            if check_collision(circle, existing_circle):
                return False
    else:
        # Fallback to brute force checking
        for existing_circle in circles:
            if check_collision(circle, existing_circle):
                return False

    return True

def compute_max_radius(x: float, y: float, circles: List[Tuple[float, float, float]]) -> float:
    """Compute the maximum radius for a circle at position (x,y) without overlapping existing circles"""
    if len(circles) == 0:
        return min(x, 1-x, y, 1-y)

    # Find minimum distance to any existing circle center
    min_distance = float('inf')
    for cx, cy, cr in circles:
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        min_distance = min(min_distance, distance)

    # Maximum radius is limited by boundaries and distance to other circles
    boundary_radius = min(x, 1-x, y, 1-y)
    collision_radius = min_distance - 1e-8  # Very small epsilon to avoid numerical issues

    return min(boundary_radius, collision_radius) if collision_radius > 0 else 0

def generate_voronoi_candidates(circles: List[Tuple[float, float, float]], 
                               num_candidates: int = 1000) -> List[Tuple[float, float]]:
    """Generate candidate positions using Voronoi diagram of existing circles"""
    if len(circles) == 0:
        # Return random points if no circles exist yet
        return [(random.uniform(0.01, 0.99), random.uniform(0.01, 0.99)) for _ in range(num_candidates)]

    # Get circle centers
    points = np.array([[cx, cy] for cx, cy, cr in circles])

    try:
        # Compute Voronoi diagram
        vor = Voronoi(points)

        candidates = []
        # Add Voronoi vertices
        for vertex in vor.vertices:
            x, y = vertex
            if 0 <= x <= 1 and 0 <= y <= 1:
                candidates.append((float(x), float(y)))

        # Add some random points around existing circles
        for i, (cx, cy, cr) in enumerate(circles):
            for _ in range(5):
                angle = random.uniform(0, 2*np.pi)
                distance = random.uniform(0.05, 0.2)
                x = cx + distance * np.cos(angle)
                y = cy + distance * np.sin(angle)
                if 0 <= x <= 1 and 0 <= y <= 1:
                    candidates.append((float(x), float(y)))

        # Add corner/edge points for better boundary coverage
        edge_points = [
            (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),
            (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5),
            (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)
        ]
        for x, y in edge_points:
            if 0 <= x <= 1 and 0 <= y <= 1:
                candidates.append((x, y))

        # If we don't have enough candidates, fill with random ones
        if len(candidates) < num_candidates:
            additional = num_candidates - len(candidates)
            for _ in range(additional):
                x = random.uniform(0.01, 0.99)
                y = random.uniform(0.01, 0.99)
                candidates.append((x, y))

        return candidates[:num_candidates]

    except:
        # Fallback to random sampling if Voronoi fails
        return [(random.uniform(0.01, 0.99), random.uniform(0.01, 0.99)) for _ in range(num_candidates)]

def place_circle_adaptive_voronoi(circles: np.ndarray, max_circles: int) -> np.ndarray:
    """Place circles using adaptive Voronoi-based approach with enhanced boundary awareness"""
    new_circles = circles.copy()
    placed = 0

    # Enhanced strategic positions prioritizing boundary placements where large circles can fit
    strategic_positions = [
        (0.05, 0.05), (0.05, 0.95), (0.95, 0.05), (0.95, 0.95),  # corners with buffer
        (0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5),      # edges with buffer
        (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75),  # diagonals
        (0.5, 0.5),  # center
        (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),          # standard corners
        (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)           # standard edges
    ]

    # Place initial strategic circles with priority to boundary positions
    for i, (x, y) in enumerate(strategic_positions[:min(16, max_circles)]):
        if placed >= max_circles:
            break
        # Try to place with maximum possible radius
        max_radius = compute_max_radius(x, y, new_circles[:placed])
        if max_radius > 0:
            new_circle = (x, y, max_radius)
            if is_valid_position(new_circle, new_circles[:placed]):
                new_circles[placed] = new_circle
                placed += 1

    # Fill remaining spots with adaptive Voronoi-based approach
    remaining = max_circles - placed
    attempt_count = 0
    max_attempts = remaining * 100  # Allow more attempts for better coverage

    # Keep track of spatial grid for efficiency
    spatial_grid = SpatialGrid(cell_size=0.15)  # Larger grid for better performance
    for i in range(placed):
        spatial_grid.add_circle(new_circles[i])

    while placed < max_circles and attempt_count < max_attempts:
        # Generate candidates based on current state
        if placed < 5:
            # Early stages: more random exploration to find good spots
            candidates = [(random.uniform(0.01, 0.99), random.uniform(0.01, 0.99)) for _ in range(300)]
        elif placed < 15:
            # Mid stages: mix of Voronoi and random
            candidates = generate_voronoi_candidates(new_circles[:placed], 500)
        else:
            # Later stages: focus on Voronoi sampling with more targeted approach
            candidates = generate_voronoi_candidates(new_circles[:placed], 1000)

        # Find the best valid circle among candidates
        best_circle = None
        best_radius = 0

        # Sample candidates more intelligently based on placement progress
        sample_size = min(100, max(20, len(candidates) // 2))
        sampled_candidates = random.sample(candidates, sample_size)

        for x, y in sampled_candidates:
            # Compute maximum possible radius for this position
            max_radius = compute_max_radius(x, y, new_circles[:placed])
            if max_radius <= best_radius:
                continue
            test_circle = (x, y, max_radius)
            if is_valid_position(test_circle, new_circles[:placed], spatial_grid):
                best_circle = test_circle
                best_radius = max_radius

        if best_circle is not None:
            new_circles[placed] = best_circle
            spatial_grid.add_circle(best_circle)
            placed += 1
        else:
            # If we can't find a valid circle, add a tiny circle and continue
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)
            test_circle = (x, y, 0.0001)
            if is_valid_position(test_circle, new_circles[:placed], spatial_grid):
                new_circles[placed] = test_circle
                spatial_grid.add_circle(test_circle)
                placed += 1

        attempt_count += 1

    return new_circles

def adaptive_local_optimization(circles: np.ndarray, iterations: int = 150) -> np.ndarray:
    """Advanced local optimization that dynamically adapts search intensity"""
    circles = circles.copy()
    
    # Dynamic iteration strategy
    for iter_num in range(iterations):
        improved = False
        
        # Vary search intensity based on iteration progress
        if iter_num < 30:
            moves_per_circle = 30
            step_size = 0.025
            radius_change_factor = 0.015
        elif iter_num < 80:
            moves_per_circle = 20
            step_size = 0.015
            radius_change_factor = 0.01
        else:
            moves_per_circle = 10
            step_size = 0.008
            radius_change_factor = 0.005

        # Try to improve each circle
        for i in range(len(circles)):
            old_x, old_y, old_r = circles[i]
            
            # Store original values
            orig_circle = circles[i].copy()
            
            # Track the best improvement found
            best_improvement = 0
            best_circle = orig_circle.copy()
            
            # Try multiple moves for this circle
            for _ in range(moves_per_circle):
                # Perturb position slightly
                new_x = old_x + random.uniform(-step_size, step_size)
                new_y = old_y + random.uniform(-step_size, step_size)
                
                # Keep within bounds
                new_x = max(0.01, min(0.99, new_x))
                new_y = max(0.01, min(0.99, new_y))
                
                # Compute new radius that fits in the area
                # Use a more intelligent approach - consider nearby circles
                nearby_circles = []
                for j in range(len(circles)):
                    if i != j:
                        dist = np.sqrt((new_x - circles[j][0])**2 + (new_y - circles[j][1])**2)
                        if dist < 0.2:  # Only consider nearby circles for radius calculation
                            nearby_circles.append(circles[j])
                
                # Calculate max radius considering all nearby circles
                max_radius = compute_max_radius(new_x, new_y, nearby_circles)
                
                # Also consider the current circle's influence on radius
                if i < len(circles) and len(nearby_circles) > 0:
                    # Add a small adjustment to make the circle slightly smaller to maintain compatibility
                    max_radius *= 0.95
                
                if max_radius > 0:
                    # Create test circle
                    test_circle = (new_x, new_y, max_radius)
                    
                    # Test if this change is valid
                    valid = True
                    for j in range(len(circles)):
                        if j != i:
                            # Test if this would cause overlap with other circles
                            if check_collision(test_circle, circles[j]):
                                valid = False
                                break
                    
                    # Check boundary constraints
                    if new_x - max_radius < 0 or new_x + max_radius > 1 or \
                       new_y - max_radius < 0 or new_y + max_radius > 1:
                        valid = False
                        
                    if valid:
                        # Calculate the improvement in sum of radii
                        # Note: We're not changing anything yet, just testing
                        old_sum = sum(circle[2] for circle in circles)
                        
                        # Test hypothetical new configuration
                        temp_circles = circles.copy()
                        temp_circles[i] = test_circle
                        
                        new_sum = sum(circle[2] for circle in temp_circles)
                        improvement = new_sum - old_sum
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_circle = test_circle

            # Apply the best improvement found if significant
            if best_improvement > 1e-8:  # Only apply if there's a meaningful improvement
                circles[i] = best_circle
                improved = True
                
        # Early stopping if no improvement
        if not improved and iter_num > 50:
            break

    return circles

def compute_total_radius(circles: np.ndarray) -> float:
    """Compute the total sum of all circle radii"""
    return sum(circle[2] for circle in circles)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))

    # Phase 1: Adaptive Voronoi-based initialization with enhanced boundary awareness
    circles = place_circle_adaptive_voronoi(circles, n)
    
    # Phase 2: Multi-pass local optimization
    circles = adaptive_local_optimization(circles, 100)
    
    # Phase 3: Additional refinement pass with different parameters
    circles = adaptive_local_optimization(circles, 50)

    return circles

# EVOLVE-BLOCK-END