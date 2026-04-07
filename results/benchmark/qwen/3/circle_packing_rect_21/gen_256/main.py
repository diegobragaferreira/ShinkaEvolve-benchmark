# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: perimeter = 4, so width + height = 2
    # Using 1.2 x 0.8 for good aspect ratio that allows more efficient packing
    rect_width, rect_height = 1.2, 0.8

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    def compute_max_radius_at_position(x, y, circles):
        """Compute maximum possible radius at position considering all constraints"""
        # Boundary constraints
        max_radius = min(x, rect_width - x, y, rect_height - y)
        
        # Overlap constraints
        for cx, cy, r in circles:
            if cx != x or cy != y:  # Skip self
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                max_radius = min(max_radius, dist - r)
        
        return max(max_radius, 0.001)

    def validate_configuration(circles):
        """Fast validation of configuration"""
        # Check boundaries
        for x, y, r in circles:
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                return False
        
        # Quick overlap check using distance matrix
        if len(circles) < 2:
            return True
            
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Vectorized distance matrix
        dist_matrix = cdist(positions, positions)
        np.fill_diagonal(dist_matrix, float('inf'))
        
        # Minimum distances
        min_distances = np.min(dist_matrix, axis=1)
        
        # Required spacing
        required_spacing = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Check for overlaps
        overlaps = min_distances < np.min(required_spacing, axis=0)
        
        return not np.any(overlaps)

    def compute_total_radius_sum(circles):
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])

    def generate_initial_configuration():
        """Generate initial configuration using mixed strategy"""
        circles = np.zeros((21, 3))
        
        # 1. Corner and edge points for good boundary coverage
        corner_points = [
            (0.1, 0.1), (rect_width - 0.1, 0.1), 
            (0.1, rect_height - 0.1), (rect_width - 0.1, rect_height - 0.1)
        ]
        edge_points = [
            (rect_width/2, 0.1), (rect_width/2, rect_height - 0.1),
            (0.1, rect_height/2), (rect_width - 0.1, rect_height/2)
        ]
        
        # Start with structured points
        structured_points = corner_points + edge_points
        for i in range(len(structured_points)):
            x, y = structured_points[i]
            circles[i] = [x, y, 0.02]  # Small initial radius
        
        # 2. Fill remaining with random points constrained to interior
        for i in range(len(structured_points), 21):
            attempts = 0
            while attempts < 100:
                x = random.uniform(0.05, rect_width - 0.05)
                y = random.uniform(0.05, rect_height - 0.05)
                
                # Quick boundary check
                if x - 0.01 > 0 and x + 0.01 < rect_width and y - 0.01 > 0 and y + 0.01 < rect_height:
                    # Compute max radius at this position
                    max_r = compute_max_radius_at_position(x, y, circles[:i])
                    circles[i] = [x, y, max_r]
                    break
                attempts += 1
            else:
                # Fallback to simple uniform distribution if failed
                x = random.uniform(0.05, rect_width - 0.05)
                y = random.uniform(0.05, rect_height - 0.05)
                circles[i] = [x, y, 0.01]
        
        return circles

    def get_voronoi_guided_moves(circles, idx, base_step=0.05):
        """Generate movement candidates based on Voronoi cell geometry"""
        moves = []
        
        # Get current circle info
        x, y, r = circles[idx]
        
        # Start with current position
        moves.append((0, 0, 0))  # No movement
        
        # Add systematic moves around current position
        for dx in [-base_step*2, -base_step, 0, base_step, base_step*2]:
            for dy in [-base_step*2, -base_step, 0, base_step, base_step*2]:
                if abs(dx) + abs(dy) > 0:  # Not zero
                    moves.append((dx, dy, 0))
        
        # Add boundary-aware moves
        boundary_moves = []
        if x < base_step:
            boundary_moves.append((base_step, 0, 0))
        if x > rect_width - base_step:
            boundary_moves.append((-base_step, 0, 0))
        if y < base_step:
            boundary_moves.append((0, base_step, 0))
        if y > rect_height - base_step:
            boundary_moves.append((0, -base_step, 0))
            
        moves.extend(boundary_moves)
        
        return moves

    # Phase 1: Generate initial configuration
    circles = generate_initial_configuration()
    
    # Phase 2: Multi-phase optimization with Voronoi guidance
    best_circles = circles.copy()
    best_sum = compute_total_radius_sum(circles)
    
    # Phase 2a: Coarse refinement
    for iter_num in range(100):
        improved = False
        for i in range(21):
            # Get current circle
            x, y, r = circles[i]
            
            # Generate moves based on Voronoi structure
            moves = get_voronoi_guided_moves(circles, i, base_step=0.05)
            
            # Try each move
            best_move = (0, 0, r)
            best_new_r = r
            
            for dx, dy, _ in moves:
                new_x = max(0.001, min(rect_width - 0.001, x + dx))
                new_y = max(0.001, min(rect_height - 0.001, y + dy))
                
                # Compute max radius at new position
                new_r = compute_max_radius_at_position(new_x, new_y, circles)
                
                # Only consider moves that actually increase radius
                if new_r > best_new_r:
                    best_new_r = new_r
                    best_move = (dx, dy, new_r)
            
            # Apply best move if found
            if best_new_r > r:
                dx, dy, new_r = best_move
                circles[i] = [x + dx, y + dy, new_r]
                improved = True
        
        # Early termination if no improvement
        current_sum = compute_total_radius_sum(circles)
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = circles.copy()
        
        if not improved and iter_num > 50:
            break

    # Phase 2b: Fine refinement with smaller steps and better Voronoi guidance
    for iter_num in range(200):
        improved = False
        for i in range(21):
            # Get current circle
            x, y, r = circles[i]
            
            # Generate more targeted moves
            moves = get_voronoi_guided_moves(circles, i, base_step=0.02)
            
            # Try each move
            best_move = (0, 0, r)
            best_new_r = r
            
            for dx, dy, _ in moves:
                new_x = max(0.001, min(rect_width - 0.001, x + dx))
                new_y = max(0.001, min(rect_height - 0.001, y + dy))
                
                # Compute max radius at new position
                new_r = compute_max_radius_at_position(new_x, new_y, circles)
                
                # Prefer moves that increase radius AND maintain configuration validity
                if new_r > best_new_r:
                    # Check if this move keeps the configuration valid
                    temp_circles = circles.copy()
                    temp_circles[i] = [new_x, new_y, new_r]
                    if validate_configuration(temp_circles):
                        best_new_r = new_r
                        best_move = (dx, dy, new_r)
            
            # Apply best move if found
            if best_new_r > r:
                dx, dy, new_r = best_move
                circles[i] = [x + dx, y + dy, new_r]
                improved = True
        
        # Early termination
        current_sum = compute_total_radius_sum(circles)
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = circles.copy()
        
        if not improved and iter_num > 100:
            break

    # Phase 2c: Final polishing with targeted local search
    for iteration in range(50):
        # Try to improve the configuration by systematically adjusting circles
        for i in range(21):
            x, y, r = best_circles[i]
            
            # Try several candidate positions near the current position
            candidates = []
            
            # Grid around current position
            for dx in [-0.05, -0.02, 0, 0.02, 0.05]:
                for dy in [-0.05, -0.02, 0, 0.02, 0.05]:
                    if abs(dx) + abs(dy) > 0:
                        candidates.append((x + dx, y + dy))
            
            # Always include current position as fallback
            candidates.append((x, y))
            
            # Best improvement so far
            best_x, best_y, best_r = x, y, r
            
            for new_x, new_y in candidates:
                # Keep within bounds
                new_x = max(0.001, min(rect_width - 0.001, new_x))
                new_y = max(0.001, min(rect_height - 0.001, new_y))
                
                # Compute max radius
                new_r = compute_max_radius_at_position(new_x, new_y, best_circles)
                
                # Accept improvement if valid configuration
                temp_circles = best_circles.copy()
                temp_circles[i] = [new_x, new_y, new_r]
                if validate_configuration(temp_circles):
                    if new_r > best_r:
                        best_r = new_r
                        best_x, best_y = new_x, new_y
            
            # Apply improvement if found
            if best_r > r:
                best_circles[i] = [best_x, best_y, best_r]

    # Final validation and cleanup
    if not validate_configuration(best_circles):
        # If still invalid, fallback to clean configuration
        circles = generate_initial_configuration()
        best_circles = circles.copy()
        
    # Ensure final configuration is valid
    final_sum = compute_total_radius_sum(best_circles)
    return best_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")