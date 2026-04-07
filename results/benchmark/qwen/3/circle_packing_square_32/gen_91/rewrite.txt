# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.spatial.distance import cdist

def generate_hexagonal_grid(n_circles, square_size=1.0):
    """Generate initial circle positions using a more refined hexagonal grid pattern"""
    sqrt3 = np.sqrt(3)
    
    # Better estimation of grid dimensions for hexagonal packing
    # Using formula for hexagonal lattice efficiency
    cols = max(1, int(np.ceil(np.sqrt(n_circles * 2 / sqrt3))))
    rows = max(1, int(np.ceil(n_circles / cols)))
    
    # Ensure we have enough space
    while rows * cols < n_circles:
        cols += 1

    # Calculate spacing based on desired circle count with better scaling
    max_radius = 0.08
    spacing_x = 2 * max_radius
    spacing_y = 2 * max_radius * sqrt3 / 2

    # Adjust spacing to fit within square with adequate margin
    while spacing_x * cols > square_size * 0.92 or spacing_y * rows > square_size * 0.92:
        max_radius *= 0.96
        spacing_x = 2 * max_radius
        spacing_y = 2 * max_radius * sqrt3 / 2

    # Generate positions with better distribution
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n_circles:
                break
            x = spacing_x * j + max_radius
            if i % 2 == 1:  # Offset every other row
                x += spacing_x / 2
            y = spacing_y * i + max_radius
            # More liberal boundary check to allow for better distribution
            if x <= square_size - max_radius and y <= square_size - max_radius:
                positions.append([x, y])
        if len(positions) >= n_circles:
            break

    # If we don't have enough points, fill with strategic random points
    while len(positions) < n_circles:
        # Prefer placing near edges for better space utilization
        x = np.random.uniform(max_radius, square_size - max_radius)
        y = np.random.uniform(max_radius, square_size - max_radius)
        # Slightly bias towards corners and edges for better coverage
        if np.random.random() < 0.3:
            x = np.random.choice([max_radius, square_size - max_radius])
            y = np.random.choice([max_radius, square_size - max_radius])
        elif np.random.random() < 0.5:
            x = np.random.choice([max_radius, square_size - max_radius])
        else:
            y = np.random.choice([max_radius, square_size - max_radius])
        positions.append([x, y])

    # Generate initial radii with variation around base value
    radii = [max_radius * (0.85 + np.random.random() * 0.15)] * min(len(positions), n_circles)

    # Fill with remaining circles if needed
    if len(positions) < n_circles:
        for _ in range(n_circles - len(positions)):
            radii.append(max_radius * (0.85 + np.random.random() * 0.15))

    return np.array(positions[:n_circles]), radii[:n_circles]

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))

    # Initialize using hexagonal grid
    positions, radii = generate_hexagonal_grid(n)

    # Fill the result array
    for i in range(n):
        circles[i][0] = positions[i][0]  # x coordinate
        circles[i][1] = positions[i][1]  # y coordinate
        circles[i][2] = radii[i]         # radius

    # Phase 1: Global expansion with smart step size decay
    max_iterations_phase1 = 2000
    improvement_threshold = 1e-6
    
    for iteration in range(max_iterations_phase1):
        improved = False
        max_radius_updates = 0
        
        # Adaptive step size with slower decay for better convergence
        step_size = max(0.0005, 0.008 * (1 - iteration / max_iterations_phase1)**1.5)
        
        # Process circles in order of decreasing radius (prioritize larger ones)
        sorted_indices = np.argsort(circles[:, 2])[::-1]
        
        for i in sorted_indices:
            old_radius = circles[i][2]
            
            # Calculate maximum possible radius for this circle with boundary checks
            max_radius = min(
                circles[i][0],  # Distance to left boundary
                1 - circles[i][0],  # Distance to right boundary
                circles[i][1],  # Distance to bottom boundary
                1 - circles[i][1]  # Distance to top boundary
            )
            
            # Find minimum distance to other circles (excluding self)
            min_distance = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.sqrt(
                        (circles[i][0] - circles[j][0])**2 +
                        (circles[i][1] - circles[j][1])**2
                    )
                    min_distance = min(min_distance, dist)
            
            # Maximum radius is limited by distance to neighbors minus current radius
            if min_distance < float('inf'):
                max_radius = min(max_radius, min_distance - old_radius)
            
            # Increase radius up to limit with adaptive step size
            new_radius = min(max_radius, old_radius + step_size)
            
            # Only update if there's meaningful improvement
            if new_radius > old_radius + improvement_threshold:
                circles[i][2] = new_radius
                improved = True
                max_radius_updates += 1
        
        # Early stopping if no significant improvement
        if not improved or max_radius_updates == 0:
            break

    # Phase 2: Enhanced overlap correction with more sophisticated algorithm
    def validate_and_correct(circles_array):
        # Step 1: Ensure all circles respect boundary constraints
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            # Use stricter boundary checking
            r = min(r, x, 1-x, y, 1-y)
            circles_array[i] = [x, y, r]
        
        # Step 2: Resolve overlaps with improved prioritization and reduction strategy
        max_overlap_iterations = 400  # Reduced for faster execution
        
        for iteration in range(max_overlap_iterations):
            changed = False
            distances = cdist(circles_array[:, :2], circles_array[:, :2])
            
            # Collect all overlapping pairs with detailed information
            overlap_pairs = []
            for i in range(len(circles_array)):
                for j in range(i+1, len(circles_array)):
                    dist = distances[i, j]
                    r_i, r_j = circles_array[i, 2], circles_array[j, 2]
                    overlap = r_i + r_j - dist
                    
                    if overlap > 0:
                        overlap_pairs.append((overlap, i, j, r_i, r_j, dist))
            
            # Sort by overlap amount (largest first), then by sum of radii (larger first)
            overlap_pairs.sort(key=lambda x: (-x[0], -(x[3] + x[4])))
            
            # Process largest overlaps first with refined reduction
            processed_pairs = set()
            for overlap, i, j, r_i, r_j, dist in overlap_pairs:
                if (i, j) in processed_pairs or (j, i) in processed_pairs:
                    continue
                    
                # Enhanced reduction strategy based on geometric properties
                # Ratio of circle sizes influences reduction priority
                size_ratio = r_i / r_j if r_j > 0 else 1.0
                
                # Determine reduction factor based on relative sizes and overlap
                if size_ratio > 2.0:
                    # Large circle dominates - reduce small circle more
                    reduction_factor = 0.7 if overlap < 0.05 else 0.8
                    reduction_i = overlap * reduction_factor * 0.3
                    reduction_j = overlap * reduction_factor * 0.7
                elif size_ratio < 0.5:
                    # Small circle dominates - reduce large circle more
                    reduction_factor = 0.7 if overlap < 0.05 else 0.8
                    reduction_i = overlap * reduction_factor * 0.7
                    reduction_j = overlap * reduction_factor * 0.3
                else:
                    # Similar sizes - reduce both equally
                    reduction_factor = 0.6 if overlap < 0.05 else 0.75
                    reduction_i = overlap * reduction_factor * 0.5
                    reduction_j = overlap * reduction_factor * 0.5
                
                # Apply reductions with safety checks
                if r_i > reduction_i and r_j > reduction_j:
                    new_r_i = max(0.001, r_i - reduction_i)
                    new_r_j = max(0.001, r_j - reduction_j)
                    
                    # Only apply if it doesn't cause new overlaps
                    # Check if resulting configuration maintains validity
                    safe = True
                    for k in range(len(circles_array)):
                        if k != i and k != j:
                            dist_i_k = np.sqrt((new_r_i - circles_array[k, 0])**2 + (circles_array[i, 1] - circles_array[k, 1])**2)
                            dist_j_k = np.sqrt((new_r_j - circles_array[k, 0])**2 + (circles_array[j, 1] - circles_array[k, 1])**2)
                            
                            if dist_i_k < new_r_i + circles_array[k, 2] or dist_j_k < new_r_j + circles_array[k, 2]:
                                safe = False
                                break
                    
                    if safe:
                        circles_array[i, 2] = new_r_i
                        circles_array[j, 2] = new_r_j
                        changed = True
                        processed_pairs.add((i, j))
            
            # Stop if no changes made
            if not changed:
                break
        
        # Step 3: Final cleanup and boundary validation with improved handling
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            # Ensure circle fits in unit square with better margin
            r = min(r, x - 0.001, 1 - x - 0.001, y - 0.001, 1 - y - 0.001)
            circles_array[i] = [x, y, r]
            
        return circles_array

    circles = validate_and_correct(circles)
    
    # Phase 3: Strategic fine-tuning with different approach
    max_fine_tune_iterations = 200
    for iteration in range(max_fine_tune_iterations):
        improved = False
        max_radius_updates = 0
        
        # Even smaller step size for fine tuning
        step_size = 0.0003
        
        # Process circles but with strategic ordering - prioritize those most likely to grow
        # Calculate potential growth capacity for each circle
        growth_potential = []
        for i in range(n):
            # Potential radius is limited by boundaries and neighbors
            potential_radius = min(
                circles[i][0],  # Distance to left boundary
                1 - circles[i][0],  # Distance to right boundary
                circles[i][1],  # Distance to bottom boundary
                1 - circles[i][1]  # Distance to top boundary
            )
            
            # Find minimum distance to other circles
            min_distance = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.sqrt(
                        (circles[i][0] - circles[j][0])**2 +
                        (circles[i][1] - circles[j][1])**2
                    )
                    min_distance = min(min_distance, dist)
            
            if min_distance < float('inf'):
                potential_radius = min(potential_radius, min_distance - circles[i][2])
            
            growth_potential.append(potential_radius - circles[i][2])
        
        # Sort by growth potential descending
        sorted_indices = np.argsort(growth_potential)[::-1]
        
        for i in sorted_indices:
            old_radius = circles[i][2]
            
            # Calculate maximum possible radius for this circle
            max_radius = min(
                circles[i][0],  # Distance to left boundary
                1 - circles[i][0],  # Distance to right boundary
                circles[i][1],  # Distance to bottom boundary
                1 - circles[i][1]  # Distance to top boundary
            )
            
            # Find minimum distance to other circles (excluding self)
            min_distance = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.sqrt(
                        (circles[i][0] - circles[j][0])**2 +
                        (circles[i][1] - circles[j][1])**2
                    )
                    min_distance = min(min_distance, dist)
            
            # Maximum radius is limited by distance to neighbors minus current radius
            if min_distance < float('inf'):
                max_radius = min(max_radius, min_distance - old_radius)
            
            # Very small increase for fine-tuning
            new_radius = min(max_radius, old_radius + step_size)
            
            # Only update if there's meaningful improvement
            if new_radius > old_radius + improvement_threshold:
                circles[i][2] = new_radius
                improved = True
                max_radius_updates += 1
        
        # Early stopping if no significant improvement
        if not improved or max_radius_updates == 0:
            break

    # Final validation pass with simplified but effective cleanup
    circles = validate_and_correct(circles)

    return circles


# EVOLVE-BLOCK-END