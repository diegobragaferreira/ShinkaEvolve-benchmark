# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import random
from typing import Tuple

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    def build_spatial_grid(circles: np.ndarray, cell_size: float = None) -> dict:
        """Build a spatial grid for efficient neighbor lookup"""
        if cell_size is None:
            # Estimate cell size based on average radius
            avg_radius = np.mean(circles[:, 2])
            cell_size = 2 * avg_radius

        grid = {}
        for i, (x, y, r) in enumerate(circles):
            # Determine which grid cell this circle belongs to
            grid_x = int(x / cell_size)
            grid_y = int(y / cell_size)

            if (grid_x, grid_y) not in grid:
                grid[(grid_x, grid_y)] = []
            grid[(grid_x, grid_y)].append(i)

        return grid, cell_size

    def get_neighbors_in_grid(grid: dict, x: float, y: float, cell_size: float,
                            search_radius: float = None) -> list:
        """Get all circle indices that could potentially collide with a point"""
        if search_radius is None:
            search_radius = cell_size

        grid_x = int(x / cell_size)
        grid_y = int(y / cell_size)

        neighbors = []
        # Check the cell itself and its 8 neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell_key = (grid_x + dx, grid_y + dy)
                if cell_key in grid:
                    neighbors.extend(grid[cell_key])

        return neighbors

    def validate_solution(circles: np.ndarray) -> bool:
        """Check if solution satisfies all constraints"""
        n = len(circles)
        # Vectorized containment check
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]

        # Check containment constraints for all circles at once
        containment_ok = (
            (radii <= x_coords) &
            (x_coords <= 1 - radii) &
            (radii <= y_coords) &
            (y_coords <= 1 - radii)
        )

        if not np.all(containment_ok):
            return False

        # Build spatial grid for efficient overlap checking
        grid, cell_size = build_spatial_grid(circles)

        # Check overlap constraints using spatial grid
        for i in range(n):
            x1, y1, r1 = circles[i]

            # Get potential neighbors from spatial grid
            neighbor_indices = get_neighbors_in_grid(grid, x1, y1, cell_size)

            # Only check overlap against relevant neighbors
            for j in neighbor_indices:
                if i != j:
                    x2, y2, r2 = circles[j]
                    # Compute squared distance for efficiency
                    dx = x1 - x2
                    dy = y1 - y2
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (r1 + r2) * (r1 + r2)

                    # Check if circles are overlapping
                    if dist_sq < min_dist_sq:
                        return False

        return True

    def calculate_sum_radii(circles: np.ndarray) -> float:
        """Calculate total sum of radii"""
        return np.sum(circles[:, 2])

    def voronoi_initialization(n_circles: int) -> np.ndarray:
        """Initialize circles using Voronoi diagram approach"""
        # Generate random points and create Voronoi diagram
        sample_points = np.random.rand(n_circles*5, 2)  # Generate extra points for better coverage
        vor = Voronoi(sample_points)

        # Select valid Voronoi vertices inside unit square
        valid_vertices = []
        for vertex in vor.vertices:
            if 0 <= vertex[0] <= 1 and 0 <= vertex[1] <= 1:
                valid_vertices.append(vertex)

        # If not enough valid vertices, add random points
        while len(valid_vertices) < n_circles:
            valid_vertices.append([np.random.rand(), np.random.rand()])

        # Take first n_circles vertices
        selected_vertices = np.array(valid_vertices[:n_circles])

        # Create circles with radius based on proximity to neighbors
        circles = []
        for i, (x, y) in enumerate(selected_vertices):
            # Calculate minimum distance to other points to determine max radius
            min_dist = float('inf')
            for j, (x2, y2) in enumerate(selected_vertices):
                if i != j:
                    d = np.sqrt((x-x2)**2 + (y-y2)**2)
                    min_dist = min(min_dist, d)

            # Set radius to half the minimum distance to neighbors, but bounded by unit square
            r = min(min_dist/2, x, 1-x, y, 1-y)
            r = max(r, 0.001)  # Minimum radius to avoid degenerate cases
            circles.append([x, y, r])

        return np.array(circles)

    def mutate_voronoi(circles: np.ndarray, mutation_strength: float = 0.05) -> np.ndarray:
        """Create a mutated version of the Voronoi-based solution"""
        # Copy current circles
        new_circles = circles.copy()

        # Choose random circles to mutate
        n_mutations = max(1, int(len(circles) * 0.2))  # Mutate about 20% of circles
        indices = np.random.choice(len(circles), size=n_mutations, replace=False)

        for idx in indices:
            x, y, r = new_circles[idx]

            # Mutate position slightly
            x += np.random.normal(0, mutation_strength)
            y += np.random.normal(0, mutation_strength)

            # Bound position to unit square
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)

            # Mutate radius
            r *= (1 + np.random.normal(0, mutation_strength/2))
            r = max(0.001, r)
            r = min(r, x, 1-x, y, 1-y)  # Ensure containment

            new_circles[idx] = [x, y, r]

        return new_circles

    def optimize_circle_positions(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Fine-tune circle positions using local optimization"""
        # Convert to flattened parameter array for optimization
        params = circles.flatten()

        def objective(params_flat: np.ndarray) -> float:
            # Reconstruct circles
            circles_new = params_flat.reshape(-1, 3)
            # Calculate negative sum of radii (we want to maximize)
            return -np.sum(circles_new[:, 2])

        def constraint_func(params_flat: np.ndarray) -> float:
            # Reconstruct circles
            circles_new = params_flat.reshape(-1, 3)

            # Boundary constraints
            violations = 0
            for i in range(len(circles_new)):
                x, y, r = circles_new[i]
                if x < r or x > 1-r or y < r or y > 1-r:
                    violations += 1

            # Overlap constraints
            for i in range(len(circles_new)):
                for j in range(i+1, len(circles_new)):
                    x1, y1, r1 = circles_new[i]
                    x2, y2, r2 = circles_new[j]
                    dist_sq = (x1-x2)**2 + (y1-y2)**2
                    min_dist_sq = (r1+r2)**2
                    if dist_sq < min_dist_sq:
                        violations += 1

            return violations

        # Try to optimize
        try:
            # Define bounds for optimization
            bounds = []
            for i in range(len(params)//3):
                bounds.extend([(0.001, 1-0.001), (0.001, 1-0.001), (0.001, 0.5)])

            result = minimize(objective, params, method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6})

            if result.success:
                # Reconstruct circles from optimized parameters
                circles_optimized = result.x.reshape(-1, 3)
                return circles_optimized
        except Exception as e:
            pass

        return circles

    # Main algorithm
    best_circles = None
    best_sum_radii = 0

    # Try multiple initializations
    for attempt in range(10):
        # Start with Voronoi-based initialization
        circles = voronoi_initialization(32)

        # Local optimization
        circles = optimize_circle_positions(circles)

        # Validate
        if validate_solution(circles):
            sum_radii = calculate_sum_radii(circles)
            if sum_radii > best_sum_radii:
                best_sum_radii = sum_radii
                best_circles = circles.copy()

        # Evolutionary improvement
        for gen in range(100):  # Increased iterations for better exploration
            # Create new candidate via mutation
            mutated = mutate_voronoi(circles, 0.02)

            # Local optimization on mutated solution
            mutated = optimize_circle_positions(mutated)

            # Validate and accept if better
            if validate_solution(mutated):
                sum_radii = calculate_sum_radii(mutated)
                if sum_radii > best_sum_radii:
                    best_sum_radii = sum_radii
                    best_circles = mutated.copy()
                    circles = mutated.copy()

    # If we still don't have a good solution, fall back to a simple approach
    if best_circles is None:
        # Simple greedy initialization
        circles = np.zeros((32, 3))
        # Place circles in a grid-like pattern with decreasing sizes
        placed = 0
        for i in range(6):
            for j in range(6):
                if placed >= 32:
                    break
                x = 0.1 + i * 0.15
                y = 0.1 + j * 0.15
                r = 0.05
                circles[placed] = [x, y, r]
                placed += 1
            if placed >= 32:
                break
        best_circles = circles

    return best_circles

# EVOLVE-BLOCK-END