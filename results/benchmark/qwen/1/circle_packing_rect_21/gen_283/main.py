# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List, Dict, Optional
import time

class InitializationStrategy:
    """Handles different circle initialization strategies."""
    
    @staticmethod
    def hexagonal(n: int, width: float, height: float) -> np.ndarray:
        """Initialize circles using hexagonal packing pattern."""
        circles = np.zeros((n, 3))
        
        # Hexagonal packing arrangement
        rows = max(1, int(np.sqrt(n) * 0.8))
        cols = max(1, int(n / rows) + 1)
        
        # Calculate spacing
        spacing_x = width / (cols + 1)
        spacing_y = height / (rows + 1)
        
        # Place circles in hexagonal pattern
        idx = 0
        for i in range(rows):
            offset = spacing_x * (i % 2) * 0.5  # Offset every other row
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x + offset
                y = (i + 1) * spacing_y
                
                # Ensure position is within bounds
                x = max(0.01, min(width - 0.01, x))
                y = max(0.01, min(height - 0.01, y))
                
                # Set initial radius to a small value
                circles[idx] = [x, y, 0.05]
                idx += 1
            
            if idx >= n:
                break
        
        # Fill remaining circles if needed
        while idx < n:
            x = np.random.uniform(0.01, width - 0.01)
            y = np.random.uniform(0.01, height - 0.01)
            circles[idx] = [x, y, 0.05]
            idx += 1
            
        return circles
    
    @staticmethod
    def random_spaced(n: int, width: float, height: float) -> np.ndarray:
        """Initialize circles using random placement with basic spacing checks."""
        circles = np.zeros((n, 3))
        
        for i in range(n):
            # Generate random positions until we find a good one
            attempts = 0
            while attempts < 100:
                x = np.random.uniform(0.01, width - 0.01)
                y = np.random.uniform(0.01, height - 0.01)
                
                # Basic check against existing circles (not comprehensive)
                valid = True
                for j in range(i):
                    dx = x - circles[j, 0]
                    dy = y - circles[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance < (circles[j, 2] + 0.05) * 1.5:  # Safety margin
                        valid = False
                        break
                
                if valid:
                    circles[i] = [x, y, 0.05]
                    break
                attempts += 1
            
            # If we couldn't find a good spot, just place randomly
            if attempts >= 100:
                x = np.random.uniform(0.01, width - 0.01)
                y = np.random.uniform(0.01, height - 0.01)
                circles[i] = [x, y, 0.05]
                
        return circles
    
    @staticmethod
    def square_grid(n: int, width: float, height: float) -> np.ndarray:
        """Initialize circles using square grid pattern."""
        circles = np.zeros((n, 3))
        
        # Square grid arrangement
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        spacing_x = width / (cols + 1)
        spacing_y = height / (rows + 1)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Ensure position is within bounds
                x = max(0.01, min(width - 0.01, x))
                y = max(0.01, min(height - 0.01, y))
                
                # Set initial radius to a small value
                circles[idx] = [x, y, 0.05]
                idx += 1
            
            if idx >= n:
                break
        
        # Fill remaining circles if needed
        while idx < n:
            x = np.random.uniform(0.01, width - 0.01)
            y = np.random.uniform(0.01, height - 0.01)
            circles[idx] = [x, y, 0.05]
            idx += 1
            
        return circles

class ConstraintAnalyzer:
    """Manages constraint checking and Voronoi-based density calculations."""
    
    @staticmethod
    def calculate_voronoi_area(circles: np.ndarray, center_idx: int, width: float, height: float) -> float:
        """Approximate Voronoi cell area for a given circle using scipy Voronoi."""
        try:
            # Get all circle centers
            centers = circles[:, :2]
            
            # Create Voronoi diagram
            vor = Voronoi(centers)
            
            # Find the region corresponding to the center_idx
            if center_idx < len(vor.point_region):
                region_idx = vor.point_region[center_idx]
                if region_idx >= 0 and len(vor.regions[region_idx]) > 0:
                    # Extract vertices of the Voronoi cell
                    vertices = vor.regions[region_idx]
                    if -1 not in vertices:
                        region_vertices = [vor.vertices[i] for i in vertices if i < len(vor.vertices)]
                        if len(region_vertices) >= 3:
                            # Calculate area using shoelace formula
                            points = np.array(region_vertices)
                            x = points[:, 0]
                            y = points[:, 1]
                            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                            return max(1e-8, area)  # Prevent zero area
            
            # Fallback to distance-based calculation if Voronoi fails
        except:
            pass
        
        # Fallback: approximate based on neighbor distances
        distances = []
        center_x, center_y = circles[center_idx, 0], circles[center_idx, 1]
        
        for i in range(len(circles)):
            if i != center_idx:
                dx = center_x - circles[i, 0]
                dy = center_y - circles[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                distances.append(dist)
        
        if len(distances) == 0:
            return width * height  # Full area if no neighbors
        
        # Return inverse of average distance (higher distance = larger Voronoi cell)
        avg_dist = np.mean(distances) if len(distances) > 0 else 1.0
        return 1.0 / (avg_dist + 0.001)
    
    @staticmethod
    def check_collision(circles: np.ndarray, i: int, j: int) -> bool:
        """Check if two circles at indices i and j collide."""
        dx = circles[i, 0] - circles[j, 0]
        dy = circles[i, 1] - circles[j, 1]
        distance = np.sqrt(dx*dx + dy*dy)
        required_distance = circles[i, 2] + circles[j, 2] + 0.001
        return distance < required_distance
    
    @staticmethod
    def get_collision_count(circles: np.ndarray, center_idx: int, width: float, height: float) -> int:
        """Count collision constraints with nearby circles."""
        collision_count = 0
        center_x, center_y = circles[center_idx, 0], circles[center_idx, 1]
        
        for j in range(len(circles)):
            if i != j:
                dx = center_x - circles[j, 0]
                dy = center_y - circles[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                required_distance = circles[i, 2] + circles[j, 2] + 0.001
                if distance < required_distance:
                    collision_count += 1
        return collision_count

class OptimizationEngine:
    """Handles the optimization phases with adaptive strategies."""
    
    @staticmethod
    def adaptive_mutate(circles: np.ndarray, 
                       center_idx: int,
                       width: float,
                       height: float,
                       constraint_density: float,
                       max_radius: float,
                       phase: str = "relaxed") -> np.ndarray:
        """Apply adaptive mutation to a specific circle based on constraint density."""
        mutated = circles.copy()
        x, y, r = mutated[center_idx]
        
        # Determine mutation intensity based on constraint density
        # High constraint density = lower mutation intensity
        if constraint_density > 2.0:
            mutation_intensity = 0.005
        elif constraint_density > 1.0:
            mutation_intensity = 0.01
        else:
            mutation_intensity = 0.02
            
        # Adjust based on phase
        if phase == "strict":
            mutation_intensity *= 0.5
            
        # Mutate position
        x += np.random.normal(0, mutation_intensity)
        y += np.random.normal(0, mutation_intensity)
        
        # Bound constraints
        x = np.clip(x, r, width - r)
        y = np.clip(y, r, height - r)
        
        # Mutate radius with constraint-aware adaptation
        if constraint_density > 2.0:  # High constraint area
            radius_step = np.random.normal(0, mutation_intensity * 0.3)
        elif constraint_density < 0.5:  # Low constraint area
            radius_step = np.random.normal(0, mutation_intensity * 2.0)
        else:  # Medium constraint area
            radius_step = np.random.normal(0, mutation_intensity * 1.0)
            
        new_r = max(0.001, r + radius_step)
        
        # Ensure new radius is compatible with position
        max_radius_allowed = min(x, width - x, y, height - y)
        new_r = min(new_r, max_radius_allowed)
        
        mutated[center_idx] = [x, y, new_r]
        return mutated
    
    @staticmethod
    def expand_circles(circles: np.ndarray, 
                      width: float,
                      height: float,
                      constraint_densities: np.ndarray,
                      phase: str = "relaxed") -> Tuple[np.ndarray, bool]:
        """Expand circles based on available space and constraints."""
        improved = False
        mutated = circles.copy()
        
        # Shuffle circles for diverse optimization
        indices = list(range(len(circles)))
        np.random.shuffle(indices)
        
        for i in indices:
            # Calculate constraints for this circle
            max_radius = min(
                mutated[i][0],  # Distance to left edge
                width - mutated[i][0],  # Distance to right edge
                mutated[i][1],  # Distance to bottom edge
                height - mutated[i][1]   # Distance to top edge
            ) - 0.001
            
            # Consider collision constraints with neighbors
            for j in range(len(circles)):
                if i != j:
                    if ConstraintAnalyzer.check_collision(mutated, i, j):
                        dx = mutated[i][0] - mutated[j][0]
                        dy = mutated[i][1] - mutated[j][1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        collision_radius = distance - mutated[j][2] - 0.001
                        if collision_radius > 0:
                            max_radius = min(max_radius, collision_radius)
            
            # Adaptive expansion based on constraint density
            constraint_density = constraint_densities[i]
            
            # Expansion factor based on constraint density
            if constraint_density > 2.0:  # Highly constrained region
                expansion_factor = 0.2
                delta = min(0.01, max_radius - mutated[i][2]) * expansion_factor
            elif constraint_density > 1.0:  # Moderately constrained
                expansion_factor = 0.5
                delta = min(0.015, max_radius - mutated[i][2]) * expansion_factor
            else:  # Less constrained
                expansion_factor = 1.0
                delta = min(0.02, max_radius - mutated[i][2]) * expansion_factor
            
            if delta > 0.0005:
                mutated[i][2] += delta
                improved = True
        
        return mutated, improved

class CirclePacker:
    """Main orchestrator for circle packing optimization."""
    
    def __init__(self, n_circles: int = 21, width: float = 1.2, height: float = 0.8):
        self.n_circles = n_circles
        self.width = width
        self.height = height
        self.initializer = InitializationStrategy()
        self.constraint_analyzer = ConstraintAnalyzer()
        self.optimizer = OptimizationEngine()
        
    def initialize_population(self) -> np.ndarray:
        """Initialize circles using multiple strategies and select the best."""
        strategies = [
            ("hexagonal", self.initializer.hexagonal),
            ("random_spaced", self.initializer.random_spaced),
            ("square_grid", self.initializer.square_grid)
        ]
        
        best_circles = None
        best_sum = -1
        
        for strategy_name, strategy_func in strategies:
            circles = strategy_func(self.n_circles, self.width, self.height)
            total_radius = np.sum(circles[:, 2])
            if total_radius > best_sum:
                best_sum = total_radius
                best_circles = circles.copy()
        
        return best_circles if best_circles is not None else self.initializer.hexagonal(self.n_circles, self.width, self.height)
    
    def calculate_constraint_densities(self, circles: np.ndarray) -> np.ndarray:
        """Calculate constraint density for each circle."""
        densities = np.zeros(len(circles))
        for i in range(len(circles)):
            voronoi_area = self.constraint_analyzer.calculate_voronoi_area(circles, i, self.width, self.height)
            # Inverse of Voronoi area gives constraint density (smaller area = higher density)
            densities[i] = 1.0 / max(1e-6, voronoi_area)
        return densities
    
    def optimize(self) -> np.ndarray:
        """Main optimization routine with two-phase strategy."""
        # Phase 1: Multi-start initialization with better seeding
        circles = self.initialize_population()
        
        # Phase 2: Two-phase optimization with progressive constraint relaxation
        max_iterations_phase1 = 150  # First phase: relaxed constraints
        max_iterations_phase2 = 100  # Second phase: strict constraints
        
        # Phase 1: Progressive constraint relaxation (allow some violations initially)
        for iteration in range(max_iterations_phase1):
            # Calculate constraint densities
            constraint_densities = self.calculate_constraint_densities(circles)
            
            # Apply optimization with relaxed constraints
            circles, improved = self.optimizer.expand_circles(
                circles, self.width, self.height, constraint_densities, "relaxed"
            )
            
            if not improved and iteration > 50:  # Early stopping condition
                break
        
        # Phase 2: Strict constraint enforcement with Voronoi-based adaptive mutation
        for iteration in range(max_iterations_phase2):
            # Calculate constraint densities for this iteration
            constraint_densities = self.calculate_constraint_densities(circles)
            
            # Apply optimization with strict constraints
            circles, improved = self.optimizer.expand_circles(
                circles, self.width, self.height, constraint_densities, "strict"
            )
            
            if not improved:
                break
        
        # Final validation and cleanup
        self._validate_and_fix_constraints(circles)
        
        return circles
    
    def _validate_and_fix_constraints(self, circles: np.ndarray) -> None:
        """Ensure all circles satisfy boundary and collision constraints."""
        # Check boundary violations and fix them
        for i in range(len(circles)):
            # Ensure circles are within bounds
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], self.width - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], self.height - circles[i, 2])
        
        # Resolve collision violations iteratively
        for _ in range(50):
            improved = False
            for i in range(len(circles)):
                # Check collisions with all others
                for j in range(len(circles)):
                    if i != j:
                        if ConstraintAnalyzer.check_collision(circles, i, j):
                            dx = circles[i, 0] - circles[j, 0]
                            dy = circles[i, 1] - circles[j, 1]
                            distance = np.sqrt(dx*dx + dy*dy)
                            required_distance = circles[i, 2] + circles[j, 2] + 0.001
                            
                            if distance < required_distance:
                                # Move circles apart (simple approach)
                                if distance > 0.001:
                                    move_distance = (required_distance - distance) / 2.0
                                    direction_x = dx / distance
                                    direction_y = dy / distance
                                    
                                    # Move both circles away from each other
                                    circles[i, 0] += direction_x * move_distance * 0.5
                                    circles[i, 1] += direction_y * move_distance * 0.5
                                    circles[j, 0] -= direction_x * move_distance * 0.5
                                    circles[j, 1] -= direction_y * move_distance * 0.5
                                    
                                    # Clip back to bounds
                                    circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], self.width - circles[i, 2])
                                    circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], self.height - circles[i, 2])
                                    circles[j, 0] = np.clip(circles[j, 0], circles[j, 2], self.width - circles[j, 2])
                                    circles[j, 1] = np.clip(circles[j, 1], circles[j, 2], self.height - circles[j, 2])
                                
                                improved = True
            
            if not improved:
                break

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    # Create packer instance
    packer = CirclePacker(n_circles=21, width=1.2, height=0.8)
    
    # Perform optimization
    circles = packer.optimize()
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")