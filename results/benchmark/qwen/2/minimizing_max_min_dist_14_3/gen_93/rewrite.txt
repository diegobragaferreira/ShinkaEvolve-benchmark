# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a sphere tiling and geometric construction approach for optimal performance.
    """
    
    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0.0
        return d_min / d_max

    def project_to_unit_cube(points):
        """Project points to unit cube [0,1]^3"""
        # Find min/max along each axis
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # Handle case where there's no variation
        ranges = max_coords - min_coords
        if np.any(ranges == 0):
            # If any dimension has no variation, return points centered at 0.5
            return np.full_like(points, 0.5)

        # Scale to [0,1] range
        normalized = (points - min_coords) / ranges

        # Ensure they're clipped to [0,1]
        return np.clip(normalized, 0, 1)

    def construct_optimal_sphere_config():
        """Construct an initial configuration based on known optimal spherical arrangements."""
        # For 14 points on sphere, we use a pattern inspired by icosahedral symmetry
        # We create 12 vertices of icosahedron plus 2 additional points
        phi = (1 + np.sqrt(5)) / 2
        vertices = [
            (-1, 0, phi), (1, 0, phi), (-1, 0, -phi), (1, 0, -phi),
            (0, phi, 1), (0, phi, -1), (0, -phi, 1), (0, -phi, -1),
            (phi, 1, 0), (-phi, 1, 0), (phi, -1, 0), (-phi, -1, 0)
        ]
        vertices = np.array(vertices)
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0])
        
        # Add two more points that are well distributed
        # Place one at north pole and one at south pole
        additional_points = np.array([[0, 0, 1], [0, 0, -1]])
        
        # Combine and ensure proper number
        all_points = np.vstack([vertices, additional_points])
        if len(all_points) > 14:
            # Select 14 points that are well spread using angular separation
            # For now, just take first 14
            return all_points[:14]
        elif len(all_points) < 14:
            # Add more points
            # Generate additional points in a structured way
            extra_needed = 14 - len(all_points)
            # Add points along latitude bands
            latitudes = np.linspace(-np.pi/2, np.pi/2, extra_needed + 2)[1:-1]
            additional = []
            for lat in latitudes:
                # Add points at this latitude
                for i in range(4):
                    lon = 2 * np.pi * i / 4
                    x = np.cos(lat) * np.cos(lon)
                    y = np.cos(lat) * np.sin(lon)
                    z = np.sin(lat)
                    additional.append([x, y, z])
            additional = np.array(additional)
            return np.vstack([all_points, additional[:extra_needed]])
        else:
            return all_points

    def spherical_constrained_optimization(points, max_iter=1000):
        """Optimize points on unit sphere using constrained optimization"""
        n_points = len(points)
        
        # Flatten the points for optimization
        initial_flat = points.flatten()
        
        def objective(flat_points):
            points_local = flat_points.reshape(-1, 3)
            # Keep points on unit sphere
            for i in range(len(points_local)):
                norm = np.linalg.norm(points_local[i])
                if norm > 0:
                    points_local[i] = points_local[i] / norm
            return -calculate_min_max_ratio(points_local)  # negative for maximization
            
        def constraint_func(flat_points):
            # Constraint: all points must lie on unit sphere
            points_local = flat_points.reshape(-1, 3)
            norms = np.linalg.norm(points_local, axis=1)
            # Return differences from unit sphere
            return norms - 1.0
            
        # Use SLSQP for constrained optimization
        try:
            result = minimize(
                objective,
                initial_flat,
                method='SLSQP',
                constraints={'type': 'eq', 'fun': constraint_func},
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8},
                tol=1e-6
            )
            refined_points = result.x.reshape(-1, 3)
            # Ensure they're on unit sphere
            for i in range(len(refined_points)):
                norm = np.linalg.norm(refined_points[i])
                if norm > 0:
                    refined_points[i] = refined_points[i] / norm
            return refined_points, -result.fun
        except:
            # Fallback to iterative refinement
            current_points = points.copy()
            best_ratio = calculate_min_max_ratio(current_points)
            best_points = current_points.copy()
            
            # Simple gradient-like approach with sphere projection
            for iter_num in range(max_iter):
                # Compute pairwise distances
                distances = pdist(current_points)
                if len(distances) == 0:
                    break
                    
                # Focus on improving ratios with small distances
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                # For each point, compute forces from neighbors
                new_points = current_points.copy()
                for i in range(n_points):
                    # Compute influence from all other points
                    force = np.zeros(3)
                    for j in range(n_points):
                        if i != j:
                            diff = current_points[i] - current_points[j]
                            dist = np.linalg.norm(diff)
                            if dist > 0:
                                # Repulsive force for nearby points
                                if dist < min_dist * 1.5:  # Only consider nearby points
                                    force -= diff / (dist * dist + 1e-8) * 0.001
                    
                    # Apply force and project back to sphere
                    new_points[i] += force
                    norm = np.linalg.norm(new_points[i])
                    if norm > 0:
                        new_points[i] = new_points[i] / norm
                
                current_points = new_points
                new_ratio = calculate_min_max_ratio(current_points)
                
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = current_points.copy()
                    
                # Stop if improvement is minimal
                if abs(new_ratio - best_ratio) < 1e-6:
                    break
                    
            return best_points, best_ratio
    
    def sphere_tiling_refinement(initial_points, max_iterations=500):
        """
        Refine using sphere tiling concept and symmetry preservation
        """
        points = initial_points.copy()
        best_ratio = calculate_min_max_ratio(points)
        best_points = points.copy()
        
        # Get the angular distribution of points for symmetry analysis
        angles = []
        for p in points:
            theta = np.arccos(p[2])  # polar angle
            phi = np.arctan2(p[1], p[0])  # azimuthal angle
            angles.append((theta, phi))
        
        # Apply a few rounds of targeted optimization
        for iteration in range(max_iterations):
            # Create a refined configuration based on geometric principles
            # Use small perturbations around symmetry-preserving directions
            new_points = points.copy()
            
            # For each point, apply rotation-invariant corrections
            for i in range(len(points)):
                # Compute average distance to neighbors
                distances = []
                for j in range(len(points)):
                    if i != j:
                        dist = np.linalg.norm(points[i] - points[j])
                        distances.append(dist)
                
                if distances:
                    avg_dist = np.mean(distances)
                    # Move points slightly away from neighbors that are too close
                    move_vector = np.zeros(3)
                    for j in range(len(points)):
                        if i != j:
                            diff = points[i] - points[j]
                            dist = np.linalg.norm(diff)
                            if dist > 0 and dist < avg_dist * 0.9:
                                move_vector += diff / (dist * dist + 1e-8)
                    
                    # Apply the correction with adaptive magnitude
                    if np.linalg.norm(move_vector) > 0:
                        move_vector = move_vector / np.linalg.norm(move_vector) * 0.01
                        new_points[i] += move_vector
                        # Project back to sphere
                        norm = np.linalg.norm(new_points[i])
                        if norm > 0:
                            new_points[i] = new_points[i] / norm
            
            points = new_points
            new_ratio = calculate_min_max_ratio(points)
            
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = points.copy()
        
        return best_points, best_ratio

    # Main execution flow
    np.random.seed(42)
    
    # Step 1: Construct initial configuration using geometric principles
    initial_points = construct_optimal_sphere_config()
    
    # Step 2: Apply constrained optimization to improve the configuration
    optimized_points, _ = spherical_constrained_optimization(initial_points, max_iter=500)
    
    # Step 3: Apply specialized refinement
    final_points, final_ratio = sphere_tiling_refinement(optimized_points, max_iterations=300)
    
    # Step 4: Final constraint satisfaction with gradient-based refinement
    try:
        final_points, _ = spherical_constrained_optimization(final_points, max_iter=200)
    except:
        pass
    
    # Normalize to unit cube [0,1]^3
    points_in_cube = project_to_unit_cube(final_points)
    
    return points_in_cube

# EVOLVE-BLOCK-END