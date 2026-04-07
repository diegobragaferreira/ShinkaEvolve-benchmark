# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import math

def fibonacci_sphere(n):
    """Generate n points evenly distributed on a unit sphere using Fibonacci spiral method."""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def icosahedron_points():
    """Generate vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1]
    ])
    # Normalize to unit sphere
    return vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

def subdivide_triangle(v1, v2, v3, depth):
    """Recursively subdivide a triangle on the sphere."""
    if depth == 0:
        return [v1, v2, v3]
    
    # Find midpoints and normalize to sphere
    m1 = (v1 + v2) / 2
    m2 = (v2 + v3) / 2
    m3 = (v3 + v1) / 2
    
    m1 = m1 / np.linalg.norm(m1)
    m2 = m2 / np.linalg.norm(m2)
    m3 = m3 / np.linalg.norm(m3)
    
    # Recursively subdivide
    triangles = []
    triangles.extend(subdivide_triangle(v1, m1, m3, depth-1))
    triangles.extend(subdivide_triangle(m1, v2, m2, depth-1))
    triangles.extend(subdivide_triangle(m3, m2, v3, depth-1))
    triangles.extend(subdivide_triangle(m1, m2, m3, depth-1))
    
    return triangles

def icosahedral_tiling(n_points):
    """Generate approximately n_points using icosahedral subdivision."""
    # Start with icosahedron
    vertices = icosahedron_points()
    
    # Subdivide to get enough points
    # Each subdivision increases vertex count by factor of 4
    depth = int(np.log2(n_points / 12)) + 1
    triangles = []
    # Connect vertices to form 20 triangular faces
    faces = [
        [0,1,2], [0,2,3], [0,3,4], [0,4,5], [0,5,1],
        [1,6,2], [2,7,3], [3,8,4], [4,9,5], [5,10,1],
        [6,1,11], [7,2,11], [8,3,11], [9,4,11], [10,5,11],
        [6,7,8], [7,8,9], [8,9,10], [9,10,6], [10,6,7]
    ]
    
    # Generate all vertices
    all_vertices = vertices.tolist()
    for face in faces:
        tri = subdivide_triangle(vertices[face[0]], vertices[face[1]], vertices[face[2]], depth)
        all_vertices.extend(tri)
    
    # Remove duplicates and normalize
    unique_vertices = []
    seen = set()
    for v in all_vertices:
        v_tuple = tuple([round(x, 10) for x in v])  # Round for comparison
        if v_tuple not in seen:
            seen.add(v_tuple)
            unique_vertices.append(v)
    
    # Take first n_points  
    points = np.array(unique_vertices[:n_points])
    # Normalize to ensure they're on unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    points = points / norms
    
    return points

def point_on_sphere_to_cartesian(point, radius=0.5):
    """Convert point on sphere to cartesian coordinates in unit cube."""
    # Map sphere point to cube [0,1]^3
    return (point + 1) / 2

def cartesian_to_sphere_point(cart_point):
    """Convert cartesian point in unit cube to unit sphere point."""
    # Map cube [0,1]^3 to sphere [-1,1]^3
    sphere_point = cart_point * 2 - 1
    # Normalize to unit sphere
    norm = np.linalg.norm(sphere_point)
    if norm > 0:
        sphere_point = sphere_point / norm
    return sphere_point

def min_max_ratio(points):
    """Calculate the ratio of minimum to maximum pairwise distances."""
    if len(points) < 2:
        return 0

    # Calculate pairwise distances
    distances = pdist(points)

    # Get min and max distances
    d_min = np.min(distances)
    d_max = np.max(distances)

    # Avoid division by zero
    if d_max == 0:
        return 0

    return d_min / d_max

def geometric_optimization_step(points, step_size=0.01, penalty_weight=1000):
    """
    Perform a geometric optimization step on the points.
    This uses a gradient-based approach on the sphere manifold.
    """
    n_points = len(points)
    if n_points < 2:
        return points
    
    # Calculate forces between all points
    forces = np.zeros_like(points)
    
    # For each pair of points, compute force (repulsion)
    for i in range(n_points):
        for j in range(i+1, n_points):
            diff = points[i] - points[j]
            dist_sq = np.dot(diff, diff)
            
            if dist_sq > 0:
                dist = np.sqrt(dist_sq)
                # Repulsive force (inverse square law)
                force_magnitude = 1.0 / (dist_sq * dist)
                force = force_magnitude * diff / dist
                forces[i] += force
                forces[j] -= force
    
    # Normalize forces for consistent step size
    force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
    force_norms = np.where(force_norms == 0, 1, force_norms)
    forces = forces / force_norms
    
    # Apply step
    new_points = points - step_size * forces
    
    # Project back to sphere
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    new_points = new_points / norms
    
    return new_points

def smooth_boundary_penalty(points, penalty_weight=1000):
    """Apply smooth penalty for points near boundaries."""
    penalty = 0
    # For each point, calculate proximity to boundaries [0,1]^3
    for i in range(len(points)):
        for j in range(3):  # x, y, z coordinates
            if points[i, j] < 0:
                penalty += penalty_weight * (0 - points[i, j])**2
            elif points[i, j] > 1:
                penalty += penalty_weight * (points[i, j] - 1)**2
    return penalty

def hybrid_objective(points_flat, penalty_weight=1000):
    """Combined objective function that balances distance ratio with boundary penalties."""
    # Reshape points
    points = points_flat.reshape(-1, 3)
    
    # Calculate the main objective
    ratio = min_max_ratio(points)
    
    # Add penalty for boundary violations
    penalty = smooth_boundary_penalty(points, penalty_weight)
    
    # Return negative since we minimize in scipy.optimize
    return -ratio + penalty

def sphere_tiling_initialization(n_points=14, method='fibonacci'):
    """Generate initial point set using spherical tiling techniques."""
    if method == 'icosahedral':
        # Use icosahedral tiling for high-quality distribution
        points = icosahedral_tiling(n_points)
    elif method == 'fibonacci':
        # Use Fibonacci spiral for good distribution
        points = fibonacci_sphere(n_points)
    else:
        # Default to Fibonacci
        points = fibonacci_sphere(n_points)
    
    # Map to unit cube [0,1]^3
    points_cube = (points + 1) / 2
    return points_cube

def sphere_tangent_space_optimize(points, max_iter=500):
    """
    Optimize using constrained optimization on the sphere manifold.
    This uses a tangent space approach similar to Riemannian optimization.
    """
    # Convert to sphere points for proper optimization
    sphere_points = np.array([cartesian_to_sphere_point(p) for p in points])
    
    # Use a hybrid approach: iterative geometric steps with local refinement
    for iteration in range(max_iter):
        # Apply geometric optimization step
        sphere_points = geometric_optimization_step(sphere_points, step_size=0.005)
        
        # Every few iterations, do more substantial correction
        if iteration % 20 == 0:
            # Refine using scipy minimize on sphere
            try:
                # Create flattened version for scipy optimization
                x0_flat = sphere_points.flatten()
                
                def objective(x_flat):
                    # Convert back to points
                    pts = x_flat.reshape(-1, 3)
                    # Project back to sphere
                    pts_unit = pts / np.linalg.norm(pts, axis=1, keepdims=True)
                    # Calculate ratio
                    ratio = min_max_ratio(pts_unit)
                    return -ratio
                
                # Use L-BFGS-B with bounds constrained to sphere
                bounds = [(-1, 1) for _ in range(len(x0_flat))]
                
                result = minimize(
                    objective,
                    x0_flat,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-10, 'gtol': 1e-10},
                    tol=1e-10
                )
                
                # Update points
                sphere_points = result.x.reshape(-1, 3)
                sphere_points = sphere_points / np.linalg.norm(sphere_points, axis=1, keepdims=True)
                
            except:
                pass  # Continue with geometric optimization
    
    # Convert back to unit cube
    final_points = np.array([point_on_sphere_to_cartesian(p) for p in sphere_points])
    return final_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Start with multiple initialization strategies
    initial_candidates = []
    
    # Strategy 1: Icosahedral tiling (very uniform)
    try:
        ico_points = sphere_tiling_initialization(14, 'icosahedral')
        initial_candidates.append(('icosahedral', ico_points))
    except:
        pass
    
    # Strategy 2: Fibonacci spiral
    try:
        fib_points = sphere_tiling_initialization(14, 'fibonacci')
        initial_candidates.append(('fibonacci', fib_points))
    except:
        pass
    
    # Strategy 3: Random points
    try:
        rand_points = np.random.rand(14, 3)
        initial_candidates.append(('random', rand_points))
    except:
        pass
    
    best_ratio = 0
    best_points = None
    
    # Try each initialization strategy
    for init_name, initial_points in initial_candidates:
        try:
            # Apply geometric optimization
            optimized_points = sphere_tangent_space_optimize(initial_points, max_iter=300)
            
            # Calculate final ratio
            final_ratio = min_max_ratio(optimized_points)
            
            # Keep the best
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            continue
    
    # If nothing worked, return a default initialization
    if best_points is None:
        # Use a simple Fibonacci approach as fallback
        points = fibonacci_sphere(14)
        best_points = (points + 1) / 2
    
    return best_points

# EVOLVE-BLOCK-END