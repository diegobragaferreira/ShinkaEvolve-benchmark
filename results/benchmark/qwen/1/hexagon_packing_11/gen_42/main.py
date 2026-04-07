# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time

# Set random seed for reproducibility
np.random.seed(42)

def create_regular_hexagon(center=(0, 0), side_length=1, rotation=0):
    """Create a regular hexagon as a Shapely polygon."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = [(center[0] + side_length * np.cos(a), 
               center[1] + side_length * np.sin(a)) for a in angles]
    return Polygon(points)

def hexagon_vertices(center, side_length, rotation):
    """Get vertices of a hexagon."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    return [(center[0] + side_length * np.cos(a), 
             center[1] + side_length * np.sin(a)) for a in angles]

def evaluate_solution(x):
    """Evaluate a candidate solution for the hexagon packing problem."""
    # Parse the solution vector into hexagon parameters
    # x[0:22]: 11 hexagons with (x, y) positions
    # x[22:33]: 11 hexagon rotations
    
    # Extract inner hexagon parameters
    inner_positions = x[:22].reshape(-1, 2)
    inner_rotations = x[22:33]
    
    # Extract outer hexagon parameters
    outer_center = (x[33], x[34])
    outer_rotation = x[35]
    outer_side_length = x[36]
    
    # Create inner hexagons
    inner_hexagons = []
    for i in range(11):
        center = tuple(inner_positions[i])
        rotation = inner_rotations[i]
        hexagon = create_regular_hexagon(center, 1, rotation)
        inner_hexagons.append(hexagon)
    
    # Create outer hexagon
    outer_hexagon = create_regular_hexagon(outer_center, outer_side_length, outer_rotation)
    
    # Check containment: all inner hexagon vertices must be inside outer hexagon
    all_contained = True
    for hexagon in inner_hexagons:
        if not outer_hexagon.contains(hexagon):
            all_contained = False
            break
    
    if not all_contained:
        # Return very bad fitness if not contained
        return 1e10
    
    # Check overlap: no two inner hexagons should overlap
    for i in range(11):
        for j in range(i+1, 11):
            if inner_hexagons[i].intersects(inner_hexagons[j]):
                # Return very bad fitness if overlapping
                return 1e10
    
    # If we reach here, the solution is valid
    # Return negative of outer hexagon side length (we want to minimize this)
    return -outer_side_length

def optimize_hexagon_packing():
    """Optimize the hexagon packing using differential evolution."""
    # Initial guess: some reasonable initial configuration
    # Start with a central hexagon, then place others in a pattern
    initial_positions = np.array([
        [0, 0],      # center
        [-2, 0],     # left
        [2, 0],      # right
        [0, 2],      # top
        [0, -2],     # bottom
        [-1, 1],     # top-left
        [1, 1],      # top-right
        [-1, -1],    # bottom-left
        [1, -1],     # bottom-right
        [-2, 1],     # far top-left
        [2, 1],      # far top-right
    ])
    
    # Initial rotations (all 0 for simplicity)
    initial_rotations = np.zeros(11)
    
    # Initial outer hexagon (large enough to start)
    initial_outer_center = [0, 0]
    initial_outer_rotation = 0
    initial_outer_side_length = 10
    
    # Combine everything into one vector
    initial_guess = np.concatenate([
        initial_positions.flatten(),
        initial_rotations,
        initial_outer_center,
        [initial_outer_rotation],
        [initial_outer_side_length]
    ])
    
    # Define bounds for each variable
    # Positions: -10 to 10
    bounds = []
    for _ in range(22):  # 11 hexagons * 2 coordinates each
        bounds.extend([(-10, 10), (-10, 10)])
    
    # Rotations: 0 to 360 degrees
    for _ in range(11):
        bounds.append((0, 360))
    
    # Outer hexagon center (x, y)
    bounds.extend([(-10, 10), (-10, 10)])
    
    # Outer hexagon rotation (0 to 360)
    bounds.append((0, 360))
    
    # Outer hexagon side length (0.1 to 20)
    bounds.append((0.1, 20))
    
    # Run optimization
    result = differential_evolution(
        evaluate_solution,
        bounds,
        maxiter=100,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=False
    )
    
    return result

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Get the optimized solution
    result = optimize_hexagon_packing()
    
    # Extract results
    x = result.x
    inner_positions = x[:22].reshape(-1, 2)
    inner_rotations = x[22:33]
    outer_center = (x[33], x[34])
    outer_rotation = x[35]
    outer_side_length = x[36]
    
    # Format inner hexagon data
    inner_hex_data = np.column_stack([inner_positions, inner_rotations])
    
    # Format outer hexagon data
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_rotation])
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
