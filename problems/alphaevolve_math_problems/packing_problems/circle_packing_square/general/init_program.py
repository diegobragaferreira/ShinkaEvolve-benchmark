# EVOLVE-BLOCK-START
import numpy as np

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing_square(num_circles: int) -> np.ndarray:
    """
    Places num_circles non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (num_circles,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # TODO: Improve
    circles = np.zeros((num_circles, 3))

    return circles


# EVOLVE-BLOCK-END
