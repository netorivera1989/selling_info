import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

def conc_env(f, num_points=1000):
    # Define the function on a discrete grid
    grid = np.linspace(0, 1, num_points)
    f_values = np.array([f(x) for x in grid])

    # Treat the function values as 2D points in the plane
    points = np.column_stack([grid, f_values])

    # Compute the convex hull of these points
    hull = ConvexHull(points)

    # Interpolate between the points of the convex hull to get a function that represents the concave envelope
    hull_points = points[hull.vertices, :]
    hull_points = np.flip(hull_points, axis=0)
    min_index = np.argmin(hull_points[:, 0])
    rolled_points = np.roll(hull_points, -min_index, axis=0)

    # Sort the points by x-coordinate
    top_boundary = [rolled_points[0]]
    for point in rolled_points[1:]:
        if point[0] >= top_boundary[-1][0]:
            top_boundary.append(point)

    top_boundary = np.array(top_boundary)

    # Return the interpolation function for the concave envelope
    return  interp1d(top_boundary[:, 0], top_boundary[:, 1], kind='linear', fill_value="extrapolate")

# Test

if __name__ == "__main__":
    def f(x):
        """ Cosine function with a shorter period """
        return x * np.cos(8 * x)
    
    # Test the function with the short period cosine
    num_points = 1000
    f_hat = conc_env(f, num_points=num_points)

    # Plotting
    plt.figure(figsize=(10, 6))
    grid = np.linspace(0, 1, num_points)
    plt.plot(grid, [f(x) for x in grid], label="Original Function f")
    plt.plot(grid, [f_hat(x) for x in grid], label="Concave Envelope", color="red")
    plt.title("Concave Envelope of f")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
