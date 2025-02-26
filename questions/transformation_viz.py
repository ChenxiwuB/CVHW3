import numpy as np
import matplotlib.pyplot as plt
import argparse

def estimate_transform(start_points, end_points):
    """
    This function estimates the transformation matrix using the given
    starting and ending points. We recommend using the least squares solution
    to solve for the transformation matrix. See the handout, Q1 of the written
    questions, or the lecture slides for how to set up these equations.
    
    :param starting_points: 2xN array of points in the starting image
    :param end_points: 2xN array of points in the ending image
    :return: Matrix M such that M * starting_points = end_points
    """
    
    # We solve for the matrix M such that M * starting_points = end_points
    # We can rewrite this as M * starting_points - end_points = 0
    
    # TODO Step 1: Transform the point coordinates to the A matrix and b vector
    # A = np.array(
    #     [
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #     ]
    # )
    # b = np.array([[0], [0], [0], [0], [0], [0], [0], [0]])

    # for i in range(4):
    #     x = start_points[0, i]
    #     y = start_points[1, i]
    #     xprime = end_points[0, i]
    #     yprime = end_points[1, i]
        
    #     A[2*i, :]   = [x, y, 0, 0]
    #     b[2*i, 0]   = xprime
        
    #     A[2*i+1, :] = [0, 0, x, y]
    #     b[2*i+1, 0] = yprime
        
    # ## TODO Step 2: Solve for the least squares solution using np.linalg.lstsq()
    # x_vec, residual, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # ## TODO Step 3: Reshape the x vector into a matrix the same size as M
    # x = x_vec.reshape((2, 2))
    
    
    # Create the A matrix (2N x 6) and b vector (2N x 1)
    A = np.zeros((2 * 4, 6))
    b = np.zeros((2 * 4, 1))
    
    for i in range(4):
        x = start_points[0, i]
        y = start_points[1, i]
        xprime = end_points[0, i]
        yprime = end_points[1, i]
        
        # Equation for x':  x * m11 + y * m12 + 1 * t1 = xprime
        A[2 * i, :] = [x, y, 1, 0, 0, 0]
        b[2 * i, 0] = xprime
        
        # Equation for y':  x * m21 + y * m22 + 1 * t2 = yprime
        A[2 * i + 1, :] = [0, 0, 0, x, y, 1]
        b[2 * i + 1, 0] = yprime
    
    # Solve for x in the least squares sense (should be exact for an affine transformation)
    x_vec, residual, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Reshape the solution vector into a 2x3 affine transformation matrix
    x = x_vec.reshape((2, 3))
    
    return x, residual


def transform(starting_points, transformation_matrix):
    N = starting_points.shape[1]
    homo_points = np.vstack((starting_points, np.ones((1, N))))
    return transformation_matrix @ homo_points


def main():
    start_points = np.array([[1, 1.5, 2, 2.5], [1, 0.5, 1, 2]])
    end_points = np.array([[-0.9, -0.1, -0.4, -1.25], [0.8, 1.3, 1.9, 2.55]])

    transformation_matrix, residual = estimate_transform(start_points, end_points)
    print(f"Your transformation matrix is\n {transformation_matrix}")
    print(f"The residual of your transformation is {residual}")

    transformed_points = transform(start_points, transformation_matrix)
    print(f"Your transformed points are\n {transformed_points}")

    fig, ax = plt.subplots()
    
    # annotate points
    start_offsets = np.array([[-0.15, 0.05, 0.1, 0], [0, -0.1, 0, 0]])
    end_offsets = np.array([[0.05, 0, 0, 0], [-0.08, 0, 0, 0]])
    for i, txt in enumerate(["a", "b", "c", "d"]):
        ax.annotate(txt, xy=(start_points[0, i] + start_offsets[0, i], start_points[1, i] + start_offsets[1, i]), fontweight='bold', fontsize=12)
    for i, txt in enumerate(["a'", "b'", "c'", "d'"]):
        ax.annotate(txt, (end_points[0, i] + end_offsets[0, i], end_points[1, i] + end_offsets[1, i]), fontweight='bold', fontsize=12)

    # plot the starting and ending points
    ax.fill(start_points[0], start_points[1], color="blue", alpha=0.5, label="Starting Points", zorder=3)
    ax.fill(end_points[0], end_points[1], alpha=0.5, color="red", label="End Points", zorder=3)
    # plot transformed points
    ax.fill(transformed_points[0], transformed_points[1], color="green", alpha=0.5, label="Your Transformation", zorder=3)
    
    plt.xlim([-1.5, 3.0])
    plt.ylim([0.0, 3.0])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
