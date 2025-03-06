
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points. See the handout, Q5
    of the written questions, or the lecture slides for how to set up these
    equations.

    Don't forget to set M_34 = 1 in this system to fix the scale.

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points
    
    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
             residual, the error in the estimation of M given the point sets
    """
    ######################
    # Do not change this #
    ######################

    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in our scanning setup
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters)
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)
    points3d = np.array(points3d)
    N = len(points2d)

    A = []
    b = []
    for i in range(N):
        X, Y, Z = points3d[i]
        u, v = points2d[i]
        
        A.append([
            -X, -Y, -Z, -1,      
             0,  0,  0,  0,      
             u*X, u*Y, u*Z       
        ])
        b.append(-u)


        A.append([
             0,  0,  0,  0,      
            -X, -Y, -Z, -1,      
             v*X, v*Y, v*Z       
        ])
        b.append(-v)

    A = np.array(A, dtype=np.float64) 
    b = np.array(b, dtype=np.float64)
  
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    M = np.array([
        [x[0],  x[1],  x[2],  x[3]],
        [x[4],  x[5],  x[6],  x[7]],
        [x[8],  x[9],  x[10], 1.0]
    ], dtype=np.float64)

    if residuals.size > 0:
        residual = residuals[0]
    else:
        residual = 0.0
        for i in range(N):
            X, Y, Z = points3d[i]
            u_true, v_true = points2d[i]
            val = M @ np.array([X, Y, Z, 1], dtype=np.float64)
            u_e = val[0] / val[2]
            v_e = val[1] / val[2]
            residual += (u_e - u_true)**2 + (v_e - v_true)**2

    return M, residual


def normalize_coordinates(points):
    """
    ============================ EXTRA CREDIT ============================
    Normalize the given Points before computing the fundamental matrix. You
    should perform the normalization to make the mean of the points 0
    and the average magnitude 1.0.

    The transformation matrix T is the product of the scale and offset matrices.

    Offset Matrix
    Find c_u and c_v and create a matrix of the form in the handout for T_offset

    Scale Matrix
    Subtract the means of the u and v coordinates, then take the reciprocal of
    their standard deviation i.e. 1 / np.std([...]). Then construct the scale
    matrix in the form provided in the handout for T_scale

    :param points: set of [n x 2] 2D points
    :return: a tuple of (normalized_points, T) where T is the [3 x 3] transformation
    matrix
    """
    ########################
    # TODO: Your code here #
    ########################
    # This is a placeholder with the identity matrix for T replace with the
    # real transformation matrix for this set of points
    T = np.eye(3)

    return points, T

def estimate_fundamental_matrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2. The fundamental matrix will transform a point into 
    a line within the second image - the epipolar line - such that F x' = l. 
    Fitting a fundamental matrix to a set of points will try to minimize the 
    error of all points x to their respective epipolar lines transformed 
    from x'. The residual can be computed as the difference from the known 
    geometric constraint that x^T F x' = 0.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Implement this function efficiently as it will be
    called repeatedly within the RANSAC part of the project.

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
            residual, the sum of the squared error in the estimation
    """
    n = points1.shape[0]
    u = points1[:, 0]
    v = points1[:, 1]
    up = points2[:, 0]
    vp = points2[:, 1]
    A = np.column_stack((up*u, up*v, up, vp*u, vp*v, vp, u, v, np.ones(n)))
    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1, :]
    F_matrix = f.reshape((3, 3))
    U_f, S_f, V_ft = np.linalg.svd(F_matrix)
    S_f[-1] = 0
    F_matrix = U_f @ np.diag(S_f) @ V_ft
    X  = np.hstack([points1, np.ones((n, 1))])
    Xp = np.hstack([points2, np.ones((n, 1))])
    vals = np.sum(Xp * (F_matrix @ X.T).T, axis=1)
    residual = np.sum(vals**2)

    return F_matrix, residual

def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Implement RANSAC to find the best fundamental matrix robustly
    by randomly sampling interest points. See the handout for a detailing of the RANSAC method.
    
    Inputs:
    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points across two images. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    Outputs:
    best_Fmatrix is the [3 x 3] fundamental matrix
    best_inliers1 and best_inliers2 are the [M x 2] subset of matches1 and matches2 that
    are inliners with respect to best_Fmatrix
    best_inlier_residual is the sum of the square error induced by best_Fmatrix upon the inlier set

    :return: best_Fmatrix, inliers1, inliers2, best_inlier_residual
    """
    # DO NOT TOUCH THE FOLLOWING LINES
    random.seed(0)
    np.random.seed(0)

    best_Fmatrix = None
    best_inliers_a = None
    best_inliers_b = None
    best_inlier_count = 0
    best_inlier_residual = np.inf

    #******Set a threshold*******
    threshold = 0.005

    N = matches1.shape[0]
    for i in range(num_iters):
        sample_indices = np.random.choice(N, 10, replace=False)
        subset1 = matches1[sample_indices, :]
        subset2 = matches2[sample_indices, :]
        #F, _ = cv2.findFundamentalMat(subset1, subset2, cv2.FM_8POINT, 1e10, 0, 1)
        F, _ = estimate_fundamental_matrix(subset1, subset2)

        if F is None or F.shape != (3, 3):
            continue

        current_inlier_idx = []
        current_residual = 0.0

        for j in range(N):
            pt1 = np.array([matches1[j, 0], matches1[j, 1], 1.0])
            pt2 = np.array([matches2[j, 0], matches2[j, 1], 1.0])
            error = abs(np.dot(pt2, np.dot(F, pt1)))
            if error < threshold:
                current_inlier_idx.append(j)
                current_residual += error**2 

        inlier_counts.append(len(current_inlier_idx))
        inlier_residuals.append(current_residual)

        if (len(current_inlier_idx) > best_inlier_count) or \
           (len(current_inlier_idx) == best_inlier_count and current_residual < best_inlier_residual):
            best_inlier_count = len(current_inlier_idx)
            best_Fmatrix = F
            best_inlier_residual = current_residual
            best_inliers_a = matches1[current_inlier_idx, :]
            best_inliers_b = matches2[current_inlier_idx, :]

    return best_Fmatrix, best_inliers_a, best_inliers_b, best_inlier_residual


def matches_to_3d(points2d_1, points2d_2, M1, M2, threshold=1.0):
    """
    Given two sets of corresponding 2D points and two projection matrices, you will need to solve
    for the ground-truth 3D points using np.linalg.lstsq().

    You may find that some 3D points have high residual/error, in which case you 
    can return a subset of the 3D points that lie within a certain threshold.
    In this case, also return subsets of the initial points2d_1, points2d_2 that
    correspond to this new inlier set. You may modify the default value of threshold above.
    All local helper code that calls this function will use this default value, but we
    will pass in a different value when autograding.

    N is the input number of point correspondences
    M is the output number of 3D points / inlier point correspondences; M could equal N.

    :param points2d_1: [N x 2] points from image1
    :param points2d_2: [N x 2] points from image2
    :param M1: [3 x 4] projection matrix of image1
    :param M2: [3 x 4] projection matrix of image2
    :param threshold: scalar value representing the maximum allowed residual for a solved 3D point

    :return points3d_inlier: [M x 3] NumPy array of solved ground truth 3D points for each pair of 2D
    points from points2d_1 and points2d_2
    :return points2d_1_inlier: [M x 2] points as subset of inlier points from points2d_1
    :return points2d_2_inlier: [M x 2] points as subset of inlier points from points2d_2
    """

    points3d_list = []
    inlier_idx = []

    N = len(points2d_1)
    for i in range(N):
        u1, v1 = points2d_1[i]
        u2, v2 = points2d_2[i]

        A = np.zeros((4, 3), dtype=float)
        b = np.zeros((4,),    dtype=float)

        A[0, :] = u1 * M1[2, 0:3] - M1[0, 0:3]
        b[0]    = M1[0, 3] - u1 * M1[2, 3]

        A[1, :] = v1 * M1[2, 0:3] - M1[1, 0:3]
        b[1]    = M1[1, 3] - v1 * M1[2, 3]

        A[2, :] = u2 * M2[2, 0:3] - M2[0, 0:3]
        b[2]    = M2[0, 3] - u2 * M2[2, 3]

        A[3, :] = v2 * M2[2, 0:3] - M2[1, 0:3]
        b[3]    = M2[1, 3] - v2 * M2[2, 3]

        X, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        homog_3d = np.append(X, 1.0)

        proj_1 = M1 @ homog_3d 
        px1 = proj_1[0] / proj_1[2]
        py1 = proj_1[1] / proj_1[2]
  
        err1 = np.sqrt((px1 - u1)**2 + (py1 - v1)**2)

        proj_2 = M2 @ homog_3d
        px2 = proj_2[0] / proj_2[2]
        py2 = proj_2[1] / proj_2[2]
        err2 = np.sqrt((px2 - u2)**2 + (py2 - v2)**2)

        total_error = err1 + err2

        if total_error < threshold:
            points3d_list.append(X)
            inlier_idx.append(i)

    points3d_inlier = np.array(points3d_list, dtype=float)
    points2d_1_inlier = points2d_1[inlier_idx, :]
    points2d_2_inlier = points2d_2[inlier_idx, :]

    return points3d_inlier, points2d_1_inlier, points2d_2_inlier

#/////////////////////////////DO NOT CHANGE BELOW LINE///////////////////////////////
inlier_counts = []
inlier_residuals = []

def visualize_ransac():
    iterations = np.arange(len(inlier_counts))
    best_inlier_counts = np.maximum.accumulate(inlier_counts)
    best_inlier_residuals = np.minimum.accumulate(inlier_residuals)

    plt.figure(1, figsize = (8, 8))
    plt.subplot(211)
    plt.plot(iterations, inlier_counts, label='Current Inlier Count', color='red')
    plt.plot(iterations, best_inlier_counts, label='Best Inlier Count', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Number of Inliers")
    plt.title('Current Inliers vs. Best Inliers per Iteration')
    plt.legend()

    plt.subplot(212)
    plt.plot(iterations, inlier_residuals, label='Current Inlier Residual', color='red')
    plt.plot(iterations, best_inlier_residuals, label='Best Inlier Residual', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title('Current Residual vs. Best Residual per Iteration')
    plt.legend()
    plt.show()