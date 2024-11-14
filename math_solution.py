import numpy as np


def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
) -> np.ndarray:
    """
    :param camera_matrix: intrinsic camera matrix, np.ndarray 3x3
    :param camera_position1: position of the first camera in world coordinates, np.ndarray 3x1
    :param camera_rotation1: rotation matrix of the first camera in world coordinates, np.ndarray 3x3
    :param camera_position2: position of the second camera in world coordinates, np.ndarray 3x1
    :param camera_rotation2: rotation matrix of the second camera in world coordinates, np.ndarray 3x3
    :param image_points1: points in the first image, np.ndarray Nx2
    :param image_points2: points in the second image, np.ndarray Nx2
    :return: triangulated 3D points, np.ndarray Nx3
    """

    # Compute projection matrices for both cameras
    proj_matrix1 = camera_matrix @ np.hstack((camera_rotation1.T, -camera_rotation1.T @ camera_position1))
    proj_matrix2 = camera_matrix @ np.hstack((camera_rotation2.T, -camera_rotation2.T @ camera_position2))
    points_3d = []

    for pt1, pt2 in zip(image_points1, image_points2):
        pt1_homog = np.array([pt1[0], pt1[1], 1.0])
        pt2_homog = np.array([pt2[0], pt2[1], 1.0])

        # Set up the linear system Ax = 0
        A = np.array([
            -pt1_homog[0] * proj_matrix1[2].T + proj_matrix1[0].T,
            pt1_homog[1] * proj_matrix1[2].T - proj_matrix1[1].T,
            -pt2_homog[0] * proj_matrix2[2].T + proj_matrix2[0].T,
            pt2_homog[1] * proj_matrix2[2].T - proj_matrix2[1].T
        ])

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        points_3d.append(X[:3] / X[3])

    points_3d = np.array(points_3d)

    return points_3d
