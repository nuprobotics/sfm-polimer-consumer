import numpy as np
import cv2
import typing
from typing import Tuple, Sequence


def get_matches(
        image1: np.ndarray,
        image2: np.ndarray,
        k_ratio: float = 0.75
) -> Tuple[Sequence[cv2.KeyPoint], Sequence[cv2.KeyPoint], Sequence[cv2.DMatch]]:
    sift = cv2.SIFT_create()

    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    kp1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    kp2, descriptors2 = sift.detectAndCompute(img2_gray, None)

    bf = cv2.BFMatcher()

    matches_1_to_2 = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches_2_to_1 = bf.knnMatch(descriptors2, descriptors1, k=2)

    good_matches_1_to_2 = [
        m[0] for m in matches_1_to_2 if len(m) == 2 and m[0].distance < k_ratio * m[1].distance
    ]
    good_matches_2_to_1 = [
        m[0] for m in matches_2_to_1 if len(m) == 2 and m[0].distance < k_ratio * m[1].distance
    ]

    final_matches = []
    for m1 in good_matches_1_to_2:
        for m2 in good_matches_2_to_1:
            if m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx:
                final_matches.append(m1)
                break

    return kp1, kp2, final_matches


def get_second_camera_position(kp1, kp2, matches, camera_matrix):
    coordinates1 = np.array([kp1[match.queryIdx].pt for match in matches])
    coordinates2 = np.array([kp2[match.trainIdx].pt for match in matches])
    E, mask = cv2.findEssentialMat(coordinates1, coordinates2, camera_matrix)
    _, R, t, mask = cv2.recoverPose(E, coordinates1, coordinates2, camera_matrix)
    return R, t, E


# Task 3
def triangulation(
        camera_matrix: np.ndarray,
        camera1_translation_vector: np.ndarray,
        camera1_rotation_matrix: np.ndarray,
        camera2_translation_vector: np.ndarray,
        camera2_rotation_matrix: np.ndarray,
        kp1: typing.Sequence[cv2.KeyPoint],
        kp2: typing.Sequence[cv2.KeyPoint],
        matches: typing.Sequence[cv2.DMatch]
) -> np.ndarray:
    """
    :param camera_matrix: intrinsic camera matrix, np.ndarray 3x3
    :param camera1_translation_vector: first camera translation vector, np.ndarray 3x1
    :param camera1_rotation_matrix: first camera rotation matrix, np.ndarray 3x3
    :param camera2_translation_vector: second camera translation vector, np.ndarray 3x1
    :param camera2_rotation_matrix: second camera rotation matrix, np.ndarray 3x3
    :param kp1: key points from the first image
    :param kp2: key points from the second image
    :param matches: sequence of matched key points between the two images
    :return: triangulated 3D points, np.ndarray Nx3
    """

    proj_matrix1 = camera_matrix @ np.hstack(
        (camera1_rotation_matrix, camera1_translation_vector))
    proj_matrix2 = camera_matrix @ np.hstack(
        (camera2_rotation_matrix, camera2_translation_vector))

    points1 = np.array([kp1[m.queryIdx].pt for m in matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in matches])

    points_3d = []

    for pt1, pt2 in zip(points1, points2):
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


# Task 4
def resection(
        image1,
        image2,
        camera_matrix,
        matches,
        points_3d
):
    pass
    # YOUR CODE HERE


def convert_to_world_frame(translation_vector, rotation_matrix):
    pass
    # YOUR CODE HERE


def visualisation(
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        camera_position3: np.ndarray,
        camera_rotation3: np.ndarray,
):
    def plot_camera(ax, position, direction, label):
        color_scatter = 'blue' if label != 'Camera 3' else 'green'
        # print(position)
        ax.scatter(position[0][0], position[1][0], position[2][0], color=color_scatter, s=100)
        color_quiver = 'red' if label != 'Camera 3' else 'magenta'

        ax.quiver(position[0][0], position[1][0], position[2][0], direction[0], direction[1], direction[2],
                  length=1, color=color_quiver, arrow_length_ratio=0.2)
        ax.text(position[0][0], position[1][0], position[2][0], label, color='black')

    camera_positions = [camera_position1, camera_position2, camera_position3]
    camera_directions = [camera_rotation1[:, 2], camera_rotation2[:, 2], camera_rotation3[:, 2]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_camera(ax, camera_positions[0], camera_directions[0], 'Camera 1')
    plot_camera(ax, camera_positions[1], camera_directions[1], 'Camera 2')
    plot_camera(ax, camera_positions[2], camera_directions[2], 'Camera 3')

    initial_elev = 0
    initial_azim = 270

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=initial_elev, azim=initial_azim)

    ax.set_xlim([-1.50, 2.0])
    ax.set_ylim([-.50, 3.0])
    ax.set_zlim([-.50, 3.0])

    ax_elev_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
    elev_slider = Slider(ax_elev_slider, 'Elev', 0, 360, valinit=initial_elev)

    ax_azim_slider = plt.axes([0.1, 0.05, 0.65, 0.03])
    azim_slider = Slider(ax_azim_slider, 'Azim', 0, 360, valinit=initial_azim)

    def update(val):
        elev = elev_slider.val
        azim = azim_slider.val
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()

    elev_slider.on_changed(update)
    azim_slider.on_changed(update)

    plt.show()


def main():
    image1 = cv2.imread('./images/image0.jpg')
    image2 = cv2.imread('./images/image1.jpg')
    image3 = cv2.imread('./images/image2.jpg')
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    camera_matrix = np.array(config["camera_matrix"], dtype=np.float32, order='C')

    key_points1, key_points2, matches_1_to_2 = get_matches(image1, image2)
    R2, t2, E = get_second_camera_position(key_points1, key_points2, matches_1_to_2, camera_matrix)
    triangulated_points = triangulation(
        camera_matrix,
        np.array([0, 0, 0]).reshape((3, 1)),
        np.eye(3),
        t2,
        R2,
        key_points1,
        key_points2,
        matches_1_to_2
    )

    R3, t3 = resection(image1, image3, camera_matrix, matches_1_to_2, triangulated_points)
    camera_position1, camera_rotation1 = convert_to_world_frame(np.array([0, 0, 0]).reshape((3, 1)), np.eye(3))
    camera_position2, camera_rotation2 = convert_to_world_frame(t2, R2)
    camera_position3, camera_rotation3 = convert_to_world_frame(t3, R3)
    visualisation(
        camera_position1,
        camera_rotation1,
        camera_position2,
        camera_rotation2,
        camera_position3,
        camera_rotation3
    )


if __name__ == "__main__":
    main()
