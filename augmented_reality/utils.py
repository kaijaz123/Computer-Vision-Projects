import numpy as np
import math
import cv2

def projection_3D(calibration_matrix, homography):
    """
    Recover the R3 column in external calibration matrix for 3D homography matrix
    by using the camera calibration matrix and calculated 2D homography matrix

    Note:
    The method is done by using F.Merono method for calibration of the Jones matrix coefficients
    """

    # The inverted homography matrix is optional
    homography = homography * (-1)

    # Compute external calibration matrix
    external_calibration_matrix = np.dot(np.linalg.inv(calibration_matrix), homography)
    R1 = external_calibration_matrix[:, 0]
    R2 = external_calibration_matrix[:, 1]
    t = external_calibration_matrix[:, 2]

    # normalise vectors
    norm = math.sqrt(np.linalg.norm(R1, 2) * np.linalg.norm(R2, 2))
    R1 = R1 / norm
    R2 = R2 / norm
    translation = t / norm

    # compute the orthonormal basis
    c = R1 + R2
    p = np.cross(R1, R2)
    d = np.cross(c, p)

    # cross product of new R1 and R2 to obtain R3
    R1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    R2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    R3 = np.cross(R1, R2)
    matrix_3d = np.stack((R1, R2, R3, translation)).T
    return np.dot(calibration_matrix, matrix_3d)


def render(img, obj, projection, query, scale_movement):
    """
    Render a loaded obj model into the current video frame
    """
    DEFAULT_COLOR = (0, 0, 0)
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3 # scale the size of projected model
    h, w = query.shape[:2]

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render the model in the middle of the project surface
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)

    return img
