import numpy as np
from typing import List, Tuple
import cv2
import matplotlib.pyplot as plt
from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

'''
Please do Not change or add any imports. 
'''

#task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    rot_xyz2XYZ = np.eye(3).astype(float)

    # Your implementation
    alpha_rad = np.radians(alpha)
    c1 = np.cos(alpha_rad)
    s1 = np.sin(alpha_rad)
    rm1 = np.array([[c1, -s1, 0],
                    [s1, c1, 0],
                    [0, 0, 1]])
    beta_rad = np.radians(beta)
    c2 = np.cos(beta_rad)
    s2 = np.sin(beta_rad)
    rm2 = np.array([[1, 0, 0],
                    [0, c2, -s2],
                    [0, s2, c2]])
    gamma_rad = np.radians(gamma)
    c3 = np.cos(gamma_rad)
    s3 = np.sin(gamma_rad)
    rm3 = np.array([[c3, -s3, 0],
                    [s3, c3, 0],
                    [0, 0, 1]])
    rot_xyz2XYZ = np.matmul(np.matmul(rm3, rm2), rm1)
    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    rot_XYZ2xyz = np.eye(3).astype(float)

    # Your implementation
    task = findRot_xyz2XYZ(alpha, beta, gamma)
    rot_XYZ2xyz = np.linalg.inv(task)
    return rot_XYZ2xyz

"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1






#--------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = np.zeros([32, 2], dtype=float)
    # Your implementation
    height, width, c = image.shape
    mp_width = width // 2
    # for left side of the chessboard
    lhs = image[:, :mp_width]
    gray_1 = cv2.cvtColor(lhs, cv2.COLOR_BGR2GRAY)
    ret_1, corners_1 = cv2.findChessboardCorners(gray_1, (4, 4), None)
    if ret_1:
        criteria_1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_1 = cv2.cornerSubPix(gray_1, corners_1, (11, 11), (-1, -1), criteria_1)
        img_coord_1 = cv2.drawChessboardCorners(lhs.copy(), (4, 4), corners_1, True)
    else:
        return None, image
    # for right side of the chessboard
    rhs = image[:, mp_width:]
    gray_2 = cv2.cvtColor(rhs, cv2.COLOR_BGR2GRAY)
    ret_2, corners_2 = cv2.findChessboardCorners(gray_2, (4, 4), None)
    if ret_2:
        criteria_2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_2 = cv2.cornerSubPix(gray_2, corners_2, (11, 11), (-1, -1), criteria_2)
        img_coord_2 = cv2.drawChessboardCorners(rhs.copy(), (4, 4), corners_2, True)
    else:
        return None, image
    corners_2[:, :, 0] += mp_width
    img_coord = np.concatenate((corners_1, corners_2), axis=0)
    img_coord = img_coord.reshape((32, 2))
    combined_image = np.hstack((lhs, rhs))
    cv2.drawChessboardCorners(combined_image.copy(), (8, 4), img_coord, True)
    return img_coord


def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''

    # Your implementation
    import numpy as np

    world_coord = np.array([
        [0, 40, 40],
        [0, 30, 40],
        [0, 20, 40],
        [0, 10, 40],
        [0, 40, 30],
        [0, 30, 30],
        [0, 20, 30],
        [0, 10, 30],
        [0, 40, 20],
        [0, 30, 20],
        [0, 20, 20],
        [0, 10, 20],
        [0, 40, 10],
        [0, 30, 10],
        [0, 20, 10],
        [0, 10, 10],
        [40, 0, 40],
        [40, 0, 30],
        [40, 0, 20],
        [40, 0, 10],
        [30, 0, 40],
        [30, 0, 30],
        [30, 0, 20],
        [30, 0, 10],
        [20, 0, 40],
        [20, 0, 30],
        [20, 0, 20],
        [20, 0, 10],
        [10, 0, 40],
        [10, 0, 30],
        [10, 0, 20],
        [10, 0, 10]
    ])
    world_coord = world_coord.astype(float)
    return world_coord


def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    # Your implementation
    x = img_coord[:, 0]
    y = img_coord[:, 1]
    X = world_coord[:, 0]
    Y = world_coord[:, 1]
    Z = world_coord[:, 2]
    A = np.zeros((64, 12), dtype=float)
    h = 0
    for i in range(64):
            if i % 2 == 0:
                A[i,0] = X[h]
                A[i,1] = Y[h]
                A[i,2] = Z[h]
                A[i,3] = 1
                A[i,4] = 0
                A[i,5] = 0
                A[i,6] = 0
                A[i,7] = 0
                A[i,8] = -x[h] * X[h]
                A[i,9] = -x[h] * Y[h]
                A[i,10] = -x[h] * Z[h]
                A[i,11] = -x[h]
            else:
                A[i,0] = 0
                A[i,1] = 0
                A[i,2] = 0
                A[i,3] = 0
                A[i,4] = X[h]
                A[i,5] = Y[h]
                A[i,6] = Z[h]
                A[i,7] = 1
                A[i,8] = -y[h] * X[h]
                A[i,9] = -y[h] * Y[h]
                A[i,10] = -y[h] * Z[h]
                A[i,11] = -y[h]
                h = h + 1
    U, S, VT = np.linalg.svd(A)
    # U is the left singular vectors
    # S is the singular values (in descending order)
    # VT is the right singular vectors (transposed)

    x = VT[11, :]
    temp_m = x.reshape((3, 4))
    lamda = 1/(np.sqrt((temp_m[2, 0]) ** 2 + (temp_m[2, 1]) ** 2 + (temp_m[2, 2]) ** 2))
    m = lamda * temp_m
    m1 = m[0, :3]
    m2 = m[1, :3]
    m3 = m[-1, :3]
    cx = np.dot(m1.T, m3)
    cy = np.dot(m2.T, m3)
    fx = np.sqrt(np.dot(m1.T, m1) - cx ** 2)
    fy = np.sqrt(np.dot(m2.T, m2) - cy ** 2)
    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)

    # Your implementation
    x = img_coord[:, 0]
    y = img_coord[:, 1]
    X = world_coord[:, 0]
    Y = world_coord[:, 1]
    Z = world_coord[:, 2]
    A = np.zeros((64, 12), dtype=float)
    h = 0
    for i in range(64):
        if i % 2 == 0:
            A[i, 0] = X[h]
            A[i, 1] = Y[h]
            A[i, 2] = Z[h]
            A[i, 3] = 1
            A[i, 4] = 0
            A[i, 5] = 0
            A[i, 6] = 0
            A[i, 7] = 0
            A[i, 8] = -x[h] * X[h]
            A[i, 9] = -x[h] * Y[h]
            A[i, 10] = -x[h] * Z[h]
            A[i, 11] = -x[h]
        else:
            A[i, 0] = 0
            A[i, 1] = 0
            A[i, 2] = 0
            A[i, 3] = 0
            A[i, 4] = X[h]
            A[i, 5] = Y[h]
            A[i, 6] = Z[h]
            A[i, 7] = 1
            A[i, 8] = -y[h] * X[h]
            A[i, 9] = -y[h] * Y[h]
            A[i, 10] = -y[h] * Z[h]
            A[i, 11] = -y[h]
            h = h + 1
    U, S, VT = np.linalg.svd(A)
    # U is the left singular vectors
    # S is the singular values (in descending order)
    # VT is the right singular vectors (transposed)

    x = VT[11, :]
    temp_m = x.reshape((3, 4))
    lamda = 1 / (np.sqrt((temp_m[2, 0]) ** 2 + (temp_m[2, 1]) ** 2 + (temp_m[2, 2]) ** 2))
    m = lamda * temp_m
    m1 = m[0, :3]
    m2 = m[1, :3]
    m3 = m[-1, :3]
    cx = np.dot(m1.T, m3)
    cy = np.dot(m2.T, m3)
    fx = np.sqrt(np.dot(m1.T, m1) - cx ** 2)
    fy = np.sqrt(np.dot(m2.T, m2) - cy ** 2)
    t_z = m[2, 3]
    t_x = (m[0, 3] - (cx * t_z))/fx
    t_y = (m[1, 3] - (cy * t_z))/fy
    T = np.array([t_x, t_y, t_z], dtype=float)
    R[2, 0] = m[2, 0]
    R[2, 1] = m[2, 1]
    R[2, 2] = m[2, 2]
    R[0, 0] = (m[0, 0] - (cx * R[2, 0]))/fx
    R[0, 1] = (m[0, 1] - (cx * R[2, 1]))/fx
    R[0, 2] = (m[0, 2] - (cx * R[2, 2]))/fx
    R[1, 0] = (m[1, 0] - (cy * R[2, 0]))/fy
    R[1, 1] = (m[1, 1] - (cy * R[2, 1]))/fy
    R[1, 2] = (m[1, 2] - (cy * R[2, 2]))/fy
    m_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    m_extrinsic = np.array([[R[0, 0], R[0, 1], R[0, 2], t_x], [R[1, 0], R[1, 1], R[1, 2], t_y], [R[2, 0], R[2, 1], R[2, 2], t_z]])
    m_intrinsic = m_intrinsic.reshape((3, 3))
    m_extrinsic = m_extrinsic.reshape((3, 4))
    projection = np.matmul(m_intrinsic, m_extrinsic)
    world_1 = np.array([0, 40, 40, 1]).reshape(-1, 1)
    image_1 = np.matmul(projection, world_1)
    final_image_1 = [[image_1[0, 0] / image_1[2, 0]], [image_1[1, 0] / image_1[2, 0]], image_1[2, 0] / image_1[2, 0]]
    print("Verified image coordinate of 1st corner: ", final_image_1)
    return R, T


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2







#---------------------------------------------------------------------------------------------------------------------