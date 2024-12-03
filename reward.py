import numpy as np

# TODO: unpack necessary coordinates from environment
def preprocess(obs):
    target_vertices = np.random.rand(8, 3) # 8 vertices each with 3 dimensions (x, y, z)
    wall_vertices = np.random.rand(6, 4, 3) # 6 planes, each containing 4 vertices each with 3 dimensions (x, y, z)
    camera_position = np.random.rand(3) # (x, y, z) of camera
    camera_vec = np.random.rand(3) # (delta_x, delta_y, delta_z) describing the vector along which the camera points
    camera_bloom = np.random.rand(2) # defines rectangular pyramid in degrees (alpha, beta): (vertical, horizontal)
    return target_vertices, wall_vertices, camera_position, camera_vec, camera_bloom

def line_segment_intersects_truncated_plane(point1, point2, truncated_plane):
    """
    Boolean function to determine whether a line segment drawn between two points intersects a truncated rectangular plane defined by four points

    Arguments:
        point1: numpy array of shape (3,) corresponding to (x1, y1, z1)
        point2: numpy array of shape (3,) corresponding to (x2, y2, z2)
        truncated_plane: numpy array of shape (4, 3) corresponding to 4 points of a rectangular plane, each with 3 dimensions

    Returns: True if line segment intersects the truncated rectangular plane, False otherwise
    """

    # calculate plane equation
    A, B, C, D = truncated_plane
    AB = A - B
    AC = A - C
    plane = np.cross(AB, AC)
    k = -1 * np.sum(plane * D)

    # calculate the line equation
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    line = np.array([[x1, y1, z1],
                     [x2-x1, y2-y1, z2-z1]])
    
    intersection = np.sum(line * plane, axis=1)
    intersection[0] += k

    # line is parallel to plane, the plane does not obstruct the view
    if intersection[1] == 0:
        return False

    t = -1 * intersection[0] / intersection[1]

    # intersection occurs on line beyond line segment's endpoints, no occlusion
    if t < 0 or t > 1:
        return False

    x_intersect = x1 + t * (x2 - x1)
    y_intersect = y1 + t * (y2 - y1)
    z_intersect = z1 + t * (z2 - z1)
    intersect = np.array([x_intersect, y_intersect, z_intersect])

    # if the intersection points lies within the truncated plane, return True, else return False
    if (np.dot(AB, intersect) >= min(np.dot(AB, A), np.dot(AB, B)) and np.dot(AB, intersect) <= max(np.dot(AB, A), np.dot(AB, B)) and
        np.dot(AC, intersect) >= min(np.dot(AC, A), np.dot(AC, C)) and np.dot(AC, intersect) <= max(np.dot(AC, A), np.dot(AC, C))):
        return True
    
    return False


def target_visible_in_conical_bloom(target_pos, camera_pos, camera_vec, camera_bloom):
    """
    Boolean function to determine whether a particular point falls within the conical bloom of the camera, given the camera's position, angle, and bloom

    Arguments:
        target_pos: numpy array of shape (3,) corresponding to (x1, y1, z1) of the target point
        camera_pos: numpy array of shape (3,) corresponding to (x2, y2, z2) of the camera
        camera_vec: numpy array of shape (3,) corresponding to (delta_x, delta_y, delta_z) of the direction vector of the camera
        camera_bloom: numpy array of shape (2,) corresponding to (alpha, beta) the bloom in each dimension of the camera frame

    Return: True if the target point is within the conical bloom of the camera, False otherwise

    """
    line_vec = target_pos - camera_pos
    line_vec = line_vec / np.linalg.norm(line_vec)

    camera_vec = camera_vec / np.linalg.norm(camera_vec)
    angle_diff = np.rad2deg(np.arccos(np.clip(np.dot(camera_vec, line_vec), -1.0, 1.0))) # in degrees

    if angle_diff <= np.min(camera_bloom):
        return True
    
    return False


def reward(obs):
    target_vertices, wall_vertices, camera_position, camera_vec, camera_bloom = preprocess(obs)

    # if any corner of the target cube is outside of frame, return 0
    for target_vertex in target_vertices:
        if not target_visible_in_conical_bloom(target_vertex, camera_position, camera_vec, camera_bloom):
            return 0

    # if any corner of the target cube is blocked by the wall, return 0
    for target_vertex in target_vertices:
        for wall_plane in wall_vertices:
            if line_segment_intersects_truncated_plane(target_vertex, camera_position, wall_plane):
                return 0
        
    # cube is entirely within bloom and is not obstructed
    return 1



