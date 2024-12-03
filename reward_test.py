import numpy as np
from reward import *

def check(expected, received):
    if expected == received:
        print("Good!")
    else:
        print(f"Expected: {expected}. Received: {received}")




point1 = np.array([0, 0, 0])
point2 = np.array([0, 0, 2])
plane = np.array([[-1, -1, 3],
                 [-1, 1, 3],
                 [1, -1, 3],
                 [1, 1, 3]])
check(False, line_segment_intersects_truncated_plane(point1, point2, plane))

point1 = np.array([0, 0, 0])
point2 = np.array([0, 0, 2])
plane = np.array([[-1, -1, 1],
                 [-1, 1, 1],
                 [1, -1, 1],
                 [1, 1, 1]])
check(True, line_segment_intersects_truncated_plane(point1, point2, plane))

point1 = np.array([1, 1, 1])
point2 = np.array([3, 3, 3])
plane = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [4, 4, 0]])
check(False, line_segment_intersects_truncated_plane(point1, point2, plane))

point1 = np.array([5, 5, 5])
point2 = np.array([-1, -1, -1])
plane = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [4, 4, 0]])
check(True, line_segment_intersects_truncated_plane(point1, point2, plane))

point1 = np.array([0, 0, 1])
point2 = np.array([4, 4, 1])
plane = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [4, 4, 0]])
check(False, line_segment_intersects_truncated_plane(point1, point2, plane))

point1 = np.array([5, 5, -1])
point2 = np.array([5, 5, 1])
plane = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [4, 4, 0]])
check(False, line_segment_intersects_truncated_plane(point1, point2, plane))

point1 = np.array([5, 5, 0])
point2 = np.array([6, 6, 0])
plane = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [4, 4, 0]])
check(False, line_segment_intersects_truncated_plane(point1, point2, plane))

point1 = np.array([0, 0, -1])
point2 = np.array([0, 0, 1])
plane = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [4, 4, 0]])
check(True, line_segment_intersects_truncated_plane(point1, point2, plane))

point1 = np.array([1, 1, 1])
point2 = np.array([4, 2, -1])
plane = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [4, 4, 0]])
check(True, line_segment_intersects_truncated_plane(point1, point2, plane))

point1 = np.array([1, 2, 1])
point2 = np.array([-4, 5, -1])
plane = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [4, 4, 0]])
check(False, line_segment_intersects_truncated_plane(point1, point2, plane))

point1 = np.array([1, 2, 1])
point2 = np.array([0, 5, -1])
plane = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [4, 4, 0]])
check(True, line_segment_intersects_truncated_plane(point1, point2, plane))

point1 = np.array([1, 2, 1])
point2 = np.array([1, 2, 2])
plane = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [4, 4, 0]])
check(False, line_segment_intersects_truncated_plane(point1, point2, plane))

point1 = np.array([1, 2, 1])
point2 = np.array([2, 2, 1])
plane = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [4, 4, 0]])
check(False, line_segment_intersects_truncated_plane(point1, point2, plane))

point1 = np.array([3.5, 3.5, 1])
point2 = np.array([3.5, 3.5, -1])
plane = np.array([[0, 2, 0], [2, 4, 0], [2, 0, 0], [4, 2, 0]])
check(False, line_segment_intersects_truncated_plane(point1, point2, plane))




target = np.array([0, 0, 2])
camera = np.array([0, 0, 0])
angle = np.array([0, 0, 1])
bloom = np.array([30, 60])
check(True, target_visible_in_conical_bloom(target, camera, angle, bloom))

target = np.array([0, 0, -2])
camera = np.array([0, 0, 0])
angle = np.array([0, 0, 1])
bloom = np.array([30, 60])
check(False, target_visible_in_conical_bloom(target, camera, angle, bloom))

target = np.array([1, 1, 0])
camera = np.array([0, 0, 0])
angle = np.array([1, 0, 0])
bloom = np.array([46, 60])
check(True, target_visible_in_conical_bloom(target, camera, angle, bloom))

target = np.array([1, 1.5, 0])
camera = np.array([0, 0, 0])
angle = np.array([0, 1, 0])
bloom = np.array([30, 30])
check(False, target_visible_in_conical_bloom(target, camera, angle, bloom))

target = np.array([0, 0, -1])
camera = np.array([0, 0, 0])
angle = np.array([0, 0, 1])
bloom = np.array([90, 90])
check(False, target_visible_in_conical_bloom(target, camera, angle, bloom))

target = np.array([0, 0, -1])
camera = np.array([0, 0, 0])
angle = np.array([0, 0, 1])
bloom = np.array([180, 180])
check(True, target_visible_in_conical_bloom(target, camera, angle, bloom))


