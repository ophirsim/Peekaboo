import robosuite as suite
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import array_to_string, new_joint
from robosuite.utils.placement_samplers import UniformRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET  # For debugging


class CustomLiftWithWall(Lift):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env_timesteps = 0
        self.episode_reward = 0.0

    def _load_model(self):
        super()._load_model()

        self.randomize_wall()
    
    def randomize_wall(self):
        wall_size = np.random.uniform(low=[0.1, 0, 0.1], high=[0.2, 0.02, 0.2])  # Width, depth, height
        self.wall_size = wall_size

        # Define a wall
        self.wall = BoxObject(
            name="wall",
            size=wall_size,  # Adjusted size to be more visible
            rgba=[0.2, 0.2, 0.7, 1.0],
        )

        # get table
        table_body_name = "table"  # Default table name in Mujoco's Lift environment
        table_xml = self.model.mujoco_arena.worldbody.find("body[@name='{}']".format(table_body_name))
        if table_xml is None:
            raise ValueError("Table body not found in the Mujoco world.")
        
        table_pos = table_xml.get("pos")
        table_pos = np.array([float(x) for x in table_pos.split()])  # Convert string to array

        # Get the Mujoco XML for the wall and place it in the worldbody but on the table
        wall_offset = np.random.uniform(low=[-0.2, -0.2, 0.0], high=[0.2, 0.2, 0.0])
        wall_pos = table_pos + wall_offset  # Centered on table and offset in height
        wall_pos[2] += wall_size[2] / 2
        wall_xml = self.wall.get_obj()
        wall_xml.set("pos", array_to_string(wall_pos))

        # randomize table orientation, but still perpecndicault to table
        # Generate a random yaw quaternion
        wall_quat = random_yaw_quaternion()
        wall_xml.set("quat", array_to_string(wall_quat))

        # Append the wall object to the worldbody
        self.model.mujoco_arena.worldbody.append(wall_xml)

        # Set the wall's position
        self.model.merge_objects([self.wall])

        # Pretty-print the worldbody XML after appending the wall
        # The wall is there as wall_main
        print("Arena worldbody after appending the wall:")
        tree_str = ET.tostring(self.model.mujoco_arena.worldbody, encoding='unicode', method='xml')
        print(tree_str)

    def randomize_camera(self, camera_name="robot0_eye_in_hand"):
        """
        Randomizes the camera orientation (quaternion) while keeping the camera position fixed.
        """
        return # We don't actually need this, but now arm randomization is the only thing controling starting position of camera
        cam_id = self.sim.model.camera_name2id(camera_name)
        
        # Generate a random quaternion for camera orientation
        random_quat = self.generate_random_unit_quaternion()
        self.sim.model.cam_quat[cam_id] = random_quat
        self.sim.forward()

    def generate_random_unit_quaternion(self):
        """
        Generates a random unit quaternion for 3D rotation.
        """
        u1, u2, u3 = np.random.uniform(0, 1, 3)
        qx = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        qy = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        qz = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        qw = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        return np.array([qw, qx, qy, qz])


    def reset(self):
        super().reset()

        self.env_timesteps = 0
        self.episode_reward = 0.0
        
        #print("reset reset reset reset reset reset reset")

        #self.randomize_wall()
        self.randomize_camera()

        observations = (
            self.viewer._get_observations(force_update=True)
            if self.viewer_get_obs
            else self._get_observations(force_update=True)
        )

        # Return new observations
        return observations
    
    def step(self, action):
        obs, reward, done, info = super().step(action)

        self.env_timesteps += 1
        if self.env_timesteps >= 500:
            done = True

        self.episode_reward += reward
        
        #print(reward)
        #print(self.episode_reward)
        #print(done)
        #print(info)

        #if done:
            #print("done done done done done done done done")

        return obs, reward, done, info
    
    def unpack_env(self):
        # Grab target cube position, size, and orientation
        # then convert this information into 8 vertices
        cube_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(self.cube.root_body)] # (x, y, z) looking from arm pov (forward-back, left-right, up-down)
        cube_quat = self.sim.data.body_xquat[self.sim.model.body_name2id(self.cube.root_body)]
        bb = self.cube.get_bounding_box_half_size()

        target_vertices = np.zeros((8, 3)) + cube_pos
        i = 0
        for x in [-bb[0], bb[0]]:
            for y in [-bb[1], bb[1]]:
                for z in [-bb[2], bb[2]]:
                    target_vertices[i] += apply_quaternion_to_vector(np.array([x, y, z]), cube_quat)
                    i += 1


        # Grab wall position, size, and orientation
        # then convert this information into 8 vertices
        # lastly, convert the 8 vertices into 6 planes
        wall = self.model.mujoco_arena.worldbody.find(f"body[@name='wall_main']")
        wall_pos = np.array([float(x) for x in wall.get('pos').split()])
        wall_quat = np.array([float(x) for x in wall.get('quat').split()])
        wall_size = self.wall_size

        wall_vertices = np.zeros((8, 3)) + wall_pos
        i = 0
        for x in [-wall_size[0], wall_size[0]]:
            for y in [-wall_size[1], wall_size[1]]:
                for z in [-wall_size[2], wall_size[2]]:
                    wall_vertices[i] += apply_quaternion_to_vector(np.array([x, y, z]), wall_quat)
                    i += 1

        v1, v2, v3, v4, v5, v6, v7, v8 = wall_vertices
        wall_planes = np.array([[v1, v2, v3, v4],
                                [v5, v6, v7, v8],
                                [v1, v2, v5, v6],
                                [v3, v4, v7, v8],
                                [v1, v3, v5, v7],
                                [v2, v4, v6, v8]])
        
        # grab the camera position
        cam_id = self.sim.model.camera_name2id("robot0_eye_in_hand")
        camera_position = self.sim.data.cam_xpos[cam_id]


        # grab the camera quaternion
        camera_quat = self.sim.model.cam_quat[cam_id]


        # grap the end effector rotation matrix, and apply it to the rotation frame of reference
        eef_site_name = "gripper0_right_grip_site"
        eef_site_id = self.sim.model.site_name2id(eef_site_name)
        rotation_matrix = self.sim.data.site_xmat[eef_site_id].reshape(3, 3)
        camera_vector = rotation_matrix @ np.array([0, 0, 1])


        # Grab the field of view of the camera in the y-direction
        # assume the x-direction field of view is the same
        camera_bloom = np.array([self.sim.model.cam_fovy[cam_id]/2, self.sim.model.cam_fovy[cam_id]/2]) # defines rectangular pyramid in degrees (alpha, beta): (vertical, horizontal)

        return target_vertices, wall_planes, camera_position, camera_vector, camera_bloom

    def reward(self, action=None):
        target_vertices, wall_planes, camera_position, camera_vector, camera_bloom = self.unpack_env()

        # if any corner of the target cube is outside of frame, return 0
        for target_vertex in target_vertices:
            if not target_visible_in_conical_bloom(target_vertex, camera_position, camera_vector, camera_bloom):
                return 0.0

        # if any corner of the target cube is blocked by the wall, return 0
        for target_vertex in target_vertices:
            for wall_plane in wall_planes:
                if line_segment_intersects_truncated_plane(target_vertex, camera_position, wall_plane):
                    return 0.0
            
        # cube is entirely within bloom and is not obstructed
        return 1.0

def random_yaw_quaternion():
    """
    Generates a quaternion for a random yaw rotation (rotation about the Z-axis only).
    This ensures the wall remains perpendicular to the table.
    """
    yaw = np.random.uniform(0, 2 * np.pi)  # Random angle in radians
    qx = 0.0
    qy = 0.0
    qz = np.sin(yaw / 2)
    qw = np.cos(yaw / 2)
    return [qw, qx, qy, qz]


def apply_quaternion_to_vector(vector, quaternion):
    def hamilton(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def inv_quaternion(quaternion):
        w, x, y, z = quaternion
        return np.array([w, -x, -y, -z])
    
    psuedo_vec = np.array([0, vector[-3], vector[-2], vector[-1]])
    return hamilton(hamilton(quaternion, psuedo_vec), inv_quaternion(quaternion))[1:]


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


def main():
    randomize_arm = False # flag to randomize arm
    placement_initializer = UniformRandomSampler(
        name="ObjectSampler",
        x_range=[-0.3, 0.3],
        y_range=[-0.3, 0.3],
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=np.array((0, 0, 0.8)),
        z_offset=0.01,
    )

    env = CustomLiftWithWall(
        robots="Panda",
        initialization_noise={'magnitude': 0.3 if randomize_arm else 0.0, 'type': 'uniform'},
        has_renderer=True,
        has_offscreen_renderer=True, #this shit always gives me an error when it's False
        use_camera_obs=False,
        use_object_obs=True,
        placement_initializer=placement_initializer,
        control_freq=20,
    )

    # Reset the environment
    for _ in range(10):
        env.reset()

        print(env.reward())

        # Render the scene from camera of choice, I picked robot0_eye_in_hand for now
        # Available "camera" names = ('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand')
        frame1 = env.sim.render(
            width=256, height=256, camera_name="robot0_eye_in_hand"
        )

        frame2 = env.sim.render(
            width=256, height=256, camera_name="agentview"
        )

        frame3 = env.sim.render(
            width=256, height=256, camera_name="frontview"
        )

        frame4 = env.sim.render(
            width=256, height=256, camera_name="birdview"
        )

        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(frame1, origin='lower')
        axarr[0,0].axis("off")

        axarr[0,1].imshow(frame2, origin='lower')
        axarr[0,1].axis("off")

        axarr[1,0].imshow(frame3, origin='lower')
        axarr[1,0].axis("off")

        axarr[1,1].imshow(frame4, origin='lower')
        axarr[1,1].axis("off")

        plt.show()

if __name__ == "__main__":
    main()