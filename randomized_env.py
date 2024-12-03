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

    def _load_model(self):
        super()._load_model()

        self.randomize_wall()
    
    def randomize_wall(self):
        wall_size = np.random.uniform(low=[0.1, 0, 0.1], high=[0.2, 0.02, 0.2])  # Width, depth, height

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

        self.randomize_wall()
        self.randomize_camera()

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


def main():
    randomize_arm = True # flag to randomize arm
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
    env.reset()

    # Render the scene from camera of choice, I picked robot0_eye_in_hand for now
    # Available "camera" names = ('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand')
    frame = env.sim.render(
        width=640, height=480, camera_name="birdview"
    )

    # Display the rendered frame
    plt.imshow(frame)
    plt.axis("off")
    plt.title("Randomized Camera View with Randomized Cube and Wall")
    plt.show()

if __name__ == "__main__":
    main()