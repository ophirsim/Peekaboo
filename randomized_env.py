import robosuite as suite
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import array_to_string, new_joint
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET  # For debugging


class CustomLiftWithWall(Lift):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()

        for body in self.model.mujoco_arena.worldbody.findall("body"):
            print("Body name:", body.get("name"))

        wall_size = np.random.uniform(low=[0.1, 0, 0.1], high=[0.25, 0.02, 0.25])  # Width, depth, height

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
        wall_offset = np.random.uniform(low=[-0.25, -0.25, 0.0], high=[0.25, 0.25, 0.0])
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

    def reset(self):
        super().reset()
        '''
        I DONT KNOW IF ANY OF THIS DOES ANYTHING SO I COMMENTED IT OUT
        ''' 
        # Set the wall's position after the environment is reset
        #wall_body = self.sim.model.body_name2id("wall_main")
        #self.sim.model.body_pos[wall_body] = [0.0, -0.3, 1.25]  # Adjusted position
        #self.sim.forward()
    
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

def randomize_camera(env, camera_name="robot0_eye_in_hand"):
    """
    Randomizes only the camera orientation (angle) while keeping the camera position fixed on the end-effector.
    """
    cam_id = env.sim.model.camera_name2id(camera_name)
    # Get the initial camera orientation
    initial_quat = env.sim.model.cam_quat[cam_id]
    print(f"Initial camera orientation (quat): {initial_quat}")
    # Randomize the camera angle (rotation around the X, Y, and Z axes)
    random_quat = np.random.uniform(low=-np.pi, high=np.pi, size=4)  # Random quaternion
    random_quat = random_quat / np.linalg.norm(random_quat)  # Normalize quaternion to make it valid
    env.sim.model.cam_quat[cam_id] = random_quat
    # Get the new camera orientation
    new_quat = env.sim.model.cam_quat[cam_id]
    print(f"New camera orientation (quat): {new_quat}")

def randomize_cube(env, cube_name="cube_main", cube_geom_name="cube_g0", table_geom_name="table_collision"):
    """
    Randomizes the cube position on the table surface.
    Ensures the cube remains on the table.
    """
    # Get table dimensions from the Mujoco model
    table_geom_id = env.sim.model.geom_name2id(table_geom_name)
    table_pos = env.sim.model.geom_pos[table_geom_id]
    table_size = env.sim.model.geom_size[table_geom_id]

    # Get cube dimensions
    cube_geom_id = env.sim.model.geom_name2id(cube_geom_name)
    cube_size = env.sim.model.geom_size[cube_geom_id]

    # Compute valid ranges for cube's x and y positions
    x_range = [
        table_pos[0] - table_size[0] + cube_size[0],
        table_pos[0] + table_size[0] - cube_size[0],
    ]
    y_range = [
        table_pos[1] - table_size[1] + cube_size[1],
        table_pos[1] + table_size[1] - cube_size[1],
    ]

    # Set cube's z position to ensure it rests on top of the table
    z_position = table_pos[2] + table_size[2] + cube_size[2]

    # Randomize x and y positions within valid ranges
    x_position = np.random.uniform(*x_range)
    y_position = np.random.uniform(*y_range)

    # Set new position for the cube
    cube_body_id = env.sim.model.body_name2id(cube_name)
    new_pos = [x_position, y_position, z_position]
    env.sim.model.body_pos[cube_body_id] = new_pos
    env.sim.forward()

    # Debugging: Print new cube position
    print(table_pos, table_size, cube_size)
    print(f"New cube position: {new_pos}")


def main():
    # Create the custom Lift environment with a wall
    env = CustomLiftWithWall(
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=True, #this shit always gives me an error when it's False
        use_camera_obs=False,
        control_freq=20,
    )

    # Reset the environment
    env.reset()
    # Randomize camera position
    randomize_camera(env)
    # Randomize the cube position
    randomize_cube_like_wall(env)

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