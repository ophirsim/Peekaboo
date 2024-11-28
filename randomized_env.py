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

        # Define a wall
        self.wall = BoxObject(
            name="wall",
            size=[0.5, 0.01, 0.5],  # Adjusted size to be more visible
            rgba=[0.2, 0.2, 0.7, 1.0],
        )

        # Get the Mujoco XML for the wall and place it in the worldbody
        wall_xml = self.wall.get_obj()
        wall_xml.set("pos", array_to_string([0.0, -0.3, 1.25]))  # Adjust wall position closer to robot, randomizable?

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

def randomize_cube(env, cube_name="cube_main"):
    """
    Randomizes the cube position on the table with significantly larger bounds.
    """
    cube_id = env.sim.model.body_name2id(cube_name)
    # Get the initial cube position
    initial_pos = env.sim.model.body_pos[cube_id]
    print(f"Initial cube position: {initial_pos}")
    # Ideally, randomize based on table coordinates but couldn't figure that out
    random_pos = np.random.uniform(low=[-3, -3, 0.01], 
                                  high=[3, 3, 0.1])
    
    # Update the cube's position
    env.sim.model.body_pos[cube_id] = random_pos
    # Get the new cube position
    new_pos = env.sim.model.body_pos[cube_id]
    print(f"New cube position: {new_pos}")
    env.sim.forward()

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
    randomize_cube(env)

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