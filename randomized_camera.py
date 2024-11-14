import robosuite as suite
from robosuite.models.robots import Panda
from robosuite.environments.manipulation.lift import Lift
import numpy as np
import matplotlib.pyplot as plt

# Helper function to randomize camera position
def randomize_camera(env, camera_name="robot0_eye_in_hand"):
    """
    Randomizes the camera position and orientation on the robotic arm.
    """
    cam_id = env.sim.model.camera_name2id(camera_name)

    # Generate random position near the end-effector
    random_position = np.random.uniform(low=[-0.5, -0.5, -0.5], high=[0.5, 0.5, 0.5])
    ee_pos = env.sim.data.get_site_xpos("gripper0_ee_x")
    env.sim.model.cam_pos[cam_id] = ee_pos + random_position

def main():
    # Create the custom environment
    env = suite.make(
        env_name="Lift",
        robots="Panda", 
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    # Reset the environment
    obs = env.reset()

    # Randomize the camera position
    randomize_camera(env)

    # Render the initial scene
    frame = env.sim.render(
        width=640, height=480, camera_name="robot0_eye_in_hand"
    )

    # Display the rendered frame
    plt.imshow(frame)
    plt.axis("off")
    plt.title("Initial Scene from Arm Camera")
    plt.show()


if __name__ == "__main__":
    main()
