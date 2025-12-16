import jax.numpy as jnp
import mujoco

from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.simulation.asynchronous import run_interactive
from hydrax.tasks.peg_in_hole import PegInHole

def main():
    # Load the model and data
    model_path = "hydrax/models/peg_in_hole/model.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Set initial position
    mj_data.qpos[0:3] = jnp.array([0.1, 0.0, 0.35])
    mj_data.qpos[3:7] = jnp.array([1.0, 0.0, 0.0, 0.0])  # Quaternion

    # Create the task
    task = PegInHole(mj_model=mj_model)

    # Create the controller
    # Note: PredictiveSampling is a simplified version of MPPI where only the best
    # sample is used.
    controller = PredictiveSampling(
        task=task,
        num_samples=1024,  # K from probdat.yaml
        num_knots=20,  # Ncnt from probdat.yaml
        plan_horizon=4.0,  # T from probdat.yaml
        noise_level=0.25,  # Noise level for sampling
        #control_bounds=(task.u_min, task.u_max),
    )

    # Run the interactive simulation asynchronously
    print("Starting asynchronous simulation...")
    run_interactive(controller, mj_model, mj_data)

# Safe import guard to prevent multiprocessing issues
if __name__ == '__main__':
    main()
