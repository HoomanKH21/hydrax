import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.task_base import Task


class PegInHole(Task):
    """
    A task for inserting a peg into a hole.
    This class defines the cost functions for the peg-in-hole task.
    The dynamics are defined by the provided MuJoCo model.
    """

    def __init__(self, mj_model: mujoco.MjModel):
        """
        Initializes the PegInHole task.

        Args:
            mj_model: The MuJoCo model to use for simulation.
        """
        super().__init__(mj_model, trace_sites=["peg_tip"])
        
        # --- Cost function weights from probdat.yaml ---
        self.stage_c_rho = 10.0
        self.stage_c_xi = 10.0
        self.terminal_c_rho = 50000.0
        self.terminal_c_xi = 1000.0
        self.vel_c_v = 0.0
        self.vel_c_omega = 0.0

        # --- Goal state from probdat.yaml ---
        # qN: [0., 0., 0.1, 1., 0., 0., 0.]
        self.goal_pos = jnp.array([0.0, 0.0, 0.1])
        self.goal_quat = jnp.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z

        # Get body ID for the peg
        self.peg_body_id = mj_model.body("peg").id

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """
        The running cost ℓ(xₜ, uₜ).

        This cost penalizes:
        1. Control effort (velocity commands for the impedance controller).
        2. Velocity of the peg.
        """
        # Penalize control effort (vref in the C++ code)
        # control is equivalent to vref in the raptico code
        linear_vel_cost = jnp.sum(jnp.square(control[:3]))
        angular_vel_cost = jnp.sum(jnp.square(control[3:]))
        
        cost = self.stage_c_rho * linear_vel_cost + self.stage_c_xi * angular_vel_cost

        # Penalize peg velocity
        peg_vel = state.qvel[:6] # 3 linear, 3 angular
        cost += self.vel_c_v * jnp.sum(jnp.square(peg_vel[:3]))
        cost += self.vel_c_omega * jnp.sum(jnp.square(peg_vel[3:]))
        
        return cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """
        The terminal cost ϕ(x_T).

        This cost penalizes the final distance and orientation error
        of the peg relative to the goal.
        """
        # Get the peg's final position and orientation
        peg_pos = state.xpos[self.peg_body_id]
        peg_quat = state.xquat[self.peg_body_id] # w, x, y, z

        # Positional error
        pos_error = jnp.sum(jnp.square(peg_pos - self.goal_pos))

        # Orientation error (quaternion dot product)
        # Ensure the shortest path rotation is taken
        quat_dot = jnp.abs(jnp.dot(peg_quat, self.goal_quat))
        quat_error = 1.0 - quat_dot**2
        
        terminal_cost = self.terminal_c_rho * pos_error + self.terminal_c_xi * quat_error
        
        return terminal_cost
