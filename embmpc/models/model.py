import numpy as np
from embmpc.utils.rk6 import odeintRK6, odeintRK6_batch, odeintRK4_batch, odeintEuler_batch
import matplotlib.pyplot as plt


class Model:

	def __init__(self):
		pass

	def _integrate(self, x_t, u_t, t_start, t_end):
		fun=self._diffequation
		odesol = odeintRK6(
			fun=fun, 
			y0=x_t, 
			t=[t_start, t_end], 
			args=(u_t,))
		return odesol[-1,:]

	def _integrate_batch(self, x_t_batch, u_t_batch, t_start, t_end):
		fun = self._diffequation_batch
		odesol = odeintRK4_batch(
			fun=fun,
			y0_batch=x_t_batch,
			t=[t_start, t_end],
			args_batch=(u_t_batch,))
		return odesol[-1]

	def plot_results(self, t, x, dxdt, u, friction_circle=False):

		plt.figure()
		plt.plot(x[0,:], x[1,:])
		plt.xlabel('x [m]')
		plt.ylabel('y [m]')
		plt.axis('equal')
		plt.grid(True)

		plt.figure()
		plt.plot(t, x[0,:], label='x')
		plt.plot(t, x[1,:], label='y')
		plt.xlabel('time [s]')
		plt.ylabel('position [m]')
		plt.grid(True)
		plt.legend()


		if dxdt is not None:
			plt.figure()
			plt.plot(t, dxdt[0,:], label='speed x')
			plt.plot(t, dxdt[1,:], label='speed y')
			if not friction_circle:
				plt.plot(t, dxdt[2,:], label='yaw rate')
			plt.plot(t, np.sqrt(dxdt[0,:]**2+dxdt[1,:]**2), '--', label='speed abs')
			plt.xlabel('time [s]')
			plt.ylabel('velocity [m/s]')
			plt.grid(True)
			plt.legend()

			plt.figure()
			plt.plot(dxdt[0,:], dxdt[1,:])
			plt.xlabel('speed x [m/s]')
			plt.ylabel('speed y [m/s]')
			plt.axis('equal')
			plt.grid(True)


		plt.figure()
		if friction_circle:
			plt.plot(t, np.arctan2(dxdt[1,:],dxdt[0,:]))
		else:
			plt.plot(t, x[2,:])
		plt.ylabel('yaw (heading) [rad]')
		plt.xlabel('time [s]')
		plt.grid(True)


		plt.figure()
		if friction_circle:
			plt.plot(t[1:], u[0,:], label='force x')
			plt.plot(t[1:], u[1,:], label='force y')
		else:
			plt.plot(t[1:], u[0,:], label='acceleration')
			plt.plot(t[1:], u[1,:], label='steering')
		plt.ylabel('inputs')
		plt.grid(True)
		plt.legend()
		plt.show()