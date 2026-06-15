import numpy as np
import matplotlib.pyplot as plt
from embmpc.utils import Projection, Spline, Spline2D

class Track:

	def __init__(self):
		self._calc_center_line()
		self._calc_track_length()
		self._calc_theta_track()

	def _calc_center_line(self):
		self.center_line = np.concatenate([
			self.x_center.reshape(1,-1), 
			self.y_center.reshape(1,-1)
			])

	def _calc_track_length(self):
		center = self.center_line

		center = np.concatenate([center, center[:,0].reshape(-1,1)], axis=1) 
		diff = np.diff(center)
		self.track_length = np.sum(np.linalg.norm(diff, 2, axis=0))

	def _calc_raceline_length(self, raceline):

		raceline = np.concatenate([raceline, raceline[:,0].reshape(-1,1)], axis=1) 
		diff = np.diff(raceline)
		return np.sum(np.linalg.norm(diff, 2, axis=0))

	def _calc_theta_track(self):
		diff = np.diff(self.center_line)
		theta_track = np.cumsum(np.linalg.norm(diff, 2, axis=0))
		self.theta_track = np.concatenate([np.array([0]), theta_track])

	def _load_raceline(self, wx, wy, n_samples, v=None, t=None, vs=None, mus=None):







		self.spline = Spline2D(wx, wy)

		x, y = wx, wy
		theta = self.spline.s

		self.x_raceline = np.array(x)
		self.y_raceline = np.array(y)
		self.raceline = np.array([x, y])

		if vs is not None :
			self.v_raceline = vs
			self.mus = mus
			self.t_raceline = t
			self.spline_v = []
			for vi in vs :


				self.spline_v.append(Spline(theta, vi))
		elif v is not None:
			self.v_raceline = v
			self.t_raceline = t
			self.spline_v = Spline(theta, v)
		
		
	def _fit_cubic_splines(self, wx, wy, n_samples):
		sp = Spline2D(wx, wy)
		self.spline = sp


		s = np.linspace(0, sp.s[-1]-0.001, n_samples)
		x, y = [], []
		for i_s in s:
			ix, iy = sp.calc_position(i_s)
			x.append(ix)
			y.append(iy)
		return x, y, s

	def _param2xy(self, theta):
		theta_track = self.theta_track
		idt = 0
		while idt<theta_track.shape[0]-1 and theta_track[idt]<=theta:
			idt+=1
		deltatheta = (theta-theta_track[idt-1])/(theta_track[idt]-theta_track[idt-1])
		x = self.x_center[idt-1] + deltatheta*(self.x_center[idt]-self.x_center[idt-1])
		y = self.y_center[idt-1] + deltatheta*(self.y_center[idt]-self.y_center[idt-1])
		return x, y

	def _xy2param(self, x, y):
		center_line = self.center_line
		theta_track = self.theta_track

		optxy, optidx = self.project(x, y, center_line)
		distxy = np.linalg.norm(optxy-center_line[:,optidx],2)
		dist = np.linalg.norm(center_line[:,optidx+1]-center_line[:,optidx],2)
		deltaxy = distxy/dist
		if optidx==-1:
			theta = theta_track[optidx] + deltaxy*(self.track_length-theta_track[optidx])
		else:
			theta = theta_track[optidx] + deltaxy*(theta_track[optidx+1]-theta_track[optidx])
		theta = theta % self.track_length
		return theta

	def project(self, x, y, raceline):
		point = [(x, y)]
		n_waypoints = raceline.shape[1]

		proj = np.empty([2,n_waypoints])
		dist = np.empty([n_waypoints])
		for idl in range(-1, n_waypoints-1):
			line = [raceline[:,idl], raceline[:,idl+1]]
			proj[:,idl], dist[idl] = Projection(point, line)
		optidx = np.argmin(dist)
		if optidx == n_waypoints-1:
			optidx = -1
		optxy = proj[:,optidx]
		return optxy, optidx

	def project_fast(self, x, y, raceline):
		point = [(x, y)]
		n_waypoints = raceline.shape[1]

		proj = np.empty([2,n_waypoints-1])
		dist = np.empty([n_waypoints-1])
		for idl in range(n_waypoints-1):
			line = [raceline[:,idl], raceline[:,idl+1]]
			proj[:,idl], dist[idl] = Projection(point, line)
		optidx = np.argmin(dist)
		optxy = proj[:,optidx]
		return optxy, optidx

	def _plot(self, color='k', grid=True, figsize=(6.4, 4.8)):
		fig = plt.figure(figsize=figsize)
		plt.grid(grid)

		plt.plot(self.x_outer, self.y_outer, color, lw=1.6)
		plt.plot(self.x_inner, self.y_inner, color, lw=1.6)

		plt.axis('equal')
		return fig

	def plot_raceline(self):
		fig = self._plot()
		plt.plot(self.x_raceline, self.y_raceline, 'b', lw=1)
		plt.show()

	def param_to_xy(self, theta, **kwargs):
		raise NotImplementedError

	def xy_to_param(self, x, y):
		raise NotImplementedError

	def find_closest_point(x, y, x_refs, y_refs):


		distances = np.sqrt((x_refs - x) ** 2 + (y_refs - y) ** 2)


		idx = np.argmin(distances)

		return idx, x_refs[idx], y_refs[idx], distances[idx]