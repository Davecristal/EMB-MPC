

import numpy as np
from embmpc.utils import Spline
from embmpc.utils import Spline2D


def ConstantSpeed(x0, v0, track, N, Ts, projidx, scale=1., curr_mu = 1.):

	raceline = track.raceline
	xy, idx = track.project_fast(x=x0[0], y=x0[1], raceline=raceline[:,projidx:projidx+10])
	projidx = idx+projidx


	start = track.raceline[:,:projidx+2]

	xref = np.zeros([2,N+1])
	xref[:2,0] = x0


	dist0 = np.sum(np.linalg.norm(np.diff(start), 2, axis=0))
	dist = dist0
	v = max(v0,.01)
	vr = 0.
	for idh in range(1,N+1):
		dist += scale*v*Ts
		dist = dist % track.spline.s[-1]

		xref[:2,idh] = track.spline.calc_position(dist)



		if curr_mu < track.mus[0] :
			v = track.spline_v[0].calc(dist)
			i=0
		elif curr_mu > track.mus[-1] :
			v = track.spline_v[-1].calc(dist)
			i=len(track.mus)-1
		else :
			i = 0
			for i in range(len(track.mus)) :
				if track.mus[i] >= curr_mu :
					break

			vb = track.spline_v[i-1].calc(dist)
			va = track.spline_v[i].calc(dist)
			v = vb*(track.mus[i]-curr_mu)/(track.mus[i]-track.mus[i-1]) + va*(curr_mu-track.mus[i-1])/(track.mus[i]-track.mus[i-1])
		if idh==1 :
			vr = v*scale


	return xref, projidx, vr