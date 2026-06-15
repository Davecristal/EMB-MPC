import argparse
import sys

if any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Idealized informed Oracle reference')
    parser.parse_args()

fval_history = []
violation_total = []
from embmpc.mpc.constraints import Boundary
import copy


def find_closest_point(x, y, raceline):

	x_refs = raceline[0]
	y_refs = raceline[1]


	distances = np.sqrt((x_refs - x) ** 2 + (y_refs - y) ** 2)


	idx = np.argmin(distances)

	return idx, x_refs[idx], y_refs[idx], distances[idx]


def update_friction(Df,Dr,curr_time,style='sudden') :
    if style == 'const_decay' :
        if curr_time > 14.3:

            Df -= Df/2600.
            Dr -= Dr/2600.
    elif style == 'sudden' :

        if curr_time > 14.3 and curr_time < 14.5:



            Df -= Df/22.
            Dr -= Dr/22.
    elif style == 'no_change' :
        return Df, Dr
    return Df, Dr

import time as tm
import numpy as np
import casadi
import _pickle as pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from embmpc.params import ORCA
from embmpc.models import Dynamic
from embmpc.tracks import ETHZ, ETHZMobil
from embmpc.mpc.planner import ConstantSpeed
from embmpc.mpc.nmpc import setupNLP
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection


import os
import subprocess
import copy

import matplotlib
matplotlib.use('Agg')


import matplotlib.pylab as pylab

import matplotlib.animation as animation

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
plt.rcParams['text.usetex'] = True





SAVE_RESULTS = False
TRACK_CONS = False




SIM_TIME = 36
SAMPLING_TIME = 0.02
HORIZON = 20
COST_Q = np.diag([1, 1])
COST_P = np.diag([0, 0])
COST_R = np.diag([5/1000, 1])

v_factor = 1

if not TRACK_CONS:
	SUFFIX = 'NOCONS-'
else:
	SUFFIX = ''




params = ORCA(control='pwm')
model = Dynamic(**params)




TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)




Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON




nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model, track, track_cons=TRACK_CONS)





states = np.zeros([n_states, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts
Ffy = np.zeros([n_steps+1])
Frx = np.zeros([n_steps+1])
Fry = np.zeros([n_steps+1])
hstates = np.zeros([n_states,horizon+1])
hstates2 = np.zeros([n_states,horizon+1])

Hs0 = []
Hs1 = []

Hs0_2 = []
Hs1_2 = []

MU_preds = []

projidx = 0
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
states[:,0] = x_init
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))


media_dir = "Oracle"
os.makedirs(media_dir, exist_ok=True)


H = .1
W = .05
dims = np.array([[-H/2.,-W/2.],[-H/2.,W/2.],[H/2.,W/2.],[H/2.,-W/2.],[-H/2.,-W/2.]])

fig_track = track.plot(color='k', grid=False)
fig_track.set_dpi(80)
plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
ax = plt.gca()
LnS, = ax.plot(states[0,0], states[1,0], '#4B0082', label='Trajectory',alpha=0.65)

xyproj, _ = track.project(x=x_init[0], y=x_init[1], raceline=track.raceline)
LnP, = ax.plot(states[0,0] + dims[:,0]*np.cos(states[2,0]) - dims[:,1]*np.sin(states[2,0])\
		, states[1,0] + dims[:,0]*np.sin(states[2,0]) + dims[:,1]*np.cos(states[2,0]), 'red', alpha=0.8)
LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=.5, lw=0.5, color='green', label="Ground truth")
LnH2, = ax.plot(hstates2[0], hstates2[1], '-b', marker='o', markersize=.5, lw=0.5, color='blue', label="Prediction")
plt.xlabel(r'$x$ [$\mathrm{m}$]')
plt.ylabel(r'$y$ [$\mathrm{m}$]')
plt.legend()

ref_speeds = []
Drs = []
Dfs = []

laps_completed = 0
lap_times = [0.,0.,0.,0.,0.]


for idt in range(n_steps-horizon):

	uprev = inputs[:,idt-1]
	x0 = states[:,idt]
	Drs.append(model.Dr)
	Dfs.append(model.Df)

	model.Df, model.Dr = update_friction(model.Df, model.Dr, idt*Ts)
	params['Df'], params['Dr'] = model.Df, model.Dr

	MU = (model.Df+model.Dr)/(9.81*params['mass'])
	MU_preds.append(copy.deepcopy(MU))


	if idt > 2 :
		xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx, curr_mu=MU_preds[-1], scale= v_factor )
	else :
		xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx)

	fval_history.append(find_closest_point(x0[0],x0[1], track.raceline)[-1])
	ref_speeds.append(v)

	if projidx > 656:

		if laps_completed > 0:
			lap_times[laps_completed] = idt * Ts
			print(lap_times)
		else:
			lap_times[laps_completed] = idt * Ts
			print(lap_times)
		laps_completed += 1
		projidx = 0


	start = tm.time()
	nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model, track, track_cons=TRACK_CONS)
	umpc, fval, xmpc, violation = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev)


	end = tm.time()
	inputs[:,idt] = umpc[:,0]
	print("iter: {}, cost: {:.5f}, time: {:.2f}".format(idt, fval, end-start))


	x_next, dxdt_next = model.sim_continuous(states[:,idt], inputs[:,idt].reshape(-1,1), [0, Ts])
	Ain, bin = Boundary(np.array(x_next[:2,-1]), track)
	flag = (np.array(Ain@np.array(x_next[:2,-1])).T > np.array(np.array( [bin[0][0], bin[1][0]] ) ) ).any()
	violation_total.append(flag*0.02 )

	states[:,idt+1] = x_next[:,-1]
	dstates[:,idt+1] = dxdt_next[:,-1]
	Ffy[idt+1], Frx[idt+1], Fry[idt+1] = model.calc_forces(states[:,idt], inputs[:,idt])


	hstates[:,0] = x0
	hstates2[:,0] = x0
	for idh in range(horizon):
		x_next, dxdt_next = model.sim_continuous(hstates[:,idh], umpc[:,idh].reshape(-1,1), [0, Ts])
		hstates[:,idh+1] = x_next[:,-1]
		hstates2[:,idh+1] = xmpc[:,idh+1]

	Hs0.append(copy.deepcopy(hstates[0]))
	Hs1.append(copy.deepcopy(hstates[1]))

	Hs0_2.append(copy.deepcopy(hstates2[0]))
	Hs1_2.append(copy.deepcopy(hstates2[1]))


	mean_cost = np.mean(fval_history)
	print(f"Mean cost at iter {idt}: {mean_cost:.3f}")






vel = np.sqrt(dstates[0,:]**2 + dstates[1,:]**2)















fig_track = track.plot(color='k', grid=False)


colors_list = ['navy', 'blue', 'orange','yellow']
custom_cmap = LinearSegmentedColormap.from_list("custom_speed", colors_list)

norm = colors.Normalize(vmin=np.min(vel[:n_steps-horizon]),
                        vmax=np.max(vel[:n_steps-horizon]))

points = np.array([states[0,:-(horizon)], states[1,:-(horizon)]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap=custom_cmap, norm=norm, linewidth=1.5, alpha=0.5)
lc.set_array(vel[:n_steps-horizon-1])
plt.gca().add_collection(lc)

sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, orientation='vertical')
cbar.set_label(r'Speed [${ \mathrm{m} }/{\mathrm{s}}$]')

ax.set_axis_off()
plt.axis('equal')
plt.axis('off')


fig_track.tight_layout(pad=0)
fig_track.subplots_adjust(left=0, right=.8, top=1, bottom=0)
plt.savefig(media_dir+'/Traj_Velocity.png', dpi=400, bbox_inches="tight")








































































colors_list = ['navy', 'blue', 'orange', 'yellow']
custom_cmap = LinearSegmentedColormap.from_list("custom_speed", colors_list)
norm = colors.Normalize(vmin=np.min(vel[:n_steps-horizon]), vmax=np.max(vel[:n_steps-horizon]))


fig_track = track.plot(color='k', grid=False)
ax = plt.gca()

points = np.array([states[0, :n_steps-horizon], states[1, :n_steps-horizon]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=custom_cmap, norm=norm, linewidth=1.5, alpha=0.5)
lc.set_array(vel[:n_steps-horizon-1])
ax.add_collection(lc)


LnP, = ax.plot(states[0,0] + dims[:,0]*np.cos(states[2,0]) - dims[:,1]*np.sin(states[2,0]),
               states[1,0] + dims[:,0]*np.sin(states[2,0]) + dims[:,1]*np.cos(states[2,0]), 'red', alpha=0.8, label='Current pose')
LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=.5, lw=0.5, color='green', label="ground truth")
LnH2, = ax.plot(hstates2[0], hstates2[1], '-b', marker='o', markersize=.5, lw=0.5, color='blue', label="prediction")


sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, orientation='vertical')
cbar.set_label(r'Speed [${ \mathrm{m} }/{\mathrm{s}}$]')


ax.set_axis_off()
plt.axis('equal')
plt.axis('off')
plt.legend()

def update(idt):
    if idt == 0:
        fig_track.tight_layout(pad=0)
    ax.set_title(f"Frame {idt}")

    new_points = np.array([states[0, :idt+1], states[1, :idt+1]]).T.reshape(-1, 1, 2)
    new_segments = np.concatenate([new_points[:-1], new_points[1:]], axis=1)
    lc.set_segments(new_segments)
    lc.set_array(vel[:idt])

    LnP.set_xdata(states[0, idt] + dims[:, 0] * np.cos(states[2, idt]) - dims[:, 1] * np.sin(states[2, idt]))
    LnP.set_ydata(states[1, idt] + dims[:, 0] * np.sin(states[2, idt]) + dims[:, 1] * np.cos(states[2, idt]))

    LnH.set_xdata(Hs0[idt])
    LnH.set_ydata(Hs1[idt])

    LnH2.set_xdata(Hs0_2[idt])
    LnH2.set_ydata(Hs1_2[idt])

    return lc, LnP, LnH, LnH2

SAVE_VIDEO = 1
if SAVE_VIDEO:
    fps = 17
    interval = 1000 / fps

    frame_numbers = range(0, n_steps-horizon,3)
    ani = animation.FuncAnimation(fig_track, update, frames=frame_numbers, interval=interval, blit=True)
    video_path = f"{media_dir}/traj_video.mp4"
    ani.save(video_path, fps=fps, extra_args=['-vcodec', 'h264_videotoolbox', '-b:v', '2000k', '-preset', 'ultrafast'])
    print(f"🎥 Smooth video saved as {video_path}")



for i in range(len(lap_times)-1,0,-1) :
	if lap_times[i] != 0. :
		lap_times[i] = lap_times[i] - lap_times[i-1]

print("lap times:",lap_times, "violation: ",np.sum(violation_total), "mean_dev", np.mean(fval_history))
