import argparse
import sys

if any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    parser = argparse.ArgumentParser(description='LLA-MPC fixed-bank non-real-time baseline')
    parser.add_argument('--n_models', type=int, default=5000, help='Number of models in the fixed bank')
    parser.parse_args()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--n_models', type=int, default=5000)
cli_args, _ = parser.parse_known_args()

from embmpc.mpc.constraints import Boundary

import time as tm
import numpy as np
import casadi
import _pickle as pickle

import matplotlib
from matplotlib.lines import lineStyles

violation_total = []


matplotlib.use('Agg')


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import subprocess


from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed

from embmpc.params import ORCA
from embmpc.models import Dynamic
from embmpc.tracks import ETHZ, ETHZMobil
from embmpc.mpc.planner import ConstantSpeed
from embmpc.mpc.nmpc import setupNLP
import multiprocessing as mp
from embmpc.mpc.evaluate_models_vectorized import evaluate_models_vectorized
import os
import imageio
import copy

import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection


import matplotlib.pylab as pylab
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

if not TRACK_CONS:
    SUFFIX = 'NOCONS-'
else:
    SUFFIX = ''




params = ORCA(control='pwm')
model = Dynamic(**params)
print("check1",id(model))

model_run = Dynamic(**params)




N_MODELS = cli_args.n_models
N_AC_STEPS = 10
smoothing_mu = 20
smoothing_mu_over_mod = 10
v_factor = .9
mu_init = 1.

class ExponentialSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.smooth_value = None

    def update(self, new_value):
        if self.smooth_value is None:
            self.smooth_value = new_value
        else:
            self.smooth_value = self.alpha * new_value + (1 - self.alpha) * self.smooth_value
        return self.smooth_value




smoother = ExponentialSmoother(alpha=0.08)


def find_closest_point(x, y, raceline):

    x_refs = raceline[0]
    y_refs = raceline[1]

    distances = np.sqrt((x_refs - x) ** 2 + (y_refs - y) ** 2)

    idx = np.argmin(distances)
    return idx, x_refs[idx], y_refs[idx], distances[idx]

def update_friction(Df,Dr,curr_time,style='const_decay') :
    if style == 'const_decay' :
        if curr_time > 14.3:
            Df -= Df/2600.
            Dr -= Dr/2600.
    elif style == 'sudden' :
        if curr_time > 3.3 and curr_time < 3.5:

            Df -= Df/22.
            Dr -= Dr/22.
    elif style == 'no_change' :
        return Df, Dr
    return Df, Dr

MODEL_BANK = []
MODEL_PARAMS = []
laps_completed = 0
lap_times = [0.,0.,0.,0.,0.]

Drs = []
Dfs = []

Drs_preds = []
Dfs_preds = []

MUs = []
MU_preds = []



variation_dict = {







    'Br': 1.5,
    'Cr': 1.5,
    'Dr': 1.5,
    'Bf': 1.5,
    'Cf': 1.5,
    'Df': 1.5,
}

fval_history = []



for i in range(N_MODELS):
    param_variation = params.copy()

    for param_name, variation_percentage in variation_dict.items():
        if param_name in param_variation:
            param_variation[param_name] *= (1 + variation_percentage * np.random.randn())

    MODEL_PARAMS.append(param_variation)
    MODEL_BANK.append(Dynamic(**param_variation))

Bfs_pass = np.array([m.Bf for m in MODEL_BANK])
Cfs_pass = np.array([m.Cf for m in MODEL_BANK])
Dfs_pass = np.array([m.Df for m in MODEL_BANK])
Brs_pass = np.array([m.Br for m in MODEL_BANK])
Crs_pass = np.array([m.Cr for m in MODEL_BANK])
Drs_pass = np.array([m.Dr for m in MODEL_BANK])

params_pass = (Bfs_pass, Cfs_pass, Dfs_pass, Brs_pass, Crs_pass, Drs_pass)




TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)




Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON


error_windows = np.zeros((N_MODELS, N_AC_STEPS))
window_count = 0















nlp_initial = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model_run, track, track_cons=TRACK_CONS)







ref_speeds = []

states = np.zeros([n_states, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts

Ffy = np.zeros([n_steps+1])
Frx = np.zeros([n_steps+1])
Fry = np.zeros([n_steps+1])

Ffy_preds = np.zeros([n_steps+1])
Frx_preds = np.zeros([n_steps+1])
Fry_preds = np.zeros([n_steps+1])

hstates = np.zeros([n_states,horizon+1])
hstates2 = np.zeros([n_states,horizon+1])

Hs0 = []
Hs1 = []

Hs0_2 = []
Hs1_2 = []


model_switches = []
model_mses = []
chosen_models = []

projidx = 0
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
states[:,0] = x_init
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))


media_dir = "LLA"
os.makedirs(media_dir, exist_ok=True)


H = .1
W = .05
dims = np.array([[-H/2.,-W/2.],[-H/2.,W/2.],[H/2.,W/2.],[H/2.,-W/2.],[-H/2.,-W/2.]])

fig_track = track.plot(color='k', grid=False)
fig_track.set_dpi(50)
plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
ax = plt.gca()
LnS, = ax.plot(states[0,0], states[1,0], '#4B0082', label='Trajectory',alpha=0.65)

xyproj, _ = track.project(x=x_init[0], y=x_init[1], raceline=track.raceline)
LnP, = ax.plot(states[0,0] + dims[:,0]*np.cos(states[2,0]) - dims[:,1]*np.sin(states[2,0])\
		, states[1,0] + dims[:,0]*np.sin(states[2,0]) + dims[:,1]*np.cos(states[2,0]), 'red', alpha=0.8, label='Current pose')
LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=.5, lw=0.5, color='green', label="ground truth")
LnH2, = ax.plot(hstates2[0], hstates2[1], '-b', marker='o', markersize=.5, lw=0.5, color='blue', label="prediction")
plt.xlabel(r'$x$ [$\mathrm{m}$]')
plt.ylabel(r'$y$ [$\mathrm{m}$]')
plt.legend()


current_model_idx = 0
uprev = np.zeros(n_inputs)



for idt in range(n_steps-horizon):
    print("checkiter", id(model))
    x0 = states[:,idt]


    model.Df, model.Dr = update_friction(model.Df, model.Dr, idt * Ts, "const_decay")
    params['Df'], params['Dr'] = model.Df, model.Dr


    if idt > N_AC_STEPS+1:
        xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx,
                                         curr_mu=MU_pred, scale=v_factor)
    else:
        xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx)

    ref_speeds.append(v)
    fval_history.append(find_closest_point(x0[0], x0[1], track.raceline)[-1])

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























    if idt <= N_AC_STEPS:
        nlp = nlp_initial
    else:
        nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R,
                  MODEL_PARAMS[current_model_idx],
                  MODEL_BANK[current_model_idx],
                  track, track_cons=TRACK_CONS)

    umpc, fval, xmpc, violation = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev)

    inputs[:,idt] = umpc[:,0]
    uprev = inputs[:,idt]









    x_next, dxdt_next = model.sim_continuous(states[:,idt], inputs[:,idt].reshape(-1,1), [0, Ts])




    states[:,idt+1] = x_next[:,-1]
    dstates[:,idt+1] = dxdt_next[:,-1]
    Ffy[idt+1], Frx[idt+1], Fry[idt+1] = model.calc_forces(states[:,idt], inputs[:,idt])
    Ffy_preds[idt+1], Frx_preds[idt+1], Fry_preds[idt+1] = MODEL_BANK[current_model_idx].calc_forces(states[:,idt], inputs[:,idt], return_slip=False)

    Drs.append(model.Dr)
    Dfs.append(model.Df)




    if idt <= N_AC_STEPS:
        Drs_preds.append(mu_init * params['mass'] * 9.8 * params['lr'] / (params['lf'] + params['lr']))
        Dfs_preds.append(mu_init * params['mass'] * 9.8 * params['lf'] / (params['lf'] + params['lr']))
        MUs.append(copy.deepcopy((model.Df + model.Dr) / (9.81 * params['mass'])))
        MU_preds.append(copy.deepcopy(mu_init))
    else:
        MUs.append(copy.deepcopy((model.Df + model.Dr) / (9.81 * params['mass'])))

        bestKDr = []
        bestKDf = []
        for best_ind in ind_best_KM:
            bestKDr.append(copy.deepcopy(MODEL_BANK[best_ind].Dr))
            bestKDf.append(copy.deepcopy(MODEL_BANK[best_ind].Df))
        Drs_preds.append(np.mean(bestKDr))
        Dfs_preds.append(np.mean(bestKDf))
        MU_pred = copy.deepcopy(np.mean(np.array(Drs_preds)[-smoothing_mu:])  + np.mean(np.array(Dfs_preds)[-smoothing_mu:]) ) / (9.81 * params['mass'])


        MU_preds.append(smoother.update(MU_pred)*.95)


    if idt > 0:


        errors = np.mean((evaluate_models_vectorized(MODEL_BANK, N_MODELS, states[:, idt], inputs[:, idt], Ts, params_pass) - states[0:4, idt + 1]) ** 2, axis=1)


        error_windows = np.roll(error_windows, -1, axis=1)
        error_windows[:, -1] = errors
        window_count = min(window_count + 1, N_AC_STEPS)


        if window_count >= N_AC_STEPS:
            avg_errors = np.mean(error_windows, axis=1)
            new_model_idx = np.argmin(avg_errors)
            ind_best_KM = avg_errors.argsort()[:smoothing_mu_over_mod]

            if new_model_idx != current_model_idx:
                current_model_idx = new_model_idx
                model_switches.append(idt)
                model_mses.append(avg_errors[new_model_idx])
                chosen_models.append(current_model_idx)


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

    end = tm.time()
    print("iter: {}, model: {}, cost: {:.5f}, time: {:.2f}, mse*10000: {:.5f}".format(
        idt,
        current_model_idx,
        fval,
        end - start,
        np.mean(error_windows[current_model_idx])*10000 if window_count > 0 else 0.0
    ))

    mean_cost = np.mean(fval_history)
    print(f"Mean cost at iter {idt}: {mean_cost:.3f}")

















def update(idt):
    if idt == 0:
        fig_track.tight_layout()

    seconds = (idt*Ts) % 60
    ax.set_title(f"Time {seconds:06.2f}")












    LnS.set_xdata(states[0, :idt + 1])
    LnS.set_ydata(states[1, :idt + 1])








    LnP.set_xdata(states[0, idt] + dims[:, 0] * np.cos(states[2, idt]) - dims[:, 1] * np.sin(states[2, idt]))
    LnP.set_ydata(states[1, idt] + dims[:, 0] * np.sin(states[2, idt]) + dims[:, 1] * np.cos(states[2, idt]))



    LnH.set_xdata(Hs0[idt])
    LnH.set_ydata(Hs1[idt])

    LnH2.set_xdata(Hs0_2[idt])
    LnH2.set_ydata(Hs1_2[idt])

    return LnS, LnP, LnH, LnH2



plt.figure(figsize=(6.4, 2.4))


time_steps = int(SIM_TIME / Ts)
model_usage = np.ones((N_MODELS, time_steps))


thickness = 12


for t in range(time_steps):
    if t in np.array(model_switches):
        idx = model_switches.index(t)

        for offset in range(-thickness // 2, thickness // 2 + 1):
            row = chosen_models[idx] + offset
            if 0 <= row < N_MODELS:
                model_usage[row, t:] = 0


        if idx > 0:
            prev_model = chosen_models[idx - 1]
            for offset in range(-thickness // 2, thickness // 2 + 1):
                row = prev_model + offset
                if 0 <= row < N_MODELS:
                    model_usage[row, t:] = 1


plt.imshow(model_usage,
           aspect='auto',
           cmap='binary',
           extent=[0, SIM_TIME, -0.5, N_MODELS - 0.5],
           interpolation='none')


cbar = plt.colorbar()
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Chosen', 'Not Chosen'])

plt.xlabel(r'Time [$\mathrm{s}$]')
plt.ylabel('Model Index')
plt.tight_layout()
plt.savefig(media_dir + '/Switching_Grid.png', dpi=1200)
















































plt.figure(figsize=(6.4, 2.4))
vel = np.sqrt(dstates[0,:]**2 + dstates[1,:]**2)
plt.plot(time[:n_steps-horizon], ref_speeds,color="#E5AE1C",linewidth=4, label='Reference')
plt.plot(time[:n_steps-horizon], vel[:n_steps-horizon],color="#0B67B2",linewidth=4, label='Actual')


plt.xlabel(r'Time [$\mathrm{s}$]')
plt.ylabel(r'Speed [$\frac{ \mathrm{m} }{\mathrm{s}}$]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(media_dir+'/Speeds.png', dpi=1200, bbox_inches="tight")




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
plt.savefig(media_dir+'/Traj_Velocity.png', dpi=1200, bbox_inches="tight")













plt.figure(figsize=(6.4, 2.4))
plt.plot(time[:n_steps-horizon],MUs,color="#E5AE1C",linewidth=4, label=r"Ground Truth")
plt.plot(time[:n_steps-horizon],MU_preds,color="#0B67B2",linewidth=4,  label=r"Predicted")
plt.grid(True)
plt.xlabel(r'Time [$\mathrm{s}$]')
plt.ylabel(r'$\mu$')
plt.legend()
plt.tight_layout()
plt.savefig(media_dir+'/MUs.png', dpi=1200, bbox_inches="tight")

np.save(media_dir+'/Time.npy', time[:n_steps-horizon])
np.save(media_dir+'/MUs.npy', MUs)
np.save(media_dir+'/MU_preds.npy', MU_preds)











































for i in range(len(lap_times)-1,0,-1) :
	if lap_times[i] != 0. :
		lap_times[i] = lap_times[i] - lap_times[i-1]
print(lap_times)
print("lap times:",lap_times, "violation: ",np.sum(violation_total), "mean_dev", np.mean(fval_history))








