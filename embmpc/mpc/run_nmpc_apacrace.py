import argparse
import sys

if any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    parser = argparse.ArgumentParser(description='APACRace baseline')
    parser.parse_args()

fval_history = []

def find_closest_point(x, y, raceline):

	x_refs = raceline[0]
	y_refs = raceline[1]


	distances = np.sqrt((x_refs - x) ** 2 + (y_refs - y) ** 2)


	idx = np.argmin(distances)

	return idx, x_refs[idx], y_refs[idx], distances[idx]

import time as tm
import numpy as np
import casadi

import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from embmpc.params import ORCA
from embmpc.models import Dynamic, Kinematic6
from embmpc.gp.utils import loadGPModel, loadGPModelVars, loadMLPModel, loadTorchModel, loadTorchModelEq, loadTorchModelImplicit
from embmpc.tracks import ETHZ, ETHZMobil
from embmpc.mpc.planner import ConstantSpeed
from embmpc.mpc.gpmpc_torch import setupNLP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import torch
import random
import os
import copy
import subprocess

import matplotlib
matplotlib.use('Agg')


import matplotlib.pylab as pylab
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from embmpc.mpc.constraints import Boundary



params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
plt.rcParams['text.usetex'] = True





GP_EPS_LEN = 410
mu_init = 1.
t_collect = 8.
LR = 0.002
BETA = 0.9
violation_total = []

def dist(a,b) :
	return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


statistics = []




ERROR_CORR = True
TRACK_CONS = False
RUN_NO = 'with_var_speeds'
ITERS_EACH_STEP = 50
LOAD_MODEL = False
ACT_FN = 'relu'
SAFEGUARD = True
MAX_STEER = 0.34
VEHICLE_MODEL = 'Kinematic'
lambda_ = 10.
lambda_2 = 3.
ALPHA = 1.
torch.manual_seed(3)
random.seed(0)
np.random.seed(0)


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

class DynamicModel(torch.nn.Module):
	def __init__(self, model, deltat = 0.01):
		super().__init__()
		if ACT_FN == 'relu' :
			self.act = torch.nn.ReLU()
		elif ACT_FN =='tanh' :
			self.act = torch.nn.Tanh()
		elif ACT_FN =='lrelu' :
			self.act = torch.nn.LeakyReLU()
		elif ACT_FN =='sigmoid' :
			self.act = torch.nn.Sigmoid()
		
		self.Rx = torch.nn.Sequential(torch.nn.Linear(1,1).to(torch.float64))
		
		self.Ry = torch.nn.Sequential(torch.nn.Linear(1,6).to(torch.float64), \
					self.act, \
					torch.nn.Linear(6,1).to(torch.float64))
		self.Ry[0].weight.data.fill_(1.)

		self.Ry[0].bias.data = torch.arange(-.6,.6,(1.2)/6.).to(torch.float64)
		self.Ry[2].weight.data.fill_(0.)
		self.Ry[2].bias.data.fill_(0.)



		self.Fy = torch.nn.Sequential(torch.nn.Linear(1,6).to(torch.float64), \
					self.act, \
					torch.nn.Linear(6,1).to(torch.float64))
		
		self.Fy[0].weight.data.fill_(1.)
		self.Fy[2].weight.data.fill_(0.)
		self.Fy[2].bias.data.fill_(0.)


		self.Fy[0].bias.data = torch.arange(-.6,.6,(1.2)/6.).to(torch.float64)
		self.deltat = deltat
		self.model = model

	def forward(self, x, debug=False):


		deltatheta = x[:,1]
		theta = x[:,2]
		pwm = x[:,0]
		out = torch.zeros_like(x[:,3:6])

		for i in range(2) :
			vx = (x[:,3] + out[:,0]).unsqueeze(1)
			vy = x[:,4] + out[:,1]
			w = x[:,5] + out[:,2]
			alpha_f = (theta - torch.atan2(w*self.model.lf+vy,vx[:,0])).unsqueeze(1)
			alpha_r = torch.atan2(w*self.model.lr-vy,vx[:,0]).unsqueeze(1)
			Ffy = self.Fy(alpha_f)[:,0]
			Fry = self.Ry(alpha_r)[:,0]
			Frx = self.Rx(vx**2)[:,0]
			Frx = (self.model.Cm1-self.model.Cm2*vx[:,0])*pwm + Frx
			
			if debug :
				print(Ffy,Fry,Frx)
			
			Frx_kin = (self.model.Cm1-self.model.Cm2*vx[:,0])*pwm
			vx_dot = (Frx-Ffy*torch.sin(theta)+self.model.mass*vy*w)/self.model.mass
			vy_dot = (Fry+Ffy*torch.cos(theta)-self.model.mass*vx[:,0]*w)/self.model.mass
			w_dot = (Ffy*self.model.lf*torch.cos(theta)-Fry*self.model.lr)/self.model.Iz
			out += torch.cat([vx_dot.unsqueeze(dim=1),vy_dot.unsqueeze(dim=1),w_dot.unsqueeze(dim=1)],axis=1)*self.deltat
		out2 = (out)
		return out2



def get_optimal_control_outer(pwm_ref,steer_ref,v,theta,x,curvature,model,nx=60,EPSILON=1e-2) :


	if SAFEGUARD==False :
		return steer_ref
	min_steer = -MAX_STEER
	min_cost = 1000000
	L = model['lf'] + model['lr']
	
	pwm, steer = np.meshgrid(np.linspace(model['min_pwm'], model['max_pwm'], nx),\
				 np.linspace(model['min_steer'], model['max_steer'], nx))
	if VEHICLE_MODEL=='Kinematic' :
		ax = ((model['Cm1']-model['Cm2']*v)*pwm - model['Cr0'] - model['Cr2']*v**2)/model['mass']
		h_left = x-EPSILON
		hd_left = v*math.sin(theta)
		hdd_left = ax*math.sin(theta)+v**2*math.cos(theta)*steer/L - v**2*curvature
		cost = (steer-steer_ref)**2 + (pwm-pwm_ref)**2
		cost += ALPHA*((hdd_left+lambda_*hd_left+lambda_**2*h_left)<0)*\
			(hdd_left+lambda_*hd_left+lambda_**2*h_left)**2
	ind = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
	min_pwm = np.linspace(model['min_pwm'], model['max_pwm'], nx)[ind[0]]
	min_steer = np.linspace(model['min_steer'], model['max_steer'], nx)[ind[1]]
	feasibility_map = (hdd_left+lambda_*hd_left+lambda_**2*h_left)>=0.
	return min_steer, min_pwm, feasibility_map, pwm, steer

def get_optimal_control_inner(pwm_ref,steer_ref,v,theta,x,curvature,\
			      theta_,x_,curvature_,model,nx=60,ny=120,EPSILON=0.) :
	if SAFEGUARD==False :
		return steer_ref
	min_steer = -MAX_STEER
	min_cost = 1000000
	print(curvature,curvature_)
	L = model['lf'] + model['lr']
	
	pwm, steer = np.meshgrid(np.linspace(model['min_pwm'], model['max_pwm'], nx),\
				 np.linspace(model['min_steer'], model['max_steer'], ny))
	if VEHICLE_MODEL=='Kinematic' :
		ax = ((model['Cm1']-model['Cm2']*v)*pwm - model['Cr0'] - model['Cr2']*v**2)/model['mass']
		h_left = -x-EPSILON
		hd_left = -v*math.sin(theta)
		hdd_left = -ax*math.sin(theta)-v**2*math.cos(theta)*steer/L + v**2*curvature
		
		h_right = x_-EPSILON
		hd_right = v*math.sin(theta_)
		hdd_right = ax*math.sin(theta_)+v**2*math.cos(theta_)*steer/L - v**2*curvature_
		
		cost = (steer-steer_ref)**2 + (pwm-pwm_ref)**2
		cost += ALPHA*((hdd_left+lambda_*hd_left+lambda_**2*h_left)<0)*\
			(hdd_left+lambda_*hd_left+lambda_**2*h_left)**2
		cost += ALPHA*((hdd_right+lambda_*hd_right+lambda_**2*h_right)<0)*\
			(hdd_right+lambda_*hd_right+lambda_**2*h_right)**2
	elif VEHICLE_MODEL=='Dynamic' :
		ax = ((model['Cm1']-model['Cm2']*v)*pwm - model['Cr0'] - model['Cr2']*v**2)/model['mass']
		h_left = -x-EPSILON
		hd_left = -v*math.sin(theta)
		hdd_left = -ax*math.sin(theta)-v**2*math.cos(theta)*steer/L + v**2*curvature
		
		h_right = x_-EPSILON
		hd_right = v*math.sin(theta_)
		hdd_right = ax*math.sin(theta_)+v**2*math.cos(theta_)*steer/L - v**2*curvature_
		
		cost = (steer-steer_ref)**2 + (pwm-pwm_ref)**2
		cost += ALPHA*((hdd_left+lambda_*hd_left+lambda_**2*h_left)<0)*\
			(hdd_left+lambda_*hd_left+lambda_**2*h_left)**2
		cost += ALPHA*((hdd_right+lambda_*hd_right+lambda_**2*h_right)<0)*\
			(hdd_right+lambda_*hd_right+lambda_**2*h_right)**2
	
	ind = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
	min_pwm = pwm[ind]
	min_steer = steer[ind]
	feasibility_map = (hdd_left+lambda_*hd_left+lambda_**2*h_left)>=0.
	feasibility_map *= (hdd_right+lambda_*hd_right+lambda_**2*h_right)>=0.
	return min_steer, min_pwm, feasibility_map, pwm, steer

def get_optimal_control(pwm_ref,steer_ref,state_left,curvature_left,\
			      state_right,curvature_right,model,nx=60,ny=120,EPSILON=0.05) :
	print(state_right)
	if SAFEGUARD==False :
		return steer_ref
	min_steer = -MAX_STEER
	min_cost = 1000000
	

	L = model['lf'] + model['lr']
	
	pwm, steer = np.meshgrid(np.linspace(model['min_pwm'], model['max_pwm'], nx),\
				 np.linspace(model['min_steer'], model['max_steer'], ny))
	
	vx = state_right[2]
	vy = state_right[3]
	omega = state_right[4]
	
	alpha_f = steer - np.arctan2(omega*model['lf']+vy,vx)
	alpha_r = np.arctan2(omega*model['lr']-vy,vx)
	
	Ffy = model['Df']*np.sin(model['Cf']*np.arctan(model['Bf']*alpha_f))
	Fry = model['Dr']*np.sin(model['Cr']*np.arctan(model['Br']*alpha_r))
	v = np.sqrt(vx**2+vy**2)
	Frx = ((model['Cm1']-model['Cm2']*vx)*pwm)
	ax = (Frx-Ffy*np.sin(steer)+model['mass']*vy*omega)/model['mass']
	ay = (Fry+Ffy*np.cos(steer)-model['mass']*vx*omega)/model['mass']
	w_dot = (Ffy*model['lf']*np.cos(steer)-Fry*model['lr'])/model['Iz']
	

	

	

	d = state_right[0]
	theta = state_right[1]
	
	print("Curv : ", curvature_right)
	curvature_right = np.clip(curvature_right,-2,0.5)
	print("Theta : ", theta)
	h_right = d-EPSILON
	hd_right = vx*np.sin(theta) + vy*np.cos(theta)
	print("hd : ",hd_right+lambda_2*h_right)
	print((omega-(vx*np.cos(theta)-vy*np.sin(theta))*curvature_right)*(vx*np.cos(theta)-vy*np.sin(theta)))
	hdd_right = (ax*math.sin(theta)+ay*math.cos(theta)) + \
				(omega-(vx*np.cos(theta)-vy*np.sin(theta))*curvature_right)*(vx*np.cos(theta)-vy*np.sin(theta))
	

	d = state_left[0]
	theta = state_left[1]

	curvature_left = np.clip(curvature_left,-0.5,10)

	h_left = -d-EPSILON
	hd_left = -(vx*np.sin(theta) + vy*np.cos(theta))


	hdd_left = -((ax*math.sin(theta)+ay*math.cos(theta)) + \
				(omega-(vx*np.cos(theta)-vy*np.sin(theta))*curvature_left)*(vx*np.cos(theta)-vy*np.sin(theta)))
	



	
	cost = 100*(steer-steer_ref)**2 + (pwm-pwm_ref)**2
	cost += ALPHA*((hdd_left+lambda_*hd_left+lambda_*lambda_2*h_left)<0)*\
		(hdd_left+lambda_*hd_left+lambda_*lambda_2*h_left)**2
	cost += ALPHA*((hdd_right+lambda_*hd_right+lambda_*lambda_2*h_right)<0)*\
		(hdd_right+lambda_*hd_right+lambda_*lambda_2*h_right)**2
	
	ind = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
	min_pwm = pwm[ind]
	min_steer = steer[ind]
	feasibility_map = (hdd_left+lambda_*hd_left+lambda_*lambda_2*h_left)>=0.
	feasibility_map *= (hdd_right+lambda_*hd_right+lambda_*lambda_2*h_right)>=0.
	return min_steer, min_pwm, feasibility_map, pwm, steer


RUN_FOLDER = 'RUN_ONLINE_' + str(RUN_NO) + '_' + str(GP_EPS_LEN) + '/'

SIM_TIME = 36
SAMPLING_TIME = 0.02
HORIZON = 20
COST_Q = np.diag([1, 1])
COST_P = np.diag([0, 0])
COST_R = np.diag([5/1000, 1])

N_collect = int(t_collect/SAMPLING_TIME)
if not TRACK_CONS:
	SUFFIX = 'NOCONS-'+RUN_NO
else:
	SUFFIX = RUN_NO




params = ORCA(control='pwm')
model = Dynamic(**params)
model_kin = Kinematic6(**params)




TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)
print(track.raceline[0][0])




MODEL_PATH = '../gp/orca/semi_mlp-v2.pickle'
model_ = DynamicModel(model)
if LOAD_MODEL :
	model_.load_state_dict(torch.load(MODEL_PATH))


model_Rx = loadTorchModelImplicit('Rx',model_.Rx)
model_Ry = loadTorchModelImplicit('Ry',model_.Ry)
model_Fy = loadTorchModelImplicit('Fy',model_.Fy)

models = {
	'Rx' : model_Rx,
	'Ry' : model_Ry,
	'Fy' : model_Fy,
	'act_fn' : ACT_FN
}
x_train = np.zeros((GP_EPS_LEN,2+3+1))

optimizer = torch.optim.SGD(model_.parameters(), lr=LR,momentum=BETA)
loss_fn = torch.nn.MSELoss()




Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON




nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, models, track, GP_EPS_LEN=GP_EPS_LEN, 
	track_cons=TRACK_CONS, error_correction=ERROR_CORR)




def load_data(data_dyn, data_kin, VARIDX):
	y_all = (data_dyn['states'][:6,1:]-data_dyn['states'][:6,:-1])
	x = np.concatenate([
		data_kin['inputs'][:,:-1].T,
		data_dyn['inputs'][1,:-1].reshape(1,-1).T,
		data_dyn['states'][3:6,:-1].T],
		axis=1)
	y = y_all[VARIDX].reshape(-1,1)

	return x, y





states = np.zeros([n_states+1, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
inputs_kin = np.zeros([n_inputs, n_steps])
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

data_dyn = {}
data_kin = {}
data_dyn['time'] = time
data_kin['time'] = time

projidx = 0
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
states[:n_states,0] = x_init
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))
states_kin = np.zeros([7,n_steps+1])
states_kin[:,0] = states[:,0]


media_dir = "APACRace"
os.makedirs(media_dir, exist_ok=True)


H = .1
W = .05
dims = np.array([[-H/2.,-W/2.],[-H/2.,W/2.],[H/2.,W/2.],[H/2.,-W/2.],[-H/2.,-W/2.]])


ref_speeds = []
Drs = []
Dfs = []
Drs_pred = []
Dfs_pred = []
Df_init = model.Df
Dr_init = model.Dr
h_outers = []
h_inners = []

MUs = []
MU_preds = []

dist_covered = 0
laps_completed = 0
lap_times = [0.,0.,0.,0.,0.]
cum_dists = [0.]

curr_cum_dist = 0.
for i in range(len(track.center_line[0])-1) :
	cum_dists.append(curr_cum_dist)
	curr_cum_dist += dist(track.center_line[:,i],track.center_line[:,i+1])

ref_errs = []
boundary_viol_time = 0.
v_factor = .9
alpha_f_max = 0.45
alpha_r_max = 0.45
for idt in range(n_steps-horizon):
	print("alpha maxes: ", alpha_f_max, alpha_r_max)
	start_g = tm.time()
	uprev = inputs[:,idt-1]
	x0 = states[:,idt]
	use_kinematic = True
	Drs.append(model.Dr)
	Dfs.append(model.Df)
	

	if idt > 0 : 
		start = tm.time()	
		min_ind = max(idt-GP_EPS_LEN-1,3)
		data_dyn['states'] = states[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
		data_dyn['dstates'] = dstates[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
		data_dyn['inputs'] = inputs[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
		
		data_kin['states'] = states_kin[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
		data_kin['inputs'] = inputs_kin[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
	
	
		end = tm.time()
		print("GP init time : ", end-start)
		
		y_trains = []
		for VARIDX in [3,4,5] :
			x_train, y_train = load_data(data_dyn,data_kin,VARIDX)
			y_trains.append(torch.tensor(y_train))
		y_train = torch.cat(y_trains,axis=1)
		x_train = torch.tensor(x_train)

	if idt > 12 and 'without' not in RUN_NO:
		start = tm.time()	
		for i in range(ITERS_EACH_STEP) :

			optimizer.zero_grad()
			outputs = model_(x_train[10:])
			loss = loss_fn(outputs, y_train[10:])
			loss.backward()

			optimizer.step()
		end = tm.time()	
		print("Iter " + str(idt) + " loss : ", loss.item(), "time : ", end-start)

	model.Df, model.Dr = update_friction(model.Df, model.Dr, idt*Ts)
	params['Df'], params['Dr'] = model.Df, model.Dr

	if idt > N_collect and 'without' not in RUN_NO:
		use_kinematic = False
		
	

	prev_projidx = projidx
	if idt > N_collect and 'without' in RUN_NO:
		v_factor = 1.2
	if idt > N_collect and 'const_speeds' not in RUN_NO and 'without' not in RUN_NO:
		xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx, curr_mu=(Dfs_pred[idt-1]+Drs_pred[idt-1])/(9.81*params['mass']),scale=v_factor)
	else :
		xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx, curr_mu=mu_init,scale=v_factor)
	ref_speeds.append(v)
	fval_history.append(find_closest_point(x0[0],x0[1], track.raceline)[-1])

	print(projidx)
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







	umpc, fval, xmpc, violation = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev, use_kinematic=use_kinematic,models=model_)


	end = tm.time()
	inputs[:,idt] = np.array([umpc[0,0], states[n_states,idt] + Ts*umpc[1,0]])
	print("iter: {}, cost: {:.5f}, time: {:.2f}".format(idt, fval, end-start))
	control_before = inputs[:,idt].copy()

	x_next, dxdt_next = model.sim_continuous(states[:n_states,idt], inputs[:,idt].reshape(-1,1), [0, Ts])

	Ain, bin = Boundary(np.array(x_next[:2,-1]), track)
	flag = (np.array(Ain@np.array(x_next[:2,-1])).T > np.array(np.array( [bin[0][0], bin[1][0]] ) ) ).any()
	violation_total.append(flag*0.02 )

	inputs_kin[:,idt] = inputs[:,idt]

	if (idt!=0) :
		inputs_kin[1,idt] = (inputs[1,idt] - inputs[1,idt-1])/Ts
	else :
		inputs_kin[1,idt] = (inputs[1,idt] - 0.)/Ts

	x_next_kin, dxdt_next_kin = model_kin.sim_continuous(states[:,idt], inputs_kin[:,idt].reshape(-1,1), [0, Ts])

	states_kin[:,idt+1] = x_next_kin[:,-1]
	
	states[:n_states,idt+1] = x_next[:,-1]
	states[n_states,idt+1] = inputs[1,idt]

	dstates[:,idt+1] = dxdt_next[:,-1]
	Ffy[idt+1], Frx[idt+1], Fry[idt+1], alpha_f_curr, alpha_r_curr = model.calc_forces(states[:,idt], inputs[:,idt],return_slip=True)


	steer = states[n_states,idt]
	hstates[:,0] = x0[:n_states]
	hstates2[:,0] = x0[:n_states]

	for idh in range(horizon):
		steer = steer + Ts*umpc[1,idh]
		hinput = np.array([umpc[0,idh], steer])
		x_next, dxdt_next = model.sim_continuous(hstates[:n_states,idh], hinput.reshape(-1,1), [0, Ts])
		hstates[:,idh+1] = x_next[:n_states,-1]
		hstates2[:,idh+1] = xmpc[:n_states,idh+1]

	Hs0.append(copy.deepcopy(hstates[0]))
	Hs1.append(copy.deepcopy(hstates[1]))

	Hs0_2.append(copy.deepcopy(hstates2[0]))
	Hs1_2.append(copy.deepcopy(hstates2[1]))

	mean_cost = np.mean(fval_history)
	print(f"Mean cost at iter {idt}: {mean_cost:.3f}")


	alpha_f_max = min(0.6,max(np.abs(alpha_f_curr),alpha_f_max))
	alpha_r_max = min(0.6,max(np.abs(alpha_r_curr),alpha_r_max))

	alpha_f = torch.tensor(np.arange(-alpha_f_max,alpha_f_max,0.01)).unsqueeze(1)
	Ffy_pred = model_.Fy(alpha_f)[:,0].detach().numpy()
	Ffy_true = params['Df']*torch.sin(params['Cf']*torch.atan(params['Bf']*alpha_f))

	if idt < N_collect :
		Dfs_pred.append(mu_init*params['mass']*9.8*params['lr']/(params['lf']+params['lr']))
	else:
		Dfs_pred.append(np.max(Ffy_pred))

	alpha_r = torch.tensor(np.arange(-alpha_r_max,alpha_r_max,0.01)).unsqueeze(1)
	Fry_pred = model_.Ry(alpha_r)[:,0].detach().numpy()
	Fry_true = params['Dr']*torch.sin(params['Cr']*torch.atan(params['Br']*alpha_r))

	if idt < N_collect :
		Drs_pred.append(mu_init*params['mass']*9.8*params['lf']/(params['lf']+params['lr']))
	else :
		Drs_pred.append(np.max(Fry_pred))

	if idt < N_collect :
		MUs.append((model.Df + model.Dr) / (9.81 * params['mass']))
		MU_preds.append(mu_init)
	else:
		MUs.append((model.Df + model.Dr) / (9.81 * params['mass']))
		MU_preds.append((Dfs_pred[-1] + Drs_pred[-1]) / (9.81 * params['mass']))







plt.figure(figsize=(6.4, 2.4))
vel = np.sqrt(dstates[0,:]**2 + dstates[1,:]**2)
plt.plot(time[:n_steps-horizon], ref_speeds,color="#E5AE1C",linewidth=4, label='Reference')
plt.plot(time[:n_steps-horizon], vel[:n_steps-horizon],color="#0B67B2",linewidth=4, label='Actual')


plt.xlabel(r'Time [$\mathrm{s}$]')
plt.ylabel(r'Speed [$\frac{ \mathrm{m} }{\mathrm{s}}$]')
plt.title('Speeds')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(media_dir+'/Speeds.png', dpi=400, bbox_inches="tight")













plt.figure(figsize=(6.4, 2.4))
plt.plot(time[:n_steps-horizon],MUs,color="#E5AE1C",linewidth=4, label=r"Ground Truth")
plt.plot(time[:n_steps-horizon],MU_preds,color="#0B67B2",linewidth=4,  label=r"Predicted")
plt.grid(True)
plt.xlabel(r'Time [$\mathrm{s}$]')
plt.ylabel(r'$\mu$')
plt.legend()
plt.tight_layout()
plt.savefig(media_dir+'/MUs.png', dpi=400, bbox_inches="tight")

np.save(media_dir+'/Time.npy', time[:n_steps-horizon])
np.save(media_dir+'/MUs.npy', MUs)
np.save(media_dir+'/MU_preds.npy', MU_preds)




















































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


for i in range(len(lap_times)-1,0,-1) :
	if lap_times[i] != 0. :
		lap_times[i] = lap_times[i] - lap_times[i-1]
print("lap times:",lap_times, "violation: ",np.sum(violation_total), "mean_dev", np.mean(fval_history))
