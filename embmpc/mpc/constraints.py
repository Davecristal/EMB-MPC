
import numpy as np


def Boundary(x0, track, eps=0.01):
	theta = track.xy_to_param(x0[0], x0[1])
	
	x_, y_ = track.param_to_xy(theta+eps)
	_x, _y = track.param_to_xy(theta-eps)
	x, y = track.param_to_xy(theta)
	
	_x_ = np.array([x,y])



	norm = np.sqrt((y_-_y)**2 + (x_-_x)**2)

	width = track.track_width/2					
	xin = x - width*(y_-_y)/norm				
	yin = y + width*(x_-_x)/norm				
	Ain = np.array([(y_-_y), -(x_-_x)])		
	bin = (y_-_y)*xin - (x_-_x)*yin


	width = -track.track_width/2				
	xout = x - width*(y_-_y)/norm				
	yout = y + width*(x_-_x)/norm				

	Aout = np.array([(y_-_y), -(x_-_x)])		

	bout = (y_-_y)*xout - (x_-_x)*yout




	A = np.concatenate([[-Ain],[Aout]])			
	b = np.concatenate([[-bin],[bout]]).reshape(-1,1)	
	return A, b									