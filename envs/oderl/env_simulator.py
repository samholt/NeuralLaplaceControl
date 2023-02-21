import torch, os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cpu'
torch.set_default_dtype(torch.float32)

import envs
import ctrl.ctrl as base
from ctrl import utils
from tqdm import tqdm

################## environment and dataset ##################
dt      = 0.1 		# mean time difference between observations
noise   = 0.0 		# observation noise std
ts_grid = 'fixed' 	# the distribution for the observation time differences: ['fixed','uniform','exp']
# ENV_CLS = envs.CTCartpole # [CTPendulum, CTCartpole, CTAcrobot]
ENV_CLS = envs.CTAcrobot # [CTPendulum, CTCartpole, CTAcrobot]
env = ENV_CLS(dt=dt, obs_trans=True, device=device, obs_noise=noise, ts_grid=ts_grid, solver='euler')
# D = utils.collect_data(env, H=5.0, N=env.N0)


################## model ##################
dynamics = 'enode'   		# ensemble of neural ODEs
# dynamics = 'benode' 		# batch ensemble of neural ODEs
# dynamics = 'ibnode'		# implicit BNN ODEs
# dynamics = 'pets'	   		# PETS
# dynamics = 'deep_pilco' 	# deep PILCO
n_ens       = 5				# ensemble size
nl_f        = 3				# number of hidden layers in the differential function
nn_f        = 200			# number of hidden neurons in each hidden layer of f
act_f       = 'elu'			# activation of f (should be smooth)
dropout_f   = 0.05			# dropout parameter (needed only for deep pilco)
learn_sigma = False			# whether to learn the observation noise or keep it fixed
nl_g        = 2				# number of hidden layers in the policy function
nn_g        = 200			# number of hidden neurons in each hidden layer of g
act_g       = 'relu'		# activation of g
nl_V        = 2				# number of hidden layers in the state-value function
nn_V        = 200			# number of hidden neurons in each hidden layer of V
act_V       = 'tanh'		# activation of V (should be smooth)

ctrl = base.CTRL(env, dynamics, n_ens=n_ens, learn_sigma=learn_sigma,
                 nl_f=nl_f, nn_f=nn_f, act_f=act_f, dropout_f=dropout_f,
                 nl_g=nl_g, nn_g=nn_g, act_g=act_g, 
                 nl_V=nl_V, nn_V=nn_V, act_V=act_V).to(device)

print('Env dt={:.3f}\nObservation noise={:.3f}\nTime increments={:s}'.\
      format(env.dt,env.obs_noise,str(env.ts_grid)))

import imageio
import PIL.Image
import pyvirtualdisplay
display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()




j = 0
filename = f'./logs/videos/out_video_{j}.mp4'
fps = int(1/0.05)
state = env.reset()
done = False

def g(state, t):
      return torch.from_numpy(env.action_space.sample())

with imageio.get_writer(filename, fps=fps) as video:
      for iter in tqdm(range(100)):
            # action = env.action_space.sample()
            returns = env.integrate_system(2, g, s0=state, return_states=True)
            state = returns[-1][-1]
            env.set_state_(state)
            # state_next, reward, done, info = env.integrate_system()
            video.append_data(env.render(mode='rgb_array', last_act=returns[1][-1]))
env.close()


# ################## learning ##################
# utils.plot_model(ctrl, D, L=30, H=2.0, rep_buf=10, fname=ctrl.name+'-train.png')
# utils.plot_test( ctrl, D, L=30, H=2.5, N=5, fname=ctrl.name+'-test.png')

# utils.train_loop(ctrl, D, ctrl.name, 50, L=30, H=2.0)

# ctrl.save(D=D,fname=ctrl.name)				# save the model & dataset
# ctrl_,D_ = base.CTRL.load(env, f'{fname}')	# load the model & dataset



















