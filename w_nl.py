import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torchlaplace import laplace_reconstruct
import time
import logging
from config import get_config, CME_reconstruction_terms
from copy import deepcopy
from overlay import setup_logger, create_env, generate_irregular_data_delay_time_multi, get_val_loss_delay_time_multi, get_val_loss_delay_precomputed, compute_val_data_delay
# from oracle import cartpole_dynamics
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger()

class ReverseGRUEncoder(nn.Module):
    # Encodes observed trajectory into latent vector
    def __init__(self, dimension_in, latent_dim, hidden_units, encode_obs_time=True):
        super(ReverseGRUEncoder, self).__init__()
        self.encode_obs_time = encode_obs_time
        if self.encode_obs_time:
            dimension_in += 1
        self.gru = nn.GRU(dimension_in, hidden_units, 2, batch_first=True)
        self.linear_out = nn.Linear(hidden_units, latent_dim)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, observed_data):
        trajs_to_encode = observed_data  # (batch_size, t_observed_dim, observed_dim)
        reversed_trajs_to_encode = torch.flip(trajs_to_encode, (1,))
        out, _ = self.gru(reversed_trajs_to_encode)
        return self.linear_out(out[:, -1, :])


class LaplaceRepresentationFunc(nn.Module):
    # SphereSurfaceModel : C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
    def __init__(self, s_dim, output_dim, latent_dim, hidden_units=64):
        super(LaplaceRepresentationFunc, self).__init__()
        self.s_dim = s_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(s_dim * 2 + latent_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, (s_dim) * 2 * output_dim),
        )

        for m in self.linear_tanh_stack.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        phi_max = torch.pi / 2.0
        self.phi_scale = phi_max - -torch.pi / 2.0

    def forward(self, i):
        out = self.linear_tanh_stack(i.view(-1, self.s_dim * 2 + self.latent_dim)).view(
            -1, 2 * self.output_dim, self.s_dim
        )
        theta = nn.Tanh()(out[:, : self.output_dim, :]) * torch.pi  # From - pi to + pi
        phi = (
            nn.Tanh()(out[:, self.output_dim :, :]) * self.phi_scale / 2.0
            - torch.pi / 2.0
            + self.phi_scale / 2.0
        )  # Form -pi / 2 to + pi / 2
        return theta, phi

class NeuralLaplaceModel(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 latent_dim,
                 hidden_units=64,
                 s_recon_terms=33,
                 ilt_algorithm="fourier",
                 encode_obs_time=False,
                 state_mean=None,
                 state_std=None,
                 action_mean=None,
                 action_std=None,
                 normalize=False,
                 normalize_time=False,
                 dt=0.05):
        super(NeuralLaplaceModel, self).__init__()
        self.ilt_algorithm = ilt_algorithm
        if ilt_algorithm == "cme":
            terms = CME_reconstruction_terms()
            s_recon_terms = terms[np.argmin(terms < s_recon_terms)-2]
        action_encoder_latent_dim = 2
        laplace_latent_dim = state_dim + action_encoder_latent_dim
        self.latent_dim = latent_dim
        self.action_encoder = ReverseGRUEncoder(
                action_dim,
                action_encoder_latent_dim,
                hidden_units // 2,
                # hidden_units,
                encode_obs_time=encode_obs_time,
            )
        self.laplace_rep_func = LaplaceRepresentationFunc(
            s_recon_terms, state_dim, laplace_latent_dim, hidden_units=hidden_units
            )
        self.encode_obs_time = encode_obs_time
        self.output_dim = state_dim
        self.normalize = normalize
        self.normalize_time = normalize_time
        self.s_recon_terms = s_recon_terms
        # if self.encode_obs_time:
        #     state_mean = np.concatenate((state_mean,np.array([1])))
        #     state_std = np.concatenate((state_std,np.array([1])))
        #     action_mean = np.concatenate((action_mean,np.array([1])))
        #     action_std = np.concatenate((action_std,np.array([1])))
        self.register_buffer("state_mean", torch.tensor(state_mean))
        self.register_buffer("state_std", torch.tensor(state_std))
        self.register_buffer("action_mean", torch.tensor(action_mean))
        self.register_buffer("action_std", torch.tensor(action_std))
        self.register_buffer("dt", torch.tensor(dt))

    def forward(self, in_batch_obs, in_batch_action, ts_pred):
        # in_batch_action = in_batch_action[:,:1,:]
        if self.normalize:
            batch_obs = (in_batch_obs - self.state_mean) / self.state_std
            batch_action = (in_batch_action - self.action_mean) / self.action_std
            if self.normalize_time:
                ts_pred = (ts_pred / (self.dt*8.0))
                # ts_pred = (((ts_pred - self.dt) / (self.dt*4.0)) + 0.05)
            # batch_action = in_batch_action.view(-1, in_batch_action.shape[2])
            # batch_action = batch_action.view(*in_batch_action.shape)
        else:
            batch_obs = in_batch_obs
            batch_action = in_batch_action / 3.0
        # p_action = batch_action.view(batch_action.shape[0],batch_action.shape[-1])
        if len(batch_action.shape) == 2:
            batch_action = batch_action.unsqueeze(1)
        p_action = self.action_encoder(batch_action)
        sa_in = torch.cat((batch_obs, p_action), axis=1)
        # if len(sa_in.shape) == 2:
        #     sa_in = sa_in.unsqueeze(1)

        # p = self.obs_encoder(sa_in)
        # p = sa_in.squeeze()
        p = sa_in
        return torch.squeeze(laplace_reconstruct(
            self.laplace_rep_func, p, ts_pred, recon_dim=self.output_dim, ilt_algorithm=self.ilt_algorithm, ilt_reconstruction_terms=self.s_recon_terms
        ))

def load_replay_buffer(fn):
    offline_dataset = np.load(fn, allow_pickle=True).item()
    return offline_dataset

def get_nl(train_env_task,
            delay,
            config={},
            retrain=False,
            start_from_checkpoint=False,
            model_seed=0, # Trained best model for
            force_retrain=False,
            wandb=None,
            print_settings=True):
    model_name = 'nl'
    # model_name = 'NL-diff-continous'
    # model_saved_name = f'{model_name}_{train_env_task}_delay-{delay}_{model_seed}_updated.pt'
    model_saved_name = f'{model_name}_{train_env_task}_delay-{delay}_ts-grid-{config.ts_grid}_{model_seed}.pt'
    model_path = f'{config.saved_models_path}{model_saved_name}'
    env = create_env(train_env_task, ts_grid=config.ts_grid, dt=config.dt * config.train_dt_multiple)
    obs_state = env.reset()
    state_dim = obs_state.shape[0]
    action_dim = env.action_space.shape[0]
    # if not retrain:
    #     s0, a0, sn, ts = generate_irregular_data_delay_time_multi(train_env_task, env, samples_per_dim=2, rand=rand, delay=delay)
    # else:    
    #     s0, a0, sn, ts = generate_irregular_data_delay_time_multi(train_env_task, env, samples_per_dim=15, rand=rand, delay=delay)

    # state_mean = s0.mean(0).detach().cpu().numpy()
    # state_std = s0.std(0).detach().cpu().numpy()
    # action_mean = a0.mean().detach().cpu().numpy()
    # ACTION_HIGH = env.action_space.high[0]
    # action_std = np.array([ACTION_HIGH/2.0])
    
    if train_env_task == 'oderl-cartpole':
        state_mean = np.array([-0.0042523 ,  0.04628198,  0.0002319 , -0.00171167,  0.04176694])
        state_std = np.array([ 4.04281527, 11.56545697,  0.70878245,  0.70542503, 17.30071496])
        action_mean = np.array([0])
        ACTION_HIGH = env.action_space.high[0]
        action_std = np.array([ACTION_HIGH/2.0])
    else:
        raise NotImplementedError

    latent_dim = state_dim
    hidden_units=config.nl_hidden_units

    model = NeuralLaplaceModel(state_dim,
                                action_dim,
                                latent_dim,
                                hidden_units=hidden_units,
                                s_recon_terms=config.nl_s_recon_terms,
                                ilt_algorithm=config.nl_ilt_algorithm,
                                encode_obs_time=config.encode_obs_time,
                                state_mean=state_mean,
                                state_std=state_std,
                                action_mean=action_mean,
                                action_std=action_std,
                                normalize=config.normalize,
                                normalize_time=config.normalize_time,
                                ).to(device)
    model.double()
    model_number_of_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f'[Model] params={model_number_of_parameters}')
    loss_func = nn.MSELoss()

    if not force_retrain:
        logger.info(f'Trying to load : {model_path}')
        if not retrain and os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))
            return model.eval()
        elif not retrain:
            raise ValueError
        if start_from_checkpoint and os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))
    if print_settings:
        logger.info(f'[RUN SETTINGS]: {config}')
    if wandb is not None:
        wandb.config.update({f"{model_name}__number_of_parameters": model_number_of_parameters}, allow_val_change=True)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=0.00001)
    # optimizer = optim.Adam(model.parameters(), lr=1e-6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=10, cooldown=100)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_scheduler_step_size, gamma=config.lr_scheduler_gamma, verbose=True)
    loss_l = []
    eval_l = []
    model.train()
    cum_loss = 0
    iters = 0

    best_loss = float("inf")
    waiting = 0
    patience = float("inf")

    batch_size = config.training_batch_size
    train_start_time = time.perf_counter()
    elapsed_time_loss_l = []
    # vs0, va0, vsn, vts = compute_val_data_delay(train_env_task, env, delay=delay, dt=config.dt)
    for epoch_i in range(config.training_epochs):
        iters = 0
        cum_loss = 0
        t0 = time.perf_counter()

        s0, a0, sn, ts = generate_irregular_data_delay_time_multi(train_env_task,
                                                                     env,
                                                                     samples_per_dim=config.train_samples_per_dim,
                                                                     rand=config.rand_sample,
                                                                     delay=delay,
                                                                     encode_obs_time=config.encode_obs_time,
                                                                     action_buffer_size=config.action_buffer_size,
                                                                     reuse_state_actions_when_sampling_times=config.reuse_state_actions_when_sampling_times)
        s0, a0, sn, ts = s0.to(device), a0.to(device), sn.to(device), ts.to(device)
        permutation = torch.randperm(s0.size()[0])
        if int(s0.size()[0]/batch_size) < config.iters_per_log:
            config.update({'iters_per_log': int(s0.size()[0]/batch_size)}, allow_val_change=True)
        for iter_i in range(int(s0.size()[0]/batch_size)):
            optimizer.zero_grad()
            indices = permutation[iter_i*batch_size:iter_i*batch_size+batch_size]
            bs0, ba0, bsn, bts = s0[indices], a0[indices], sn[indices], ts[indices]
            pred_sd = model(bs0, ba0, bts)
            bsd = bsn - bs0
            loss = loss_func(pred_sd.squeeze(), bsd.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()
            cum_loss += loss.item()
            iters += 1
            # if iter_i % config.iters_per_log == 0 and not iter_i == 0:
            if iter_i % (config.iters_per_log - 1) == 0 and not iter_i == 0:
                val_loss = get_val_loss_delay_time_multi(model, train_env_task, env, delay=delay, dt=config.dt, encode_obs_time=config.encode_obs_time, action_buffer_size=config.action_buffer_size)
                # val_loss = get_val_loss_delay_precomputed(model, vs0, va0, vsn, vts)
                track_loss = cum_loss / iters
                elapsed_time = time.perf_counter() - train_start_time
                if config.sweep_mode and elapsed_time > config.end_training_after_seconds:
                    elapsed_time_loss_mean = np.array(elapsed_time_loss_l).mean()
                    logger.info(f'Ending training: elapsed_time_loss_mean: {elapsed_time_loss_mean}')
                    wandb.log({'elapsed_time_loss_mean': elapsed_time_loss_mean})
                    break
                elapsed_time_loss = elapsed_time * val_loss
                elapsed_time_loss_l.append(elapsed_time_loss)
                # logger.info(f'[epoch={epoch_i+1:04d}|iter={iter_i+1:04d}/{int(s0.size()[0]/batch_size):04d}] train_loss={track_loss} \t\t| val_loss={val_loss} \t\t| s/it={(time.perf_counter() - t0)/config.iters_per_log:.5f}')
                logger.info(f'[epoch={epoch_i+1:04d}|iter={iter_i+1:04d}/{int(s0.size()[0]/batch_size):04d}|t:{int(elapsed_time)}/{config.end_training_after_seconds if config.sweep_mode else 0}] train_loss={track_loss} \t\t| val_loss={val_loss} \t\t| elapsed_time_loss={elapsed_time_loss} \t\t| s/it={(time.perf_counter() - t0)/config.iters_per_log:.5f}')
                t0 = time.perf_counter()
                if wandb is not None:
                    wandb.log({"loss": track_loss, "epoch": epoch_i, "val_loss": val_loss, "model_name": model_name, "elapsed_time_loss": elapsed_time_loss})
                cum_loss = 0
                iters = 0

                # Early stopping procedure
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = deepcopy(model.state_dict())
                    torch.save(model.state_dict(), model_path)
                    waiting = 0
                elif waiting > patience:
                    break
                else:
                    waiting += 1
            if iter_i % (config.iters_per_evaluation - 1) == 0 and not iter_i == 0:
                evaluate_model(model, model_name, train_env_task, wandb, config)
        if config.sweep_mode and elapsed_time > config.end_training_after_seconds:
            break
        scheduler.step()
        loss_l.append(loss.item())

    logger.info(f'[Training Finished] model: {model_name} \t|[epoch={epoch_i+1:04d}|iter={iter_i+1:04d}/{int(s0.size()[0]/batch_size):04d}] train_loss={track_loss:.5f} \t| val_loss={val_loss:.5f} \t| s/it={(time.perf_counter() - t0)/config.iters_per_log:.5f}')
    evaluate_model(model, model_name, train_env_task, wandb, config)
    model.load_state_dict(best_model)
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    results = {'val_loss': val_loss, 'train_loss': loss.item(), 'best_val_loss': best_loss} # 'loss_l': np.array(loss_l)}
    return model.eval(), results

def evaluate_model(model, model_name, train_env_task, wandb, config):
    if config.sweep_mode:
        seed_all(0)
    from mppi_with_model import mppi_with_model_evaluate_single_step
    eval_result = mppi_with_model_evaluate_single_step(model_name=model_name,
                                        env_name=train_env_task,
                                        roll_outs=config.mppi_roll_outs,
                                        time_steps=config.mppi_time_steps,
                                        lambda_=config.mppi_lambda,
                                        sigma=config.mppi_sigma,
                                        dt=config.dt,
                                        action_delay=delay,
                                        encode_obs_time=config.encode_obs_time,
                                        config=config,
                                        model=model,
                                        # save_video=config.save_video,
                                        save_video=False,
                                        )
    total_reward =  eval_result['total_reward']
    logger.info(f'[Evaluation Result] Total reward {total_reward}')
    if wandb is not None:
        wandb.log({"total_reward": total_reward})

if __name__ == '__main__':
    import wandb, sys
    defaults = get_config()
    # defaults['training_epochs'] = 10000
    # defaults['train_samples_per_dim'] = 10
    # defaults['train_dt_multiple'] = 1

    # # Optomized hyper parameters
    # defaults['clip_grad_norm'] = 10
    # defaults['learning_rate'] = 1e-5
    # defaults['nl_hidden_units'] = 256
    # defaults['nl_s_recon_terms'] = 65
    # defaults['normalize_time'] = True

    # defaults['learning_rate'] = 0.00001
    # defaults['training_batch'] = 16
    # defaults['clip_grad_norm'] = 10
    # defaults['nl_hidden_units'] = 32
    # defaults['learning_rate'] = 0.001
    # defaults['nl_s_recon_terms'] = 33
    # defaults['nl_ilt_algorithm'] = "cme"
    # defaults['training_batch_size'] = 16
    # defaults['normalize_time'] = True
    # defaults['normalize'] = True

    # defaults['clip_grad_norm'] = 100
    # defaults['nl_hidden_units'] = 128
    # defaults['learning_rate'] = 1e-2
    # defaults['training_batch_size'] = 64
    # defaults['learning_rate'] = 0.0001
    # defaults['nl_s_recon_terms'] = 65
    # # defaults['nl_ilt_algorithm'] = "cme"
    # defaults['normalize_time'] = True
    # defaults['normalize'] = True

    # 11 model_seed - nicely trained

    wandb.init(config=defaults, project=defaults['wandb_project']) #, mode="disabled")
    config = wandb.config
    logger = setup_logger(__file__, log_folder=config.log_folder)
    from config import seed_all
    seed_all(0)
    logger.info('Training a new nl model')
    logger.info(get_nl('oderl-cartpole',
                        config=config,
                        retrain=True,
                        model_seed=config.model_seed,
                        start_from_checkpoint=True,
                        force_retrain=True,
                        wandb=wandb))
    wandb.finish()
    # wandb.close()



# Comments
# Can speed up by removing .double(), how we load data, set gradients to zero, and where any casting is done to device etc, create tensors directly