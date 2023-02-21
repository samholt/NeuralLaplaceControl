import torch
from torch import nn, Tensor
import sys
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from functools import partial
from time import time
from matplotlib import cm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from time import time, strftime
from tqdm import tqdm
import logging
from copy import deepcopy
import pandas as pd
import random
import seaborn as sn
from matplotlib.pyplot import loglog

pd.set_option('mode.chained_assignment', None)
# SCALE = 7
SCALE = 13
# HEIGHT_SCALE = 1.5
HEIGHT_SCALE = 0.5
sn.set(rc={'figure.figsize': (SCALE, int(HEIGHT_SCALE * SCALE))})
# sn.set(font_scale=1.4)
sn.set(font_scale=2.0)
sn.set_style(style='white')
# sn.color_palette("tab10")
sn.color_palette("colorblind")
# plt.style.use('tableau-colorblind10')


# LEGEND_Y_CORD = -0.70  # * (HEIGHT_SCALE / 2.0)
LEGEND_Y_CORD = -0.75  # * (HEIGHT_SCALE / 2.0)
SUBPLOT_ADJUST = 1 / HEIGHT_SCALE  # -(0.05 + LEGEND_Y_CORD)
LEGEND_X_CORD = 0.45

# plt.gcf().subplots_adjust(bottom=(1-1/HEIGHT_SCALE), left=0.15, top=0.99)
plt.gcf().subplots_adjust(bottom=0.40, left=0.2, top=0.95)
# LINE_WIDTH = 3

PLOT_FROM_CACHE = False
PLOT_SAFTEY_MARGIN = 1.25

N = 3  # Siginficant Figures for Results
DP = 5

np.random.seed(999)
torch.random.manual_seed(999)

def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False

def string_to_float_dict(d):
    return {k: float(v) if is_float(v) else v for k, v in d.items()}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X_METRIC='nevals'
# Y_METRIC='nmse_test'
Y_METRIC='nmse_train'
# Y_METRIC='r_best'
import ast

import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, 2*h # m-h, m+h

def confidence_interval(prob, n):
    return 1.96 * np.sqrt( (prob * (1 - prob)) / n)

N = 5 # Significant figure

LOG_PATH = "./process_results/files/main_table_results.txt"

GENERATE_FIGS = False

HEADINGS = ['val_loss', 'train_loss', 'best_val_loss', 'total_reward', 'delay', 'model_name', 'seed', 'planner']
HEADINGS_NEW = ['val_loss', 'train_loss', 'best_val_loss', 'total_reward', 'delay', 'model_name', 'seed', 'planner', 'model_env']
ENVS = ['oderl-pendulum', 'oderl-cartpole', 'oderl-acrobot']
env_inx = 0

name_map = {'delta_t_rnn+mpc' : '$\Delta t-$RNN',
            'latent_ode+mpc' : 'Latent-ODE',
            'nl+mpc' : 'NLC \textbf{(Ours)}',
            'node+mpc' : 'NODE',
            'oracle+mpc' : 'Oracle',
            'random+mpc' : 'Random'}

custom_method_order = {'delta_t_rnn+mpc' : 2,
                        'latent_ode+mpc' : 3,
                        'nl+mpc' : 5,
                        'node+mpc' : 4,
                        'oracle+mpc' : 1,
                        'random+mpc' : 0}

def name_mapper(name):
    return name_map[name]

NORMALIZE = True
CHANGE_COLUMN_HEADINGS = True
if __name__ == '__main__':
    with open(LOG_PATH) as f:
        lines = f.readlines()
    
    # datasets = {}
    pd_l = []
    df_tmp = [] # Drop last entry if not completed
    delay = None
    training = False
    lines_to_skip = 5
    lines_seen = 0
    delay = 0

    for line in tqdm(lines):
        if '[Model Completed evaluation mppi] {' in line and not training:
            result_dict = line.split('[Model Completed evaluation mppi] ')[1].strip()
            result_dict = result_dict.replace('nan', '\'nan\'')
            result_dict = ast.literal_eval(result_dict)
            pd_l.append(result_dict)
        if '[Model Completed evaluation q] {' in line and not training:
            result_dict = line.split('[Model Completed evaluation q] ')[1].strip()
            result_dict = result_dict.replace('nan', '\'nan\'')
            result_dict = ast.literal_eval(result_dict)
            pd_l.append(result_dict)

    dfm = pd.DataFrame(pd_l)
    dfm[['total_reward', 'delay', 'seed']] = dfm[['total_reward', 'delay', 'seed']].apply(pd.to_numeric, errors='coerce')
    # dfm[['val_loss', 'train_loss', 'best_val_loss', 'total_reward', 'delay', 'seed']] = dfm[['val_loss', 'train_loss', 'best_val_loss', 'total_reward', 'delay', 'seed']].apply(pd.to_numeric, errors='coerce')
    dfm['name'] = dfm['model_name'] + '+' + dfm['planner']
    t = dfm.groupby(['delay', 'env_name', 'name', 'seed']).agg('mean')['total_reward']
    
    delay_results = {}
    finals_t = []
    for delay in [d for d in dfm['delay'].unique() if d >= 1]:
        b = t.unstack(level=0)[delay]
        if NORMALIZE:
            # if delay == 1:
            #     print('')
            best_policy = b.unstack(level=-1).mean(1).unstack()['oracle+mpc']
            # best_policy = b.unstack(level=-1).mean(1).unstack().max(1)
            random_policy = b.unstack(level=-1).mean(1).unstack()['random+mpc']
            # random_policy = b.unstack(level=-1).mean(1).unstack().min(1)
        bi = b.unstack()
        delay_l = []
        # for env_name in b.unstack(level=0).columns:
        for env_name in ['oderl-cartpole', 'oderl-pendulum', 'oderl-acrobot']:
            if NORMALIZE:
                vals = (b.unstack(level=0)[env_name] - random_policy[env_name]) / (best_policy[env_name] - random_policy[env_name])
                vm = vals.unstack().mean(1) * 100.0
                vstd = vals.unstack().std(1) * 100.0
                vstd[vm<0] = 0
                vm[vm < 0] = 0
            else:
                vals = b.unstack(level=0)[env_name]
                vm = vals.unstack().mean(1)
                vstd = vals.unstack().std(1)
            res = vm.round(2).astype('string') + '$\pm$' + vstd.round(2).astype('string')
            res.name = env_name
            delay_l.append(res)
        final = pd.concat(delay_l, axis=1).transpose()
        final.index = final.index + f'_d={delay}'
        # if delay != 0:
        finals_t.append(final.transpose())
        # print(f'DELAY: {delay}')
        # str_p = final.to_latex(escape=False).replace('\\textbackslash', '\\')
        str_p = final.to_latex(escape=False)
        str_p = str_p.replace('<NA>', 'NA')
        # print(str_p)
        # print('')
        delay_results[delay] = final
    final_df = pd.concat(finals_t, axis=1)
    final_df = final_df[['+mpc' in s for s in final_df.index]]
    final_df = final_df.drop('rnn+mpc', errors='ignore')
    final_df = final_df.sort_values(by=['name'], key=lambda x: x.map(custom_method_order))
    final_df.index = final_df.index.map(name_mapper)
    str_p = final_df.to_latex(escape=False)
    str_p = str_p.replace('<NA>', 'NA')
    if CHANGE_COLUMN_HEADINGS:
        lines = str_p.split('\n')
        lines[0] = r'\begin{tabular}{c|ccc|ccc|ccc}'
        lines[2] = r'                                & \multicolumn{3}{c}{Action Delay~$\tau=\bar{\Delta}$ s}      & \multicolumn{3}{c}{Action Delay~$\tau=2\bar{\Delta}$ s}      &  \multicolumn{3}{c}{Action Delay~$\tau=3\bar{\Delta}$ s}               \\'
        lines[3] = r'        Dynamics Model                  & Cartpole & Pendulum & Acrobot                   & Cartpole & Pendulum & Acrobot                     & Cartpole & Pendulum & Acrobot \\        '
        lines.insert(-4, r'\midrule')
        str_p = '\n'.join(lines)
    print(str_p)
    print('')