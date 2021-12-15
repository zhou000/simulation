import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neural_exploration import *
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import load_trace
import sim_fixed_env as env
import copy
sns.set()

VIDEO_BIT_RATE = [344, 742, 1064, 2437, 4583, 6636]  # Kbps ************* VBR video bit_rate
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
# REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
BITS_IN_BYTE = 8.0
TOTAL_VIDEO_CHUNKS = 1250
BITRATE_LEVELS = 6
VMAF_SMOOTH_PENALTY = 1
VMAF_REBUF_PENALTY = 10000
VIDEO_SIZE_FILE = '../simulation_video_size/synthetic_video_size_BBB_ED/video_size_'



# VIDEO_VMAF_FILE = '../simulation_vmaf/BBB_ED_vmaf_1s/vmaf_'
# TEST_TRACES = sys.argv[1]
# LOG_FILE = sys.argv[2]
# alpha = float(sys.argv[3])
# VMAF_REBUF_PENALTY_1 = float(sys.argv[4])
# QUAITY_WEIGHT = float(sys.argv[5])

# debug:
VIDEO_VMAF_FILE = '../simulation_vmaf/BBB_ED_vmaf_1s/vmaf_'
LOG_FILE = '../test_results/log_NeuralUCB0'
TEST_TRACES = '../norway_bus_times1/'
# TEST_TRACES = '../norway_bus_times3/'
alpha = 5
VMAF_REBUF_PENALTY_1 = 100
QUAITY_WEIGHT = 1
# VMAF_REBUF_PENALTY_1 = 1
# QUAITY_WEIGHT = 3

S_INFO = 5  # throughput, bit_rate, buffer_size, chunk_size, penalty_sm
# X_INFO = 7 + COARSE_DIM * 2  # throughput*5, vmaf, buffer_size, chunk_size*COARSE_DIM , vmaf*COARSE_DIM
X_INFO = 9  # throughput*5, bit_rate, buffer_size, chunk_size, penalty_sm
X_D = X_INFO
X_LEN = 8
Y_LEN = 8  # take how many rewards in the past
S_LEN = 8  # take how many frames in the past
A_DIM = 6

video_vmaf = {}
video_sizes = {}

def get_chunk_vmaf(quality, index):
    if (index < 0 or index > TOTAL_VIDEO_CHUNKS):
        return 0
    vmaf_list = video_vmaf.get(quality)
    return vmaf_list[index]

def get_chunk_size(quality, index):
    if (index < 0 or index > TOTAL_VIDEO_CHUNKS):
        return 0
    size_list = video_sizes.get(quality)
    return size_list[index]


# Bandit settings
T = int(5e2)        # 5*10^2 = 500
# n_arms = 4
n_arms = 6
n_features = 16
noise_std = 0.1

confidence_scaling_factor = noise_std

# n_sim = 2
n_sim = 1

SEED = 42
np.random.seed(SEED)

# # LinUCB on linear rewards
# ### mean reward function
# a = np.random.randn(n_features)
# a /= np.linalg.norm(a, ord=2)
# h = lambda x: 10*np.dot(a, x)

# bandit = ContextualBandit(T, n_arms, n_features, h, noise_std=noise_std, seed=SEED)

# regrets = np.empty((n_sim, T))

# for i in range(n_sim):
#     bandit.reset_rewards()
#     model = LinUCB(bandit,
#                    reg_factor=1.0,
#                    delta=0.1,
#                    confidence_scaling_factor=confidence_scaling_factor,
#                   )
#     model.run()
#     regrets[i] = np.cumsum(model.regrets)




# Neural network settings
p = 0.2
hidden_size = 64
epochs = 100
train_every = 10
# train_every = 1
# train_every = 5
confidence_scaling_factor = 1.0
use_cuda = False

# hidden_size = 20
# n_layers = 2
# reg_factor = 1.0
# delta = 0.01
# confidence_scaling_factor = -1.0
# training_window = 100
# p = 0.0
# learning_rate = 0.01
# epochs = 1
# # train_every = 1
# train_every = 10
# throttle = 1
# use_cuda = False


# NeuralUCB on nonlinear rewards
a = np.random.randn(n_features)     # context
a /= np.linalg.norm(a, ord=2)
h = lambda x: np.cos(10*np.pi*np.dot(x, a))
# reward_func = lambda x: np.cos(10*np.pi*np.dot(x, a))

bandit = ContextualBandit(T, n_arms, n_features, h, noise_std=noise_std, seed=SEED)

regrets = np.empty((n_sim, T))

for i in range(n_sim):
    # bandit.reset_rewards()
    model = NeuralUCB(bandit,
                      hidden_size=hidden_size,
                      reg_factor=1.0,
                      delta=0.1,
                      confidence_scaling_factor=confidence_scaling_factor,
                      training_window=100,
                      p=p,
                      learning_rate=0.01,
                      epochs=epochs,
                      train_every=train_every,
                      use_cuda=use_cuda,
                     )
    model.run()
    regrets[i] = np.cumsum(model.regrets)



