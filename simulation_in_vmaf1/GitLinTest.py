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
import itertools

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

def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = np.dot(A_inv, u)
    A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))
    return A_inv

# Bandit settings
T = int(5e2)  # 5*10^2 = 500
# n_arms = 4
n_arms = 6
n_features = 16  # number of parameters
noise_std = 0.1

confidence_scaling_factor = noise_std

# n_sim = 2
n_sim = 1

SEED = 42
np.random.seed(SEED)

# # Neural network settings
# p = 0.2
# hidden_size = 64
# epochs = 100
# train_every = 10
# confidence_scaling_factor = 1.0
# use_cuda = False


# LinUCB on nonlinear rewards
### mean reward function
a = np.random.randn(n_features)
a /= np.linalg.norm(a, ord=2)
h = lambda x: 10 * np.dot(a, x)

ctx = np.random.randn(T, n_arms, n_features)
ctx /= np.repeat(np.linalg.norm(ctx, axis=-1, ord=2), n_features).reshape(T, n_arms, n_features)

# bandit = ContextualBandit(T, ctx, n_arms, n_features, h, noise_std=noise_std, seed=SEED)
bandit = ContextualBandit(T, n_arms, n_features, h, noise_std=noise_std, seed=SEED)

"""Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """
rewards = np.array(
    [
        h(ctx[t, k]) + noise_std * np.random.randn()
        for t, k in itertools.product(range(T), range(n_arms))
    ]
).reshape(T, n_arms)

# to be used only to compute regret, NOT by the algorithm itself
best_rewards_oracle = np.max(rewards, axis=1)
best_actions_oracle = np.argmax(rewards, axis=1)


regrets = np.empty((n_sim, T))
total_regret = 0


for i in range(n_sim):
    # bandit.reset_rewards()

    delta = 0.01
    bound_theta = 1.0

    """Initialize upper confidence bounds and related quantities.
            """
    exploration_bonus = np.empty((T, n_arms))
    mu_hat = np.empty((T, n_arms))
    upper_confidence_bounds = np.ones((T, n_arms))

    """Initialize regrets.
            """
    regrets = np.empty(T)

    """Initialize cache of actions.
            """
    actions = np.empty(T).astype('int')

    """Initialize n_arms square matrices representing the inverses
            of exploration bonus matrices.
            """
    reg_factor = 1.0
    approximator_dim = n_features
    A_inv = np.array(
        [
            np.eye(approximator_dim) / reg_factor for _ in range(n_arms)
        ]
    )

    """Initialize the gradient of the approximator w.r.t its parameters.
            """
    grad_approx = np.zeros((n_arms, approximator_dim))

    # randomly initialize linear predictors within their bounds
    theta = np.random.uniform(-1, 1, (n_arms, n_features)) * bound_theta

    # initialize reward-weighted features sum at zero
    b = np.zeros((n_arms, n_features))

    for t in range(T):
        # context for each arm in round t
        context = ctx[t]

        # # update confidence of all arms based on observed features at time t
        # model.update_confidence_bounds(context)

        """LinUCB confidence interval multiplier.
        """
        # bound_features = np.max(np.linalg.norm(bandit.features, ord=2, axis=-1))
        bound_features = np.max(np.linalg.norm(context, ord=2, axis=-1))
        # bound_features = np.max(np.linalg.norm(ctx, ord=2, axis=-1))
        confidence_multiplier = confidence_scaling_factor * np.sqrt(n_features * np.log(
            1 + t * bound_features ** 2 / (reg_factor * n_features)) + 2 * np.log(
            1 / delta)) + np.sqrt(reg_factor) * bound_theta

        """LinUCB confidence interval multiplier.
                """
        grad_approx = context
        # UCB exploration bonus
        exploration_bonus[t] = np.array(
            [
                confidence_multiplier * np.sqrt(
                    np.dot(grad_approx[a], np.dot(A_inv[a], grad_approx[a].T))) for a in range(n_arms)
            ]
        )
        # # update reward prediction mu_hat
        # self.predict(context)
        mu_hat[t] = np.array(
            [
                # np.dot(self.bandit.features[self.iteration, a], self.theta[a]) for a in self.bandit.arms
                np.dot(context[a], theta[a]) for a in range(n_arms)
            ]
        )
        # estimated combined bound for reward
        upper_confidence_bounds[t] = mu_hat[t] + exploration_bonus[t]

        # # pick action with the highest boosted estimated reward
        # model.action = model.sample_action()
        action = np.argmax(upper_confidence_bounds[t]).astype('int')
        # model.actions[t] = model.action
        actions[t] = action

        reward_t = rewards[t, action]

        # # update approximator
        # if t % model.train_every == 0:
        #     model.train(context)
        """Update linear predictor theta.
        """
        theta = np.array(
            [
                np.matmul(A_inv[a], b[a]) for a in range(n_arms)
            ]
        )
        # b[action] += context[action]*rewards[t, action]
        b[action] += context[action]*reward_t

        # # update exploration indicator A_inv
        # model.update_A_inv()
        A_inv[action] = inv_sherman_morrison(
            grad_approx[action], A_inv[action]
        )

        # compute regret
        regrets[t] = best_rewards_oracle[t] - rewards[t, action]
        # # increment counter
        # model.iteration += 1

        total_regret += regrets[t]
    print(total_regret)

    # regrets[i] = np.cumsum(regrets)

