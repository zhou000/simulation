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
LOG_FILE = '../test_results/log_GitLinUCB0'
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
T = TOTAL_VIDEO_CHUNKS + 1 # 1251
n_arms = A_DIM
n_features = X_INFO  # number of parameters
noise_std = 0.1

confidence_scaling_factor = noise_std


SEED = 42
np.random.seed(SEED)

reg_factor = 1.0
delta = 0.01
bound_theta = 1.0

# # LinUCB on nonlinear rewards
# ### mean reward function
# a = np.random.randn(n_features)
# a /= np.linalg.norm(a, ord=2)
# h = lambda x: 10 * np.dot(a, x)
#
# ctx = np.random.randn(T, n_arms, n_features)
# ctx /= np.repeat(np.linalg.norm(ctx, axis=-1, ord=2), n_features).reshape(T, n_arms, n_features)
#
# """Generate rewards for each arm and each round,
#         following the reward function h + Gaussian noise.
#         """
# rewards = np.array(
#     [
#         h(ctx[t, k]) + noise_std * np.random.randn()
#         for t, k in itertools.product(range(T), range(n_arms))
#     ]
# ).reshape(T, n_arms)
#
# # to be used only to compute regret, NOT by the algorithm itself
# best_rewards_oracle = np.max(rewards, axis=1)
# best_actions_oracle = np.argmax(rewards, axis=1)
#
# regrets = np.empty((n_sim, T))
# total_regret = 0




def main():

    for bitrate in range(BITRATE_LEVELS):
        video_vmaf[bitrate] = []
        with open(VIDEO_VMAF_FILE + str(bitrate)) as f:
            for line in f:
                video_vmaf[bitrate].append(float(line.split()[0]))

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'wb')
    # print ("log_file: ", log_file)
    # print ('all_file_names: ', all_file_names)

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


    # Aa = np.zeros((A_DIM, X_D, X_D))
    # Aa_inv = np.zeros((A_DIM, X_D, X_D))
    # ba = np.zeros((A_DIM, X_D, 1))
    # theta = np.zeros((A_DIM, X_D, 1))
    #
    # for i in range(A_DIM):
    #     Aa[i] = np.identity(X_D)
    #     Aa_inv[i] = np.identity(X_D)
    #     ba[i] = np.zeros((X_D, 1))
    #     theta[i] = np.zeros((X_D, 1))

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    X_batch = [np.zeros((X_INFO, X_LEN))]
    Y_batch = [np.zeros(Y_LEN)]

    video_count = 0

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real

        temp_env = copy.deepcopy(net_env)

        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, next_2_video_chunk_sizes, avg_chunk_sizes, \
        end_of_video, video_chunk_remain, video_chunk_num = net_env.get_video_chunk(bit_rate, last_bit_rate)
        # next_5_chunk_quality, buffer_remain_ratio, rush_flag= net_env.get_video_chunk(bit_rate, last_bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # the result of argmax(UCB)
        # max_a = post_data['lastquality']
        max_a = bit_rate

        # # -- linear reward --
        # # reward is video quality - rebuffer penalty - smoothness
        # reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
        #          - REBUF_PENALTY * rebuf \
        #          - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
        #                                    VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        #
        # penalty_sm = SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
        #                                      VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        # # penalty_sm = SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
        # #                                      VIDEO_BIT_RATE[last_bit_rate])
        # #
        # # penalty_sm = penalty_sm / float(np.max(VIDEO_BIT_RATE))     # normalized

        # # -- log scale reward --
        # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
        # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))
        #
        # reward = log_bit_rate \
        #          - REBUF_PENALTY * rebuf \
        #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

        # -- VMAF reward 1 --
        # reward = vmaf - VMAF_SMOOTH_PENALTY * np.abs(vmaf_i - vmaf_j) - VMAF_REBUF_PENALTY * rebuffering_time
        # where VMAF_REBUF_PENALTY = 100
        video_quality = QUAITY_WEIGHT * get_chunk_vmaf(bit_rate, video_chunk_num)

        penalty_rb = VMAF_REBUF_PENALTY_1 * rebuf

        penalty_sm = VMAF_SMOOTH_PENALTY * abs(
            get_chunk_vmaf(bit_rate, video_chunk_num) - get_chunk_vmaf(last_bit_rate, video_chunk_num - 1))

        reward = video_quality - penalty_rb - penalty_sm
        reward = reward / 100


        # # -- VMAF reward 2 --
        # # reward = vmaf^2 - VMAF_SMOOTH_PENALTY * (vmaf_i - vmaf_j)^2 - VMAF_REBUF_PENALTY * rebuffering_time
        # # where VMAF_REBUF_PENALTY = 100^2
        # video_quality = get_chunk_vmaf(bit_rate, video_chunk_num) * get_chunk_vmaf(bit_rate, video_chunk_num)
        #
        # penalty_rb = VMAF_REBUF_PENALTY * rebuf
        #
        # smoothness_dev = get_chunk_vmaf(bit_rate, video_chunk_num) - get_chunk_vmaf(last_bit_rate, video_chunk_num - 1)
        # penalty_sm = VMAF_SMOOTH_PENALTY * smoothness_dev * smoothness_dev
        #
        # reward = video_quality - penalty_rb - penalty_sm
        # reward = reward / M_IN_K    # normalize

        # regret = 0
        # if video_chunk_num > 0:
        #     test_rewards = []
        #     for i in range(BITRATE_LEVELS):
        #         test_bitrate = i
        #         # print ("the id of temp_env: ", id(temp_env))
        #         test_env = copy.deepcopy(temp_env)
        #         # print ("the id of test_env: ", id(test_env))
        #         test_delay, test_sleep_time, test_buffer_size, test_rebuf, \
        #         test_video_chunk_size, test_next_video_chunk_sizes, test_next_2_video_chunk_sizes, test_avg_chunk_sizes, \
        #         test_end_of_video, test_video_chunk_remain, test_video_chunk_num = test_env.get_video_chunk(test_bitrate, last_bit_rate)
        #
        #         test_video_quality = QUAITY_WEIGHT * get_chunk_vmaf(test_bitrate, video_chunk_num)
        #
        #         test_penalty_rb = VMAF_REBUF_PENALTY_1 * test_rebuf
        #
        #         test_penalty_sm = VMAF_SMOOTH_PENALTY * abs(
        #             get_chunk_vmaf(test_bitrate, video_chunk_num) - get_chunk_vmaf(last_bit_rate, video_chunk_num - 1))
        #
        #         test_reward = test_video_quality - test_penalty_rb - test_penalty_sm
        #         test_reward = test_reward / 100
        #         test_rewards.append(test_reward)
        #
        #     oracle_A = np.argmax(test_rewards)
        #     optimal = test_rewards[oracle_A]
        #     regret = optimal - reward



        r_batch.append(reward)

        last_quality = last_bit_rate
        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        # log_file.write(str(time_stamp / M_IN_K) + '\t' +
        str_log = (str(video_chunk_num) + '\t' +
                       # str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(video_quality) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       # str(last_quality) + '\t' +
                       # str(buffer_remain_ratio) + '\t' +
                       # str(rush_flag) + '\t' +
                       str(reward) + '\n')
        str_log = str_log.encode()
        log_file.write(str_log)
        # log_file.write(str(video_chunk_num) + '\t' +
        #                # str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
        #                str(video_quality) + '\t' +
        #                str(buffer_size) + '\t' +
        #                str(rebuf) + '\t' +
        #                str(video_chunk_size) + '\t' +
        #                str(delay) + '\t' +
        #                # str(last_quality) + '\t' +
        #                # str(buffer_remain_ratio) + '\t' +
        #                # str(rush_flag) + '\t' +
        #                str(reward) + '\n')
        #                # str(reward) + '\t' +
        #                # str(regret) + '\n')
        log_file.flush()


        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        if len(X_batch) == 0:
            x_context = [np.zeros((X_INFO, X_LEN))]
        else:
            x_context = np.array(X_batch[-1], copy=True)

        if len(Y_batch) == 0:
            y_reward = [np.zeros(Y_LEN)]
        else:
            y_reward = np.array(Y_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)
        x_context = np.roll(x_context, -1, axis=1)
        y_reward = np.roll(y_reward, -1)

        # state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))    # normalized to 0-1
        # state[0, -1] = VIDEO_BIT_RATE[bit_rate] / M_IN_K  # /1000   last quality
        # state[0, -1] = video_quality / M_IN_K  # /1000
        state[0, -1] = video_quality / 100
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        # state[1, -1] = buffer_size  # 1 sec
        # state[1, -1] = post_data['buffer'] / BUFFER_NORM_FACTOR  # /10s
        state[2, -1] = video_chunk_size / M_IN_K  # kilo byte
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        # state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        # state[3, -1] = float(video_chunk_size) / float(video_chunk_fetch_time) / M_IN_K  # kilo byte / ms
        # state[4, -1] = penalty_sm / M_IN_K  # /1000
        state[4, -1] = penalty_sm / 100



        y_reward[-1] = reward  # linear reward

        if video_chunk_num > 0:
            # x_context[:, -1] = max_context
            #
            # x = x_context[:, -1]
            # x = np.array(x).reshape((X_D, 1))
            #
            # # update the Aa, ..., theta
            # Aa[max_a] += np.outer(x, x)
            # Aa_inv[max_a] = np.linalg.inv(Aa[max_a])
            # ba[max_a] += reward * x
            # theta[max_a] = Aa_inv[max_a].dot(ba[max_a])

            reward_t = reward

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
            b[action] += context[action] * reward_t

            # # update exploration indicator A_inv
            # model.update_A_inv()
            A_inv[action] = inv_sherman_morrison(
                grad_approx[action], A_inv[action]
            )


        # *************************************************
        context = np.zeros((n_arms, n_features))
        # the desicion for next video chunk:
        UCB_A = []
        # BB_buffer = float(post_data['buffer'])
        # bitrate_last_chunk = VIDEO_BIT_RATE[post_data['lastquality']]
        # bitrate_last_chunk = VIDEO_BIT_RATE[bit_rate]
        vmaf_last_chunk = get_chunk_vmaf(bit_rate, video_chunk_num)
        future_contexts = []

        # the context of different actions for the next video chunk:
        for i in range(0, A_DIM):
            # bitrate_i = VIDEO_BIT_RATE[i] / float(np.max(VIDEO_BIT_RATE))    # normalized to 0-1
            # bitrate_i = VIDEO_BIT_RATE[i] / M_IN_K  # /1000
            video_vmaf_i = get_chunk_vmaf(i, video_chunk_num + 1)
            video_quality_i = QUAITY_WEIGHT * video_vmaf_i / 100  # vmaf1
            # video_quality_i = video_vmaf_i * video_vmaf_i / M_IN_K  # /1000     # vmaf2
            # start_buffer = float(post_data['buffer']) / BUFFER_NORM_FACTOR  # /10s
            start_buffer = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            # start_buffer = buffer_size   # 1 sec
            # video_chunk_size_i = get_chunk_size(i, video_chunk_num) / M_IN_K / M_IN_K  # M byte
            video_chunk_size_i = next_video_chunk_sizes[i] / M_IN_K / M_IN_K  # M byte
            throughput_1 = state[3, -1] * BUFFER_NORM_FACTOR
            throughput_2 = state[3, -2] * BUFFER_NORM_FACTOR
            throughput_3 = state[3, -3] * BUFFER_NORM_FACTOR
            throughput_4 = state[3, -4] * BUFFER_NORM_FACTOR
            throughput_5 = state[3, -5] * BUFFER_NORM_FACTOR
            smoothness_dev_i = abs(video_vmaf_i - vmaf_last_chunk)
            penalty_sm_i = SMOOTH_PENALTY * smoothness_dev_i / 100
            # penalty_sm_i = SMOOTH_PENALTY * smoothness_dev_i * smoothness_dev_i / M_IN_K

            cx_i = np.array(
                [video_quality_i, start_buffer, video_chunk_size_i, throughput_1, throughput_2, throughput_3,
                 throughput_4, throughput_5, penalty_sm_i])
            context[i] = cx_i
            future_contexts.append(cx_i)

            # x = cx_i
            # x = np.array(x).reshape((X_D, 1))
            # x_t = np.transpose(x)
            # index = i
            # UCB_i = np.matmul(np.transpose(theta[index]), x) + alpha * np.sqrt(
            #     np.matmul(x_t, Aa_inv[index].dot(x)))
            # UCB_A.append(UCB_i)

        """LinUCB confidence interval multiplier.
            """
        # bound_features = np.max(np.linalg.norm(bandit.features, ord=2, axis=-1))
        bound_features = np.max(np.linalg.norm(context, ord=2, axis=-1))
        # bound_features = np.max(np.linalg.norm(ctx, ord=2, axis=-1))
        confidence_multiplier = confidence_scaling_factor * np.sqrt(n_features * np.log(
            1 + video_chunk_num * bound_features ** 2 / (reg_factor * n_features)) + 2 * np.log(
            1 / delta)) + np.sqrt(reg_factor) * bound_theta
        # confidence_multiplier = 0.1

        """LinUCB confidence interval multiplier.
                """
        grad_approx = context
        # UCB exploration bonus
        exploration_bonus[video_chunk_num] = np.array(
            [
                confidence_multiplier * np.sqrt(
                    np.dot(grad_approx[a], np.dot(A_inv[a], grad_approx[a].T))) for a in range(n_arms)
            ]
        )
        # # update reward prediction mu_hat
        # self.predict(context)
        mu_hat[video_chunk_num] = np.array(
            [
                # np.dot(self.bandit.features[self.iteration, a], self.theta[a]) for a in self.bandit.arms
                np.dot(context[a], theta[a]) for a in range(n_arms)
            ]
        )
        # estimated combined bound for reward
        upper_confidence_bounds[video_chunk_num] = mu_hat[video_chunk_num] + exploration_bonus[video_chunk_num]

        # # pick action with the highest boosted estimated reward
        # model.action = model.sample_action()
        action = np.argmax(upper_confidence_bounds[video_chunk_num]).astype('int')
        # model.actions[t] = model.action
        actions[video_chunk_num] = action

        max_A = action
        max_context = future_contexts[action]
        idx = max_A

        # max_A = np.argmax(UCB_A)
        # max_context = future_contexts[max_A]
        # idx = max_A

        bit_rate = idx

        s_batch.append(state)
        X_batch.append(x_context)
        Y_batch.append(y_reward)

        # entropy_record.append(a3c.compute_entropy(action_prob[0]))

        if end_of_video:
            log_file.write(('\n').encode())
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]
            del X_batch[:]
            del Y_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            X_batch.append(np.zeros((X_INFO, X_LEN)))
            Y_batch.append(np.zeros(Y_LEN))
            entropy_record = []

            video_count += 1

            if video_count >= len(all_file_names):
                break

            # another trace
            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'wb')

            # Aa = np.zeros((A_DIM, X_D, X_D))
            # Aa_inv = np.zeros((A_DIM, X_D, X_D))
            # ba = np.zeros((A_DIM, X_D, 1))
            # theta = np.zeros((A_DIM, X_D, 1))
            #
            # for i in range(A_DIM):
            #     Aa[i] = np.identity(X_D)
            #     Aa_inv[i] = np.identity(X_D)
            #     ba[i] = np.zeros((X_D, 1))
            #     theta[i] = np.zeros((X_D, 1))

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


if __name__ == '__main__':
    main()






#
#
#
# for t in range(T):
#     # context for each arm in round t
#     context = ctx[t]
#
#     # # update confidence of all arms based on observed features at time t
#     # model.update_confidence_bounds(context)
#
#     """LinUCB confidence interval multiplier.
#     """
#     # bound_features = np.max(np.linalg.norm(bandit.features, ord=2, axis=-1))
#     bound_features = np.max(np.linalg.norm(context, ord=2, axis=-1))
#     # bound_features = np.max(np.linalg.norm(ctx, ord=2, axis=-1))
#     confidence_multiplier = confidence_scaling_factor * np.sqrt(n_features * np.log(
#         1 + t * bound_features ** 2 / (reg_factor * n_features)) + 2 * np.log(
#         1 / delta)) + np.sqrt(reg_factor) * bound_theta
#
#     """LinUCB confidence interval multiplier.
#             """
#     grad_approx = context
#     # UCB exploration bonus
#     exploration_bonus[t] = np.array(
#         [
#             confidence_multiplier * np.sqrt(
#                 np.dot(grad_approx[a], np.dot(A_inv[a], grad_approx[a].T))) for a in range(n_arms)
#         ]
#     )
#     # # update reward prediction mu_hat
#     # self.predict(context)
#     mu_hat[t] = np.array(
#         [
#             # np.dot(self.bandit.features[self.iteration, a], self.theta[a]) for a in self.bandit.arms
#             np.dot(context[a], theta[a]) for a in range(n_arms)
#         ]
#     )
#     # estimated combined bound for reward
#     upper_confidence_bounds[t] = mu_hat[t] + exploration_bonus[t]
#
#     # # pick action with the highest boosted estimated reward
#     # model.action = model.sample_action()
#     action = np.argmax(upper_confidence_bounds[t]).astype('int')
#     # model.actions[t] = model.action
#     actions[t] = action
#
#     reward_t = rewards[t, action]
#
#     # # update approximator
#     # if t % model.train_every == 0:
#     #     model.train(context)
#     """Update linear predictor theta.
#     """
#     theta = np.array(
#         [
#             np.matmul(A_inv[a], b[a]) for a in range(n_arms)
#         ]
#     )
#     # b[action] += context[action]*rewards[t, action]
#     b[action] += context[action]*reward_t
#
#     # # update exploration indicator A_inv
#     # model.update_A_inv()
#     A_inv[action] = inv_sherman_morrison(
#         grad_approx[action], A_inv[action]
#     )
#
#     # compute regret
#     regrets[t] = best_rewards_oracle[t] - rewards[t, action]
#     # # increment counter
#     # model.iteration += 1
#
#     total_regret += regrets[t]
# print(total_regret)



