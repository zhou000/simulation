import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
# import tensorflow as tf
import load_trace
import sim_fixed_env as env
import copy

# VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
VIDEO_BIT_RATE = [344, 742, 1064, 2437, 4583, 6636]  # Kbps ************* VBR video bit_rate
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
# REBUF_PENALTY = 2.66  # 1 sec rebuffering -> 3 Mbps     logreward
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
# REBUF_PENALTY = 6.6  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
BITS_IN_BYTE = 8.0
# LOG_FILE = '../test_results/log_sim_LinUCB0'
# TEST_LOG_FOLDER = '../test_results/'
# TEST_TRACES = '../longer_traces/'
# TEST_TRACES = './simulation_traces/'
# TEST_TRACES = '../cooked_test_traces/'
TOTAL_VIDEO_CHUNKS = 1250
BITRATE_LEVELS = 6
VMAF_SMOOTH_PENALTY = 1
VMAF_REBUF_PENALTY = 10000

# COARSE_DIM = 1
# COARSE_DIM = 2
# COARSE_DIM = 3
# COARSE_DIM = 4
COARSE_DIM = 5
VIDEO_SIZE_FILE = '../simulation_video_size/synthetic_video_size_BBB_ED/video_size_'
VIDEO_CHUNCK_LEN_IN_SECOND = 1.0  # second


# VIDEO_VMAF_FILE = '../simulation_vmaf/BBB_ED_vmaf_1s/vmaf_'
# TEST_TRACES = sys.argv[1]
# LOG_FILE = sys.argv[2]
# alpha = float(sys.argv[3])
# VMAF_REBUF_PENALTY_1 = float(sys.argv[4])
# QUAITY_WEIGHT = float(sys.argv[5])

# debug:
VIDEO_VMAF_FILE = '../simulation_vmaf/BBB_ED_vmaf_1s/vmaf_'
LOG_FILE = '../test_results/log_SimPCoarseUCB0'
TEST_TRACES = '../norway_bus_times1/'
# TEST_TRACES = '../norway_bus_times3/'
# alpha = 5
alpha = 1
# alpha = 0.01
VMAF_REBUF_PENALTY_1 = 100
QUAITY_WEIGHT = 1
# VMAF_REBUF_PENALTY_1 = 1
# QUAITY_WEIGHT = 3
# adaptive_alpha = alpha

# S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_INFO = 5  # throughput, bit_rate, buffer_size, chunk_size, penalty_sm
# S_INFO = 5  # throughput*5, bit_rate, buffer_size, chunk_size, penalty_sm, sr_time, target_buf
# X_INFO = 9  # throughput*5, bit_rate, buffer_size, chunk_size, penalty_sm
# X_INFO = 7 + COARSE_DIM * 2  # throughput*5, vmaf, buffer_size, chunk_size*COARSE_DIM , vmaf*COARSE_DIM
X_INFO = 9  # throughput*5, buffer_size, sum_chunk_size , sum_vmaf, sum_sm_penalty
X_D = X_INFO
X_LEN = 8
Y_LEN = 8  # take how many rewards in the past
S_LEN = 8  # take how many frames in the past
# A_DIM = 6     # 6 basic combos
# A_DIM = 11  # 6 basic combos, 5 mixed combos
A_DIM = 5  # 5 mixed combos

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

def get_combos(COARSE_DIM, index, temp_vmaf):


    # scheme3 ****************  5 mixed combos
    combos = {}
    # for bitrate in range(BITRATE_LEVELS):
    #     combos[bitrate] = []
    #     for i in range(COARSE_DIM):
    #         combos[bitrate].append(bitrate)

    if (index < 0 or index > TOTAL_VIDEO_CHUNKS or index + COARSE_DIM > TOTAL_VIDEO_CHUNKS):
        for bitrate in range(A_DIM):
            combos[bitrate] = []
            for i in range(COARSE_DIM):
                combos[bitrate].append(bitrate + 1)
        return combos


    for bitrate in range(A_DIM):
        temp_vmaf_two_rows = np.zeros((2, COARSE_DIM))
        temp_vmaf_two_rows[0] = temp_vmaf[bitrate,:]
        temp_vmaf_two_rows[1] = temp_vmaf[bitrate+1,:]

        combos[bitrate] = []
        the_vmaf = np.mean(temp_vmaf_two_rows)

        for i in range(COARSE_DIM):
            list_of_vmaf = temp_vmaf_two_rows[:, i]
            temp_abs_list = []
            for j in range(2):
                temp_abs_list.append(abs(list_of_vmaf[j] - the_vmaf))
            combo_idx = np.argmin(temp_abs_list) + bitrate
            combos[bitrate].append(combo_idx)
    return combos

    # # scheme 2 *******************   6 basic combos + 5 mixed combos
    # combos = {}
    # for bitrate in range(BITRATE_LEVELS):
    #     combos[bitrate] = []
    #     for i in range(COARSE_DIM):
    #         combos[bitrate].append(bitrate)
    #
    # if (index < 0 or index > TOTAL_VIDEO_CHUNKS or index + COARSE_DIM > TOTAL_VIDEO_CHUNKS):
    #     for bitrate in range(BITRATE_LEVELS, A_DIM):
    #         combos[bitrate] = combos[bitrate - 5]
    #     return combos
    #
    #
    # for bitrate in range(BITRATE_LEVELS, A_DIM):
    #     idx = bitrate - BITRATE_LEVELS
    #     temp_vmaf_two_rows = np.zeros((2, COARSE_DIM))
    #     temp_vmaf_two_rows[0] = temp_vmaf[idx,:]
    #     temp_vmaf_two_rows[1] = temp_vmaf[idx+1,:]
    #
    #     combos[bitrate] = []
    #     the_vmaf = np.mean(temp_vmaf_two_rows)
    #
    #     for i in range(COARSE_DIM):
    #         list_of_vmaf = temp_vmaf_two_rows[:, i]
    #         temp_abs_list = []
    #         for j in range(2):
    #             temp_abs_list.append(abs(list_of_vmaf[j] - the_vmaf))
    #         combo_idx = np.argmin(temp_abs_list) + idx
    #         combos[bitrate].append(combo_idx)
    # return combos

    # scheme 1 ************************     6 mixed combos
    # combos = {}
    # if (index < 0 or index > TOTAL_VIDEO_CHUNKS or index + COARSE_DIM > TOTAL_VIDEO_CHUNKS):
    #     for bitrate in range(BITRATE_LEVELS):
    #         combos[bitrate] = []
    #         for i in range(COARSE_DIM):
    #             combos[bitrate].append(bitrate)
    #     return combos
    #
    #
    # for bitrate in range(BITRATE_LEVELS):
    #
    #     combos[bitrate] = []
    #     if bitrate == 0:
    #         the_vmaf = max(temp_vmaf[bitrate, :])
    #     elif bitrate < BITRATE_LEVELS - 1:
    #         the_vmaf = np.mean(temp_vmaf[bitrate, :])
    #     else:
    #         the_vmaf = min(temp_vmaf[bitrate, :])
    #
    #     for i in range(COARSE_DIM):
    #         list_of_vmaf = temp_vmaf[:, i]
    #         temp_abs_list = []
    #         for j in range(BITRATE_LEVELS):
    #             temp_abs_list.append(abs(list_of_vmaf[j] - the_vmaf))
    #         combo_idx = np.argmin(temp_abs_list)
    #         combos[bitrate].append(combo_idx)
    # return combos

def main():

    # for bitrate in xrange(BITRATE_LEVELS):
    for bitrate in range(BITRATE_LEVELS):
        video_vmaf[bitrate] = []
        with open(VIDEO_VMAF_FILE + str(bitrate)) as f:
            for line in f:
                video_vmaf[bitrate].append(float(line.split()[0]))

    for bitrate in range(BITRATE_LEVELS):
        video_sizes[bitrate] = []
        with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
            for line in f:
                video_sizes[bitrate].append(float(line.split()[0]))

    np.random.seed(RANDOM_SEED)

    # assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'wb')
    # print ("log_file: ", log_file)
    # print ('all_file_names: ', all_file_names)

    # print ('alpha: ', type(alpha))

    Aa = np.zeros((A_DIM, X_D, X_D))
    Aa_inv = np.zeros((A_DIM, X_D, X_D))
    ba = np.zeros((A_DIM, X_D, 1))
    theta = np.zeros((A_DIM, X_D, 1))

    for i in range(A_DIM):
        Aa[i] = np.identity(X_D)
        Aa_inv[i] = np.identity(X_D)
        ba[i] = np.zeros((X_D, 1))
        theta[i] = np.zeros((X_D, 1))

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
                       # str(reward) + '\n')
                       str(reward) + '\t' + '\t' + '\t' + '\t' +
                       str(bit_rate) + '\n')
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
        #                # str(reward) + '\n')
        #                str(reward) + '\t' + '\t' + '\t' + '\t' +
        #                str(bit_rate) + '\n')
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

        # if video_chunk_num > 0 and video_chunk_num % COARSE_DIM == 0:
        if video_chunk_num > 0:
            x_context = np.roll(x_context, -1, axis=1)
            x_context[:, -1] = max_context

            # calculate the reward of SimCoarseUCB
            sum_video_quality = max_context[-3]
            sum_sm_penalty = max_context[-1]
            sim_sum_video_quality = sum_video_quality - (video_quality / 100)   # vmaf1 / 100
            sim_sum_sm_penalty = sum_sm_penalty - (penalty_sm / 100)    # sm / 100
            sim_throughput = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms = M byte / s
            sim_future_chunk_size = matrix_chunk_size[max_A]   # M byte
            sim_buffer = buffer_size    # seconds
            sim_rebuffering = 0     # seconds
            for i in range(1, COARSE_DIM):      # except the first chunk of this combo
                temp_chunk_size = sim_future_chunk_size[i]
                sim_download_time = temp_chunk_size / sim_throughput

                if (sim_buffer < sim_download_time):
                    sim_rebuffering += (sim_download_time - sim_buffer)
                    sim_buffer = 0
                else:
                    sim_buffer -= sim_download_time
                sim_buffer += VIDEO_CHUNCK_LEN_IN_SECOND

            sim_rebuf_penalty = VMAF_REBUF_PENALTY_1 * sim_rebuffering / 100
            sim_QoE = sim_sum_video_quality - sim_sum_sm_penalty - sim_rebuf_penalty

            coarse_reward = reward + sim_QoE     # true QoE + sim_QoE

            # coarse_reward = 0
            # for i in range(COARSE_DIM):
            #     idx = -1 - i
            #     coarse_reward += y_reward[idx]

            x = x_context[:, -1]
            x = np.array(x).reshape((X_D, 1))
            # update the Aa, ..., theta
            # Aa[max_a] += np.outer(x, x)
            # Aa_inv[max_a] = np.linalg.inv(Aa[max_a])
            # ba[max_a] += coarse_reward * x
            # theta[max_a] = Aa_inv[max_a].dot(ba[max_a])
            Aa[max_A] += np.outer(x, x)
            Aa_inv[max_A] = np.linalg.inv(Aa[max_A])
            ba[max_A] += coarse_reward * x
            theta[max_A] = Aa_inv[max_A].dot(ba[max_A])

        # *****************************************************

        # the desicion for next video chunk:
        UCB_A = []
        # BB_buffer = float(post_data['buffer'])
        # bitrate_last_chunk = VIDEO_BIT_RATE[post_data['lastquality']]
        # bitrate_last_chunk = VIDEO_BIT_RATE[bit_rate]
        vmaf_last_chunk = get_chunk_vmaf(bit_rate, video_chunk_num)

        # print (video_chunk_num)

        # if video_chunk_num % COARSE_DIM == 0:
        if video_chunk_num >= 0:

            # chunk_combos = {}
            temp_vmaf = np.zeros((BITRATE_LEVELS, COARSE_DIM))
            for bitrate in range(BITRATE_LEVELS):
                # temp_vmaf[bitrate] = []
                for i in range(COARSE_DIM):
                    temp_chunk_vmaf = get_chunk_vmaf(bitrate, video_chunk_num + i + 1)
                    # temp_vmaf[bitrate].append(temp_chunk_vmaf)
                    temp_vmaf[bitrate, i] = temp_chunk_vmaf
            chunk_combos = get_combos(COARSE_DIM, video_chunk_num, temp_vmaf)

            # print ("video_chunk_num", video_chunk_num, " chunk_combos: ", chunk_combos)

            future_contexts = []

            # the context of different actions for the next video chunk:
            matrix_chunk_size = np.zeros((A_DIM, COARSE_DIM))
            for i in range(0, A_DIM):
                # print (video_chunk_num, chunk_combos)
                combo = chunk_combos[i]
                # bitrate_i = VIDEO_BIT_RATE[i] / float(np.max(VIDEO_BIT_RATE))    # normalized to 0-1
                # bitrate_i = VIDEO_BIT_RATE[i] / M_IN_K  # /1000
                video_quality_last_chunk = QUAITY_WEIGHT * vmaf_last_chunk / 100
                start_buffer = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
                throughput_1 = state[3, -1] * BUFFER_NORM_FACTOR
                throughput_2 = state[3, -2] * BUFFER_NORM_FACTOR
                throughput_3 = state[3, -3] * BUFFER_NORM_FACTOR
                throughput_4 = state[3, -4] * BUFFER_NORM_FACTOR
                throughput_5 = state[3, -5] * BUFFER_NORM_FACTOR

                chunk_vmaf = np.zeros(COARSE_DIM)
                video_quality = np.zeros(COARSE_DIM)
                future_chunk_size = np.zeros(COARSE_DIM)
                chunk_sm_penalty = np.zeros(COARSE_DIM)
                for j in range(COARSE_DIM):
                    bitrate_in_combo = combo[j]
                    chunk_vmaf[j] = get_chunk_vmaf(bitrate_in_combo, video_chunk_num + 1 + j)
                    # chunk_vmaf[j] = get_chunk_vmaf(i, video_chunk_num + 1 + j)
                    video_quality[j] = QUAITY_WEIGHT * chunk_vmaf[j] / 100  # vmaf1 / 100
                    future_chunk_size[j] = get_chunk_size(bitrate_in_combo, video_chunk_num + 1 + j) / M_IN_K / M_IN_K   # M byte
                    # future_chunk_size[j] = get_chunk_size(i, video_chunk_num + 1 + j) / M_IN_K / M_IN_K   # M byte
                    matrix_chunk_size[i][j] = future_chunk_size[j]
                    if j > 0:
                        temp_smoothness_dev = abs(chunk_vmaf[j] - chunk_vmaf[j-1])
                    else:
                        temp_smoothness_dev = abs(chunk_vmaf[j] - vmaf_last_chunk)
                    chunk_sm_penalty[j] = SMOOTH_PENALTY * temp_smoothness_dev / 100    # sm / 100

                sum_video_quality = np.sum(video_quality)
                sum_future_chunk_size = np.sum(future_chunk_size)
                sum_sm_penalty = np.sum(chunk_sm_penalty)

                cx_i = np.array(
                    [start_buffer, throughput_1, throughput_2, throughput_3, throughput_4, throughput_5,
                     sum_video_quality, sum_future_chunk_size, sum_sm_penalty])

                future_contexts.append(cx_i)

                x = cx_i
                x = np.array(x).reshape((X_D, 1))
                x_t = np.transpose(x)
                index = i
                UCB_i = np.matmul(np.transpose(theta[index]), x) + alpha * np.sqrt(
                    np.matmul(x_t, Aa_inv[index].dot(x)))
                # UCB_i = np.matmul(np.transpose(theta[index]), x) + adaptive_alpha * np.sqrt(
                #     np.matmul(x_t, Aa_inv[index].dot(x)))
                UCB_A.append(UCB_i)

            max_A = np.argmax(UCB_A)
            max_combo = chunk_combos[max_A]
            max_context = future_contexts[max_A]
            # idx = max_A
            idx = chunk_combos[max_A][0]

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

            Aa = np.zeros((A_DIM, X_D, X_D))
            Aa_inv = np.zeros((A_DIM, X_D, X_D))
            ba = np.zeros((A_DIM, X_D, 1))
            theta = np.zeros((A_DIM, X_D, 1))
            # adaptive_alpha = alpha

            for i in range(A_DIM):
                Aa[i] = np.identity(X_D)
                Aa_inv[i] = np.identity(X_D)
                ba[i] = np.zeros((X_D, 1))
                theta[i] = np.zeros((X_D, 1))


if __name__ == '__main__':
    main()
