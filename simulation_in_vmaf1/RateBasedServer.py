import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import tensorflow as tf
import load_trace
import sim_fixed_env as env
import time
import itertools

# VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
VIDEO_BIT_RATE = [344, 742, 1064, 2437, 4583, 6636]  # Kbps ************* VBR video bit_rate
# BITRATE_REWARD_MAP = {0: 0, 300: 1, 750: 2, 1200: 3, 1850: 12, 2850: 15, 4300: 20}
BITRATE_REWARD_MAP = {0: 0, 344: 1, 742: 2, 1064: 3, 2437: 12, 4583: 15, 6636: 20}
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
# CHUNK_TIL_VIDEO_END_CAP = 48.0
CHUNK_TIL_VIDEO_END_CAP = 1250.0
# CHUNK_TIL_VIDEO_END_CAP = 4599.0
TOTAL_VIDEO_CHUNKS = 1250
# TOTAL_VIDEO_CHUNKS = 4599
M_IN_K = 1000.0
# REBUF_PENALTY = 2.66  # 1 sec rebuffering -> 3 Mbps     logreward
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
# REBUF_PENALTY = 6.6  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
BITRATE_LEVELS = 6
# LOG_FILE = '../test_results/log_sim_LinUCB0'
# TEST_LOG_FOLDER = '../test_results/'
# TEST_TRACES = '../longer_traces/'
# TEST_TRACES = './simulation_traces/'
# TEST_TRACES = '../cooked_test_traces/'

VMAF_SMOOTH_PENALTY = 1
VMAF_REBUF_PENALTY = 10000
VMAF_REBUF_PENALTY_1 = 100

VIDEO_VMAF_FILE = '../simulation_vmaf/BBB_ED_vmaf_1s/vmaf_'
TEST_TRACES = sys.argv[1]
LOG_FILE = '../test_results/log_RB_server'


# Debug
# VIDEO_VMAF_FILE = '../simulation_vmaf/BBB_ED_vmaf_1s/vmaf_'
# TEST_TRACES = '../cooked_test_traces/'
# # TEST_TRACES = '../long_traces/'
# LOG_FILE = '../test_results/log_RB_server'

# S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_INFO = 6  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, download_time, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
MPC_FUTURE_CHUNK_COUNT = 5
A_DIM = 6

CHUNK_COMBO_OPTIONS = []

# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []

p_rb = 1
horizon = 5
past_throughput = []
past_download_time = []
BITS_IN_BYTE = 8.0

video_vmaf = {}

def get_chunk_vmaf(quality, index):
    if (index < 0 or index > TOTAL_VIDEO_CHUNKS):
        return 0
    vmaf_list = video_vmaf.get(quality)
    return vmaf_list[index]



def main():

    for bitrate in xrange(BITRATE_LEVELS):
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

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    # X_batch = [np.zeros((X_INFO, X_LEN))]
    # Y_batch = [np.zeros(Y_LEN)]

    video_count = 0

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real

        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, next_2_video_chunk_sizes, avg_chunk_sizes, \
        end_of_video, video_chunk_remain, video_chunk_num = net_env.get_video_chunk(bit_rate, last_bit_rate)
        # next_5_chunk_quality, buffer_remain_ratio, rush_flag= net_env.get_video_chunk(bit_rate, last_bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # rebuffer_time = float(post_data['RebufferTime'] - self.input_dict['last_total_rebuf'])
        #
        # # --linear reward--
        # reward = VIDEO_BIT_RATE[post_data['lastquality']] / M_IN_K \
        #          - REBUF_PENALTY * rebuffer_time / M_IN_K \
        #          - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[post_data['lastquality']] -
        #                                    self.input_dict['last_bit_rate']) / M_IN_K

        # # -- linear reward --
        # # reward is video quality - rebuffer penalty - smoothness
        # reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
        #          - REBUF_PENALTY * rebuf \
        #          - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
        #                                    VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        #
        # penalty_sm = SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
        #                                      VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

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
        video_quality = get_chunk_vmaf(bit_rate, video_chunk_num)

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

        r_batch.append(reward)

        last_quality = last_bit_rate
        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        # log_file.write(str(time_stamp / M_IN_K) + '\t' +
        log_file.write(str(video_chunk_num) + '\t' +
                       # str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(video_quality) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # compute bandwidth measurement
        video_chunk_fetch_time = delay
        # video_chunk_size = post_data['lastChunkSize']

        # compute number of video chunks left
        # video_chunk_remain = TOTAL_VIDEO_CHUNKS - self.input_dict['video_chunk_coount']
        video_chunk_remain = TOTAL_VIDEO_CHUNKS - video_chunk_num
        # self.input_dict['video_chunk_coount'] += 1
        # video_chunk_num += 1  # +1 in env

        # dequeue history record
        state = np.roll(state, -1, axis=1)


        # state[0, -1] = VIDEO_BIT_RATE[post_data['lastquality']] / float(np.max(VIDEO_BIT_RATE))
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / M_IN_K  # /1000   last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[4, -1] = float(delay) / M_IN_K    # second
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(
            CHUNK_TIL_VIDEO_END_CAP)

        RB_bitrate = 0
        RB_action = 0

        # throughput prediction
        tmp_sum = 0
        tmp_time = 0

        past_throughput = state[3, -5:] * M_IN_K * BITS_IN_BYTE       # kilo bits / second
        while past_throughput[0] == 0.0:
            past_throughput = past_throughput[1:]

        past_download_time = state[4, -5:]      # second
        while past_download_time[0] == 0.0:
            past_download_time = past_download_time[1:]

        for throughput, download_time in zip(past_throughput, past_download_time):
            tmp_sum += download_time / throughput
            tmp_time += download_time

        RB_bitrate = p_rb * (tmp_time / tmp_sum)


        for i in range(len(VIDEO_BIT_RATE)-1, -1, -1):
            if RB_bitrate >= VIDEO_BIT_RATE[i]:
                RB_action = i
                break
            RB_action = i



        bit_rate = RB_action

        s_batch.append(state)


        # entropy_record.append(a3c.compute_entropy(action_prob[0]))

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)

            entropy_record = []

            video_count += 1

            if video_count >= len(all_file_names):
                break

            # another trace
            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'wb')




if __name__ == '__main__':
    main()
