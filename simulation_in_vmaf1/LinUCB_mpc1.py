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
VIDEO_SIZE_FILE = '../simulation_video_size/synthetic_video_size_BBB_ED/video_size_'
# VIDEO_SIZE_FILE = '../simulation_video_size/4600chunks/video_size_'
VIDEO_VMAF_FILE = '../simulation_vmaf/BBB_ED_vmaf_1s/vmaf_'
# LOG_FILE = '../test_results/log_sim_LinUCB0'
# TEST_LOG_FOLDER = '../test_results/'
# TEST_TRACES = '../longer_traces/'
# TEST_TRACES = './simulation_traces/'
# TEST_TRACES = '../cooked_test_traces/'

VMAF_SMOOTH_PENALTY = 1

VMAF_REBUF_PENALTY = 10000

TEST_TRACES = sys.argv[1]
LOG_FILE = sys.argv[2]
alpha = float(sys.argv[3])
VMAF_REBUF_PENALTY_1 = float(sys.argv[4])
QUAITY_WEIGHT = float(sys.argv[5])

# # debug
# TEST_TRACES = '../norway_bus_times1/'
# LOG_FILE = '../test_results/log_LinMPC'
# VMAF_REBUF_PENALTY_1 = 100
# QUAITY_WEIGHT = 1
# alpha = 1


S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
MPC_FUTURE_CHUNK_COUNT = 5
# MPC_FUTURE_CHUNK_COUNT = 3
A_DIM = 6
X_INFO = 8  # throughput*5, bit_rate, buffer_size, chunk_size, penalty_sm
X_D = X_INFO
X_LEN = 8
Y_LEN = 8  # take how many rewards in the past
UCB_DIM = 11   # the number of action of LinUCB
# UCB_ACTION = [0.5, 0.65, 0.8, 0.95, 1.1, 1.25, 1.4, 1.55, 1.7, 1.85, 2]

UCB_ACTION = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
BITS_IN_BYTE = 8

# alpha = 5
# alpha = 1
# alpha = 0.1

CHUNK_COMBO_OPTIONS = []
VIDEO_CHUNCK_LEN_IN_SECOND = 1.0  # second
# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []

video_size = {}
video_vmaf = {}


# for bitrate in xrange(BITRATE_LEVELS):
#     video_size[bitrate] = []
#     print (VIDEO_SIZE_FILE + str(bitrate))
#     with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
#         for line in f:
#             video_size[bitrate].append(int(line.split()[0]))

def get_chunk_size(quality, index):
    # if (index < 0 or index > 48):
    if (index < 0 or index > TOTAL_VIDEO_CHUNKS):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 8 is highest and this pertains to video1)
    # sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index],
    #          1: size_video5[index], 0: size_video6[index]}
    chunk_list = video_size.get(quality)
    # return video_size[quality, index]
    return chunk_list[index]


def get_chunk_vmaf(quality, index):
    if (index < 0 or index > TOTAL_VIDEO_CHUNKS):
        return 0
    vmaf_list = video_vmaf.get(quality)
    return vmaf_list[index]


def main():
    START_time = time.time()

    for bitrate in xrange(BITRATE_LEVELS):
        video_vmaf[bitrate] = []
        with open(VIDEO_VMAF_FILE + str(bitrate)) as f:
            for line in f:
                video_vmaf[bitrate].append(float(line.split()[0]))

    for bitrate in xrange(BITRATE_LEVELS):
        video_size[bitrate] = []
        # print (VIDEO_SIZE_FILE + str(bitrate))
        with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
            for line in f:
                video_size[bitrate].append(int(line.split()[0]))

    # for combo in itertools.product([0, 1, 2, 3, 4, 5], repeat=5):
    #     CHUNK_COMBO_OPTIONS.append(combo)

    combos = [[0,0,0,0,0],[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4],[5,5,5,5,5]]
    for i in range(A_DIM):
        combo = combos[i]
        CHUNK_COMBO_OPTIONS.append(combo)

    # combos = [[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]
    # for i in range(A_DIM):
    #     combo = combos[i]
    #     CHUNK_COMBO_OPTIONS.append(combo)

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

    Aa = np.zeros((UCB_DIM, X_D, X_D))
    Aa_inv = np.zeros((UCB_DIM, X_D, X_D))
    ba = np.zeros((UCB_DIM, X_D, 1))
    theta = np.zeros((UCB_DIM, X_D, 1))

    for i in range(UCB_DIM):
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
    TOTAL_downloading_time = 0
    TOTAL_decision_time = 0

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real

        start_time_stamp = time.time()


        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, next_2_video_chunk_sizes, avg_chunk_sizes, \
        end_of_video, video_chunk_remain, video_chunk_num = net_env.get_video_chunk(bit_rate, last_bit_rate)
        # next_5_chunk_quality, buffer_remain_ratio, rush_flag= net_env.get_video_chunk(bit_rate, last_bit_rate)

        time_stamp_1 = time.time()

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
        # reward = reward / M_IN_K  # normalize

        r_batch.append(reward)

        last_quality = last_bit_rate
        last_bit_rate = bit_rate

        # # log time_stamp, bit_rate, buffer_size, reward
        # # log_file.write(str(time_stamp / M_IN_K) + '\t' +
        # log_file.write(str(video_chunk_num) + '\t' +
        #                str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
        #                str(buffer_size) + '\t' +
        #                str(rebuf) + '\t' +
        #                str(video_chunk_size) + '\t' +
        #                str(delay) + '\t' +
        #                str(reward) + '\n')
        # log_file.flush()

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(video_chunk_num) + '\t' +
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

        if len(X_batch) == 0:
            x_context = [np.zeros((X_INFO, X_LEN))]
        else:
            x_context = np.array(X_batch[-1], copy=True)

        if len(Y_batch) == 0:
            y_reward = [np.zeros(Y_LEN)]
        else:
            y_reward = np.array(Y_batch[-1], copy=True)

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
        x_context = np.roll(x_context, -1, axis=1)
        y_reward = np.roll(y_reward, -1)

        # state[0, -1] = VIDEO_BIT_RATE[post_data['lastquality']] / float(np.max(VIDEO_BIT_RATE))
        # state[0, -1] = VIDEO_BIT_RATE[bit_rate] / M_IN_K  # /1000   last quality
        # state[0, -1] = video_quality / M_IN_K  # /1000
        state[0, -1] = video_quality / 100
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(
            CHUNK_TIL_VIDEO_END_CAP)
        curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if (len(past_bandwidth_ests) > 0):
            curr_error = abs(past_bandwidth_ests[-1] - state[3, -1]) / float(state[3, -1])
        past_errors.append(curr_error)

        time_stamp_2 = time.time()

        # past_throughput = state[3, -X_LEN:] * M_IN_K * BITS_IN_BYTE     # kilo bits / second
        past_throughput = state[3, -X_LEN:]  # kilo byte / ms
        context = past_throughput
        x_context[:, -1] = context
        if video_chunk_num > 0:
            # y_reward = -abs(predict_throughput - state[3,-1] * M_IN_K * BITS_IN_BYTE)
            y_reward = -abs(predict_throughput - state[3, -1])
            x = x_context[:, -1]
            x = np.array(x).reshape((X_D, 1))
            # update the Aa, ..., theta
            Aa[max_A] += np.outer(x, x)
            Aa_inv[max_A] = np.linalg.inv(Aa[max_A])
            ba[max_A] += reward * x
            theta[max_A] = Aa_inv[max_A].dot(ba[max_A])

        RB_bitrate = 0
        RB_action = 0
        while past_throughput[0] == 0.0:
            past_throughput = past_throughput[1:]
        mean_throughput = np.mean(past_throughput)
        UCB_A = []
        for i in range(UCB_DIM):
            cx_i = context
            x = cx_i
            x = np.array(x).reshape((X_D, 1))
            x_t = np.transpose(x)
            index = i
            UCB_i = np.matmul(np.transpose(theta[index]), x) + alpha * np.sqrt(
                np.matmul(x_t, Aa_inv[index].dot(x)))
            UCB_A.append(UCB_i)
        max_A = np.argmax(UCB_A)
        predict_throughput = mean_throughput * UCB_ACTION[max_A]
        RB_bitrate = predict_throughput * M_IN_K * BITS_IN_BYTE

        # # pick bitrate according to MPC
        # # first get harmonic mean of last 5 bandwidths
        # past_bandwidths = state[3, -5:]  # kilo byte / ms
        # while past_bandwidths[0] == 0.0:
        #     past_bandwidths = past_bandwidths[1:]
        # # if ( len(state) < 5 ):
        # #    past_bandwidths = state[3,-len(state):]
        # # else:
        # #    past_bandwidths = state[3,-5:]
        # bandwidth_sum = 0
        # for past_val in past_bandwidths:
        #     bandwidth_sum += (1 / float(past_val))
        # harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))
        #
        # # future bandwidth prediction
        # # divide by 1 + max of last 5 (or up to 5) errors
        # max_error = 0
        # error_pos = -5
        # if (len(past_errors) < 5):
        #     error_pos = -len(past_errors)
        # max_error = float(max(past_errors[error_pos:]))
        # future_bandwidth = harmonic_bandwidth / (1 + max_error)
        # past_bandwidth_ests.append(harmonic_bandwidth)

        future_bandwidth = predict_throughput

        # future chunks length (try 4 if that many remaining)
        # last_index = int(post_data['lastRequest'])
        last_index = video_chunk_num
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if (TOTAL_VIDEO_CHUNKS - last_index < 5):
            future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -100000000
        best_combo = ()
        # start_buffer = float(post_data['buffer'])
        start_buffer = buffer_size
        # start = time.time()
        for full_combo in CHUNK_COMBO_OPTIONS:
            combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 2 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer
            bitrate_sum = 0
            smoothness_diffs = 0
            # last_quality = int(post_data['lastquality'])
            last_quality = int(bit_rate)
            for position in range(0, len(combo)):
                chunk_quality = combo[position]
                index = last_index + position + 1  # e.g., if last chunk is 3, then first iter is 3+0+1=4
                download_time = (get_chunk_size(chunk_quality,
                                                index) / 1000000.) / future_bandwidth  # this is MB/MB/s --> seconds
                # download_time = (video_size[chunk_quality,
                #                                 index] / 1000000.) / future_bandwidth  # this is MB/MB/s --> seconds
                # need to get value form dict in python not in 2d-array

                if (curr_buffer < download_time):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                # curr_buffer += 4
                curr_buffer += VIDEO_CHUNCK_LEN_IN_SECOND

                # # linear reward
                # bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                # smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])

                # # log reward
                # log_bit_rate = np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0]))
                # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_quality] / float(VIDEO_BIT_RATE[0]))
                # bitrate_sum += log_bit_rate
                # smoothness_diffs += abs(log_bit_rate - log_last_bit_rate)

                # hd reward
                # bitrate_sum += BITRATE_REWARD[chunk_quality]
                # smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])

                # vmaf reward 1
                video_vmaf_i = get_chunk_vmaf(chunk_quality, index)
                video_quality_i = QUAITY_WEIGHT * video_vmaf_i / 100
                bitrate_sum += video_quality_i

                vmaf_last_chunk = get_chunk_vmaf(last_quality, index - 1)
                smoothness_dev_i = abs(video_vmaf_i - vmaf_last_chunk)
                penalty_sm_i = SMOOTH_PENALTY * smoothness_dev_i / 100
                smoothness_diffs += penalty_sm_i


                # # vmaf reward 2
                # video_vmaf_i = get_chunk_vmaf(chunk_quality, index)
                # video_quality_i = video_vmaf_i * video_vmaf_i / M_IN_K  # /1000
                # bitrate_sum += video_quality_i
                #
                # vmaf_last_chunk = get_chunk_vmaf(last_quality, index - 1)
                # smoothness_dev_i = video_vmaf_i - vmaf_last_chunk
                # penalty_sm_i = SMOOTH_PENALTY * smoothness_dev_i * smoothness_dev_i / M_IN_K
                # smoothness_diffs += penalty_sm_i

                last_quality = chunk_quality
            # compute reward for this combination (one reward per 5-chunk combo)
            # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

            # # linear reward
            # reward = (bitrate_sum / 1000.) - (4.3 * curr_rebuffer_time) - (smoothness_diffs / 1000.)

            # # log reward
            # reward = (bitrate_sum) - (2.66*curr_rebuffer_time) - (smoothness_diffs)

            # hd reward
            # reward = bitrate_sum - (8 * curr_rebuffer_time) - (smoothness_diffs)

            # vmaf reward 1
            reward = bitrate_sum - (VMAF_REBUF_PENALTY_1 * curr_rebuffer_time / 100) - smoothness_diffs

            # # vmaf reward 2
            # reward = bitrate_sum - (VMAF_REBUF_PENALTY * curr_rebuffer_time / 1000) - smoothness_diffs

            if (reward > max_reward):
                max_reward = reward
                best_combo = combo

        # send data to html side (first chunk of best combo)
        send_data = 0  # no combo had reward better than -1000000 (ERROR) so send 0
        best_combo = tuple(best_combo)
        if (best_combo != ()):  # some combo was good
            # print ('chunk num: ', video_chunk_num, "  ", best_combo[0], ' full combo: ', best_combo)
            # x = best_combo[0]
            send_data = best_combo[0]
            # print (send_data)
            # print (send_data)

        end = time.time()
        # print "TOOK: " + str(end-start)

        time_stamp_3 = time.time()
        TOTAL_downloading_time += delay / M_IN_K
        TOTAL_decision_time += time_stamp_3 - time_stamp_2
        # print ("Chunk num: ", video_chunk_num, " Downloading time: ", str(delay), " Processing time1: ", str(time_stamp_1-start_time_stamp), " Processing time2: ", str(time_stamp_2-time_stamp_1), " Processing time3: ", str(time_stamp_3-time_stamp_2))


        bit_rate = send_data

        s_batch.append(state)
        X_batch.append(x_context)
        Y_batch.append(y_reward)

        # entropy_record.append(a3c.compute_entropy(action_prob[0]))

        if end_of_video:
            # print ("TOTAL_downloading_time: ", TOTAL_downloading_time, " TOTAL_decision_time", TOTAL_decision_time)
            # TOTAL_downloading_time = 0
            # TOTAL_decision_time = 0

            log_file.write('\n')
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

            Aa = np.zeros((UCB_DIM, X_D, X_D))
            Aa_inv = np.zeros((UCB_DIM, X_D, X_D))
            ba = np.zeros((UCB_DIM, X_D, 1))
            theta = np.zeros((UCB_DIM, X_D, 1))

            for i in range(UCB_DIM):
                Aa[i] = np.identity(X_D)
                Aa_inv[i] = np.identity(X_D)
                ba[i] = np.zeros((X_D, 1))
                theta[i] = np.zeros((X_D, 1))




    END_time = time.time()
    TOTAL_time = END_time - START_time

if __name__ == '__main__':
    main()
