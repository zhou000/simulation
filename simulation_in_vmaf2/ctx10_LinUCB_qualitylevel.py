import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import tensorflow as tf
import load_trace
import sim_fixed_env as env

# VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
VIDEO_BIT_RATE = [344, 742, 1064, 2437, 4583, 6636]  # Kbps ************* VBR video bit_rate
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
# REBUF_PENALTY = 2.66  # 1 sec rebuffering -> 3 Mbps     logreward
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
# REBUF_PENALTY = 6.6  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
# LOG_FILE = '../test_results/log_sim_LinUCB0'
# TEST_LOG_FOLDER = '../test_results/'
# TEST_TRACES = '../longer_traces/'
# TEST_TRACES = '../longer_traces/'
# TEST_TRACES = './simulation_traces/'
# TEST_TRACES = '../cooked_test_traces/'

TEST_TRACES = sys.argv[1]
LOG_FILE = sys.argv[2]
alpha = float(sys.argv[3])

# debug:
# LOG_FILE = '../test_results/log_ctx14_LinUCB0'
# TEST_TRACES = '../long_traces/'
# alpha = 5

# S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_INFO = 5  # throughput, bit_rate, buffer_size, chunk_size, penalty_sm
# S_INFO = 5  # throughput*5, bit_rate, buffer_size, chunk_size, penalty_sm, sr_time, target_buf
# X_INFO = 9  # throughput*5, bit_rate, buffer_size, chunk_size, penalty_sm
X_INFO = 10  # throughput*3, bit_rate, buffer_size, chunk_size, penalty_sm, video_quality_level*3
X_D = X_INFO
X_LEN = 9
Y_LEN = 9  # take how many rewards in the past
S_LEN = 8  # take how many frames in the past
A_DIM = 6

# VIDEO_QUALITY_LEVEL_FILE = '../simulation_video_size/video_quality_level/video_quality_level'
VIDEO_QUALITY_LEVEL_FILE = '../simulation_video_size/video_quality_level/4600chunks'
# TOTAL_VIDEO_CHUNKS = 1250
TOTAL_VIDEO_CHUNKS = 4599

video_quality_level = {}


def get_chunk_quality(video_quality_level, index):
    # if (index < 0 or index > 48):
    if (index < 0 or index > TOTAL_VIDEO_CHUNKS):
        return 0
    # print video_quality_level
    # print index
    return video_quality_level[index]


def main():
    video_quality_level = []

    with open(VIDEO_QUALITY_LEVEL_FILE) as f:
        for line in f:
            video_quality_level.append(int(line.split()[0]))

    # print video_quality_level

    # video_quality_level = np.array(video_quality_level)

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

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

        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain, video_chunk_num = net_env.get_video_chunk(bit_rate, last_bit_rate)
        # next_5_chunk_quality, buffer_remain_ratio, rush_flag= net_env.get_video_chunk(bit_rate, last_bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # the result of argmax(UCB)
        # max_a = post_data['lastquality']
        max_a = bit_rate

        # -- linear reward --
        # reward is video quality - rebuffer penalty - smoothness
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        penalty_sm = SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                             VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        # penalty_sm = SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
        #                                      VIDEO_BIT_RATE[last_bit_rate])
        #
        # penalty_sm = penalty_sm / float(np.max(VIDEO_BIT_RATE))     # normalized

        # # -- log scale reward --
        # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
        # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))
        #
        # reward = log_bit_rate \
        #          - REBUF_PENALTY * rebuf \
        #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

        r_batch.append(reward)

        last_quality = last_bit_rate
        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        # log_file.write(str(time_stamp / M_IN_K) + '\t' +
        log_file.write(str(video_chunk_num) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       # str(last_quality) + '\t' +
                       # str(buffer_remain_ratio) + '\t' +
                       # str(rush_flag) + '\t' +
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

        # dequeue history record
        state = np.roll(state, -1, axis=1)
        x_context = np.roll(x_context, -1, axis=1)
        y_reward = np.roll(y_reward, -1)

        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # normalized to 0-1
        # state[0, -1] = VIDEO_BIT_RATE[bit_rate] / M_IN_K  # /1000   last quality
        # state[0, -1] = VIDEO_BIT_RATE[post_data['lastquality']] / M_IN_K  # /1000
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        # state[1, -1] = buffer_size  # 1 sec
        # state[1, -1] = post_data['buffer'] / BUFFER_NORM_FACTOR  # /10s
        state[2, -1] = video_chunk_size / M_IN_K  # kilo byte
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        # state[3, -1] = float(video_chunk_size) / float(video_chunk_fetch_time) / M_IN_K  # kilo byte / ms
        state[4, -1] = penalty_sm  # /1000

        y_reward[-1] = reward  # linear reward

        x_context[0, -1] = state[0, -1]  # bitrate
        x_context[1, -1] = state[1, -1]  # buffer
        x_context[2, -1] = state[2, -1] / M_IN_K  # chunksize     M byte
        x_context[3, -1] = state[3, -2]  # throughput    kilo byte / ms
        x_context[4, -1] = state[3, -3]
        x_context[5, -1] = state[3, -4]
        # x_context[6, -1] = state[3, -4]
        # x_context[7, -1] = state[3, -5]
        # x_context[3, -1] = state[3, -1] / 10  # throughput    kilo byte / ms  /10
        # x_context[4, -1] = state[3, -2] / 10
        # x_context[5, -1] = state[3, -3] / 10
        # x_context[6, -1] = state[3, -4] / 10
        # x_context[7, -1] = state[3, -5] / 10
        x_context[6, -1] = state[4, -1]  # sm penalty
        x_context[7, -1] = get_chunk_quality(video_quality_level, video_chunk_num)  # quality level for the video chunk
        x_context[8, -1] = get_chunk_quality(video_quality_level, video_chunk_num + 1)  # *** next video chunk
        x_context[9, -1] = get_chunk_quality(video_quality_level, video_chunk_num + 2)  # +2
        # x_context[12, -1] = get_chunk_quality(video_quality_level, video_chunk_num + 3)  # +3
        # x_context[13, -1] = get_chunk_quality(video_quality_level, video_chunk_num + 4)  # +4

        x = x_context[:, -1]
        x = np.array(x).reshape((X_D, 1))
        # update the Aa, ..., theta
        Aa[max_a] += np.outer(x, x)
        Aa_inv[max_a] = np.linalg.inv(Aa[max_a])
        ba[max_a] += reward * x
        theta[max_a] = Aa_inv[max_a].dot(ba[max_a])

        # the desicion for next video chunk:
        UCB_A = []
        # BB_buffer = float(post_data['buffer'])
        # bitrate_last_chunk = VIDEO_BIT_RATE[post_data['lastquality']]
        bitrate_last_chunk = VIDEO_BIT_RATE[bit_rate]

        # the context of different actions for the next video chunk:
        for i in range(0, A_DIM):
            bitrate_i = VIDEO_BIT_RATE[i] / float(np.max(VIDEO_BIT_RATE))  # normalized to 0-1
            # bitrate_i = VIDEO_BIT_RATE[i] / M_IN_K  # /1000
            # start_buffer = float(post_data['buffer']) / BUFFER_NORM_FACTOR  # /10s
            start_buffer = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            # start_buffer = buffer_size   # 1 sec
            # video_chunk_size_i = get_chunk_size(i, video_chunk_num) / M_IN_K / M_IN_K  # M byte
            video_chunk_size_i = next_video_chunk_sizes[i] / M_IN_K / M_IN_K  # M byte
            throughput_1 = state[3, -1]
            throughput_2 = state[3, -2]
            throughput_3 = state[3, -3]
            # throughput_4 = state[3, -4]
            # throughput_5 = state[3, -5]
            # throughput_1 = state[3, -1] / 10
            # throughput_2 = state[3, -2] / 10
            # throughput_3 = state[3, -3] / 10
            # throughput_4 = state[3, -4] / 10
            # throughput_5 = state[3, -5] / 10
            penalty_sm_i = SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[i] - bitrate_last_chunk) / M_IN_K
            # penalty_sm_i = SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[i] - bitrate_last_chunk)
            # penalty_sm_i = penalty_sm_i / float(np.max(VIDEO_BIT_RATE))     # normalized
            quality_level_1 = get_chunk_quality(video_quality_level, video_chunk_num + 1)
            quality_level_2 = get_chunk_quality(video_quality_level, video_chunk_num + 2)
            quality_level_3 = get_chunk_quality(video_quality_level, video_chunk_num + 3)
            # quality_level_4 = get_chunk_quality(video_quality_level, video_chunk_num + 4)
            # quality_level_5 = get_chunk_quality(video_quality_level, video_chunk_num + 5)
            cx_i = np.array(
                [bitrate_i, start_buffer, video_chunk_size_i, throughput_1, throughput_2, throughput_3, penalty_sm_i,
                 quality_level_1, quality_level_2, quality_level_3])

            x = cx_i
            x = np.array(x).reshape((X_D, 1))
            x_t = np.transpose(x)
            index = i
            UCB_i = np.matmul(np.transpose(theta[index]), x) + alpha * np.sqrt(
                np.matmul(x_t, Aa_inv[index].dot(x)))
            UCB_A.append(UCB_i)

        max_A = np.argmax(UCB_A)
        idx = max_A

        bit_rate = idx

        s_batch.append(state)
        X_batch.append(x_context)
        Y_batch.append(y_reward)

        # entropy_record.append(a3c.compute_entropy(action_prob[0]))

        if end_of_video:
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

            Aa = np.zeros((A_DIM, X_D, X_D))
            Aa_inv = np.zeros((A_DIM, X_D, X_D))
            ba = np.zeros((A_DIM, X_D, 1))
            theta = np.zeros((A_DIM, X_D, 1))

            for i in range(A_DIM):
                Aa[i] = np.identity(X_D)
                Aa_inv[i] = np.identity(X_D)
                ba[i] = np.zeros((X_D, 1))
                theta[i] = np.zeros((X_D, 1))


if __name__ == '__main__':
    main()
