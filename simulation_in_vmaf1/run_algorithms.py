import os
import time

TEST_LOG_FOLDER = '../test_results/'
# TEST_TRACES = '../cooked_test_traces/'
# TEST_TRACES = '../long_traces/'
TEST_TRACES = '../norway_bus_times1/'
# TEST_TRACES = '../norway_bus_times3/'

# VMAF_REBUF_PENALTY_1 = 1
# VMAF_REBUF_PENALTY_1 = 10
# VMAF_REBUF_PENALTY_1 = 25
# VMAF_REBUF_PENALTY_1 = 50
VMAF_REBUF_PENALTY_1 = 100

QUAITY_WEIGHT = 1
# QUAITY_WEIGHT = 3


os.system('rm -r ' + TEST_LOG_FOLDER)
os.system('mkdir ' + TEST_LOG_FOLDER)


# for i in range(len(Algorithms)):
#     cmd = "python " + Algorithms[i] + TEST_TRACES
#     print cmd
#     os.system(cmd)



# # for the LinUCB with 13d context, i.e., ctx13_LinUCB0-5
# LOG_FILE = '../test_results/log_ctx13_LinUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# alpha = [5, 1, 0.1, 0.01, 0.001]
# # alpha = [5, 1, 0.1, 0.01]
# alg = './ctx13_LinUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
#     print (cmd)
#     os.system(cmd)
#
#
# # for the LinUCB with 12d context, i.e., ctx12_LinUCB0-5
# LOG_FILE = '../test_results/log_ctx12_LinUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# alpha = [5, 1, 0.1, 0.01, 0.001]
# # alpha = [5, 1, 0.1, 0.01]
# alg = './ctx12_LinUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
#     print (cmd)
#     os.system(cmd)



#
# for the LinUCB with 9d context, i.e., ctx9_LinUCB0-5
LOG_FILE = '../test_results/log_ctx9_LinUCB'
# alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
alpha = [5, 1, 0.1, 0.01, 0.001]
# alpha = [5, 1, 0.1, 0.01]
# alpha = [5]
# alpha = 0.0001
alg = './ctx9_LinUCB.py '

for i in range(len(alpha)):
    cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
    print (cmd)
    os.system(cmd)



#
# # for the LinUCB with 7d context, i.e., ctx7_LinUCB0-5
# LOG_FILE = '../test_results/log_ctx7_LinUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# alpha = [5, 1, 0.1, 0.01, 0.001]
# # alpha = [5, 1, 0.1, 0.01]
# alg = './ctx7_LinUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
#     print (cmd)
#     os.system(cmd)
#
#
# # for the LinUCB with 5d context, i.e., ctx5_LinUCB0-5
# LOG_FILE = '../test_results/log_ctx5_LinUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# alpha = [5, 1, 0.1, 0.01, 0.001]
# # alpha = [5, 1, 0.1, 0.01]
# alg = './ctx5_LinUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
#     print (cmd)
#     os.system(cmd)
#
#
# # for the care1_LinUCB with 3d context, i.e., care1_LinUCB0-5
# LOG_FILE = '../test_results/log_care1_LinUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# alpha = [5, 1, 0.1, 0.01, 0.001]
# # alpha = [5, 1, 0.1, 0.01]
# # alpha = 0.0001
# alg = './care1_LinUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
#     print (cmd)
#     os.system(cmd)
#
#
# # # for the care3_LinUCB with 3d context, i.e., care3_LinUCB0-5
# # LOG_FILE = '../test_results/log_care3_LinUCB'
# # # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# # alpha = [5, 1, 0.1, 0.01, 0.001]
# # # alpha = [5, 1, 0.1, 0.01]
# # # alpha = 0.0001
# # alg = './care3_LinUCB.py '
# #
# # for i in range(len(alpha)):
# #     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1)
# #     print cmd
# #     os.system(cmd)
# #
# #
# #
# # # for the care5_LinUCB with 3d context, i.e., care5_LinUCB0-5
# # LOG_FILE = '../test_results/log_care5_LinUCB'
# # # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# # alpha = [5, 1, 0.1, 0.01, 0.001]
# # # alpha = [5, 1, 0.1, 0.01]
# # # alpha = 0.0001
# # alg = './care5_LinUCB.py '
# #
# # for i in range(len(alpha)):
# #     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1)
# #     print cmd
# #     os.system(cmd)
#
#
#
# # for the ctx3_LinUCB with 3d context, i.e., ctx3_LinUCB0-5
# LOG_FILE = '../test_results/log_ctx3_LinUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# alpha = [5, 1, 0.1, 0.01, 0.001]
# # alpha = [5, 1, 0.1, 0.01]
# # alpha = 0.0001
# alg = './ctx3_LinUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
#     print (cmd)
#     os.system(cmd)


# # robust_MPC0
# LOG_FILE = '../test_results/log_robustMPC0'
# alg = './sim_robust_mpc0.py '
#
# cmd_robust_MPC0 = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
# print (cmd_robust_MPC0)
# begin_time = time.time()
# os.system(cmd_robust_MPC0)
# end_time = time.time()
# running_time = end_time - begin_time
# print ('running_time for MPC0: ', running_time)
#
#
# # BB_server
# # LOG_FILE = '../test_results/log_BB_server'
# alg = './BufferBasedServer.py '
# cmd_BB_server = "python " + alg + TEST_TRACES + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
# os.system(cmd_BB_server)
#
#
# # RB_server
# # LOG_FILE = '../test_results/log_RB_server'
# alg = './RateBasedServer.py '
# cmd_RB_server = "python " + alg + TEST_TRACES + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
# os.system(cmd_RB_server)

# # Lin_RB
# LOG_FILE = '../test_results/log_LinRB'
# alpha = [5, 1, 0.1, 0.01, 0.001]
# alg = './LinUCB_RB1.py '
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
#     print (cmd)
#     os.system(cmd)
#
#
# # LinMPC
# LOG_FILE = '../test_results/log_LinMPC'
# alpha = [5, 1, 0.1, 0.01, 0.001]
# alg = './LinUCB_mpc1.py '
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
#     print (cmd)
#     os.system(cmd)


# # CoarseUCB
# LOG_FILE = '../test_results/log_CoarseUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# alpha = [5, 1, 0.1, 0.01, 0.001]
# alg = './CoarseUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
#     print (cmd)
#     os.system(cmd)

# # SimCoarseUCB
# LOG_FILE = '../test_results/log_SimCoarseUCB'
# alg = './SimCoarseUCB.py '
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# # alpha = [5, 1, 0.1, 0.01, 0.001]
# alpha = [1]
# # alpha = 1
# # cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(0) + ' ' + str(alpha) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
# # print (cmd)
# # os.system(cmd)
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
#     print (cmd)
#     os.system(cmd)
# # # alpha = 1
# # # COARSE_DIM = [4, 5, 6, 7, 8, 9, 10]
# # # alg = './SimCoarseUCB.py '
# # # for i in range(len(COARSE_DIM)):
# # #     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT) + ' ' + str(COARSE_DIM[i])
# # #     print (cmd)
# # #     os.system(cmd)


# # NeuralUCB
# LOG_FILE = '../test_results/log_NeuralUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# # alpha = [5, 1, 0.1, 0.01, 0.001]
# alpha = [1, 0.1, 0.01, 0.001, 0.0001, 0]
# alg = './NeuralUCB.py '
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
#     print (cmd)
#     os.system(cmd)



# # NeuralCoarseUCB
# LOG_FILE = '../test_results/log_NeuralCoarseUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# # alpha = [5, 1, 0.1, 0.01, 0.001]
# # alpha = [0.001, 0.0001]
# alg = './NeuralCoarseUCB.py '
# for i in range(10):
#     if i < 5:
#         alpha = 0.0001
#     else:
#         alpha = 0.001
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
#     print (cmd)
#     os.system(cmd)


# # SimPCoarseUCB
# LOG_FILE = '../test_results/log_SimPCoarseUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# alpha = [5, 1, 0.1, 0.01, 0.001]
# alg = './SimPCoarseUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
#     print (cmd)
#     os.system(cmd)


# # CaredCoarseUCB
# LOG_FILE = '../test_results/log_CaredCoarseUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# alpha = [5, 1, 0.1, 0.01, 0.001]
# alg = './CaredCoarseUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i]) + ' ' + str(VMAF_REBUF_PENALTY_1) + ' ' + str(QUAITY_WEIGHT)
#     print (cmd)
#     os.system(cmd)


# # for the BB9_LinUCB with 9d context, i.e., BB9_LinUCB0-5
# LOG_FILE = '../test_results/log_BB9_LinUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# # alpha = [5, 1, 0.1, 0.01, 0.001]
# alpha = [5, 1, 0.1, 0.01]
# # alpha = [5]
# # alpha = 0.0001
# alg = './BB9_LinUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i])
#     print cmd
#     os.system(cmd)


# # for the LinUCB with 14d context, with 14 chunks context in history i.e., ctx14_14_LinUCB0-5
# LOG_FILE = '../test_results/log_ctx14_14_LinUCB'
# alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# # alpha = [5, 1, 0.1, 0.01]
# alg = './ctx14_14_LinUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i])
#     print cmd
#     os.system(cmd)



# # for the LinUCB with 10d context, i.e., ctx10_LinUCB0-5
# LOG_FILE = '../test_results/log_ctx10_LinUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# alpha = [5, 1, 0.1, 0.01, 0.001]
# # alpha = [5, 1, 0.1, 0.01]
# alg = './ctx10_LinUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i])
#     print cmd
#     os.system(cmd)


# # for the LinUCB with 15d context, i.e., ctx15_LinUCB0-5
# LOG_FILE = '../test_results/log_ctx15_LinUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# alpha = [5, 1, 0.1, 0.01, 0.001]
# # alpha = [5, 1, 0.1, 0.01]
# alg = './ctx15_LinUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i])
#     print cmd
#     os.system(cmd)




# # for the LinUCB with 14d context, i.e., ctx14_LinUCB0-5
# LOG_FILE = '../test_results/log_ctx14_LinUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# alpha = [5, 1, 0.1, 0.01, 0.001]
# # alpha = [5, 1, 0.1, 0.01]
# alg = './ctx14_LinUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i])
#     print cmd
#     os.system(cmd)


# # for the BB3_LinUCB with 3d context, i.e., BB3_LinUCB0-5
# LOG_FILE = '../test_results/log_BB3_LinUCB'
# # alpha = [5, 1, 0.1, 0.01, 0.001, 0.0001]
# alpha = [5, 1, 0.1, 0.01]
# # alpha = 0.0001
# alg = './BB3_LinUCB.py '
#
# for i in range(len(alpha)):
#     cmd = "python " + alg + TEST_TRACES + ' ' + LOG_FILE + str(i) + ' ' + str(alpha[i])
#     print cmd
#     os.system(cmd)



# # robust_MPC1
# LOG_FILE = '../test_results/log_robustMPC1'
# alg = './sim_robust_mpc1.py '
#
# cmd_robust_MPC1 = "python " + alg + TEST_TRACES + ' ' + LOG_FILE
# print cmd_robust_MPC1
# begin_time = time.time()
# os.system(cmd_robust_MPC1)
# end_time = time.time()
# running_time = end_time - begin_time
# print ('running_time for MPC1: ', running_time)