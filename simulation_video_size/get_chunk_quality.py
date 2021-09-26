import os
import numpy as np

# TOTAL_VIDEO_CHUNCK = 49
# TOTAL_VIDEO_CHUNCK = 1251
TOTAL_VIDEO_CHUNCK = 4600
BITRATE_LEVELS = 6
QUALITY_LEVELS = 4  # Q1 is simple scene, Q4 is complex scene
# VIDEO_SIZE_FILE = './synthetic_video_size_BBB_ED/video_size_'
VIDEO_SIZE_FILE = './4600chunks/video_size_'
# video_size_0 is 240p, video_size_5 is 1440p
# OUTPUT = 'video_quality_level/video_quality_level'
OUTPUT = 'video_quality_level/4600chunks'


video_size = {}  # in bytes
for bitrate in xrange(BITRATE_LEVELS):
    video_size[bitrate] = []
    print (VIDEO_SIZE_FILE + str(bitrate))
    with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
        for line in f:
            video_size[bitrate].append(int(line.split()[0]))


# size_video1 = [1990396, 2022752, 1791162, 1578094, 1745447, 1506337, 1921137, 4076926, 4120285, 4796865, 4305978,
#                3607047, 2508016, 1905270, 3090816, 3030914, 1895602, 1830889, 1282559, 1509759, 2342524, 2532716,
#                1860386, 2028583, 4977855, 3866551, 2189837, 4421514, 8184547, 11164105, 7346171, 8225748, 10083912,
#                8817722, 3277718, 4395630, 4628333, 2426752, 1622178, 1629064, 1343951, 1325054, 1508791, 1084865,
#                2845482, 3640973, 1964055, 1402393, 994005]
# size_video2 = [1146167, 1205448, 1108469, 1105767, 1167898, 930625, 1173183, 2758758, 2504921, 3297255, 3063272,
#                2568309, 1677653, 1198125, 1992396, 1999759, 1317286, 1253677, 863691, 1033690, 1640811, 1850449,
#                1406727, 1528201, 3626824, 2850183, 1741643, 3261415, 5839767, 7676663, 5305126, 6037747, 7121651,
#                6332164, 2467852, 3283336, 3101464, 1682907, 982605, 1043936, 816658, 843785, 981505, 681810, 1831110,
#                2326202, 1202358, 876752, 635782]
# size_video3 = [532471, 613986, 519850, 510818, 558129, 450777, 575002, 1555907, 1258359, 1730305, 1614587, 1279723,
#                838639, 606116, 1062342, 1051782, 618830, 620050, 439259, 489999, 909492, 1056134, 869967, 943159,
#                2057506, 1591363, 953491, 1826316, 3134601, 4225368, 2887492, 3230674, 3813774, 3341851, 1276866,
#                1846699, 1662750, 950209, 475012, 511138, 399562, 421840, 515693, 372101, 961248, 1244005, 620150,
#                444245, 293094]
# size_video4 = [208763, 255768, 197181, 216123, 199200, 162995, 246365, 779126, 549777, 714882, 660038, 500321, 334155,
#                276150, 509273, 495131, 213866, 244038, 188370, 214767, 424835, 501738, 419198, 454395, 986993, 733872,
#                434832, 834446, 1414281, 1765912, 1248573, 1359749, 1575451, 1430893, 529758, 831656, 760631, 446062,
#                208251, 220328, 162289, 177247, 230090, 173347, 448092, 612541, 261453, 192522, 126822]
# size_video5 = [145365, 186972, 153146, 157474, 155273, 124967, 181041, 598585, 412092, 512197, 464357, 352637, 239827,
#                211239, 392577, 376945, 144179, 172475, 137492, 164296, 293223, 347218, 271325, 299583, 663190, 510726,
#                297138, 585943, 961765, 1214105, 818038, 873040, 1076229, 919447, 354234, 570429, 548296, 314214, 137783,
#                147922, 119190, 128692, 166600, 133067, 332767, 442367, 185739, 143181, 95339]
# size_video6 = [70240, 87117, 67601, 80583, 69776, 54500, 94123, 336407, 216714, 227729, 202079, 161582, 112727, 109402,
#                199870, 190940, 56223, 82553, 70655, 82660, 145494, 167756, 117017, 134624, 297780, 256080, 148477,
#                274134, 426848, 524783, 341286, 376760, 462212, 406573, 164541, 261521, 273718, 140811, 61004, 64942,
#                55883, 61864, 84791, 70271, 163799, 218083, 94045, 73164, 48000]

# bitrate_1 = np.array(size_video1)
# bitrate_2 = np.array(size_video2)
# bitrate_3 = np.array(size_video3)
# bitrate_4 = np.array(size_video4)
# bitrate_5 = np.array(size_video5)
# bitrate_6 = np.array(size_video6)

bitrate_1 = np.array(video_size[5])
bitrate_2 = np.array(video_size[4])
bitrate_3 = np.array(video_size[3])
bitrate_4 = np.array(video_size[2])
bitrate_5 = np.array(video_size[1])
bitrate_6 = np.array(video_size[0])


# all chunks in an array
video_chunks = np.vstack((bitrate_1, bitrate_2, bitrate_3, bitrate_4, bitrate_5, bitrate_6))
# print video_chunks

# the average bitrates are saved in avg
avg = np.zeros(BITRATE_LEVELS)

# sort all video chunks
total_chunks = BITRATE_LEVELS * TOTAL_VIDEO_CHUNCK
sort_list = np.zeros(total_chunks).reshape(BITRATE_LEVELS, TOTAL_VIDEO_CHUNCK)
# the sorted video chunks' index
sort_index = np.zeros(total_chunks).reshape(BITRATE_LEVELS, TOTAL_VIDEO_CHUNCK)
# print sort_list

# count the number of chunks in each quality level
NO_in_each_QL = np.zeros(BITRATE_LEVELS * QUALITY_LEVELS).reshape(BITRATE_LEVELS, QUALITY_LEVELS)
# print NO_in_each_QL

# count all kind of bitrates' quality level for each chunk
count_QL_4_each_chunk = np.zeros(TOTAL_VIDEO_CHUNCK * QUALITY_LEVELS).reshape(TOTAL_VIDEO_CHUNCK, QUALITY_LEVELS)

# the first method to distinguish each chunk's quality level:
# compare chunk size among all video chunks,
# based on the average of all video chunk size (1/2avg, avg, 3/2avg, others),
# for each chunk, from Q1 to Q4, while some kind of quality level (Q2) has 3 (or more) chunks with different bit-rate,
# that video chunk will be seem as Q2, the prior of each quality level is: Q1 >= Q2 >= Q3 >= Q4
# for example: count_QL_4_each_chunk[j] = [0, 3, 3, 0], then the quality of the j-th video chunk is Q2
QL_scheme_1 = np.zeros(TOTAL_VIDEO_CHUNCK, int)

# compare video size among all video chunks
for i in range(BITRATE_LEVELS):
    avg[i] = np.average(video_chunks[i])
    sort_list[i] = np.sort(video_chunks[i])
    sort_index[i] = np.argsort(video_chunks[i])
    # QL = 0
    for j in range(TOTAL_VIDEO_CHUNCK):
        if video_chunks[i][j] < avg[i] / 2:
            NO_in_each_QL[i][0] += 1
            count_QL_4_each_chunk[j][0] += 1
        elif video_chunks[i][j] < avg[i]:
            NO_in_each_QL[i][1] += 1
            count_QL_4_each_chunk[j][1] += 1
        elif video_chunks[i][j] < avg[i] * 3 / 2:
            NO_in_each_QL[i][2] += 1
            count_QL_4_each_chunk[j][2] += 1
        else:
            NO_in_each_QL[i][3] += 1
            count_QL_4_each_chunk[j][3] += 1

with open(OUTPUT, 'wb') as f:
    for j in range(TOTAL_VIDEO_CHUNCK):
        for q in range(QUALITY_LEVELS):
            if count_QL_4_each_chunk[j][q] >= 3:
                QL_scheme_1[j] = q + 1
                break
        f.write(str(QL_scheme_1[j]) + '\n')

print avg
# print sort_list
# print NO_in_each_QL
# print sort_index
# print count_QL_4_each_chunk
print QL_scheme_1

# print avg1
# algorithm 1, compare video_chunk_size among 49 video chunks
# for chunk_No in xrange(TOTAL_VIDEO_CHUNCK):
#     for i in xrange(BITRATE_LEVELS):

# bitrate_1.sort()sim/get_chunk_quality.py:44

# print bitrate_1
# print size_video1
# print (np.argsort(bitrate_1))
# print (np.argsort(size_video1))


# for i in xrange(TOTAL_VIDEO_CHUNCK / 4):
#     print (np.argsort(size_video1))


# print TOTAL_VIDEO_CHUNCK / 4, TOTAL_VIDEO_CHUNCK / 2, TOTAL_VIDEO_CHUNCK * 3/4, TOTAL_VIDEO_CHUNCK

# for i in range(0, Q1):
#     print i
# print '-----------------------------------'
# for j in range(Q1, Q2):
#     print j
# print '-----------------------------------'
# for k in range(Q2, Q3):
#     print k
# print '-----------------------------------'
# for l in range(Q3, Q4):
#     print l
