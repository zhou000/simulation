from sklearn.metrics.pairwise import rbf_kernel
import numpy as np


class KernelUCB():

    def __init__(self, narms, ndims, gamma, eta, kern):
        # Set number of arms
        self.narms = narms
        # Number of context features
        self.ndims = ndims
        # regularization parameter
        self.eta = eta
        # exploration parameter
        self.gamma = gamma
        # kernel function
        self.kern = kern
        # u_n_t values
        self.u = np.zeros(self.narms)
        # sigma_n_t values
        self.sigma = np.zeros(self.narms)
        # list of contexts of chosen actions to the moment
        self.pulled = []
        # list of rewards corresponding to chosen actions to the moment
        self.rewards = []
        # define a dictionary to store kernel matrix inverse in each tround
        self.Kinv = {}
        return

    def play(self, tround, context):
        self.tround = tround
        # get the flattened context and reshape it to an array of shape (narms,ndims)
        context = np.reshape(context, (self.narms, self.ndims))

        if self.tround == 0:
            # playing action 1 for first round and setting u_0[0] to 1.0
            self.u[0] = 1.0
        else:

            # ========================================
            #    Calculating all possible k_x ...
            # ========================================

            # To perform kernel UCB in the least and efficient time as possible I propose to
            # calculate k_x for all of the contexts and not just for chosen context (x_t)
            # this will be hugely beneficiary to calculating sigma_n_t step in for loop

            # calculate the kernel between each of the contexts of narms and the pulled
            # contexts of chosen arms to the moment

            # self.pulled is just a list of arrays, and hence reshaping it to a valid
            # numpy array of shape (tround+1,ndims). Since tround is starting from zero
            # it is being added by 1 to give valid shape in each round especially for
            # the first round

            k_x = self.kern(context, np.reshape(self.pulled, (self.tround, self.ndims)))

            # ===============================
            #    MAIN LOOP ...
            # ===============================

            for i in range(self.narms):
                self.sigma[i] = np.sqrt(self.kern(context[i].reshape(1, -1), context[i].reshape(1, -1)) - k_x[i].T.dot(
                    self.Kinv[self.tround - 1]).dot(k_x[i]))
                self.u[i] = k_x[i].T.dot(self.Kinv[self.tround - 1]).dot(self.y) + (self.eta / np.sqrt(self.gamma)) * \
                            self.sigma[i]

        # tie breaking arbitrarily
        action = np.random.choice(np.where(self.u == max(self.u))[0])
        # np.argmax returns values 0-9, we want to compare with arm indices in dataset which are 1-10
        # Hence, add 1 to action before returning
        return action + 1

    def update(self, arm, reward, context):
        # get the flattened context and reshape it to an array of shape (narms,ndims)
        context = np.reshape(context, (self.narms, self.ndims))
        # append the context of choesn arm (index = [arm]) with the previous list of contexts (self.pulled)
        # the obserbved context is being reshaped into a column vector simultanesously for future kernel calculations
        self.pulled.append(context[arm].reshape(1, -1))
        # set currently observed context of chosen arm as x_t
        x_t = context[arm].reshape(1, -1)

        # ========================================
        #    Calculating all possible k_x ...
        # ========================================

        # To perform kernel UCB in the least and efficient time as possible I propose to
        # calculate k_x for all of the contexts and not just for chosen context (x_t)
        # this will be hugely beneficiary to calculating sigma_n_t step in for loop

        # calculate the kernel between each of the contexts of narms and the pulled
        # contexts of chosen arms to the moment

        # self.pulled is just a list of arrays, and hence reshaping it to a valid
        # numpy array of shape (tround+1,ndims). Since tround is starting from zero
        # it is being added by 1 to give valid shape in each round especially for
        # the first round
        k_x = self.kern(context, np.reshape(self.pulled, (self.tround + 1, self.ndims)))

        # append the observed reward value of chosen action to the previous list of rewards
        self.rewards.append(reward)
        # generate array of y. Since tround is starting from zero
        # it is being added by 1 to give valid shape in each round especially for
        # the first round
        self.y = np.reshape(self.rewards, (self.tround + 1, 1))

        # building inverse of kernel matrix for first round is different from consequent rounds.
        if self.tround == 0:
            self.Kinv[self.tround] = 1.0 / (self.kern(x_t, x_t) + self.gamma)
        else:
            # set inverse of kernel matrix as the kernel matrix inverse of the previous round
            Kinv = self.Kinv[self.tround - 1]
            # set b as k_(x_t) excluding the kernel value of the current round
            b = k_x[arm][:-1]
            # reshape b into the valid numpy column vector
            b = b.reshape(self.tround, 1)
            # compute b.T.dot(kernel matrix inverse)
            bKinv = np.dot(b.T, Kinv)
            # compute (kernel matrix inverse).dot(b)
            Kinvb = np.dot(Kinv, b)

            # ==========================================================================
            #    Calculating components of current Kernel matrix inverse (Kinv_tround)
            # ==========================================================================

            K22 = 1.0 / (k_x[arm][-1] + self.gamma - np.dot(bKinv, b))
            K11 = Kinv + K22 * np.dot(Kinvb, bKinv)
            K12 = -K22 * Kinvb
            K21 = -K22 * bKinv
            K11 = np.reshape(K11, (self.tround, self.tround))
            K12 = np.reshape(K12, (self.tround, 1))
            K21 = np.reshape(K21, (1, self.tround))
            K22 = np.reshape(K22, (1, 1))
            # stack components into an array of shape(self.tround, self.tround)
            self.Kinv[self.tround] = np.vstack((np.hstack((K11, K12)), np.hstack((K21, K22))))


# def offlineEvaluate(mab, arms, rewards, contexts, nrounds=None):
#     # array to contain chosen arms in offline mode
#     chosen_arms = np.zeros(nrounds)
#     # rewards of each chosen arm
#     reward_arms = np.zeros(nrounds)
#     # cumulative reward at each iteration
#     cumulative_reward = np.zeros(nrounds)
#     # initialize tround to zero
#     T = 0
#     # initialize overall cumulative reward to zero
#     G = 0
#     # History or memory of offline evaluator
#     history = []
#     # play once and get the initial action
#     action = mab.play(T, contexts[0, :])
#
#     # ===============================
#     #    MAIN LOOP ...
#     # ===============================
#     for i in range(np.shape(data)[0]):
#         action = mab.play(T, contexts[i, :])
#         if T < nrounds:
#             # update parameters and play only when chosen arm from bandit matches data
#             if action == arms[i]:
#                 # append the current context of chosen arm to the previous history (list)
#                 history.append(contexts[i, :])
#                 # get the reward of chosen arm at round T
#                 reward_arms[T] = rewards[i]
#                 # the returned action is between 1-10, setting to python encoding ==> 0-9
#                 mab.update(action - 1, rewards[i], contexts[i, :])
#                 # update overall cumulative reward
#                 G += rewards[i]
#                 # update cumulative reward of round T
#                 cumulative_reward[T] = G
#                 # store chosen arm at round T
#                 chosen_arms[T] = action
#                 T += 1
#         else:
#             # if desired tround ends, terminate the loop
#             break
#     return reward_arms, chosen_arms, cumulative_reward
#
# if __name__ == '__main__':
#     data = np.loadtxt('./dataset.txt')
#     arms = data[:, 0]
#     rewards = data[:, 1]
#     contexts = data[:, 2:102]
#     mab = KernelUCB(10, 10, 1.0, 1.0, rbf_kernel)
#     results_KernelUCB, chosen_arms_KernelUCB, cumulative_reward_KernelUCB = offlineEvaluate(mab, arms, rewards,
#                                                                                             contexts, 800)
#     print('KernelUCB average reward', np.mean(results_KernelUCB))
