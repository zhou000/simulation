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
import torch
import torch.nn as nn
sns.set()

def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = np.dot(A_inv, u)
    A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))
    return A_inv

class Model(nn.Module):
    """Template for fully connected neural network for scalar approximation.
    """
    def __init__(self,
                 input_size=1,
                 hidden_size=2,
                 n_layers=1,
                 activation='ReLU',
                 p=0.0,
                 ):
        super(Model, self).__init__()

        self.n_layers = n_layers

        if self.n_layers == 1:
            self.layers = [nn.Linear(input_size, 1)]
        else:
            size = [input_size] + [hidden_size, ] * (self.n_layers-1) + [1]
            self.layers = [nn.Linear(size[i], size[i+1]) for i in range(self.n_layers)]
        self.layers = nn.ModuleList(self.layers)

        # dropout layer
        self.dropout = nn.Dropout(p=p)

        # activation function
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise Exception('{} not an available activation'.format(activation))

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.dropout(self.activation(self.layers[i](x)))
        x = self.layers[-1](x)
        return x


# Bandit settings
T = int(5e2)        # 5*10^2 = 500
# n_arms = 4
n_arms = 6
n_features = 16     # number of parameters
noise_std = 0.1

confidence_scaling_factor = noise_std

# n_sim = 2
n_sim = 1

SEED = 42
np.random.seed(SEED)

# Neural network settings
p = 0.2
hidden_size = 64
epochs = 100
train_every = 10
confidence_scaling_factor = 1.0
use_cuda = False


a = np.random.randn(n_features)     # not context, just for reward function generation
a /= np.linalg.norm(a, ord=2)
h = lambda x: np.cos(10*np.pi*np.dot(x, a))

ctx = np.random.randn(T, n_arms, n_features)
ctx /= np.repeat(np.linalg.norm(ctx, axis=-1, ord=2), n_features).reshape(T, n_arms, n_features)

# bandit = ContextualBandit(T, n_arms, n_features, h, noise_std=noise_std, seed=SEED)

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
    # model = NeuralUCB(bandit,
    #                   hidden_size=hidden_size,
    #                   reg_factor=1.0,
    #                   delta=0.1,
    #                   confidence_scaling_factor=confidence_scaling_factor,
    #                   training_window=100,
    #                   p=p,
    #                   learning_rate=0.01,
    #                   epochs=epochs,
    #                   train_every=train_every,
    #                   use_cuda=use_cuda,
    #                  )
    # model.run()
    # regrets[i] = np.cumsum(model.regrets)



    # hidden_size = 20
    hidden_size = 64
    n_layers = 2
    reg_factor = 1.0
    # delta = 0.01
    delta = 0.1
    # confidence_scaling_factor = -1.0
    confidence_scaling_factor = 1.0
    training_window = 100
    # p = 0.0
    p = 0.2
    learning_rate = 0.01
    # epochs = 1
    epochs = 100
    # train_every = 1
    train_every = 10
    throttle = 1
    use_cuda = False

    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    # neural network
    model = Model(input_size=n_features,
                       hidden_size=hidden_size,
                       n_layers=n_layers,
                       p=p
                       ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # maximum L2 norm for the features across all arms and all rounds
    bound_features = np.max(np.linalg.norm(ctx, ord=2, axis=-1))

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
    # approximator_dim = n_features
    """Sum of the dimensions of all trainable layers in the network.
            """
    approximator_dim = sum(w.numel() for w in model.parameters() if w.requires_grad)
    A_inv = np.array(
        [
            np.eye(approximator_dim) / reg_factor for _ in range(n_arms)
        ]
    )

    """Initialize the gradient of the approximator w.r.t its parameters.
            """
    grad_approx = np.zeros((n_arms, approximator_dim))


    for t in range(T):
        # context for each arm in round t
        context = ctx[t]

        # # update confidence of all arms based on observed features at time t
        # model.update_confidence_bounds(context)

        """NeuralUCB confidence interval multiplier.
        """
        bound_features = np.max(np.linalg.norm(context, ord=2, axis=-1))
        # bound_features = np.max(np.linalg.norm(ctx, ord=2, axis=-1))
        # confidence_multiplier = confidence_scaling_factor * np.sqrt(n_features * np.log(
        #     1 + t * bound_features ** 2 / (reg_factor * n_features)) + 2 * np.log(
        #     1 / delta)) + np.sqrt(reg_factor) * bound_theta
        confidence_multiplier = confidence_scaling_factor*np.sqrt(approximator_dim * np.log(
                1 + t * bound_features ** 2 / (reg_factor * approximator_dim)
            ) + 2 * np.log(1 / delta)
        )

        """LinUCB confidence interval multiplier.
                """
        # self.update_output_gradient()
        # grad_approx = context
        """Get gradient of network prediction w.r.t network weights.
                """
        for a in range(n_arms):
            x = torch.FloatTensor(
                context[a].reshape(1, -1)
            ).to(device)
            model.zero_grad()
            y = model(x)
            y.backward()
            grad_approx[a] = torch.cat(
                [w.grad.detach().flatten() / np.sqrt(hidden_size) for w in model.parameters() if
                 w.requires_grad]
            ).to(device)
        # UCB exploration bonus
        exploration_bonus[t] = np.array(
            [
                confidence_multiplier * np.sqrt(
                    np.dot(grad_approx[a], np.dot(A_inv[a], grad_approx[a].T))) for a in range(n_arms)
            ]
        )
        # # update reward prediction mu_hat
        # self.predict(context)
        # mu_hat[t] = np.array(
        #     [
        #         # np.dot(self.bandit.features[self.iteration, a], self.theta[a]) for a in self.bandit.arms
        #         np.dot(context[a], theta[a]) for a in range(n_arms)
        #     ]
        # )
        model.eval()
        mu_hat[t] = model.forward(torch.FloatTensor(context).to(device)).detach().squeeze()
        # mu_hat[t] = model.forward(torch.FloatTensor(ctx[t]).to(device)).detach().squeeze()
        # estimated combined bound for reward
        upper_confidence_bounds[t] = mu_hat[t] + exploration_bonus[t]

        # # pick action with the highest boosted estimated reward
        # model.action = model.sample_action()
        action = np.argmax(upper_confidence_bounds[t]).astype('int')
        # model.actions[t] = model.action
        actions[t] = action

        reward_t = rewards[t, action]

        # # update approximator
        if t % train_every == 0:
            """Train neural approximator.
                    """
            iterations_so_far = range(np.max([0, t - training_window]), t + 1)
            actions_so_far = actions[np.max([0, t - training_window]):t + 1]

            x_train = torch.FloatTensor(ctx[iterations_so_far, actions_so_far]).to(device)
            y_train = torch.FloatTensor(rewards[iterations_so_far, actions_so_far]).squeeze().to(device)

            # train mode
            model.train()
            for _ in range(epochs):
                y_pred = model.forward(x_train).squeeze()
                loss = nn.MSELoss()(y_train, y_pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # """Update linear predictor theta.
        # """
        # theta = np.array(
        #     [
        #         np.matmul(A_inv[a], b[a]) for a in range(n_arms)
        #     ]
        # )
        # # b[action] += context[action]*rewards[t, action]
        # b[action] += context[action] * reward_t

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


