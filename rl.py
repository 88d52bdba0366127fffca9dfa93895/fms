"""
This is a Reinforcement learning approach for fms framework

How to run: python3 -W ignore rl.py

"""
import os
import numpy as np
import pandas as pd

import gym
from gym import spaces
from subprocess import Popen, PIPE
from scipy.stats import norm
from pandas.tools.plotting import autocorrelation_plot
import scipy.stats.stats as st
from gym import error, spaces


def f_autocorr(x, t=1):
    return np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))


class VN_Market(gym.Env):
    """
    Vietnam market environment
    """
    def __init__(self):
        """
        Init the environment, this is a `_reset` function however we don't need
        the `reset` function, so I put the code here
        """
        input_path = 'data/vnindex.csv'
        df = pd.read_csv(input_path)
        df['return'] = df['close'].pct_change()
        for index, row in df.iterrows():
            if index < 100: continue
            data = df.iloc[index - 100:index]['return']
            mu, sigma = norm.fit(data)
            skew, kurtosis = st.skew(data), st.kurtosis(data)
            autocorr = f_autocorr(data.abs())[0, 1]
            df.loc[index, 'mu'] = mu
            df.loc[index, 'sigma'] = sigma
            df.loc[index, 'skew'] = skew
            df.loc[index, 'kurtosis'] = kurtosis
            df.loc[index, 'autocorr'] = autocorr
        #
        df.to_csv(input_path, index=False)
        self.df = df
        # self.df = pd.read_csv(input_path)
        self.sim_df = pd.DataFrame()
        # init parameters for fms
        self.total_number = 10000
        self.init_price = 100000
        # The Space object corresponding to valid observations
        self.obs_mu = [-0.0102, -0.0011, 0.0001, 0.0016, 0.0140]
        self.obs_sigma = [0.0028, 0.0084, 0.0120, 0.0159, 0.0492]
        self.obs_skew = [-2.0660, -0.2824, 0.0388, 0.3409, 2.6633]
        self.obs_kurtosis = [-1.47, -0.24, 0.34, 1.40, 16.19]
        self.observation_space = None
        self.observation_space_n = \
            len(self.obs_mu) * \
            len(self.obs_sigma) * \
            len(self.obs_skew) * \
            len(self.obs_kurtosis)
        # A tuple corresponding to the min and max possible rewards
        self.reward_range = (-np.inf, 0)
        #
        self.zero_pct = 0.3
        self.herding_pct = 0.3

    def _get_index(self, _list: list, _value: float):
        return _list.index(min(_list, key=lambda x: abs(x - _value)))

    def _get_observation(self, real_values: dict):
        """
        Get the index of observation from the given real values as dictionary
        """
        mu_index = self._get_index(self.obs_mu, real_values['mu'])
        sigma_index = self._get_index(self.obs_sigma, real_values['sigma'])
        skew_index = self._get_index(self.obs_skew, real_values['skew'])
        kurtosis_index = self._get_index(self.obs_kurtosis,
                                         real_values['kurtosis'])
        index = mu_index * (len(self.obs_sigma) *
                            len(self.obs_skew) *
                            len(self.obs_kurtosis)) + \
            sigma_index * (len(self.obs_skew) *
                           len(self.obs_kurtosis)) + \
            skew_index * len(self.obs_kurtosis) + \
            kurtosis_index
        return index

    def _step(self, action: tuple, i: int):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (float): an action provided by the environment
            i (int): current index of action

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        real_values = {
            'mu': self.df.iloc[i]['mu'],
            'sigma': self.df.iloc[i]['sigma'],
            'skew': self.df.iloc[i]['skew'],
            'kurtosis': self.df.iloc[i]['kurtosis'],
        }
        info = self._do_action(action=action, real_values=real_values)
        reward_dict = self._get_reward(real_values=real_values, i=i)
        observation = self._get_observation(real_values=real_values)
        error = reward_dict.get('error', 0)
        # we never end the process
        done = False
        # store the data (action, observation and reward)
        info.update(reward_dict)
        self.sim_df = self.sim_df.append(info, ignore_index=True)
        return observation, error, done, info

    def _do_action(self, action: tuple, real_values: dict):
        """ Converts the action space into a fms' action then do it """
        zero_action = action[0]
        herding_action = action[1]
        self.zero_pct = self.zero_pct + zero_action
        self.herding_pct = self.herding_pct + herding_action
        threshold_pct = 1 - self.zero_pct - self.herding_pct
        info = {
            'zero_action': zero_action,
            'herding_action': herding_action,
            'zero_pct': self.zero_pct,
            'herding_pct': self.herding_pct,
        }
        #
        zero_number = str(int(self.zero_pct * self.total_number))
        herding_number = str(int(self.herding_pct * self.total_number))
        threshold_number = str(int(threshold_pct * self.total_number))
        #
        with open('config.yml.origin', 'r') as f:
            config = f.read()
        #
        config = config.replace('{seed}', str(np.random.randint(99999)))
        config = config.replace('{zero_number}', zero_number)
        config = config.replace('{herding_number}', herding_number)
        config = config.replace('{threshold_number}', threshold_number)
        #
        with open('config.yml', 'w') as f:
            f.write(config)
        #
        #
        with open('zerointelligencetrader.py.origin', 'r') as f:
            zero_agent = f.read()
        #
        zero_agent = zero_agent.replace('{mu}', str(real_values['mu']))
        zero_agent = zero_agent.replace('{sigma}', str(real_values['sigma']))
        #
        zero_file = 'fms/agents/zerointelligencetrader.py'
        with open(zero_file, 'w') as f:
            f.write(zero_agent)
        #
        process = Popen(
            ['python2.7', 'startfms.py', 'run', 'config.yml'],
            stdout=PIPE,
            stderr=PIPE)
        stdout, stderr = process.communicate()
        if len(stdout) != 0:
            print('STDOUT', stdout)
        if len(stderr) != 0:
            print('STDERR', stderr)
            import sys; sys.exit(0)
        #
        return info

    def _get_reward(self, real_values: dict, i: int):
        """
        Get the reward returned after previous action
        """
        df = pd.read_csv('output.csv', skiprows=[0], sep=';')
        last_return = df['price'].values[-1] / self.init_price - 1
        reward = {'return': last_return}
        if i < 100:# + 1
            return reward
        #
        returns = self.sim_df.tail(99)['return'].dropna().values + [last_return]
        mu, sigma = norm.fit(returns)
        skew, kurtosis = st.skew(returns), st.kurtosis(returns)
        # autocorr = f_autocorr(np.abs(returns))[0, 1]
        reward.update({
            'mu': mu,
            'sigma': sigma,
            'skew': skew,
            'kurtosis': kurtosis,
            # 'autocorr': autocorr,
        })
        # error = {
        #     k: np.abs((reward[k] - real_values[k])**2 / real_values[k])
        #     for k, v in reward.items() if k != 'return'
        # }
        sub_df = self.df.iloc[i-100:i]
        error = {
            k: ((reward[k] - sub_df[k].mean()) / sub_df[k].std())**2
            for k, v in reward.items() if k != 'return'
        }
        reward['error'] = -sum(error.values())
        os.remove('output.csv')
        return reward

    def _reset(self):
        """
        Reset all to the index 100
        """
        i = 100
        real_values = {
            'mu': self.df.iloc[i]['mu'],
            'sigma': self.df.iloc[i]['sigma'],
            'skew': self.df.iloc[i]['skew'],
            'kurtosis': self.df.iloc[i]['kurtosis'],
        }
        observation = self._get_observation(real_values=real_values)
        self.sim_df = self.df[['return']].head(99)
        self.zero_pct = 0.3
        self.herding_pct = 0.3
        info = {
            'zero_pct': self.zero_pct,
            'herding_pct': self.herding_pct,
        }
        return observation, info

    def _render(self, mode='human', close=False): pass

    def _seed(self, seed=None): pass


class VN_FMS(object):
    """
    FMS agent that will interact with VN Market and then simulate in the FMS
    """
    def __init__(self, observation_space_n):
        self.zero_actions = [-0.025, 0, 0.025]
        self.herding_actions = [-0.025, 0, 0.025]
        self.action_space_n = len(self.zero_actions) * len(self.herding_actions)
        self.action_space = list(range(self.action_space_n))
        #
        self.df = pd.DataFrame()
        # tabular Q-learning
        self.Q = np.zeros([observation_space_n, self.action_space_n])
        #
        self.learning_rate = 0.1
        self.eps = 0.05            # Epsilon in epsilon greedy policies
        self.discount = 0.95
        self.n_iter = 1000         # Number of iterations

    def _get_action(self, index: int):
        """
        Get the action values from the given index
        """
        zero_index = int(index / len(self.zero_actions))
        herding_index = int(index % len(self.herding_actions))
        action = (self.zero_actions[zero_index],
                  self.herding_actions[herding_index])
        return action

    def _is_valid_action(self, info: dict, action: tuple):
        zero_pct = info['zero_pct'] + action[0]
        herding_pct = info['herding_pct'] + action[1]
        if zero_pct <= 0 or zero_pct >= 1: return False
        if herding_pct <= 0 or herding_pct >= 1: return False
        if herding_pct > 1 - zero_pct: return False
        return True

    def _random_action(self, info: dict):
        valid_action_space = [i for i in self.action_space
                              if self.Q[observation, i] != -np.inf]
        if len(valid_action_space) == 0:
            print('INVALID ACTION SPACE len = 0', self.Q[observation])
            action_index = np.random.choice(self.action_space)
        else:
            action_index = np.random.choice(valid_action_space)
        action = self._get_action(index=action_index)
        if not self._is_valid_action(info=info, action=action):
            return self._random_action(info=info)
        print('RANDOM', action)
        return action_index, action

    def act(self, observation: int, info: dict, i: int):
        """
        Agent's action
        Args:
            observation (int): index of observation from environment
        """
        if np.random.random() < self.eps:
            return self._random_action(info=info)
        action_index = np.argmax(self.Q[observation])
        action = self._get_action(index=action_index)
        #
        if not self._is_valid_action(info=info, action=action):
            self.Q[observation, action_index] = -np.inf
            return self.act(observation=observation, info=info, i=i)
        #
        return action_index, action

    def learning(
            self,
            action: int,
            observation: int,
            next_observation: int,
            reward: float,
    ):
        """
        Update Q-matrix
        """
        if reward is np.nan: return False
        # Q-learning
        self.Q[observation, action] = self.Q[observation, action] + \
            self.learning_rate * \
            (reward +
             self.discount * np.max(self.Q[next_observation, :]) -
             self.Q[observation, action])
        return True


if __name__ == "__main__":
    # initialize gym environment and our FMS agent
    env = VN_Market()
    agent = VN_FMS(observation_space_n=env.observation_space_n)
    # backup Q-lerning
    Q_dict = dict()
    error_dict = dict()
    # iterate through first 1000 days for training, 3000 days later for testing
    for _iter in range(10, 20):
        observation, info = env.reset()
        for i in range(101, 1000):
            action_index, action = agent.act(
                observation=observation,
                info=info,
                i=i
            )
            next_observation, reward, done, info = env._step(action=action, i=i)
            _ = agent.learning(
                action=action_index,
                observation=observation,
                next_observation=next_observation,
                reward=reward
            )
            observation = next_observation
            try:
                print('+++ iter #{}, index #{}: reward {}'.format(
                    _iter, i, reward))
                print(env.sim_df.tail(1)[[
                    'zero_pct', 'herding_pct', 'error']].round(3))
            except Exception as e:
                print(e)
        #
        Q_dict[_iter] = agent.Q.copy()
        error_dict[_iter] = env.sim_df['error'].dropna().copy()
    #
    agent.eps = -1
    env.sim_df = env.df[['return']].head(99)
    observation, info = env.reset()
    for i in range(1000, 4000):
        action_index, action = agent.act(observation=observation, info=info, i=i)
        next_observation, reward, done, info = env._step(action=action, i=i)
        observation = next_observation
        try:
            print('+++ iter {}, #{}'.format(_iter, i), int(reward))
            print(env.sim_df.tail(1)[[
                'zero_pct', 'herding_pct', 'error']].round(3))
        except Exception as e:
            pass
    #
    env.sim_df.to_csv('simulation_rl.csv', index=False)
    data = env.sim_df['return'].dropna()
    mu, sigma = norm.fit(data)
    skew, kurtosis = st.skew(data), st.kurtosis(data)
    autocorr = f_autocorr(data.abs())[0, 1]
    print('SIM', mu, sigma, skew, kurtosis, autocorr)

"""
>>> env.df.iloc[1000:4000].describe()
            return           mu        sigma         skew     kurtosis  \
count  3000.000000  3000.000000  3000.000000  3000.000000  3000.000000
mean      0.000462     0.000478     0.013941    -0.033289     1.094297
std       0.015165     0.002530     0.005415     0.594328     2.069665
min      -0.058717    -0.008770     0.004127    -2.081689    -1.063509
25%      -0.007164    -0.000663     0.010211    -0.300301    -0.102979
50%       0.000428     0.000350     0.012682    -0.010333     0.429744
75%       0.008054     0.001413     0.018259     0.220564     1.442771
max       0.080489     0.008277     0.026457     2.913808    13.470942

>>> ((mu - env.df.iloc[1000:4000]['mu'].mean()) / env.df.iloc[1000:4000]['mu'].std())**2 + \
((sigma - env.df.iloc[1000:4000]['sigma'].mean()) / env.df.iloc[1000:4000]['sigma'].std())**2 + \
((skew - env.df.iloc[1000:4000]['skew'].mean()) / env.df.iloc[1000:4000]['skew'].std())**2 + \
((kurtosis - env.df.iloc[1000:4000]['kurtosis'].mean()) / env.df.iloc[1000:4000]['kurtosis'].std())**2
a = [88.138970755904481, 82.839828050631141, 85.777736452203527, 86.9724456718657]
85.9322452327 1.97094973805
"""
