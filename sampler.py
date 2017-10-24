import numpy as np
import tensorflow as tf
import scipy.signal

class Sampler(object):
    def __init__(self,
                 policy,
                 env,
                 gru_unit_size=16,
                 num_step=10,
                 num_layers=1,
                 max_step=2000,
                 batch_size=10000,
                 discount=0.99,
                 n_step_TD=5,
                 summary_writer=None):
        self.policy = policy
        self.env = env
        self.gru_unit_size = gru_unit_size
        self.num_step = num_step
        self.num_layers = num_layers
        self.max_step = max_step
        self.batch_size = batch_size
        self.discount = discount
        self.n_step_TD = n_step_TD
        self.summary_writer = summary_writer
        self.state = self.env.reset()
        self.init_state = tuple(
                [np.zeros((1, self.gru_unit_size)) for _ in range(self.num_layers)])

    def discounted_x(self, x):
        return scipy.signal.lfilter([1], [1, -self.discount], x[::-1])[::-1]

    def compute_monte_carlo_returns(self, rewards):
        return self.discounted_x(rewards)

    def flush_summary(self, value, tag="rewards"):
        global_step = self.policy.session.run(self.policy.global_step)
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.summary_writer.add_summary(summary, global_step)
        self.summary_writer.flush()

    def n_step_returns(self, rewards, values, final_value):
        """
        >>> n_step_returns([1., 1, 1, 1, 1, 0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 0)
        = np.array([2.3, 2.4, 2.5, 2.6, 1.0, 0.0])
        """
        n = min(len(rewards), self.n_step_TD)

        filter_ = self.discount ** np.arange(n)

        filtered_reward = scipy.signal.lfilter(filter_, [1.], rewards[::-1])[::-1]

        discounts = np.concatenate([[self.discount ** n] * (len(rewards) - n + 1),
                                     self.discount ** np.arange(n - 1, 0, -1)])

        shifted_values = np.concatenate([values[n:], [final_value] * n])

        n_step_value =  filtered_reward + discounts * shifted_values

        return n_step_value

    def return_estimate(self, rewards, final_value):
       """
       >>> return_estimate([1., 1., 1.], 5)
       = np.array([1 + self.discount + self.discount ** 2 + self.discount ** 3 * 5,
                                        1 + self.discount + self.discount ** 2 * 5,
                                                             1 + self.discount * 5])
       """
       reward_plus = rewards + [final_value]
       return  self.discounted_x(reward_plus)[:-1]


    def collect_one_episode(self, render=False):
        states, actions, rewards, values, dones = [], [], [], [], []
        init_states = tuple([] for _ in range(self.num_layers))

        for t in range(self.max_step):
            if render:
                self.env.render()
            action, final_state, value = self.policy.sampleAction(
                                        self.state[np.newaxis, np.newaxis, :],
                                        self.init_state)
            next_state, reward, done, _ = self.env.step(action)
            # appending the experience
            states.append(self.state)
            actions.append(action)
            rewards.append(reward)
            values.append(value[0, 0])
            for i in range(self.num_layers):
                init_states[i].append(self.init_state[i][0])
            dones.append(done)
            # going to next state
            self.state = next_state
            self.init_state = final_state
            if done:
                break

        if done:
            final_value = 0.0
        else:
            _, _, final_value = self.policy.sampleAction(
                                        self.state[np.newaxis, np.newaxis, :],
                                        self.init_state)
            final_value = final_value[0, 0]

        self.state = self.env.reset()
        self.init_state = tuple([np.zeros((1, self.gru_unit_size)) for _ in range(self.num_layers)])
        self.flush_summary(np.sum(rewards))

        # NB. configure for Monter Carlo or n-step returns
        returns = self.compute_monte_carlo_returns(rewards)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        # returns = self.n_step_returns(rewards, values, final_value)
        #returns = self.return_estimate(rewards, final_value)
        advantages = np.array(returns) - np.array(values)
        episode = dict(
                    states = np.array(states),
                    actions = np.array(actions),
                    rewards = np.array(rewards),
                    monte_carlo_returns = np.array(returns),
                    advantages = advantages,
                    init_states = tuple(np.array(init_states[i])
                                   for i in range(self.num_layers)),
                    )
        return self.expand_episode(episode)

    def collect_one_batch(self):
        episodes = []
        len_samples = 0
        while len_samples < self.batch_size:
            episode = self.collect_one_episode()
            episodes.append(episode)
            len_samples += np.sum(episode["seq_len"])
        # prepare input
        states = np.concatenate([episode["states"] for episode in episodes])
        actions = np.concatenate([episode["actions"] for episode in episodes])
        rewards = np.concatenate([episode["rewards"] for episode in episodes])
        monte_carlo_returns = np.concatenate([episode["monte_carlo_returns"]
                                 for episode in episodes])
        advantages = np.concatenate([episode["advantages"]
                                      for episode in episodes])

        init_states = tuple(
                       np.concatenate([episode["init_states"][i]
                                       for episode in episodes])
                       for i in range(self.num_layers))
        seq_len = np.concatenate([episode["seq_len"] for episode in episodes])
        batch = dict(
                    states = states,
                    actions = actions,
                    rewards = rewards,
                    monte_carlo_returns = monte_carlo_returns,
                    advantages = advantages,
                    init_states = init_states,
                    seq_len = seq_len
                    )
        return batch

    def expand_episode(self, episode):
        episode_size = len(episode["rewards"])
        if episode_size % self.num_step:
            batch_from_episode = (episode_size // self.num_step + 1)
        else:
            batch_from_episode = (episode_size // self.num_step)

        extra_length = batch_from_episode * self.num_step - episode_size
        last_batch_size = episode_size - (batch_from_episode - 1) * self.num_step

        batched_episode = {}
        for key, value in episode.items():
            if key == "init_states":
                truncated_value = tuple(value[i][::self.num_step] for i in
                                        range(self.num_layers))
                batched_episode[key] = truncated_value
            else:
                expanded_value = np.concatenate([value, np.zeros((extra_length,) +
                                                     value.shape[1:])])
                batched_episode[key] = expanded_value.reshape((-1, self.num_step) +
                                                         value.shape[1:])

        seq_len = [self.num_step] * (batch_from_episode - 1) + [last_batch_size]
        batched_episode["seq_len"] = np.array(seq_len)
        return batched_episode

    def samples(self):
        return self.collect_one_batch()
