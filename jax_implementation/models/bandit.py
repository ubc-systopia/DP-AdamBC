import jax.numpy as jnp
import jax.random
from numpy.linalg import inv
from abc import ABC, abstractmethod


class MAB(ABC):

    @abstractmethod
    def play(self, key, tround, context):
        # Current round of t (for my implementations average mean reward array
        # at round t is passed to this function instead of tround itself)
        self.tround = tround
        # Context: features of contextual bandits
        self.context = context
        # choose an arm which yields maximum value of average mean reward, tie breaking randomly
        chosen_arm = jax.random.choice(key, jnp.where(self.tround == max(self.tround))[0])
        return chosen_arm
        pass

    @abstractmethod
    def update(self, arm, reward, context):
        # get the chosen arm
        self.arm = arm
        # get the context (may be None)
        self.context = context
        # update the overall step of the model
        self.step_n += 1
        # update the step of individual arms
        self.step_arm = self.step_arm.at[self.arm].add(1)
        # update average mean reward of each arm
        self.AM_reward = self.AM_reward.at[self.arm].set(((self.step_arm[self.arm] - 1) / float(self.step_arm[self.arm])
                                                          * self.AM_reward[self.arm] + (1 / float(self.step_arm[self.arm])) * reward))
        return
        pass


class EpsGreedy(MAB):

    def __init__(self, narms, epsilon, Q0=jnp.inf):
        # Set number of arms
        self.narms = narms
        # Exploration probability
        self.epsilon = epsilon
        # Q0 values
        self.Q0 = jnp.ones(self.narms) * jnp.inf
        # Total step count
        self.step_n = 0
        # Step count for each arm
        self.step_arm = jnp.zeros(self.narms)
        # Mean reward for each arm
        self.AM_reward = jnp.zeros(self.narms)
        super().__init__()
        return

    # Play one round and return the action (chosen arm)
    def play(self, key, tround, context=None):
        # Generate random number
        p = jax.random.uniform(key)

        # if self.epsilon == 0 and self.step_n == 0:
        if self.epsilon == 0 and tround == 0:
            action = jax.random.choice(key, self.narms)
        elif p < self.epsilon:
            action = jax.random.choice(key, self.narms)
        else:
            # Q0 values are initially set to np.inf. Hence, choose an arm with maximum Q0 value (
            # for all of them is np.inf, and therefore will play all of the arms at least one time)

            if len(jnp.where(self.Q0 == 0)[0]) < self.narms:
                # choose an arm with maximum Q0 value
                action = jax.random.choice(key, jnp.where(self.Q0 == max(self.Q0))[0])
                # after the arm is chosen, set the corresponding Q0 value to zero
                self.Q0 = self.Q0.at[action].set(0)
            else:
                # Now, after that we ensure that there is no np.inf in Q0 values and all of them are set to zero
                # we return to play based on average mean rewards
                action = super(EpsGreedy, self).play(key, self.AM_reward, context)

        return action

    def update(self, arm, reward, context=None):
        super(EpsGreedy, self).update(arm, reward, context)
        return
