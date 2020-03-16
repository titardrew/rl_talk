import random


##########################################################################
########                        TASK 0                            ########
##########################################################################
# Implement ReplayBuffer class. See docstrings for details               #

class ReplayBuffer(object):
    def __init__(self, capacity):
        """
        Arguments:
            capacity: Max number of elements in buffer
        """
        self.capacity = capacity
        self.buffer = []
        self.pointer = 0

    def push(self, s0, a, s1, r, d):
        """Push an element to the buffer.

        Arguments:
            s0: State before action
            a: Action picked by the agent
            s1: State after performing the action
            r: Reward recieved is state s1.
            d: Whether the episode terminated after in the state s1.

        If the buffer is full, start to rewrite elements
        starting from the oldest ones.
        """
        element = (s0, a, s1, r, d)
        if len(self.buffer) < self.capacity:
            self.buffer.append(element)
        else:
            self.buffer[self.pointer] = element

        self.pointer = (self.pointer + 1) % self.capacity

    def sample(self, batch_size):
        """Return `batch_size` randomly chosen elements."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return size of the buffer."""
        return len(self.buffer)

##########################################################################
########                        TASK 0                            ########
##########################################################################
