import numpy as np
import random, sys


class PUFGenerator(object):
    
    def __init__(self, num_stages, ff_loops, overlaps):
        self.num_stages = num_stages
        self.ff_loops = ff_loops
        self.overlaps = overlaps
        self.init_arbiter_delays()
        self.generate_ff_loops()

    def generate(self, num_challenges):
        C = self.generate_challenges(num_challenges, self.num_stages)
        C, r = self.simulate_challenges(C)
        return C, r

    def init_arbiter_delays(self):
        # Generate the random Arbiter PUF
        self.puf_arbiter = np.random.normal(300, 40, size=(self.num_stages, 4))

        # Removes negative delays
        self.puf_arbiter[self.puf_arbiter < 0] = 300.

    def generate_ff_loops(self):
        self.ff_loops_set = {}

        # loops_set stores a mapping between the starting and ending positions of each FF-loop.
        all_stages = list(range(self.num_stages))
        all_stages = all_stages[1:]

        # Randomly select the starting and ending point of each loop
        if self.overlaps:

            # Select the source bits (starting points of the FF-loops)
            starting_loop_bit = random.sample(all_stages[:len(all_stages)//2], self.ff_loops)

            # Get the remaining bits
            remaining_bits = set([x for x in list(range(self.num_stages)) if x not in starting_loop_bit])

            # For each starting position choose an ending position and
            # remove it from the remaining ones.
            for b in starting_loop_bit:
                t = random.choice([x for x in remaining_bits if x > b])
                remaining_bits.remove(t)
                self.ff_loops_set[b] = t
            print(self)

        else:

            # Not Overlapping: the starting points are in even positions and the ending points in odd positions.
            while True:
                swaps = random.sample(all_stages, 2 * self.ff_loops)
                swaps = sorted(swaps)
                if self.num_stages - swaps[-1] < 10:
                    del swaps
                else:
                    break

            for i in range(int(len(swaps) / 2)):
                self.ff_loops_set[swaps[2 * i]] = swaps[2 * i + 1]

        return self

    def generate_challenges(self, num_challenges, num_stages):
        challenges = [[random.randint(0, 1) for _ in range(num_stages)] for _ in range(num_challenges)]

        challenges = np.asarray(challenges, dtype=np.int8)
        return challenges

    def simulate_challenges(self, challenges):
        responses_list = []
        for challenge in challenges:
            responses_list.append(self.simulate_one_challenge(challenge))
        return challenges, np.asarray(responses_list, dtype=np.int8)

    def simulate_one_challenge(self, challenge):
        delay_top = 0.
        delay_bottom = 0.
        top_is_up = True
        ch_aux = list(challenge)
        for bit_, time_diff in enumerate(self.puf_arbiter):
            # Compute the FF-loop value based on the bit-swaps mask
            # Namely, if the current stage has a FF-loop starting point,
            # update the ending stage by using the current delays of the
            # top and bottom paths.
            if bit_ in self.ff_loops_set:
                ch_aux[self.ff_loops_set[bit_]] = 1 if delay_top < delay_bottom else 0

            # Compute the delays as usual
            input_value = ch_aux[bit_]
            if top_is_up:
                if input_value == 1:
                    delay_top += time_diff[0]
                    delay_bottom += time_diff[3]
                elif input_value == 0:
                    delay_top += time_diff[1]
                    delay_bottom += time_diff[2]
                    top_is_up = not top_is_up
            elif not top_is_up:
                if input_value == 1:
                    delay_top += time_diff[3]
                    delay_bottom += time_diff[0]
                elif input_value == 0:
                    delay_top += time_diff[2]
                    delay_bottom += time_diff[1]
                    top_is_up = not top_is_up
        if delay_top < delay_bottom:
            return 1
        else:
            return -1
