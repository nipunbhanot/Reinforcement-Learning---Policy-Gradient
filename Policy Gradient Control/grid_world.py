import numpy as np
import copy

MAP1 = ["s0",
        "0g"]


MAP2 = ["s0100",
        "00100",
        "00100",
        "00000",
        "0000g"]

MAP3 = ["s0100000",
        "00100000",
        "00100100",
        "00100100",
        "00000100",
        "0000010g"]

class GridWorld(object):
    EMPTY = 0
    HOLE = 1
    START = 2
    GOAL = 3

    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3

    def __init__(self, map_string):
        self._parse_string(map_string)
        self.reset()
        self.max_steps = self.get_num_states()
        self.goals_reached = 0

    def get_num_states(self):
        return self.n_rows * self.n_cols

    def get_num_actions(self):
        return 4

    # Resets the grid world to the starting position
    def reset(self):
        self.loc = copy.deepcopy(self.start)
        self.step_iter = 0
        return self._flatten_idx(self.loc)

    # Takes an action "u", which is one of
    # [GridWorld.ACTION_UP, GridWorld.ACTION_RIGHT, GridWorld.ACTION_DOWN, GridWorld.ACTION_LEFT]
    # this function will return a tuple of
    # (next_state, reward, done)
    # where
    # next state is the state of the system after taking action "u"
    # reward is the one-step reward
    # done is a boolean saying whether or not the episode has ended. 
    # if done is true, you have to call reset() before you can call step() again
    def step(self, u):
        fall_reward = -100
        goal_reward = 100
        step_reward = -1

        if u == GridWorld.ACTION_UP:
            self.loc[0] -= 1
        elif u == GridWorld.ACTION_DOWN:
            self.loc[0] += 1
        elif u == GridWorld.ACTION_RIGHT:
            self.loc[1] += 1
        elif u == GridWorld.ACTION_LEFT:
            self.loc[1] -= 1
        else:
            raise Exception("Not a valid action")

        out_of_bounds = False
        if self.loc[0] < 0:
            self.loc[0] = 0
            out_of_bounds = True
        if self.loc[0] >= self.n_rows:
            self.loc[0] = self.n_rows - 1
            out_of_bounds = True
        if self.loc[1] < 0:
            self.loc[1] = 0
            out_of_bounds = True
        if self.loc[1] >= self.n_cols:
            self.loc[1] = self.n_cols - 1
            out_of_bounds = True

        self.step_iter += 1


        goal_reached = (self.loc == self.goal)
        if (out_of_bounds):
            return self._flatten_idx(self.loc), fall_reward, True
        if self.map[self.loc[0], self.loc[1]] == GridWorld.HOLE:
            return self._flatten_idx(self.loc), fall_reward, True
        if goal_reached:
            self.goals_reached += 1
            return self._flatten_idx(self.loc), goal_reward, True
        if self.step_iter == self.max_steps:
            return self._flatten_idx(self.loc), step_reward, True
        return self._flatten_idx(self.loc), step_reward, False

    def print(self):
        print_str = ""
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if self.loc == [row, col]:
                    print_str += "*"
                else:
                    print_str += str(self.map[row, col])
            print_str += "\n"
        print(print_str)

    def _flatten_idx(self, idx):
        flattened = idx[0] * self.n_cols + idx[1]
        return flattened

    def _parse_string(self, map_string):
        assert(len(map_string) > 0)
        assert(len(map_string[0]) > 0)
        
        self.n_rows = len(map_string)
        self.n_cols = len(map_string[0])

        self.map = np.zeros((self.n_rows, self.n_cols), dtype=np.int8)
        symbol_dict = {
            "0" : GridWorld.EMPTY,
            "1" : GridWorld.HOLE,
            "s" : GridWorld.START,
            "g" : GridWorld.GOAL}

        for row_idx, row in enumerate(map_string):
            assert(len(row) == self.n_cols)
            for col_idx in range(self.n_cols):
                assert(row[col_idx] in symbol_dict.keys())
                self.map[row_idx, col_idx] = symbol_dict[row[col_idx]]
                if row[col_idx] == 's':
                    self.start = [row_idx, col_idx]
                if row[col_idx] == 'g':
                    self.goal = [row_idx, col_idx]
                


# example of how to use this grid world
if __name__ == "__main__":
    env = GridWorld(MAP2) # choose one of [MAP1, MAP2, MAP3]
    env.print()

    # keep going right until the episode has finished
    done = False
    while not done:
        state, reward, done = env.step(GridWorld.ACTION_RIGHT)
        env.print()


