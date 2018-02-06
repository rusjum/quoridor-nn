import gym
import quoridor
from gym import spaces

ROWS = 9
COLS = 9
NUMBER_OF_PLAYERS = 2


class QuoridorEnv(gym.Env):
    def __init__(self, player=0):
        self.init_player = player
        self.game = quoridor.QuoridorGame(NUMBER_OF_PLAYERS, ROWS, COLS)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(ROWS * COLS + NUMBER_OF_PLAYERS * 2 + 1)
        self.player = player
        self.reward_range = [0, 1]
        self.metadata['render.modes'] = ['ansi']

    def _step(self, action):
        # observation (object): agent's observation of the current environment
        #     reward (float) : amount of reward returned after previous action
        #     done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        #     info

        self.game.do_move(action, self.player)
        reward = self._calculate_reward()
        self.player = (self.player + 1) % NUMBER_OF_PLAYERS
        return self.game.get_game_state(self.player), reward, self.game.is_finished()[0], 'Action: {}'.format(action)

    def _reset(self):
        self.game = quoridor.QuoridorGame(NUMBER_OF_PLAYERS, ROWS, COLS)
        self.player = (self.init_player + 1) % NUMBER_OF_PLAYERS
        self.init_player = (self.init_player + 1) % NUMBER_OF_PLAYERS
        return self.game.get_game_state(self.player)

    def _isDone(self):
        return self.game.is_finished()

    #  would be nice to have at least ansi mode
    def _render(self, mode, close):
        return self.game.render()

    def _seed(self, seed=None):
        return [seed]

    def _calculate_reward(self):
        finished = self.game.is_finished()
        if finished[0]:
            if finished[1] == self.player:
                return 1
            # max for -1 if there is no path
            return max(0, (max(self.game.sx, self.game.sy) - self.game.shortest_path_for_player_to_win(self.player)) * 0.1)
        return 0

gym.envs.register(
    id='quoridor-v0',
    entry_point='quoridor_env:QuoridorEnv'
)