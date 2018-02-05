import numpy as np
from queue import Queue

# moves 0 up, 1 right, 2 down, 3 left, rest -> border on a position

MOVE_UP = 0
MOVE_RIGHT = 1
MOVE_DOWN = 2
MOVE_LEFT = 3

DIR_RIGHT = 0
DIR_DOWN = 1
DIR_RIGHT_AND_DOWN = 2

ALL_FREE = (1 << (DIR_RIGHT_AND_DOWN + 1)) - 1

class QuoridorGame:
    @staticmethod
    def init_positions(num_of_players, rows, cols):
        positions = np.zeros(num_of_players)
        positions[0] = ((rows - 1) // 2) * cols
        positions[1] = ((rows - 1) // 2) * cols + cols - 1
        if num_of_players > 2:
            positions[2] = cols // 2
            positions[3] = cols // 2 + (rows - 1) * cols
        return positions

    def __init__(self, num_of_players, sx, sy):
        self.num_of_players = num_of_players
        self.sx = sx
        self.sy = sy
        self.positions = self.init_positions(num_of_players, sx, sy)
        self.board = np.full((sx, sy), ALL_FREE)
        self.dominoes = np.full(num_of_players, 5)

    def init_from_state(self, game_state):
        self.positions = game_state[:self.num_of_players]
        self.dominoes = game_state[self.num_of_players:self.num_of_players * 2]
        board = game_state[self.num_of_players * 2:self.sx * self.sy + self.num_of_players * 2]
        self.board = np.reshape(board, (self.sx, self.sy))

    def do_move(self, move, player):
        if move < 4:
            self.do_step(player, move)
        else:
            if self.dominoes[player] > 0:
                self.add_border(move - 4, player)

        return np.concatenate((np.copy(self.positions), np.copy(self.dominoes),
                               np.reshape(np.copy(self.board), self.sx * self.sy)))

    def do_step(self, player, move):
        player_x, player_y = self.to_coordinates(self.positions[player])
        new_x, new_y = self.calculate_new_position(player_x, player_y, move)
        if self.in_board(new_x, new_y) and self.connected((player_x, player_y), (new_x, new_y)):
            self.positions[player] = self.to_position(new_x, new_y)

    # step over is not here yet :(
    def add_border(self, location, player):
        from_nodes, to_nodes = self.find_nodes(location)
        if self.connected(from_nodes[0], to_nodes[0]) and self.connected(from_nodes[1], to_nodes[1]) \
                and self.connected(from_nodes[0], to_nodes[1]):
            self.remove_edge(from_nodes[0], to_nodes[0])
            self.remove_edge(from_nodes[1], to_nodes[1])
            self.remove_edge(from_nodes[0], to_nodes[1])
            self.dominoes[player] -= 1

    def get_game_state(self, player):
        return np.concatenate((np.full(1, player), np.copy(self.positions), np.copy(self.dominoes), np.copy(self.board).reshape(self.sy * self.sx)))

    def find_nodes(self, location):
        # horizontal
        if location < (self.sx - 1) * (self.sy - 1):
            x = location // (self.sy - 1)
            y = location % (self.sy - 1)
            return ((x, y), (x, y + 1)), \
                   ((x + 1, y), (x + 1, y + 1))
        # vertical
        else:
            location -= (self.sx - 1) * (self.sy - 1)
            x = location // (self.sy - 1)
            y = location % (self.sy - 1)
            return ((x, y), (x + 1, y)), \
                   ((x, y + 1), (x + 1, y + 1))

    def shortest_path_for_player_to_win(self, player):
        dests = [
            (-1,          self.sy - 1),
            (-1,          0          ),
            (self.sx - 1, -1         ),
            (0,           -1         )
        ]
        player_pos = self.positions[player]
        x, y = self.to_coordinates(player_pos)
        return self.shortest_path(x, y, dests[player][0], dests[player][1])


    def is_finished(self):
        any_can_reach = False
        for player in range(self.num_of_players):
            l = self.shortest_path_for_player_to_win(player)
            if l != -1:
                any_can_reach = True
            if l == 0:
                return True, player
        return (not any_can_reach), -1

    def render(self):
        res = ""
        for i in range(self.sy * 2):
            y = i // 2
            row_string = ""
            for x in range(self.sx):
                position = self.to_position(x, y)
                player = self.find_player(position)
                if i % 2 == 0:
                    if player == 0:
                        row_string += '·'
                    else:
                        row_string += str(player)
                    if x < self.sx - 1 and not self.connected((x, y), (x + 1, y)):
                        row_string += '|'
                    else:
                        row_string += ' '
                else:
                    if y < self.sy - 1 and not self.connected((x, y), (x, y + 1)):
                        row_string += '—'
                    else:
                        row_string += ' '
                    row_string += ' '
            res += row_string + '\n'
        return res

    def remove_edge(self, from_coords, to_coords):
        direction = self.find_direction(from_coords, to_coords)
        val = int(self.board[int(from_coords[0]), int(from_coords[1])])
        val &= ~(1 << direction)
        self.board[int(from_coords[0]), int(from_coords[1])] = val

    def connected(self, coords_1, coords_2):
        if coords_1 == coords_2:
            return True
        if coords_1[0] > coords_2[0] or coords_1[1] > coords_2[1]:
            coords_1, coords_2 = coords_2, coords_1
        direction = self.find_direction(coords_1, coords_2)
        node = int(self.board[int(coords_1[0]), int(coords_1[1])])
        return (node & (1 << direction)) > 0

    def to_coordinates(self, position):
        return int(position) // self.sy, int(position) % self.sy

    def to_position(self, x, y):
        return x * self.sy + y

    def in_board(self, x, y):
        return 0 <= y < self.sy and 0 <= x < self.sx

    def num_of_possible_moves(self):
        return 4 + (self.sx - 1) * (self.sy - 1) * 2

    # you can specify -1 for one of the destination coordinates if you don't care about it
    def shortest_path(self, x1, y1, x2, y2):
        q = Queue()
        visited = []
        q.put((x1, y1, 0))
        while not q.empty():
            x, y, l = q.get()
            if (x == x2 or x2 == -1) and (y == y2 or y2 == -1):
                return l
            p = self.to_position(x, y)
            visited.append(p)
            for i in range(4):
                nx, ny = QuoridorGame.calculate_new_position(x, y, i)
                np = self.to_position(nx, ny)
                # check if valid and reachable from current point
                if not self.in_board(nx, ny) or not self.connected((x, y), (nx, ny)):
                    continue
                if np not in visited:
                    q.put((nx, ny, l + 1))
        return -1

    @staticmethod
    def calculate_new_position(x, y, move):
        if move == MOVE_UP:
            y -= 1
        elif move == MOVE_DOWN:
            y += 1
        elif move == MOVE_RIGHT:
            x += 1
        elif move == MOVE_LEFT:
            x -= 1
        return x, y

    @staticmethod
    def find_direction(from_coords, to_coords):
        x1, y1 = from_coords
        x2, y2 = to_coords
        if x2 > x1 and y2 > y1:
            return DIR_RIGHT_AND_DOWN
        if x2 > x1:
            return DIR_RIGHT
        if y2 > y1:
            return DIR_DOWN
        assert False

    def find_player(self, position):
        res = 0
        for i in range(self.num_of_players):
            if position == self.positions[i]:
                res += i + 1

        return res
