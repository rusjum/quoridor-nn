import numpy as np

# moves 0 up, 1 right, 2 down, 3 left, rest -> border on a position


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

    def __init__(self, num_of_players, rows, cols):
        self.num_of_players = num_of_players
        self.rows = rows
        self.cols = cols
        self.positions = self.init_positions(num_of_players, rows, cols)
        self.board = np.full((rows, cols), 7)
        self.dominoes = np.full(num_of_players, 5)

    def init_from_state(self, game_state):
        self.positions = game_state[:self.num_of_players]
        self.dominoes = game_state[self.num_of_players:self.num_of_players * 2]
        board = game_state[self.num_of_players * 2:self.rows * self.cols + self.num_of_players * 2]
        self.board = np.reshape(board, (self.cols, self.rows))

    def do_move(self, move, player):
        if move < 4:
            self.do_step(player, move)
        else:
            if self.dominoes[player] > 0:
                self.add_border(move - 4, player)

        return np.concatenate((np.copy(self.positions), np.copy(self.dominoes),
                               np.reshape(np.copy(self.board), self.rows * self.cols)))

    def do_step(self, player, move):
        player_row, player_col = self.to_coordinates(self.positions[player])
        new_row, new_col = self.calculate_new_position(player_row, player_col, move)
        if self.in_board(new_row, new_col) and self.connected((player_row, player_col), (new_row, new_col)):
            self.positions[player] =  self.to_position(new_row, new_col)

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
        return np.concatenate((np.full(1, player), np.copy(self.positions), np.copy(self.dominoes), np.copy(self.board).reshape(self.cols * self.rows)))

    def find_nodes(self, location):
        if location < (self.rows - 1) * (self.cols - 1):
            row = location // (self.cols - 1)
            col = location % (self.cols - 1)
            return ((row, col), (row, col + 1)), \
                   ((row + 1, col), (row + 1, col + 1))
        else:
            location -= (self.rows - 1) * (self.cols - 1)
            row = location // (self.rows - 1)
            col = location % (self.rows - 1)
            return ((row, col), (row + 1, col)), \
                   ((row, col + 1), (row + 1, col + 1))

    def is_finished(self):
        coordinates = self.to_coordinates(self.positions[0])
        if coordinates[1] == self.cols - 1:
            return True, 0
        coordinates = self.to_coordinates(self.positions[1])
        if coordinates[1] == 0:
            return True, 1
        if self.num_of_players > 2:
            coordinates = self.to_coordinates(self.positions[2])
            if coordinates[0] == self.rows - 1:
                return True, 2
            coordinates = self.to_coordinates(self.positions[3])
            if coordinates[0] == 0:
                return True, 3
        return False, -1

    def render(self):
        res = ""
        for i in range(self.rows * 2):
            row = i // 2
            row_string = ""
            for j in range(self.cols):
                position = row * self.cols + j
                player = self.find_player(position)
                if i % 2 == 0:
                    if player == 0:
                        row_string += '·'
                    else:
                        row_string += str(player)
                    if j < self.cols - 1 and not self.connected((row, j), (row, j + 1)):
                        row_string += '|'
                    else:
                        row_string += ' '
                else:
                    if row < self.rows - 1 and not self.connected((row, j), (row + 1, j)):
                        row_string += '—'
                    else:
                        row_string += ' '
                    row_string += ' '
            res += row_string + '\n'
        return res

    def remove_edge(self, from_node, to_node):
        direction = self.find_direction(from_node, to_node)
        val = int(self.board[int(from_node[0]), int(from_node[1])])
        val &= ~(1 << direction)
        self.board[int(from_node[0]), int(from_node[1])] = val

    def connected(self, pos_1, pos_2):
        if pos_1 == pos_2:
            return True
        if pos_1[0] > pos_2[0] or pos_1[1] > pos_2[1]:
            t = pos_1
            pos_1 = pos_2
            pos_2 = t
        direction = self.find_direction(pos_1, pos_2)
        node = int(self.board[int(pos_1[0]), int(pos_1[1])])
        return (node & (1 << direction)) > 0

    def to_coordinates(self, position):
        return int(position) // self.cols, int(position) % self.cols

    def to_position(self, x, y):
        return x * self.cols + y

    def in_board(self, col, row):
        return 0 <= col < self.cols and 0 <= row < self.rows

    def num_of_possible_moves(self):
        return 4 + (self.rows - 1) * (self.cols - 1) * 2

    @staticmethod
    def calculate_new_position(row, col, move):
        if move == 0:
            row -= 1
        elif move == 1:
            col += 1
        elif move == 2:
            row += 1
        elif move == 3:
            col -= 1
        return row, col

    @staticmethod
    def find_direction(from_node, to_node):
        direction = 0
        if from_node[1] < to_node[1] and from_node[0] < to_node[0]:
            direction = 2
        elif from_node[1] < to_node[1]:
            direction = 1

        return direction

    def find_player(self, position):
        res = 0
        for i in range(self.num_of_players):
            if position == self.positions[i]:
                res += i + 1

        return res
