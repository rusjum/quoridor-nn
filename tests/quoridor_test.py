import numpy as np
import quoridor


def test_should_return_init_game_state():
    positions = np.array([0, 1])
    dominoes = np.array([5, 5])
    board = np.full(4, quoridor.ALL_FREE)
    expected = np.concatenate((np.zeros(1), positions, dominoes, board))

    game = quoridor.QuoridorGame(2, 2, 2)
    assert np.array_equal(game.get_game_state(0), expected)


def test_should_return_init_from_game_state():
    positions = np.array([2, 2])
    dominoes = np.array([0, 0])
    board = np.full(4, 1)
    game_state = np.concatenate((positions, dominoes, board))
    expected = np.concatenate((np.zeros(1), game_state))

    game = quoridor.QuoridorGame(2, 2, 2)
    assert not np.array_equal(game.get_game_state(0), game_state)
    game.init_from_state(game_state)
    assert np.array_equal(game.get_game_state(0), expected)


def test_should_calculate_correct_number_of_moves():
    game = quoridor.QuoridorGame(2, 3, 3)
    assert game.num_of_possible_moves() == 12


def test_should_not_change_player_position_if_no_connection():
    positions = np.array([0, 0])
    dominos = np.array([5, 5])
    board = np.zeros(9)
    game_state = np.concatenate((positions, dominos, board))
    game = quoridor.QuoridorGame(2, 3, 3)
    game.init_from_state(game_state)
    game.do_move(1, 0)
    assert np.array_equal(game.get_game_state(0), np.concatenate((np.zeros(1), game_state)))


def test_should_not_change_player_position_if_wrong_direction():
    positions = np.array([3, 5])
    dominos = np.array([5, 5])
    board = np.full(9, quoridor.ALL_FREE)
    expected = np.concatenate((np.zeros(1), positions, dominos, board))
    game = quoridor.QuoridorGame(2, 3, 3)
    game.do_move(quoridor.MOVE_UP, 0)
    assert np.array_equal(game.get_game_state(0), expected)


def test_should_change_player_position_if_it_is_possible():
    positions = np.array([4, 5])
    dominos = np.array([5, 5])
    board = np.full(9, quoridor.ALL_FREE)
    expected = np.concatenate((np.zeros(1), positions, dominos, board))
    game = quoridor.QuoridorGame(2, 3, 3)
    game.do_move(quoridor.MOVE_DOWN, 0)
    assert np.array_equal(game.get_game_state(0), expected)


def test_should_put_border():
    positions = np.array([0, 1])
    expected_board = np.array([2, 6, 7, 7])
    expected_dominos = np.array([4, 5])
    expected = np.concatenate((np.zeros(1), positions, expected_dominos, expected_board))
    game = quoridor.QuoridorGame(2, 2, 2)
    game.do_move(4, 0)
    assert np.array_equal(game.get_game_state(0), expected)


def test_should_not_put_border_on_the_same_place():
    game = quoridor.QuoridorGame(2, 2, 2)
    game.do_move(4, 0)
    res1 = game.get_game_state(np.zeros(1))
    game.do_move(4, 0)
    res2 = game.get_game_state(np.zeros(1))
    assert np.array_equal(res1, res2)


def test_should_not_put_border_on_the_next_place():
    game = quoridor.QuoridorGame(2, 3, 3)
    game.do_move(4, 0)
    res1 = game.get_game_state(np.zeros(1))
    game.do_move(5, 0)
    res2 = game.get_game_state(np.zeros(1))
    assert np.array_equal(res1, res2)


def test_should_put_border_on_the_next_available():
    positions = np.array([4, 7])
    expected_dominos = np.array([3, 5])
    expected_board = np.full(16, 7)
    expected_board[0] = expected_board[2] = 2
    expected_board[1] = expected_board[3] = 6
    expected = np.concatenate((np.zeros(1), positions, expected_dominos, expected_board))
    game = quoridor.QuoridorGame(2, 4, 4)
    game.do_move(4, 0)
    res1 = game.get_game_state(0)
    game.do_move(6, 0)
    res2 = game.get_game_state(0)
    assert not np.array_equal(res1, res2)
    assert np.array_equal(expected, res2)


def test_should_not_put_if_vertical_blocked():
    game = quoridor.QuoridorGame(2, 2, 2)
    game.do_move(4, 0)
    res1 = game.get_game_state(0)
    game.do_move(5, 0)
    res2 = game.get_game_state(0)
    assert np.array_equal(res1, res2)


def test_game_finish():
    positions = np.array([0, 1, 0, 0])
    dominoes = np.array([5, 5, 5, 5])
    board = np.full(4, quoridor.ALL_FREE)
    game_state = np.concatenate((positions, dominoes, board))
    game = quoridor.QuoridorGame(4, 2, 2)
    game.init_from_state(game_state)
    assert game.is_finished()[0]

#
#
# .|. .
#
# .|. .
#   -
# . . .
#
def test_shortest_path():
    positions = np.array([0, 1])
    dominoes = np.array([5, 5])
    board = np.full(9, quoridor.ALL_FREE)
    game_state = np.concatenate((positions, dominoes, board))
    game = quoridor.QuoridorGame(2, 3, 3)
    game.init_from_state(game_state)
    game.remove_edge((0, 0), (1, 0))
    game.remove_edge((0, 1), (1, 1))
    game.remove_edge((1, 1), (1, 2))
    assert game.shortest_path(0, 0, 0, 0) == 0
    assert game.shortest_path(0, 0, 2, 0) == 6
    assert game.shortest_path(0, 0, 1, 0) == 7
    assert game.shortest_path(0, 0, 1, 1) == 6
    assert game.shortest_path(0, 1, 1, 1) == 5


#
# . . .
#
# . . .
# - - -
# . . .
#
def test_game_finish_if_all_locked():
    game = quoridor.QuoridorGame(2, 3, 3)
    game.remove_edge((0, 1), (0, 2))
    game.remove_edge((1, 1), (1, 2))
    game.remove_edge((2, 1), (2, 2))
    assert game.is_finished()[0]

#
# . . .
#
# . . .
# - -
# . . .
#
def test_game_not_finished_if_not_all_locked():
    game = quoridor.QuoridorGame(2, 3, 3)
    game.remove_edge((0, 1), (0, 2))
    game.remove_edge((1, 1), (1, 2))
    print(game.render())
    assert not game.is_finished()[0]