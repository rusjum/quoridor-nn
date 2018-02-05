import numpy as np
import quoridor


def test_should_return_init_game_state():
    positions = np.array([0, 1])
    dominoes = np.array([5, 5])
    board = np.full(4, 7)
    expected = np.concatenate((positions, dominoes, board))

    game = quoridor.QuoridorGame(2, 2, 2)
    assert np.array_equal(game.get_game_state(), expected)


def test_should_return_init_from_game_state():
    positions = np.array([2, 2])
    dominoes = np.array([0, 0])
    board = np.full(4, 1)
    game_state = np.concatenate((positions, dominoes, board))

    game = quoridor.QuoridorGame(2, 2, 2)
    assert not np.array_equal(game.get_game_state(), game_state)
    game.init_from_state(game_state)
    assert np.array_equal(game.get_game_state(), game_state)


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
    assert np.array_equal(game.get_game_state(), game_state)


def test_should_not_change_player_position_if_wrong_direction():
    positions = np.array([3, 5])
    dominos = np.array([5, 5])
    board = np.full(9, 7)
    expected = np.concatenate((positions, dominos, board))
    game = quoridor.QuoridorGame(2, 3, 3)
    game.do_move(3, 0)
    assert np.array_equal(game.get_game_state(), expected)


def test_should_change_player_position_if_it_is_possible():
    positions = np.array([6, 5])
    dominos = np.array([5, 5])
    board = np.full(9, 7)
    expected = np.concatenate((positions, dominos, board))
    game = quoridor.QuoridorGame(2, 3, 3)
    game.do_move(2, 0)
    assert np.array_equal(game.get_game_state(), expected)


def test_should_put_border():
    positions = np.array([0, 1])
    expected_board = np.array([2, 6, 7, 7])
    expected_dominos = np.array([4, 5])
    expected = np.concatenate((positions, expected_dominos, expected_board))
    game = quoridor.QuoridorGame(2, 2, 2)
    game.do_move(4, 0)
    assert np.array_equal(game.get_game_state(), expected)


def test_should_not_put_border_on_the_same_place():
    game = quoridor.QuoridorGame(2, 2, 2)
    game.do_move(4, 0)
    res1 = game.get_game_state()
    game.do_move(4, 0)
    res2 = game.get_game_state()
    assert np.array_equal(res1, res2)


def test_should_not_put_border_on_the_next_place():
    game = quoridor.QuoridorGame(2, 3, 3)
    game.do_move(4, 0)
    res1 = game.get_game_state()
    game.do_move(5, 0)
    res2 = game.get_game_state()
    assert np.array_equal(res1, res2)


def test_should_put_border_on_the_next_available():
    positions = np.array([4, 7])
    expected_dominos = np.array([3, 5])
    expected_board = np.full(16, 7)
    expected_board[0] = expected_board[2] = 2
    expected_board[1] = expected_board[3] = 6
    expected = np.concatenate((positions, expected_dominos, expected_board))
    game = quoridor.QuoridorGame(2, 4, 4)
    game.do_move(4, 0)
    res1 = game.get_game_state()
    game.do_move(6, 0)
    res2 = game.get_game_state()
    assert not np.array_equal(res1, res2)
    assert np.array_equal(expected, res2)


def test_should_not_put_if_vertical_blocked():
    game = quoridor.QuoridorGame(2, 2, 2)
    game.do_move(4, 0)
    res1 = game.get_game_state()
    game.do_move(5, 0)
    res2 = game.get_game_state()
    assert np.array_equal(res1, res2)


def test_game_finish():
    positions = np.array([0, 1, 0, 0])
    dominoes = np.array([5, 5, 5, 5])
    board = np.full(4, 7)
    game_state = np.concatenate((positions, dominoes, board))
    game = quoridor.QuoridorGame(4, 2, 2)
    game.init_from_state(game_state)
    assert game.is_finished()[0]