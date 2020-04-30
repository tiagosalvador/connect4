import numpy as np
import random
import torch
import copy
from collections import OrderedDict #need for alphabeta_v2

class config():
    def __init__(self):
        self.columns = 7
        self.rows = 6
        self.inarow = 4

#########################
##### Random Player #####
#########################

## Plays a random move from the possible moves
def player_random(board):
    actions = board.actions()
    return random.choice(actions)

#######################
##### Left Player #####
#######################

## Plays on the left most possible collumn
def player_left(board):
    actions = board.actions()
    return actions[0]

##########################
##### Player Level 1 #####
##########################

## Plays on winning move if one is available, otherwise plays randomly.
def player_lvl1(board):
    actions = board.actions()
    # Selects a winning move, if one is available.
    for a in board.actions():
        temp = copy.deepcopy(board)
        temp.play(a)
        if temp.check_winner():
            return a
    # Otherwise, plays randomly from the valid moves
    return random.choice(actions)

##########################
##### Player Level 2 #####
##########################

## Plays on winning move if one is available, blocks the opponent from winning in the next play, otherwise plays randomly.
def player_lvl2(board):
    actions = board.actions()
    # Selects a winning move, if one is available.
    for a in board.actions():
        temp = copy.deepcopy(board)
        temp.play(a)
        if temp.check_winner():
            return a
    # Otherwise, it selects a move to block the opponent from winning,
    # if the opponent has a move that it can play in its next turn to win the game.
    for a in board.actions():
        temp = copy.deepcopy(board)
        temp.player = torch.remainder(temp.player,2)+1
        temp.play(a)
        if temp.check_winner():
            return a
    # Otherwise, plays randomly from the valid moves
    return random.choice(actions)

##############################
##### Look ahead Players #####
##############################

# Calculates score if agent drops piece in selected column
def score_move(grid, col, mark, config, heuristic):
    next_grid = drop_piece(grid, col, mark, config)
    score = heuristic(next_grid, mark, config)
    return score

# Helper function for score_move: gets board at next step if agent drops piece in selected column
def drop_piece(grid, col, mark, config):
    next_grid = grid.clone()
    for row in range(config.rows-1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid

# Helper function for score_move: calculates value of heuristic for grid
def get_heuristic_lvl1(grid, mark, config):
    num_threes = count_windows(grid, 3, mark, config)
    num_fours = count_windows(grid, 4, mark, config)
    num_threes_opp = count_windows(grid, 3, mark%2+1, config)
    score = num_threes - 1e2*num_threes_opp + 1e6*num_fours
    return score

# Helper function for get_heuristic: checks if window satisfies heuristic conditions
def check_window(window, num_discs, piece, config):
    return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)

# Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
def count_windows(grid, num_discs, piece, config):
    num_windows = 0
    # horizontal
    # we only need to check rows that have enough discs
    check_rows = [i for i in range(config.rows) if (grid[i]==piece).sum()>=num_discs]
    for row in check_rows:
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # vertical
    # we only need to check columns that have enough discs
    check_columns = [i for i in range(config.columns) if (grid[:,i]==piece).sum()>=num_discs]
    for col in check_columns:
        for row in range(config.rows-(config.inarow-1)):
            window = list(grid[row:row+config.inarow, col])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    return num_windows

############################
##### One-step Players #####
############################

#################################
##### One-step Player Lvl 1 #####
#################################

def player_onestep_lvl1(board):
    config_player = config()
    # Get list of valid moves
    valid_moves = board.actions()
    grid = board.current_board
    # Use the heuristic to assign a score to each possible board in the next turn
    scores = dict(zip(valid_moves, [score_move(grid, col, board.player, config_player, get_heuristic_lvl1) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)

#################################
##### One-step Player Lvl 2 #####
#################################

def get_heuristic_lvl2(grid, mark, config):
    num_twos = count_windows(grid, 2, mark, config)
    num_threes = count_windows(grid, 3, mark, config)
    num_fours = count_windows(grid, 4, mark, config)
    num_twos_opp = count_windows(grid, 2, mark%2+1, config)
    num_threes_opp = count_windows(grid, 3, mark%2+1, config)
    A = 1e6
    B = 100
    C = 1
    D = -10
    E = -1000
    score = A*num_fours + B*num_threes + C*num_twos + D*num_twos_opp + E*num_threes_opp
    return score

def player_onestep_lvl2(board):
    config_player = config()
    # Get list of valid moves
    valid_moves = board.actions()
    grid = board.current_board
    # Use the heuristic to assign a score to each possible board in the next turn
    scores = dict(zip(valid_moves, [score_move(grid, col, board.player, config_player, get_heuristic_lvl2) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)

##########################
##### N-step Players #####
##########################

# Helper function for minimax: checks if agent or opponent has four in a row in the window
def is_terminal_window(window, config):
    return window.count(1) == config.inarow or window.count(2) == config.inarow

# Helper function for minimax: checks if game has ended
def is_terminal_node(grid, config):
    # Check for draw
    if list(grid[0, :]).count(0) == 0:
        return True
    # Check for win: horizontal, vertical, or diagonal
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if is_terminal_window(window, config):
                return True
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if is_terminal_window(window, config):
                return True
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return True
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return True
    return False

###############################
##### Minmax Player Lvl 1 #####
###############################

# Uses minimax to calculate value of dropping piece in selected column
def score_move_minmax(grid, col, mark, config, nsteps, heuristic):
    next_grid = drop_piece(grid, col, mark, config)
    score = minimax(next_grid, nsteps-1, False, mark, config, heuristic)
    return score

# Minimax implementation
def minimax(node, depth, maximizingPlayer, mark, config, heuristic):
    is_terminal = is_terminal_node(node, config)
    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
    if depth == 0 or is_terminal:
        return heuristic(node, mark, config)
    if maximizingPlayer:
        value = -np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark, config)
            value = max(value, minimax(child, depth-1, False, mark, config, heuristic))
        return value
    else:
        value = np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark%2+1, config)
            value = min(value, minimax(child, depth-1, True, mark, config, heuristic))
        return value

# How deep to make the game tree: higher values take longer to run!
N_STEPS = 3

def player_minmax_lvl1(board):
    config_player = config()
    # Get list of valid moves
    valid_moves = board.actions()
    grid = board.current_board
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move_minmax(grid, col, board.player, config_player, N_STEPS, get_heuristic_lvl1) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)

##################################
##### Alphabeta Player Lvl 1 #####
##################################

# Uses minmax with alpha-beta prunning to calculate value of dropping piece in selected column
def score_move_alphabeta(grid, col, mark, config, nsteps, alphabeta, heuristic):
    next_grid = drop_piece(grid, col, mark, config)
    score = alphabeta(next_grid, nsteps-1, -np.Inf, np.Inf, False, mark, config, heuristic)
    return score

# Minimax with alpha-beta prunning implementation
def alphabeta_v1(node, depth, alpha, beta, maximizingPlayer, mark, config, heuristic):
    is_terminal = is_terminal_node(node, config)
    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
    if depth == 0 or is_terminal:
        return heuristic(node, mark, config)
    if maximizingPlayer:
        value = -np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark, config)
            value = max(value, alphabeta_v1(child, depth-1, alpha, beta, False, mark, config, heuristic))
            alpha = max(alpha,value)
            if alpha >= beta:
                break
        return value
    else:
        value = np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark%2+1, config)
            value = min(value, alphabeta_v1(child, depth-1, alpha, beta, True, mark, config, heuristic))
            beta = min(beta,value)
            if alpha >= beta:
                break
        return value

def player_alphabeta_lvl1(board):
    config_player = config()
    # Get list of valid moves
    valid_moves = board.actions()
    grid = board.current_board
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move_alphabeta(grid, col, board.player, config_player, N_STEPS, alphabeta_v1, get_heuristic_lvl1) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)

########################################################
##### Alphabeta Player Lvl 1 with move exploration #####
########################################################

# Minimax with alpha-beta prunning implementation
def alphabeta_v2(node, depth, alpha, beta, maximizingPlayer, mark, config, heuristic):
    is_terminal = is_terminal_node(node, config)
    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
    if depth == 0 or is_terminal:
        return heuristic(node, mark, config)
    if maximizingPlayer:
        value = -np.Inf
        scores = dict(zip(valid_moves, [score_move(node, col, mark, config, heuristic) for col in valid_moves]))
        sorted_scores = OrderedDict(sorted(scores.items(), key=lambda x: x[1], reverse = True))
        valid_moves = [key for key in sorted_scores.keys()]
        for col in valid_moves:
            child = drop_piece(node, col, mark, config)
            value = max(value, alphabeta_v2(child, depth-1, alpha, beta, False, mark, config, heuristic))
            alpha = max(alpha,value)
            if alpha >= beta:
                break
        return value
    else:
        value = np.Inf
        scores = dict(zip(valid_moves, [score_move(node, col, mark, config, heuristic) for col in valid_moves]))
        sorted_scores = OrderedDict(sorted(scores.items(), key=lambda x: x[1], reverse = False))
        valid_moves = [key for key in sorted_scores.keys()]
        for col in valid_moves:
            child = drop_piece(node, col, mark%2+1, config)
            value = min(value, alphabeta_v2(child, depth-1, alpha, beta, True, mark, config,heuristic))
            beta = min(beta,value)
            if alpha >= beta:
                break
        return value

def player_alphabeta_lvl1_v2(board):
    config_player = config()
    # Get list of valid moves
    valid_moves = board.actions()
    grid = board.current_board
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move_alphabeta(grid, col, board.player, config_player, N_STEPS, alphabeta_v2, get_heuristic_lvl1) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)

def player_alphabeta_lvl2_v1(board):
    config_player = config()
    # Get list of valid moves
    valid_moves = board.actions()
    grid = board.current_board
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move_alphabeta(grid, col, board.player, config_player, 5, alphabeta_v1, get_heuristic_lvl1) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)

def player_alphabeta_lvl2_v2(board):
    config_player = config()
    # Get list of valid moves
    valid_moves = board.actions()
    grid = board.current_board
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move_alphabeta(grid, col, board.player, config_player, 5, alphabeta_v2, get_heuristic_lvl1) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)
