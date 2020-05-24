import argparse

# Needed for logging
import csv
import os
import logging
from timeit import default_timer as timer
import time, datetime

import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import copy

from connect4 import connect4board, get_win_percentages, play_connect4
from connect4players import player_random, player_left
from connect4players import player_lvl1, player_lvl2
from connect4players import player_onestep_lvl1, player_onestep_lvl2
from connect4players import player_minmax_lvl1, player_alphabeta_lvl1, player_alphabeta_lvl1_v2

###########################
##### Import networks #####
###########################
from networks import NetworkA, NetworkB, NetworkC, NetworkD


###########################
##### Board encodings #####
###########################
def process_board_onehot(board,mark):
    empty_places = torch.zeros(board.current_board.size())
    p1_places = torch.zeros(board.current_board.size())
    p2_places = torch.zeros(board.current_board.size())
    empty_places[board.current_board==0] = 1
    p1_places[board.current_board==mark] = 1
    p2_places[board.current_board==(mark%2+1)] = 1
    input_model = torch.cat((empty_places,p1_places,p2_places)).reshape(1,-1)
    return input_model.reshape(1,3,6,7).to(device)

def process_board_integer(board,mark):
    input_model = torch.zeros(board.current_board.size())
    input_model[board.current_board==mark] = 1
    input_model[board.current_board==(mark%2+1)] = -1
    return input_model.reshape(1,1,6,7).to(device)


###########################################
##### Neural Network Trained Player 1 #####
###########################################
def player_nn(board):
    with torch.no_grad():
        valid_moves = board.actions()
        best = -np.Inf
        for a in valid_moves:
                temp = copy.deepcopy(board)
                temp.play(a)
                temp_best = player1_net(process_board(temp,board.player))
                if temp_best > best:
                    best = temp_best
                    action_to_do = a
        return action_to_do

###########################
##### Parse arguments #####
###########################

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Connect 4 Training DQN')

parser.add_argument('--self-play-batch_size', type=int, default=1000, metavar='self_play_batch_size',
                    help='number of self play games in a training round (default: 1000)')
parser.add_argument('--benchmark-batch_size', type=int, default=100, metavar='benchmark_batch_size',
                    help='number of games for benchmarking (default: 100)')
parser.add_argument('--training_rounds', type=int, default=100, metavar='training_rounds', help='number of training rounds (default: 100)')
parser.add_argument('--memory-size', type=int, default=1000, metavar='memory_size',
                    help='memory size (default: 1000)')
parser.add_argument('--network', type=str, default='B', metavar='network',
                    help='network (default: NetworkB)')
parser.add_argument('--exploration', type=str, default='cte', metavar='exploration',
                    help='exploration (default: cte)')
parser.add_argument('--loss', type=str, default='MSE', metavar='loss',
                    help='loss (default: MSE)')
parser.add_argument('--encoding', type=str, default='integer', metavar='encoding',
                    help='encoding (default: integer)')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='optimizer',
                    help='optimizer (default: Adam)')
parser.add_argument('--experience', type=str, default='no', metavar='optimizer',
                    help='experience (default: no)')

args = parser.parse_args()

# Choosing Network type
if args.network == 'A':
    network = NetworkA
elif args.network == 'B':
    network = NetworkB
elif args.network == 'B':
    network = NetworkB
elif args.network == 'C':
    network = NetworkC
elif args.network == 'D':
    network = NetworkD
else:
    raise Exception("Network must be 'A', 'B', 'C' or 'D'.")

# Choosing exploration strategy
if args.exploration == 'cte':
    exploration_strategy = args.exploration
elif args.exploration == 'decay_episode':
    exploration_strategy = args.exploration
elif args.exploration == 'decay_training':
    exploration_strategy = args.exploration
elif args.exploration == 'sinosoidal_episode':
    exploration_strategy = args.exploration
elif args.exploration == 'no':
    exploration_strategy = args.exploration
elif args.exploration == 'Boltzmann':
    exploration_strategy = args.exploration
else:
    raise Exception("Not an available exploration strategy.")

# Choosing loss function
if args.loss == 'MSE':
    loss_fn = nn.MSELoss(reduction='mean')
elif args.loss == 'Huber':
    loss_fn = F.smooth_l1_loss
else:
    raise Exception("Not an available loss function.")

# Choosing board encoding
if args.encoding == 'integer':
    process_board = process_board_integer
    inputs = 1
elif args.encoding == 'onehot':
    process_board = process_board_onehot
    inputs = 3
else:
    raise Exception("Not an available loss function.")

# Choosing Optimizer
if args.optimizer == 'Adam':
    optimizer_fn = optim.Adam
elif args.optimizer == 'RMSprop':
    optimizer_fn = optim.RMSprop
elif args.optimizer == 'Rprop':
    optimizer_fn = optim.Rprop
else:
    raise Exception("Not an available loss function.")


self_play_batch_size = args.self_play_batch_size
benchmark_batch_size = args.benchmark_batch_size
training_rounds = args.training_rounds
memory_size = args.memory_size


#########################
##### Memory Buffer #####
#########################
Transition = namedtuple('Transition',('state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

##############################################
##### Action Selection Neural Net Player #####
##############################################
def select_action(board,net,mark):
    sample = random.random()
    if exploration_strategy == 'Boltzmann':
        with torch.no_grad():
            valid_moves = board.actions()
            best = -np.Inf
            p = torch.zeros(len(valid_moves))
            for i,a in enumerate(valid_moves):
                temp = copy.deepcopy(board)
                temp.play(a)
                p[i] = net(process_board(temp,mark))
            
            action_to_do = random.choices(valid_moves,weights=F.softmax(p,dim=0))[0]
            return torch.tensor([[action_to_do]], device=device, dtype=torch.long)
    else:
        if exploration_strategy == 'cte':
            eps_threshold = 0.9
        elif exploration_strategy == 'decay_episode':
            eps_threshold = math.exp(-2.0*i_episode/self_play_batch_size)
        elif exploration_strategy == 'decay_training':
            eps_threshold = math.exp(-2.0*i_training_round/training_rounds)
        elif exploration_strategy == 'sinosoidal_episode':
            eps_threshold = 0.5*math.cos(2*i_episode/self_play_batch_size*math.pi)+0.5
        elif exploration_strategy == 'no':
            eps_threshold = 0

        if sample < eps_threshold:
            if len(board.actions()) == 0:
                print(board.actions())
                print(board.current_board)
            return torch.tensor([[random.choice(board.actions())]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                valid_moves = board.actions()
                best = -np.Inf
                for a in valid_moves:
                    temp = copy.deepcopy(board)
                    temp.play(a)
                    temp_best = net(process_board(temp,mark))
                    if temp_best > best:
                        best = temp_best
                        action_to_do = a
                return torch.tensor([[action_to_do]], device=device, dtype=torch.long)

##########################
##### Optimize Model #####
##########################
def optimize_model(net,memory,optimizer):
    if len(memory) < BATCH_SIZE:
        return memory, None
    if args.experience == 'yes':
        transitions = memory.sample(BATCH_SIZE)
    elif args.experience == 'no':
        transitions = memory.memory
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    reward_batch = torch.cat(batch.reward)
    
    expected_state_values = reward_batch
    
    # Compute loss
    state_values = net(state_batch)
    loss = loss_fn(state_values.reshape(-1), expected_state_values)
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # If the Player does not use its previous experience, the memory is reset.
    if args.experience == 'no':
        memory = ReplayMemory(memory_size)
    return memory,loss

#####################
##### Self-play #####
#####################
def play_game(memory_p1,memory_p2):
    # Start a new game
    board = connect4board()
    # Every other game, player 2 starts
    if i_episode%2 == 0:
        board.player = torch.ones(1).to(device)%2+1
    states_p1 = []
    states_p2 = []
    done = None
    for t in count():
        if board.player == 1:
            action = select_action(board,player1_net,1)
            board.play(action[0][0])
            states_p1.append(copy.deepcopy(process_board(board,1)))
            # Check if there is a winner or if the game ended in a draw
            if board.check_winner():
                done = True
                reward = 1.0
            elif len(board.current_board[board.current_board==0]) == 0:
                done = True
                reward = 0.0
        else:
            action = select_action(board,player2_net,2)
            board.play(action[0][0])
            states_p2.append(copy.deepcopy(process_board(board,2)))
            # Check if there is a winner or if the game ended in a draw
            if board.check_winner():
                done = True
                reward = -1.0
            elif len(board.current_board[board.current_board==0]) == 0:
                done = True
                reward = 0.0
        if done:
            reward_game = reward
            for idx, state in reversed(list(enumerate(states_p1))):
                memory_p1.push(state,torch.tensor([reward], device=device))
                reward *= GAMMA
            reward = -reward_game
            for idx, state in reversed(list(enumerate(states_p2))):
                memory_p2.push(state,torch.tensor([reward], device=device))
                reward *= GAMMA

            return reward_game,t,memory_p1,memory_p2

############################
##### Evaluate Players #####
############################
def evaluate_players(player1, player2, n_rounds = 10):
    # In approximately half the games, player1 starts
    output = [play_connect4(player1,player2) for i in range(n_rounds//2)]
    output = np.array(output)
    outcomes = output[:,0]
    time_p1_per_action = output[:,1]/output[:,2]
    time_p2_per_action = output[:,3]/output[:,4]
    wins_p1 = sum(outcomes == 1)
    wins_p2 = sum(outcomes == -1)
    invalid_p1 = sum(outcomes == 2)
    invalid_p2 = sum(outcomes == -2)
    output = [play_connect4(player2,player1) for i in range(n_rounds-n_rounds//2)]
    output = np.array(output)
    outcomes = output[:,0]
    time_p1_per_action += output[:,3]/output[:,4]
    time_p2_per_action += output[:,1]/output[:,2]
    wins_p1 += sum(outcomes == -1)
    wins_p2 += sum(outcomes == 1)
    invalid_p1 += sum(outcomes == -2)
    invalid_p2 += sum(outcomes == 2)
    ties = n_rounds-wins_p1-wins_p2
    wins_p1 = np.round(wins_p1/n_rounds, 2)
    wins_p2 = np.round(wins_p2/n_rounds, 2)
    ties = np.round(ties/n_rounds, 2)
    invalid_p1 = np.round(invalid_p1/n_rounds, 2)
    invalid_p2 = np.round(invalid_p2/n_rounds, 2)
    return wins_p1, wins_p2, ties, invalid_p1, invalid_p2

if __name__=="__main__":
    
    BATCH_SIZE = 100 # minimum memory size for update
    GAMMA = 0.9 # discount factor
    
    # Size of board
    screen_height = 6
    screen_width = 7
    n_actions = 7
    
    best_wins_p1 = 0 # winnest percentage of the best model
    
    # Setup a Neural Net for each play with their respective optimizers and memory buffers
    player1_net = network(screen_height, screen_width, inputs).to(device)
    player2_net = network(screen_height, screen_width, inputs).to(device)
    optimizer_p1 = optimizer_fn(player1_net.parameters())
    optimizer_p2 = optimizer_fn(player2_net.parameters())
    memory_p1 = ReplayMemory(memory_size)
    memory_p2 = ReplayMemory(memory_size)
    
    
    #########################
    ##### Logging Setup #####
    #########################
    fmt = '{:.4f}'
    preffix = '{0:%Y-%m-%d--%H-%M-%S--}'.format(datetime.datetime.now())
    folder_name_training = './training/'+args.network+'_'+args.loss+'_'+args.exploration+'_'+args.encoding+'_'+args.optimizer+'_'+args.experience
    try:
        os.makedirs(folder_name_training)
    except:
        print("Folder already exists")
    list_opponents = {
            'random':player_random,
            'left':player_left,
            'lvl1':player_lvl1,
            'lvl2':player_lvl2}# 'onestep':player_onestep_lvl1,'minmax':player_minmax_lvl1}
    traincolumns = ['training_round','random','left','lvl1','lvl2',
                    'wins_p1','ties','game_length','loss_p1','loss_p2']
    file_name_training = preffix+'training.csv'
    trainlog = os.path.join(folder_name_training,file_name_training)
    with open(trainlog,'w') as f:
        logger = csv.DictWriter(f, traincolumns)
        logger.writeheader()

    logger_output = logging.getLogger()
    logger_output.setLevel(logging.INFO) # process everything, even if everything isn't printed

    # Print to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger_output.addHandler(ch)

    # Print to file
    file_name_log = preffix+'log.log'
    log = os.path.join(folder_name_training,file_name_log)
    fh = logging.FileHandler(log)
    fh.setLevel(logging.INFO) # or any level you want
    logger_output.addHandler(fh)

    # Print out arguments
    logger_output.info('Training Neural Net on Connect 4')
    for p in vars(args).items():
        logger_output.info('  '+p[0]+': '+str(p[1]))
    logger_output.info('')

    #############################
    ##### Initial Benchmark #####
    #############################
    logger_output.info("Benchmark Stats Training Round 0")
    row = {'training_round':0}
    for name_opponent in list_opponents:
        # given the deterministic way the Left player plays, we only need to play 2 games, one for each player starting
        n_rounds = 2 if name_opponent == 'left' else benchmark_batch_size
        wins_p1, wins_p2, ties, invalid_p1, invalid_p2 = evaluate_players(player_nn, list_opponents[name_opponent], n_rounds = n_rounds)
        row[name_opponent] = wins_p1
        logger_output.info("W: %1.2f, T: %1.2f (versus %s)" % (wins_p1,ties,name_opponent))
    row['wins_p1'] = np.NaN
    row['ties'] = np.NaN
    row['game_length'] = np.NaN
    row['loss_p1'] = np.NaN
    row['loss_p2'] = np.NaN
    with open(trainlog,'a') as f:
            logger = csv.DictWriter(f, traincolumns)
            logger.writerow(row)

    for i_training_round in range(1,training_rounds+1):
        start = timer()
        accumulated_wins = 0
        accumulated_ties = 0
        loss_round_p1 = []
        loss_round_p2 = []
        accumulated_length = 0
        for i_episode in range(self_play_batch_size):
            #####################
            ##### Self-play #####
            #####################
            outcome,game_length,memory_p1,memory_p2 = play_game(memory_p1,memory_p2)

            ###########################
            ##### Optimize Models #####
            ###########################`
            memory_p1,loss_p1 = optimize_model(player1_net,memory_p1,optimizer_p1)
            memory_p2,loss_p2 = optimize_model(player2_net,memory_p2,optimizer_p2)
            
            
            ##################################
            ##### Collect Training Stats #####
            ##################################
            if outcome == 1.0:
                accumulated_wins += 1
            elif outcome == 0:
                accumulated_ties += 1
            if loss_p1 is not None:
                loss_round_p1.append(loss_p1.item())
            if loss_p2 is not None:
                loss_round_p2.append(loss_p2.item())
            accumulated_length += game_length

        ####################################
        ##### Training Round Benchmark #####
        ####################################
        logger_output.info("Training round %d stats"%(i_training_round))
        accumulated_wins = accumulated_wins/self_play_batch_size
        accumulated_ties = accumulated_ties/self_play_batch_size
        accumulated_length = accumulated_length/self_play_batch_size
        logger_output.info("Wins %1.2f, Ties %1.2f, Losses %1.2f, Game Length %1.2f moves" %(accumulated_wins,accumulated_ties,1-accumulated_wins-accumulated_ties,accumulated_length))
        logger_output.info("Loss p1: %f, Loss p2: %f" %(sum(loss_round_p1)/len(loss_round_p1),sum(loss_round_p2)/len(loss_round_p2)))
        logger_output.info("Benchmark Stats Training Round %d"%(i_training_round))
        row = {'training_round':i_training_round}
        for name_opponent in list_opponents:
            # given the deterministic way the Left player plays, we only need to play 2 games, one for each player starting
            n_rounds = 2 if name_opponent == 'left' else benchmark_batch_size
            wins_p1, wins_p2, ties, invalid_p1, invalid_p2 = evaluate_players(player_nn, list_opponents[name_opponent], n_rounds = n_rounds)
            row[name_opponent] = wins_p1
            logger_output.info("W: %1.2f, T: %1.2f (versus %s)" % (wins_p1,ties,name_opponent))
        row['wins_p1'] = accumulated_wins
        row['ties'] = accumulated_ties
        row['game_length'] = accumulated_length
        row['loss_p1'] = sum(loss_round_p1)/len(loss_round_p1)
        row['loss_p2'] = sum(loss_round_p2)/len(loss_round_p2)
        with open(trainlog,'a') as f:
            logger = csv.DictWriter(f, traincolumns)
            logger.writerow(row)
        end = timer()

        # if current model is better than the current best, update
        if wins_p1 > best_wins_p1:
            best_wins_p1 = wins_p1
            torch.save(player1_net.state_dict(), os.path.join(folder_name_training,preffix+'best_net.pt'))
        if (end-start) < 60:
            logger_output.info("Training round %d completed in %d seconds." % (i_training_round,np.floor(end-start)))
        else:
            logger_output.info("Training round %d completed in %d minutes and %d seconds." % (i_training_round,np.floor((end-start)/60),round(end-start-60*np.floor((end-start)/60))))
        seconds_left = (training_rounds-i_training_round)*(end-start)
        expected_completion = '{0:%H:%M %Y-%m-%d}'.format(datetime.datetime.now()+datetime.timedelta(seconds=seconds_left))
        logger_output.info("Expected completion: %s."%expected_completion)
        start = timer()

    print('Training complete')
