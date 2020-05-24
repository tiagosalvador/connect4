import numpy as np
import torch

from timeit import default_timer as timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

class connect4board():
    def __init__(self):
        self.init_board = torch.zeros([6,7]).to(device)#.type(torch.int)
        self.player = torch.ones(1).to(device)#.type(torch.int)
        self.current_board = self.init_board
        self.last_played = np.zeros((2,1)).astype(int)
    
    def actions(self):
        r = (self.current_board[0,:] == 0).nonzero().reshape(-1)
        return r
    
    def play(self,a):
        if any(a == self.actions()):
            r = torch.arange(0,self.current_board.shape[0])
            i = torch.max(r[self.current_board[:,a] == 0])
            self.current_board[i,a] = self.player
            self.last_played[0] = i
#            self.last_played[1] = a
            self.last_played[1] = a.cpu().data.numpy()
            self.player = torch.remainder(self.player,2)+1
        else:
            return None
    
    def check_winner(self):
        p = torch.remainder(self.player,2)+1
        i = self.last_played[0,0]
        a = self.last_played[1,0]
        A = self.current_board
        ## check vertical
        for x in range(-3,1):
            r = A[max(i+x,0):(i+4+x),a]
            if all([len(r)==4,all(r == p)]):
                return True
        
        ## check horizontal
        for x in range(-3,1):
            r = A[i,max(a+x,0):(a+4+x)]
            if all([len(r)==4,all(r == p)]):
                return True
        
        ## check diagonal
        diag = torch.diag(A,int(a-i))
        for x in range(-3,1):
            r = diag[max(i+x,0):(i+4+x)]
            if all([len(r)==4,all(r == p)]):
                return True
        
        ## check anti-diagonal
        offset = (6-a)-i
        diag = A.flip([1]).diag(int(offset))
        if offset > 0:
            y = i
        else:
            y = 6-a
        for x in range(-3,1):
            r = diag[max(y+x,0):(y+4+x)]
            if all([len(r)==4,all(r == p)]):
                return True
        return 0

def play_connect4(player1,player2):
    board = connect4board()
    
    time_p1 = 0
    time_p2 = 0
    calls_p1 = 0
    calls_p2 = 0
    
    while True:
        actions = board.actions()
        if board.player == 1:
            calls_p1 +=1
            start = timer()
            a = player1(board)
            end = timer()
            time_p1 += end - start # Time in seconds
        else:
            calls_p2 +=1
            start = timer()
            a = player2(board)
            end = timer()
            time_p2 += end - start # Time in seconds
        if sum(board.actions() == a):
            board.play(a)
        else:
            if board.player == 1:
                r = 2
            else:
                r = -2
            break
        # Check if there is a winner
        if board.check_winner():
            if board.player == 2:
                r = 1
            else:
                r = -1
            break
        # Check if the game ended in a draw
        if len(board.current_board[board.current_board==0]) == 0:
            r = 0
            break
    return [r, time_p1, calls_p1, time_p2, calls_p2]

def get_win_percentages(player1, player2, n_rounds = 10):
    # In approximately half the games, player1 starts
    output = [play_connect4(player1,player2) for i in range(n_rounds//2)]
    output = np.array(output)
    outcomes = output[:,0]
    time_p1_per_action = output[:,1]/output[:,2]
    time_p2_per_action = output[:,3]/output[:,4]
    wins_p1 = sum(outcomes == 1)
    wins_p2 = sum(outcomes == -1)
    np.mean(output[:,1]/output[:,2])
    # In remaning number of games, player2 starts
    output = [play_connect4(player2,player1) for i in range(n_rounds-n_rounds//2)]
    output = np.array(output)
    outcomes = output[:,0]
    time_p2_per_action = np.concatenate((time_p2_per_action,output[:,1]/output[:,2]))
    time_p1_per_action = np.concatenate((time_p1_per_action,output[:,3]/output[:,4]))
    wins_p1 += sum(outcomes == -1)
    wins_p2 += sum(outcomes == 1)
    ties = n_rounds-wins_p1-wins_p2
    print("Agent 1 Win Percentage:", np.round(wins_p1/n_rounds, 2))
    print("Agent 2 Win Percentage:", np.round(wins_p2/n_rounds, 2))
    print("Draws Percentage:", np.round(ties/n_rounds, 2))
    print("Agent 1 Playing time per action", np.mean(time_p1_per_action))
    print("Agent 2 Playing time per action", np.mean(time_p2_per_action))
