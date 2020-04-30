# Connect 4
A simple implementation of the Connect 4 game - from game theory solvers to deep learning based solvers

## Code

The file `connect4.py` implements the Connect 4 board. The file `connect4players.py` constains the implementation of the following players:
- Random player:
  - Plays a random move from the possible moves.
- Left player:
  - Plays on the left most possible collumn.
- Player Lvl 1:
  - Plays on winning move if one is available.
  - Otherwise plays randomly.
- Player Lvl 2:
  - Plays on winning move if one is available.
  - Otherwise, it selects a move to block the opponent from winning, if such move exits.
  - Otherwise plays randomly.
- One-step Player Lvl 1:
  - Uses an heuristic to score the valid moves and plays the highest scoring one.
- One-step Player Lvl 2:
  - Same as One-step Player Lvl 1 but with an improved heuristic.
- Minmax Player Lvl 1:
  - Looks 3 moves ahead with a minmax algorithm to decide its next move. Uses the same heuristic has One-step Player Lvl 1.
- Minmax Player Lvl 2:
  - Same as Minmax Player Lvl 1 but with the improved heuristic of One-step Player Lvl 2.
- Alpha-beta Player Lvl 1:
  - Same as Minmax Player Lvl 1 but with alpha-beta prunning.
- Alpha-beta Player Lvl 2:
  - Same as Minmax Player Lvl 2 but with alpha-beta prunning.

The Jupyter Notebook `Connect4.ipynb` contains an analysis of the quality of the players.

## To do list
- [X] Implement base players.
- [x] Implement minmax players.
- [x] Implement alpha-beta prunning for minmax algorithm.
- [ ] Implement move exploration for minmax algorithm.
- [ ] Implement DQN algorithm.

## References
- Kaggle Micro Course: [Intro to Game AI and Reinforcement Learning](https://www.kaggle.com/learn/intro-to-game-ai-and-reinforcement-learning).
