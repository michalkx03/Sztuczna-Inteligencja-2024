from exceptions import AgentException
from copy import deepcopy

class MinMaxAgent:
    def __init__(self, my_token='o', max_depth=3):
        self.my_token = my_token
        self.opponent_token = 'x' if my_token == 'o' else 'o'
        self.max_depth = max_depth

    def heuristic_evaluation(self, connect4):
        player_token = self.my_token
        opponent_token = self.opponent_token

        def evaluate_line(line):
            player_count = line.count(player_token)
            opponent_count = line.count(opponent_token)
            if player_count == 4:
                return 1
            elif opponent_count == 4:
                return -1
            else:
                return ((opponent_count/4) - (player_count/4))/2

        evaluation = 0
        for row in connect4.board:
            for start in range(connect4.width - 3):
                line = row[start:start + 4]
                evaluation += evaluate_line(line)

        for col in range(connect4.width):
            column = [connect4.board[row][col] for row in range(connect4.height)]
            for start in range(connect4.height - 3):
                line = column[start:start + 4]
                evaluation += evaluate_line(line)

        for start_row in range(connect4.height - 3):
            for start_col in range(connect4.width - 3):
                line1 = [connect4.board[start_row + i][start_col + i] for i in range(4)]
                line2 = [connect4.board[start_row + i][start_col + 3 - i] for i in range(4)]
                evaluation += evaluate_line(line1)
                evaluation += evaluate_line(line2)

        return evaluation


    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')

        _, best_move = self.minimax(connect4, 0, True)
        return best_move

        
    def minimax(self, connect4, depth, maximizing_player):
        if depth == self.max_depth or connect4.game_over:
            if connect4.wins == self.my_token:
                return 1, None
            elif connect4.wins == self.opponent_token:
                return -1, None
            else:
                return 0, None

        if maximizing_player:
            max_val = float('-inf')
            best_move = None
            for move in connect4.possible_drops():
                connect4_copy = deepcopy(connect4)
                connect4_copy.drop_token(move)
                val, _ = self.minimax(connect4_copy, depth + 1, False)
                if val > max_val:
                    max_val = val
                    best_move = move
            return max_val, best_move
        else:
            min_val = float('inf')
            best_move = None
            for move in connect4.possible_drops():
                connect4_copy = deepcopy(connect4)
                connect4_copy.drop_token(move)
                val, _ = self.minimax(connect4_copy, depth + 1, True)
                if val < min_val:
                    min_val = val
                    best_move = move
            return min_val, best_move
