# 2048 game implementation in Python
# first i initialize the board and add two random tiles
# added compress logic -->  [2, --, 2, 4] becomes [2, 2, 4, --] {this is basic for left move, for right move we can reverse the row and apply the same logic}
# added merge logic --> [2, 2, 4, --] becomes [4, --, 4, --]       {this is basic for left move, for right move we can reverse the row and apply the same logic}
# added move left logic
#added move right logic
# for move up and down, implement the transpose logic
# added move up and down logic    
# can_move function checks if any moves are possible (left, right, up, down). 
# added add_tile function to add a new tile to the board in a random empty cell.
# define moves 

# ------------------------------------------

# adding heuristic evaluation function 
## added score_empty_cell function to count empty cells
## added score_monotonicity function to check if the tiles are in increasing order


import random
import math 
from copy import deepcopy
import time
from typing import List, Tuple, Dict, Optional, Union, Any


def init_board() -> List[List[int]]:
    """Initialize a new 2048 game board with two random tiles."""
    board = [[0] * 4 for _ in range(4)]
    add_tile(board)
    add_tile(board)
    return board


def print_board(board: List[List[int]]) -> None:
    """Print the game board in a formatted way."""
    max_cell = max(max(row) for row in board)
    width = max(6, len(str(max_cell)) + 2)  # Dynamic width based on largest number
    
    horizontal_line = "-" * (width * 4 + 5)
    print("\n" + horizontal_line)
    
    for row in board:
        print("|", end=" ")
        for cell in row:
            cell_str = "--" if cell == 0 else str(cell)
            print(cell_str.center(width - 2), end=" | ")
        print("\n" + horizontal_line)


def compress(row: List[int]) -> List[int]:
    """Compress a row by removing zeros and moving numbers to the left.
    Example: [2, 0, 2, 4] becomes [2, 2, 4, 0]
    """
    new_row = [num for num in row if num != 0]
    new_row += [0] * (len(row) - len(new_row))
    return new_row


def merge(row: List[int]) -> Tuple[List[int], int]:
    """Merge adjacent identical numbers in a row.
    Example: [2, 2, 4, 0] becomes [4, 0, 4, 0]
    """
    # Create a copy to avoid modifying the original
    row = row.copy()
    score = 0
    
    for i in range(len(row) - 1):
        if row[i] == row[i + 1] and row[i] != 0:
            row[i] *= 2
            score += row[i]
            row[i + 1] = 0
            
    return row, score


def move_left(board: List[List[int]]) -> Tuple[List[List[int]], int]:
    """Move all tiles to the left, merge if possible."""
    new_board = []
    total_score = 0
    
    for row in board:
        compressed = compress(row)
        merged, score = merge(compressed)
        final = compress(merged)
        new_board.append(final)
        total_score += score
        
    return new_board, total_score


def move_right(board: List[List[int]]) -> Tuple[List[List[int]], int]:
    """Move all tiles to the right, merge if possible."""
    new_board = []
    total_score = 0
    
    for row in board:
        reversed_row = row[::-1]
        compressed = compress(reversed_row)
        merged, score = merge(compressed)
        final = compress(merged)[::-1]
        new_board.append(final)
        total_score += score
        
    return new_board, total_score


def transpose(board: List[List[int]]) -> List[List[int]]:
    """Transpose the board (swap rows and columns)."""
    return [list(row) for row in zip(*board)]


def move_up(board: List[List[int]]) -> Tuple[List[List[int]], int]:
    """Move all tiles up, merge if possible."""
    transposed = transpose(board)
    moved, score = move_left(transposed)
    return transpose(moved), score


def move_down(board: List[List[int]]) -> Tuple[List[List[int]], int]:
    """Move all tiles down, merge if possible."""
    transposed = transpose(board)
    moved, score = move_right(transposed)
    return transpose(moved), score


def get_empty_cells(board: List[List[int]]) -> List[Tuple[int, int]]:
    """Get a list of coordinates of empty cells (0s) in the board."""
    return [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]


def has_equal_adjacent_tiles(board: List[List[int]]) -> bool:
    """Check if there are any adjacent equal tiles (horizontally or vertically)."""
    # Check horizontal adjacencies
    for r in range(4):
        for c in range(3):
            if board[r][c] != 0 and board[r][c] == board[r][c+1]:
                return True
    
    # Check vertical adjacencies
    for r in range(3):
        for c in range(4):
            if board[r][c] != 0 and board[r][c] == board[r+1][c]:
                return True
                
    return False


def can_move(board: List[List[int]]) -> bool:
    """Check if any moves are possible (left, right, up, down)."""
    # If there are empty cells, a move is possible
    if get_empty_cells(board):
        return True
    
    # If no empty cells, check if there are any adjacent equal tiles
    return has_equal_adjacent_tiles(board)


def is_valid_move(original_board: List[List[int]], direction: str) -> bool:
    """Check if a move in the given direction would change the board."""
    new_board = None
    
    if direction == 'w':
        new_board, _ = move_up(deepcopy(original_board))
    elif direction == 'a':
        new_board, _ = move_left(deepcopy(original_board))
    elif direction == 's':
        new_board, _ = move_down(deepcopy(original_board))
    elif direction == 'd':
        new_board, _ = move_right(deepcopy(original_board))
    
    if new_board is None:
        return False
        
    return not board_equal(original_board, new_board)


def available_moves(board: List[List[int]]) -> List[str]:
    """Return a list of valid move directions."""
    valid_moves = []
    for direction in ['w', 'a', 's', 'd']:
        if is_valid_move(board, direction):
            valid_moves.append(direction)
    return valid_moves


def add_tile(board: List[List[int]]) -> List[List[int]]:
    """Add a new tile (2 or 4) to a random empty cell."""
    empty_cells = get_empty_cells(board)
    if empty_cells:
        r, c = random.choice(empty_cells)
        board[r][c] = 2 if random.random() < 0.9 else 4
    return board


def make_move(board: List[List[int]], direction: str) -> Tuple[List[List[int]], int, bool]:
    """Make a move in the specified direction and return new board, score and if move was valid."""
    # Create a deep copy of the board to check if the move is valid
    original_board = deepcopy(board)
    new_board, score = None, 0
    
    if direction == 'w':
        new_board, score = move_up(board)
    elif direction == 'a':
        new_board, score = move_left(board)
    elif direction == 's':
        new_board, score = move_down(board)
    elif direction == 'd':
        new_board, score = move_right(board)
    else:
        return board, 0, False
    
    # Check if the move actually changed the board
    return new_board, score, not board_equal(original_board, new_board)


def board_equal(board1: List[List[int]], board2: List[List[int]]) -> bool:
    """Check if two boards are equal."""
    for i in range(4):
        for j in range(4):
            if board1[i][j] != board2[i][j]:
                return False
    return True


# -------- Heuristic evaluation functions --------

def score_empty_cells(board: List[List[int]]) -> int:
    """Score based on number of empty cells."""
    return len(get_empty_cells(board))


def score_monotonicity(board: List[List[int]]) -> int:
    """Score based on monotonicity (tiles in increasing order)."""
    score = 0

    # Check rows
    for row in board:
        for i in range(3):
            # If tiles are in increasing order
            if row[i] <= row[i+1] or row[i] == 0 or row[i+1] == 0:
                score += 1
                
    # Check columns
    for j in range(4):
        for i in range(3):
            # If tiles are in increasing order
            if board[i][j] <= board[i+1][j] or board[i][j] == 0 or board[i+1][j] == 0:
                score += 1
                
    return score


def score_smoothness(board: List[List[int]]) -> float:
    """Score based on smoothness (difference between adjacent tiles)."""
    smoothness = 0 
    for i in range(4):
        for j in range(4):
            if board[i][j] != 0:
                # Check horizontal smoothness
                if j < 3 and board[i][j+1] != 0:
                    smoothness -= abs(math.log2(board[i][j]) - math.log2(board[i][j+1]))
                # Check vertical smoothness
                if i < 3 and board[i+1][j] != 0:
                    smoothness -= abs(math.log2(board[i][j]) - math.log2(board[i+1][j]))
    return smoothness


def score_max_tile(board: List[List[int]]) -> float:
    """Score based on max tile value."""
    max_tile = max(max(row) for row in board)
    return math.log2(max_tile) if max_tile > 0 else 0


def score_corner_max(board: List[List[int]]) -> int:
    """Score extra if max tile in the corner."""
    max_tile = 0
    max_pos = None

    for i in range(4):
        for j in range(4):
            if board[i][j] > max_tile:
                max_tile = board[i][j]
                max_pos = (i, j)
    
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]  
    return 4 if max_pos in corners else 0


def score_snake_pattern(board: List[List[int]]) -> float:
    """Score based on snake pattern (zigzag from top-left to bottom-right)."""
    # Define the ideal pattern: snake-like path from top-left
    # 0→1→2→3
    #       ↓
    # 7←6←5←4
    # ↓
    # 8→9→...
    snake_order = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 3), (1, 2), (1, 1), (1, 0),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 3), (3, 2), (3, 1), (3, 0)
    ]
    
    # Get values in the snake pattern
    values = [board[r][c] for r, c in snake_order if board[r][c] != 0]
    
    # If not enough values, return low score
    if len(values) < 4:
        return 0
        
    # Check if values are in descending order
    score = 0
    for i in range(len(values) - 1):
        if values[i] >= values[i + 1]:
            score += 1
            
    return score


def evaluate_board(board: List[List[int]]) -> float:
    """Combined heuristic score for board evaluation."""
    # Get available moves
    valid_moves = available_moves(board)
    
    # If no valid moves, return negative infinity
    if not valid_moves:
        return float('-inf')  # Game over
    
    weights = {
        'empty_cells': 10.0,
        'monotonicity': 4.0,
        'smoothness': 2.0,
        'max_tile': 8.0,
        'corner_max': 12.0,
        'snake_pattern': 6.0,
        'move_options': 5.0  # New heuristic to value boards with more options
    }
    
    scores = {
        'empty_cells': score_empty_cells(board),
        'monotonicity': score_monotonicity(board),
        'smoothness': score_smoothness(board),
        'max_tile': score_max_tile(board),
        'corner_max': score_corner_max(board),
        'snake_pattern': score_snake_pattern(board),
        'move_options': len(valid_moves)  # Value having more move options
    }

    total_score = sum(weights[key] * scores[key] for key in weights)
    return total_score


def expectimax(board: List[List[int]], depth: int, is_max_player: bool, alpha: float = None, beta: float = None) -> Tuple[float, Optional[str]]:
    """Find best move using expectimax algorithm with optional alpha-beta pruning.
    
    For max player (AI): choose the move with highest expected value
    For chance player (adding tiles): calculate expected value of random tile placements
    """
    # Get valid moves
    valid_moves = available_moves(board)
    
    # Base cases
    if depth == 0:
        return evaluate_board(board), None
    elif not valid_moves:
        return float('-inf'), None
    
    if is_max_player:
        max_value = float('-inf')
        best_move = None
        
        # Try each possible move
        for move in valid_moves:
            new_board, score, _ = make_move(deepcopy(board), move)
            
            # Recursively evaluate this move
            move_value, _ = expectimax(new_board, depth - 1, False, alpha, beta)
            move_value += score  # Add the immediate score from merging
            
            if move_value > max_value:
                max_value = move_value
                best_move = move
                
                # Alpha-beta pruning (if enabled)
                if alpha is not None and beta is not None:
                    alpha = max(alpha, max_value)
                    if alpha >= beta:
                        break  # Beta cutoff
                        
        return max_value, best_move
    else:
        # Game's turn - evaluate chance nodes (adding 2 or 4 tiles)
        empty_cells = get_empty_cells(board)
        if not empty_cells:
            return evaluate_board(board), None
            
        expected_value = 0
        
        # Calculate expectation by considering all possible new tile placements
        # We limit this to a sample of cells for efficiency in deeper searches
        sample_cells = empty_cells
        if len(empty_cells) > 4 and depth > 2:
            # Sample a subset of empty cells for efficiency
            sample_cells = random.sample(empty_cells, min(4, len(empty_cells)))
        
        for r, c in sample_cells:
            # Add a 2 tile (90% probability)
            new_board_with_2 = deepcopy(board)
            new_board_with_2[r][c] = 2
            value_with_2, _ = expectimax(new_board_with_2, depth - 1, True, alpha, beta)
            
            # Add a 4 tile (10% probability)
            new_board_with_4 = deepcopy(board)
            new_board_with_4[r][c] = 4
            value_with_4, _ = expectimax(new_board_with_4, depth - 1, True, alpha, beta)
            
            # Weighted average based on spawn probabilities
            cell_expected_value = 0.9 * value_with_2 + 0.1 * value_with_4
            expected_value += cell_expected_value / len(sample_cells)
            
        return expected_value, None


def get_best_move(board: List[List[int]], depth: int = 3, use_pruning: bool = True) -> Optional[str]:
    """Get the best move using expectimax with the given search depth and optional pruning."""
    # First check if any valid moves exist
    valid_moves = available_moves(board)
    if not valid_moves:
        return None
        
    # Use expectimax to find the best move
    if use_pruning:
        _, best_move = expectimax(board, depth, True, float('-inf'), float('inf'))
    else:
        _, best_move = expectimax(board, depth, True)
        
    # Double check that the move is valid
    if best_move not in valid_moves:
        # Fallback to first valid move if something went wrong
        print(f"Warning: AI returned invalid move {best_move}. Falling back to {valid_moves[0]}")
        return valid_moves[0]
        
    return best_move


def run_ai_game(use_pruning: bool = True, target_tile: int = 2048, delay: float = 0.0) -> Tuple[int, int, int]:
    """Run an AI-controlled game of 2048.
    
    Args:
        use_pruning: Whether to use alpha-beta pruning
        target_tile: Target tile value to reach
        delay: Delay between moves in seconds
        
    Returns:
        Tuple of (max_tile, score, move_count)
    """
    board = init_board()
    print("Initial Board:")
    print_board(board)
    
    move_count = 0
    max_tile = 0
    total_score = 0
    
    while can_move(board):
        move_count += 1
        print(f"\nMove #{move_count}")
        
        # Debug: print available moves
        valid_moves = available_moves(board)
        print(f"Valid moves: {valid_moves}")
        
        if not valid_moves:
            print("No valid moves available!")
            break
        
        # Get the best move using expectimax
        start_time = time.time()
        
        # Adaptive depth based on number of empty cells
        empty_count = len(get_empty_cells(board))
        if empty_count >= 10:
            depth = 3  # Faster search when many empty cells
        elif empty_count >= 6:
            depth = 4  # Medium depth for mid-game
        else:
            depth = 5  # Deeper search when fewer empty cells
            
        best_move = get_best_move(board, depth, use_pruning)
        end_time = time.time()
        
        if best_move is None:
            print("AI couldn't find a valid move!")
            # Fallback to first valid move
            if valid_moves:
                best_move = valid_moves[0]
                print(f"Falling back to first valid move: {best_move}")
            else:
                print("Game over - no valid moves!")
                break
            
        print(f"AI chooses: {best_move} (took {end_time - start_time:.2f}s with depth {depth})")
        
        # Make the move
        board, move_score, valid_move = make_move(board, best_move)
        total_score += move_score
        
        # Verify the move was valid
        if not valid_move:
            print(f"Warning: Move {best_move} didn't change the board!")
            # Try a different move as a fallback
            for alt_move in valid_moves:
                if alt_move != best_move:
                    board, alt_score, alt_valid = make_move(deepcopy(board), alt_move)
                    if alt_valid:
                        total_score += alt_score
                        print(f"Fallback to: {alt_move}")
                        break
        
        # Add a new tile
        board = add_tile(board)
        
        # Update max tile
        current_max = max(max(row) for row in board)
        max_tile = max(max_tile, current_max)
        
        # Print current state
        print_board(board)
        print(f"Current max tile: {current_max}")
        print(f"Current score: {total_score}")
        
        if current_max >= target_tile:
            print(f"Congratulations! {target_tile} tile reached!")
            break
        
        # Optional: add delay to make the game visible
        if delay > 0:
            time.sleep(delay)
    
    print("\nGame Over!")
    print(f"Total moves: {move_count}")
    print(f"Max tile achieved: {max_tile}")
    print(f"Final score: {total_score}")
    
    return max_tile, total_score, move_count


def debug_board_state(board_state: List[List[int]]) -> None:
    """Debug a specific board state to troubleshoot issues."""
    print("Debugging board state:")
    print_board(board_state)
    
    # Check valid moves
    valid_moves = available_moves(board_state)
    print(f"Valid moves: {valid_moves}")
    
    # Validate can_move function
    should_be_able_to_move = can_move(board_state)
    print(f"can_move says: {should_be_able_to_move}")
    
    # Check each direction individually
    for direction in ['w', 'a', 's', 'd']:
        is_valid = is_valid_move(board_state, direction)
        print(f"Direction {direction} valid: {is_valid}")
        
        # For debugging, show what the board would look like after this move
        new_board, score, _ = make_move(deepcopy(board_state), direction)
        print(f"After moving {direction}:")
        print_board(new_board)
        print(f"Score from move: {score}")
        print("---")
    
    # Check heuristic scores
    print("Heuristic scores:")
    print(f"Empty cells: {score_empty_cells(board_state)}")
    print(f"Monotonicity: {score_monotonicity(board_state)}")
    print(f"Smoothness: {score_smoothness(board_state)}")
    print(f"Max tile: {score_max_tile(board_state)}")
    print(f"Corner max: {score_corner_max(board_state)}")
    print(f"Snake pattern: {score_snake_pattern(board_state)}")
    print(f"Total evaluation: {evaluate_board(board_state)}")


if __name__ == '__main__':
    # Option 1: Run the AI game
    run_ai_game(use_pruning=True, delay=0.2)
    
    # Option 2: Debug a specific board state
    # Example from your input
    # problematic_board = [
    #     [2, 32, 4, 4],
    #     [4, 16, 128, 32],
    #     [2, 512, 256, 1024],
    #     [16, 32, 4, 2]
    # ]
    # debug_board_state(problematic_board)