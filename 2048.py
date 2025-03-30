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
# finally adding the main function to run the game loop and handle user input for moves.


import random

def init_board():
    board = [[0] * 4 for _ in range(4)]

    def add_new_tile():
        empty_cells = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]
        if empty_cells:
            r, c = random.choice(empty_cells)
            board[r][c] = 2 if random.random() < 0.9 else 4
         # random.random() generates a random float between 0 and 1

    add_new_tile()
    add_new_tile()
    return board

def print_board(board):
    for row in board:
        print("\t".join(str(cell) if cell != 0 else "--" for cell in row))

def compress(row):
    #    Example: [2, 0 2, 4] becomes [2, 2, 4, 0]
    new_row = [num for num in row if num != 0]
    new_row += [0] * (len(row) - len(new_row))
    return new_row

def merge(row):
    #    Example: [2, 2, 4, 0] becomes [4, 4, 0, 0]
    for i in range(len(row) - 1):
        if row[i] == row[i + 1] and row[i] != 0:
            row[i] *= 2
            row[i + 1] = 0
    return row

def move_left(board):
    # For each row, compress, merge, and compress again
    new_board = []
    for row in board:
        compressed = compress(row)
        merged = merge(compressed)
        final = compress(merged)
        new_board.append(final)
    return new_board

def move_right(board):
    # For each row, reverse it, apply move_left, and reverse it back
    new_board = []
    for row in board:
        reversed_row = row[::-1]
        compressed = compress(reversed_row)
        merged = merge(compressed)
        final = compress(merged)[::-1]
        new_board.append(final)
    return new_board

def transpose(board):
    # Transpose the board (swap rows and columns)
    return [list(row) for row in zip(*board)]

def move_up(board):
    # Transpose the board, move left, then transpose again
    transposed = transpose(board)
    moved = move_left(transposed)
    return transpose(moved)

def move_down(board):
    # Transpose the board, move right, then transpose again
    transposed = transpose(board)
    moved = move_right(transposed)
    return transpose(moved)

def can_move(board):
    # Check if any moves are possible (left, right, up, down)
    for row in board:
        if 0 in row:
            return True
    for row in board:
        for c in range(3):
            if row[c] == row[c + 1] and row[c] != 0:
                return True
    for c in range(4):
        for r in range(3):
            if board[r][c] == board[r + 1][c] and board[r][c] != 0:
                return True
    return False

def add_tile(board):
    # Add a new tile (2 or 4) to a random empty cell
    empty_cells = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]
    if empty_cells:
        r, c = random.choice(empty_cells)
        board[r][c] = 2 if random.random() < 0.9 else 4
    return board

if __name__ == '__main__':
    board = init_board()
    print("Welcome to 2048!")
    print_board(board)
    
    while True:
        # Check if any moves are possible. If not, end the game.
        if not can_move(board):
            print("Game Over! No more moves available.")
            break
        
        # Get user input for the move.
        move = input("Enter move (w = up, a = left, s = down, d = right, q = quit): ").strip().lower()
        if move == 'q':
            print("Quitting the game.")
            break
        
        # Make a copy of the board to check if the move changes anything.
        prev_board = [row[:] for row in board]
        
        # Map input to the corresponding move function.
        if move == 'w':
            board = move_up(board)
        elif move == 'a':
            board = move_left(board)
        elif move == 's':
            board = move_down(board)
        elif move == 'd':
            board = move_right(board)
        else:
            print("Invalid move. Please try again.")
            continue
        
        # If the move changes the board, add a new tile.
        if board != prev_board:
            add_tile(board)
        else:
            print("Move did not change the board. Try a different move.")
        
        # Print the updated board.
        print_board(board)