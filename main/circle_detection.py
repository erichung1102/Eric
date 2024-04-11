def is_surrounded(board, flag, row, col, circle):
    """
        Check if 0 cell on the board is completely surrounded by 1.

        For each 0 search the surrounding 0,
        if there is a 0 at the boundary during the search indicates that it is not surrounded
    """

    def reach_edge(i, j):
        if i == 0 or i == len(board) - 1:
            return True
        elif j == 0 or j == len(board[0]) - 1:
            return True
        else:
            return False

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in directions:
        new_row = row + dr
        new_col = col + dc
        if new_row < 0 or new_col < 0 or new_row == len(board) or new_col == len(board[0]):
            continue
        if reach_edge(new_row, new_col) and board[new_row][new_col] == 0:
            flag[new_row][new_col] = -1
            circle.clear()
            return False
        else:
            if board[new_row][new_col] == 0 and flag[new_row][new_col] == 0:
                flag[new_row][new_col] = -1
                circle.add((new_row, new_col))
                if not is_surrounded(board, flag, new_row, new_col, circle):
                    circle.clear()
                    return False
                else:
                    continue
            else:
                continue
    return True


def is_circle(board):
    flag = board
    circle = set()
    for i in range(1, len(board)-1):
        for j in range(1, len(board[0])-1):
            if board[i][j] == 0 and flag[i][j] == 0:
                # For each unvisited zero, mark it and check if surrounded
                flag[i][j] = -1
                circle.add((i, j))
                if not is_surrounded(board, flag, i, j, circle):
                    circle.clear()
                else:
                    continue
    return len(circle)  # return the number of 0 surrounded by 1


if __name__ == "__main__":

    # test
    board = [
        [1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    print(is_circle(board))