class Board:
    def __init__(self, board_string):
        # Load Board
        self.board = dict()
        for i in range(81):
            self.board[i] = int(board_string[i])

    def display_board(self):
        i = 0
        output = " - - - - - - - - - - - -\n"
        for row in range(9):
            for col in range(9):
                if i % 3 == 0:
                    output += "| "
                output += str(self.board[i]) + " "
                i += 1
            output += "|\n"
            if row % 3 == 2:
                output += " - - - - - - - - - - - -\n"

        print output

    @staticmethod
    def get_cell_index(col, row):
        return row * 9 + col

    @staticmethod
    def get_cell_coordinates(index):
        col = index % 9
        row = index // 9
        return col, row

    def get_cell_value_by_index(self, index):
        return self.board[index]

    def get_cell_value_by_coordinates(self, col, row):
        return self.get_cell_value_by_index(self.get_cell_index(col, row))

    def reset_cell_value_by_coordinates(self, col, row):
        index = self.get_cell_index(col, row)
        return self.reset_cell_value_by_index(index)

    def reset_cell_value_by_index(self, index):
        self.board[index] = 0

    def set_cell_value_by_index(self, index, value):
        col, row = self.get_cell_coordinates(index)
        return self.set_cell_value_by_coordinates(col, row, value)

    def set_cell_value_by_coordinates(self, col, row, value):
        if self.is_cell_value_allowed_by_coordinates(col, row, value):
            self.board[self.get_cell_index(col, row)] = value
            return True
        else:
            return False

    def is_cell_value_allowed_by_coordinates(self, col, row, value):

        # Validate Cols Constraints
        for i in range(9):
            if i == col:
                continue
            second = row * 9 + i
            if self.board[second] == value:
                return False

        # Validate Rows Constraints
        for i in range(9):
            if i == row:
                continue
            second = i * 9 + col
            if self.board[second] == value:
                return False

        # Arcs for Groups Constraints
        group_col = (col // 3) * 3
        group_row = (row // 3) * 3
        for i in range(3):
            for j in range(3):
                if col != group_col + i and row != group_row + j:
                    second = (group_row + j) * 9 + (group_col + i)
                    if self.board[second] == value:
                        return False
        return True

    def is_cell_value_allowed_by_index(self, index, value):
        col, row = self.get_cell_coordinates(index)
        return self.is_cell_value_allowed_by_coordinates(col, row, value)
