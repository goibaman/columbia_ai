from collections import OrderedDict


class CSP:

    def __init__(self, board):
        self. board = board
        self.domains = []
        self.constraints = OrderedDict()
        self.last_algoritm = 0

    def generate_domains(self):
        self.domains = []
        for i in range(81):
            self.domains.append([])
            if self.board.get_cell_value_by_index(i) == 0:
                self.domains[i] = range(1, 10)
            else:
                self.domains[i] = [self.board.get_cell_value_by_index(i)]

    def generate_constraints_arcs(self):
        self.constraints = OrderedDict()
        # Explore all cells and get arcs.
        for col in range(9):
            for row in range(9):
                first = row * 9 + col

                # Arcs for Cols Constraints
                for i in range(9):
                    if i == col:
                        continue
                    second = row * 9 + i
                    self.constraints[(first, second)] = 0

                # Arcs for Rows Constraints
                for i in range(9):
                    if i == row:
                        continue
                    second = i * 9 + col
                    self.constraints[(first, second)] = 0

                # Arcs for Groups Constraints
                group_col = (col // 3) * 3
                group_row = (row // 3) * 3
                for i in range(3):
                    for j in range(3):
                        if col != group_col + i and row != group_row + j:
                            second = (group_row + j) * 9 + (group_col + i)
                            self.constraints[(first, second)] = 0

    def regenerate_constraints_arcs(self, first, exclude):
        # Explore all cells and get arcs.
        first_col = first % 9
        first_row = first // 9

        # Arcs for Cols Constraints
        for i in range(9):
            if i == first_col:
                continue
            second = first_row * 9 + i
            if second != exclude:
                self.constraints[(second, first)] = 0

        # Arcs for Rows Constraints
        for i in range(9):
            if i == first_row:
                continue
            second = i * 9 + first_col
            if second != exclude:
                self.constraints[(second, first)] = 0

        # Arcs for Groups Constraints
        group_col = (first_col // 3) * 3
        group_row = (first_row // 3) * 3
        for i in range(3):
            for j in range(3):
                if first_col != group_col + i and first_row != group_row + j:
                    second = (group_row + j) * 9 + (group_col + i)
                    if second != exclude:
                        self.constraints[(second, first)] = 0

    def ac3(self):
        while len(self.constraints):
            key, value = self.constraints.popitem(last=False)
            first, second = key

            if self.revise(first, second):
                if len(self.domains[first]) == 0:
                    return False

                # Add all the arcs related to this variable again to the queue.
                self.regenerate_constraints_arcs(first, second)

        return True

    def revise(self, first, second):

        # Removed Variable Revised because can't be two invalidation rules at the
        # same time for two cells. This way, return True directly if not Valid.
        for value in self.domains[first]:

            # Validate if there is a value on second domain that is different, therefore valid.
            valid = False
            for i in self.domains[second]:
                if value != i:
                    valid = True
                    break

            # If there is no valid values on second domain, remove the current value from first domain
            # and mark to revise again.
            if not valid:
                self.domains[first].remove(value)
                return True

    def display_domains(self):
        i = 0
        output = " - - - - - - - - - - - -\n"
        for row in range(9):
            for col in range(9):
                if i % 3 == 0:
                    output += "| "
                output += str(self.domains[i]) + " "
                i += 1
            output += "|\n"
            if row % 3 == 2:
                output += " - - - - - - - - - - - -\n"

        print output

    def is_domain_solved(self):
        for i in range(81):
            if len(self.domains[i]) != 1:
                return False
        return True

    def get_result(self):
        output = ""
        if self.last_algoritm == 0:
            for i in range(81):
                output += str(self.domains[i][0])
            return output + " AC3"
        else:
            for i in range(81):
                output += str(self.board.get_cell_value_by_index(i))
            return output + " BTS"

    def is_solved(self):
        for i in range(81):
            # If Value is 0, then is not solved.
            if self.board.get_cell_value_by_index(i) == 0:
                return False
        return True

    def mrv(self):
        cell_index = -1
        for i in range(81):
            if self.board.get_cell_value_by_index(i) == 0:
                if cell_index == -1:
                    cell_index = i
                else:
                    if len(self.domains[i]) < len(self.domains[cell_index]):
                        cell_index = i
        return cell_index

    def bts(self):
        self.last_algoritm = 1
        self.update_board_with_single_domains()
        return self.recursive_bts(self)

    # def order_domain_values(self, index):
    #     self.lcv(index)
    #     return
    #
    # def lcv(self, index):
    #     col, row = self.board.get_cell_coordinates(index)
    #
    #     ordered_list = []
    #
    #
    #     for value in self.domains[index]:
    #         removed = 0.0
    #
    #         # Validate Cols Constraints
    #         for i in range(9):
    #             if i == col:
    #                 continue
    #             second = row * 9 + i
    #             if value in self.domains[second]:
    #                 if self.domains[second] > 0:
    #                     removed += len(self.domains[second])
    #
    #         # Validate Rows Constraints
    #         for i in range(9):
    #             if i == row:
    #                 continue
    #             second = i * 9 + col
    #             if value in self.domains[second]:
    #                 if self.domains[second] > 0:
    #                     removed += len(self.domains[second])
    #
    #                     # Arcs for Groups Constraints
    #         group_col = (col // 3) * 3
    #         group_row = (row // 3) * 3
    #         for i in range(3):
    #             for j in range(3):
    #                 if col != group_col + i and row != group_row + j:
    #                     second = (group_row + j) * 9 + (group_col + i)
    #                     if value in self.domains[second]:
    #                         if self.domains[second] > 0:
    #                             removed += len(self.domains[second])
    #
    #         ordered_list.append((removed, value))
    #
    #     ordered_list.sort(reverse=True, key=lambda tup: tup[0])
    #
    #     self.domains[index] = []
    #     for value in ordered_list:
    #         self.domains[index].append(value[1])
    #
    #     return

    @staticmethod
    def recursive_bts(assignment):

        # This assignment is a solution?
        if assignment.is_solved():
            return True

        # Choose the variable with minimum remaining value
        cell_index = assignment.mrv()
        if cell_index == -1:
            return False

        # For each domain value for this variable, set a new variable and recurse
        # assignment.order_domain_values(cell_index)
        for value in assignment.domains[cell_index]:
            if assignment.board.set_cell_value_by_index(cell_index, value):
                #assignment.board.display_board()
                removed_subdomain = assignment.forward_checking(cell_index)
                # Reduce domains with AC3
                if assignment.inference():
                    if assignment.is_domain_solved():
                        assignment.update_board_with_single_domains()
                        return True

                    if assignment.recursive_bts(assignment):
                        return True
                assignment.restore_domain(removed_subdomain)
                assignment.board.reset_cell_value_by_index(cell_index)
        return False

    def update_board_with_single_domains(self):
        for i in range(81):
            if len(self.domains[i]) == 1:
                if self.board.get_cell_value_by_index(i) == 0:
                    self.board.set_cell_value_by_index(i, self.domains[i][0])

    def forward_checking(self, index):
        removed = dict()
        col, row = self.board.get_cell_coordinates(index)

        # Validate Cols Constraints
        for i in range(9):
            if i == col:
                continue
            second = row * 9 + i
            if self.board.get_cell_value_by_index(index) in self.domains[second]:
                if second in removed.keys():
                    removed[second].append()
                else:
                    removed[second] = [self.board.get_cell_value_by_index(index)]
                    self.domains[second].remove(self.board.get_cell_value_by_index(index))

        # Validate Rows Constraints
        for i in range(9):
            if i == row:
                continue
            second = i * 9 + col
            if self.board.get_cell_value_by_index(index) in self.domains[second]:
                if second in removed.keys():
                    removed[second].append()
                else:
                    removed[second] = [self.board.get_cell_value_by_index(index)]
                    self.domains[second].remove(self.board.get_cell_value_by_index(index))

        # Arcs for Groups Constraints
        group_col = (col // 3) * 3
        group_row = (row // 3) * 3
        for i in range(3):
            for j in range(3):
                if col != group_col + i and row != group_row + j:
                    second = (group_row + j) * 9 + (group_col + i)
                    if self.board.get_cell_value_by_index(index) in self.domains[second]:
                        if second in removed.keys():
                            removed[second].append()
                        else:
                            removed[second] = [self.board.get_cell_value_by_index(index)]
                            self.domains[second].remove(self.board.get_cell_value_by_index(index))

        return removed

    def restore_domain(self, subdomain):
        for key, value in subdomain.items():
            self.domains[key].extend(value)

    def inference(self):
        self.generate_constraints_arcs()
        self.generate_domains()
        return self.ac3()
