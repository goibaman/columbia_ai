from board import Board
from csp import CSP
import sys

def main(board_string):

    # Creates the given Board
    board = Board(board_string)
    # board.display_board()

    # Creates de CSP.
    csp = CSP(board)
    csp.generate_domains()
    csp.generate_constraints_arcs()
    file = open("output.txt", "w")
    if csp.ac3():
        if csp.is_domain_solved():
            file.write(csp.get_result())
            print csp.get_result()
        else:
            if csp.bts():
                file.write(csp.get_result())
                print csp.get_result()
    else:
        print "Could not solve"
    file.close()

if __name__ == '__main__':

    main(sys.argv[1])


