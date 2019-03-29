import time

from random import randint

from BaseAI import BaseAI

import math

class PlayerAI(BaseAI):
    def getMove(self, grid):
        timeLimit = 0.2
        maxTime = time.clock() + timeLimit
        startDepth = 1

        # Choose a randomized move
        moves = grid.getAvailableMoves()
        bestMove = moves[randint(0, len(moves) - 1)] if moves else None

        while maxTime > time.clock():
            move = self.maximize(grid, startDepth, maxTime, float('-inf'), float('inf'))

            if(move[0] != 5):
                bestMove = move[0]

            startDepth += 1
        return bestMove


    def maximize(self, state, depth, maxTime, alpha, beta):

        #Initialize move and alpha
        maxEvaluation = (None, float('-inf'))

        # Get Available Moves.
        moves = state.getAvailableMoves()

        # Verifies if the node is terminal
        if len(moves) <= 0 or depth <= 0:
            evaluation = (None, self.evaluate(state))
            return evaluation

        # Get the Child Nodes
        for move in moves:
            if(time.clock() >= maxTime):
                evaluation = (5, float('-inf'))
                return evaluation

            childState = state.clone()
            childState.move(move)
            evaluation = self.minimize(childState, depth, maxTime, alpha, beta)
            if evaluation[0] == 5:
                return evaluation

            if evaluation[1] > maxEvaluation[1]:
                maxEvaluation = (move, evaluation[1])

                if maxEvaluation[1] >= beta:
                    break

                if maxEvaluation[1] > alpha:
                    alpha = maxEvaluation[1]

        return maxEvaluation

    def minimize(self, state, depth, maxTime, alpha, beta):
        # Initialize move and beta
        minEvaluation = (None, float('inf'))

        # Get available cells
        availableCells = state.getAvailableCells()

        # Verifies if the node is terminal
        if len(availableCells) <= 0:
            evaluation = (None, self.evaluate(state))
            return evaluation

        # Get the Child Nodes
        for cell in availableCells:
            if (time.clock() >= maxTime):
                evaluation = (5, float('inf'))
                return evaluation

            for value in range(0,2):
                childState = state.clone()
                childState.setCellValue(cell, (value + 1) * 2)
                evaluation = self.maximize(childState, depth - 1, maxTime, alpha, beta)
                if evaluation[0] == 5:
                    return evaluation

                if evaluation[1] < minEvaluation[1]:
                    minEvaluation = ((cell, value), evaluation[1])

                    if minEvaluation[1] <= alpha:
                        break

                    if minEvaluation[1] > beta:
                        beta = minEvaluation[1]

        return minEvaluation

    def evaluate(self, state):

        weightSmoothness = 0.1
        weightMonotonicity = 1.0
        weightEmptyCellCount = 2.7
        weightMaxCellValue = 1.0

        emptyness = self.emptyCellsCount(state)
        if emptyness == 0:
            emptyness = 1
        return weightSmoothness * self.smoothness(state) + weightMonotonicity * self.monotonicity(state) + weightEmptyCellCount * math.log(emptyness, 10) + weightMaxCellValue * self.maxCellValue(state)

    def emptyCellsCount(self, state):

        count = 0
        for cellIndex in range(0, state.size**2):
            col = cellIndex % state.size
            row = cellIndex // state.size
            if state.map[row][col] == 0:
                count += 1

        return count

    def maxCellValue(self, state):

        max = 0
        for cellIndex in range(0, state.size**2):
            col = cellIndex % state.size
            row = cellIndex // state.size
            if state.map[row][col] > max:
                max = state.map[row][col]

        return max

    def smoothness(self, state):

        smoothness = 0;
        for row in range(0, state.size):
            for col in range(0, state.size):
                if state.map[row][col] != 0:

                    # Search Down
                    targetRow = row + 1
                    while targetRow < state.size and state.map[targetRow][col] == 0:
                        targetRow += 1

                    if not state.crossBound((targetRow, col)):
                        smoothness -= abs(math.log(state.map[row][col], 2) - math.log(state.map[targetRow][col], 2))

                    # Search Left
                    targetCol = col + 1
                    while targetCol < state.size and state.map[row][targetCol] == 0:
                        targetCol += 1

                    if not state.crossBound((row, targetCol)):
                        smoothness -= abs(math.log(state.map[row][col], 2) - math.log(state.map[row][targetCol], 2))

        return smoothness;

    def monotonicity(self, state):

        # Distance by directions
        up = 0
        down = 0
        left = 0
        right = 0

        # Vertical
        for col in range(0, state.size):
            node = 0
            nextNode = 1;
            while nextNode < state.size:
                if state.map[nextNode][col] == 0:
                    nextNode += 1
                    if nextNode >= state.size:
                        up -= math.log(state.map[node][col], 2) if state.map[node][col] > 0 else 0
                    continue

                if state.map[node][col] > state.map[nextNode][col]:
                    up += math.log(state.map[nextNode][col], 2) - math.log(state.map[node][col], 2)
                else:
                    down += math.log(state.map[node][col], 2) - math.log(state.map[nextNode][col], 2) if state.map[node][col] > 0 else 0

                node = nextNode
                nextNode += 1

        # Horizontal
        for row in range(0, state.size):
            node = 0
            nextNode = 1;
            while nextNode < state.size:
                if state.map[row][nextNode] == 0:
                    nextNode += 1
                    if nextNode >= state.size:
                        left -= math.log(state.map[row][node], 2) if state.map[row][node] > 0 else 0
                    continue

                if state.map[row][node] > state.map[row][nextNode]:
                    left += math.log(state.map[row][nextNode], 2) - math.log(state.map[row][node], 2)
                else:
                    right += math.log(state.map[row][node], 2) - math.log(state.map[row][nextNode], 2) if state.map[row][node] > 0 else 0

                node = nextNode
                nextNode += 1

        return max(up, down) + max(left, right)