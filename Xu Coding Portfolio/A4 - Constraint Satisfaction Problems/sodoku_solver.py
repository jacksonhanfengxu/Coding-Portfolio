# Author: Hanfeng Xu
# Date: Nov 10, 2021
# Introduction: Solved traditional 9x9 soduku puzzles with two
#               implementations including 1)constraint satisfaction
#               problems and 2)uninformed search method. 
import copy
import time


class SodokuSolver:
    """ Sodoku Solver Class

    We formulate this problem as CSP and using uninformed search method to solve it.
    Sudoku puzzle is represented as a two-dimensional array, with -1 for positions that have no pre-filled elements.
    """

    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.domains = dict()
        self.initialize_domains()

    def initialize_domains(self):
        """
        Initialize the domains of all variables.

        :return: None
        """
        for r in range(9):
            for c in range(9):
                self.domains[(r, c)] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for r in range(9):
            for c in range(9):
                if self.puzzle[r][c] != -1:
                    self.update_domains(r, c, self.puzzle[r][c])

    def update_domains(self, r, c, t):
        """
        Update corresponding domains of variables after we assign puzzle[r][c] the value t.

        :param r: the row index
        :param c: the column index
        :param t: the number we filled in
        :return: the list of cells we have modified in this update
        """
        record = []
        for i in range(9):
            d = self.domains.get((r, i))
            if t in d:
                if (r, i) not in record:
                    record.append((r, i))
            d = self.domains.get((i, c))
            if t in d:
                if (i, c) not in record:
                    record.append((i, c))
            s_row, s_col = (r // 3) * 3 + i // 3, (c // 3) * 3 + i % 3
            d = self.domains.get((s_row, s_col))
            if t in d:
                if (s_row, s_col) not in record:
                    record.append((s_row, s_col))
        for idx1, idx2 in record:
            self.domains.get((idx1, idx2)).remove(t)
        return record

    def forward_checking(self, r, c, t):
        return self.update_domains(r, c, t)

    def rollback(self, records, t):
        """
        This is used for backtracking the state of the domains.

        :param records: the list return by update_domain() method
        :param t: the value we assigned to the cell
        :return: None
        """
        for idx1, idx2 in records:
            self.domains.get((idx1, idx2)).append(t)

    def check_consistency(self, r, c, t):
        """
        Check if puzzle[r][c] = t violates the constraints.

        :param r: the row index
        :param c: the column index
        :param t: the number we filled in
        :return: whether puzzle[r][c] = t violates the constraints or not
        """
        for i in range(9):
            s_row, s_col = (r // 3) * 3 + i // 3, (c // 3) * 3 + i % 3
            if self.puzzle[r][i] == t or self.puzzle[i][c] == t or self.puzzle[s_row][s_col] == t:
                return False
        return True

    def is_assignment_complete(self, puzzle):
        """
        Check if we have reached at the final state

        :param puzzle: the puzzle we are working on
        :return: Whether we have reached at the final state or not
        """
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] == -1:
                    return False
        return True

    def get_next_variable(self):
        """
        Get the next variable for next assignment

        :return: the position of next variable
        """
        for r in range(9):
            for c in range(9):
                if self.puzzle[r][c] == -1:
                    return r, c
        return -1, -1

    def backtrack(self):
        """
        Main routine for search with backtracking.

        :return: None
        """
        if self.is_assignment_complete(self.puzzle):
            return self.puzzle
        r, c = self.get_next_variable()
        domain = self.domains.get((r, c))
        if len(domain) == 0:
            return "Failure"
        for t in range(1, 10):
            if self.check_consistency(r, c, t):
                self.puzzle[r][c] = t
                records = self.forward_checking(r, c, t)
                result = self.backtrack()
                if result != "Failure":
                    return result
                # back to the previous state
                self.puzzle[r][c] = -1
                # rollback changes we made to the domains
                self.rollback(records, t)
        return "Failure"

    def check_conflicts(self, puzzle, r, c, t):
        """
        Return the conflict set after we make assignment puzzle[r][c] = t

        :param puzzle: the sodoku puzzle
        :param r: the row index
        :param c: the column index
        :param t: the number we filled in
        :return: the conflict set after we make assignment puzzle[r][c] = t
        """
        ans = []
        for i in range(9):
            if puzzle[r][i] == t:
                ans.append((r, i))
            if puzzle[i][c] == t:
                ans.append((i, c))
            s_row, s_col = (r // 3) * 3 + i // 3, (c // 3) * 3 + i % 3
            if puzzle[s_row][s_col] == t:
                ans.append((s_row, s_col))
        ans.append((r, c))
        return ans

    def backjump(self, puzzle):
        """
        Main routine for search with conflict-directed backjump

        :param puzzle: the sodoku puzzle
        :return: Whether we succeed or not, the solved puzzle, the conflict set
        """
        if self.is_assignment_complete(puzzle):
            return True, puzzle, []

        # Conflict set tracking new conflicting assignments + subsequent coords
        conflict_set = []
        r, c = self.get_next_variable()

        # Observing board
        for t in range(1, 10):
            if self.check_consistency(r, c, t):
                puzzle[r][c] = t
                result, update_puzzle, new_conflicts = self.backjump(puzzle.copy())
                if result:
                    return True, update_puzzle, []
            else:
                new_conflicts = self.check_conflicts(puzzle, r, c, t)
                if (r, c) not in new_conflicts:
                    return False, puzzle, new_conflicts
                else:
                    # do the union operation as textbook mentioned
                    conflict_set = self.union(conflict_set, new_conflicts)
                    conflict_set.remove((r, c))

            puzzle[r][c] = -1
        return False, puzzle, conflict_set

    def print_sodoku(self):
        """
        Help routine for printing the sodoku

        :return: None
        """
        for r in range(9):
            print(self.puzzle[r])

    def union(self, s1, s2):
        """
        Helper routine. Just act as if the set union operation.
        :param s1: the first list
        :param s2: the second list
        :return: a list contain elements from s1 and s2
        """
        ans = copy.deepcopy(s1)
        for ele in s2:
            if ele not in s1:
                ans.append(ele)
        return ans

    def solve(self):
        """
        Solve the problem with basic implementation

        :return: None
        """
        self.backtrack()
        self.print_sodoku()

    def solve_with_cdj(self):
        """
        Solve the problem with conflict-directed backjumping

        :return: None
        """
        succeed, puzzle, st = self.backjump(self.puzzle)
        if succeed:
            self.puzzle = puzzle
            self.print_sodoku()
        else:
            print("Error!!! Please check the Conflict-Directed Backjumping Algorithm")

    #   Use for debug!
    # def print_domains(self):
    #     print(self.domains)


if __name__ == '__main__':
    sodoku_easy = [
        [6, -1, 8, 7, -1, 2, 1, -1, -1],
        [4, -1, -1, -1, 1, -1, -1, -1, 2],
        [-1, 2, 5, 4, -1, -1, -1, -1, -1],
        [7, -1, 1, -1, 8, -1, 4, -1, 5],
        [-1, 8, -1, -1, -1, -1, -1, 7, -1],
        [5, -1, 9, -1, 6, -1, 3, -1, 1],
        [-1, -1, -1, -1, -1, 6, 7, 5, -1],
        [2, -1, -1, -1, 9, -1, -1, -1, 8],
        [-1, -1, 6, 8, -1, 5, 2, -1, 3]
    ]
    sodoku_evil = [
        [-1, 7, -1, -1, 4, 2, -1, -1, -1],
        [-1, -1, -1, -1, -1, 8, 6, 1, -1],
        [3, 9, -1, -1, -1, -1, -1, -1, 7],
        [-1, -1, -1, -1, -1, 4, -1, -1, 9],
        [-1, -1, 3, -1, -1, -1, 7, -1, -1],
        [5, -1, -1, 1, -1, -1, -1, -1, -1],
        [8, -1, -1, -1, -1, -1, -1, 7, 6],
        [-1, 5, 4, 8, -1, -1, -1, -1, -1],
        [-1, -1, -1, 6, 1, -1, -1, 5, -1]
    ]
    start = time.time()
    solver = SodokuSolver(sodoku_easy)
    print("The ans for the easy puzzle with basic implementation is: ")
    solver.solve()
    end = time.time()
    t1 = end - start
    start = time.time()
    solver = SodokuSolver(sodoku_evil)
    print("The ans for the evil puzzle with basic implementation is: ")
    solver.solve()
    end = time.time()
    t2 = end - start
    start = time.time()
    solver = SodokuSolver(sodoku_easy)
    print("The ans for the easy puzzle with Conflict-Directed Backjumping is: ")
    solver.solve_with_cdj()
    end = time.time()
    t3 = end - start
    start = time.time()
    solver = SodokuSolver(sodoku_evil)
    print("The ans for the evil puzzle with Conflict-Directed Backjumping is: ")
    solver.solve_with_cdj()
    end = time.time()
    t4 = end - start
    print("---------------------------------------------------------------------------------")
    print("                 BASIC IMPLEMENTATION                   CONFLICT-DIRECTED BACKJUMPING")
    print(f"EASY                     {t1:^2.2f}s                                     {t3:^2.2f}s")
    print(f"EVIL                     {t2:^2.2f}s                                     {t4:^2.2f}s")
