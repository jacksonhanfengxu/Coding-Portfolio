# Author: Hanfeng Xu
# Date: Oct 13, 2021
# Introduction: Defined and Implemented the pancake problem as a informed
#               search problem. The goal is for the cook to have them in
#               the “correct” order for the customer, that is, the large
#               on the bottom up to the smallest on top
import time

class MinPriorityQueue:
    """Data structure: Priority Queue

    This class represents a priority queue with the
    minimum on the top. It is designed for this specific
    problem. Notice that our priority queue is 1-index
    based.

    Attributes:
            pq: the underlying list that stores the actual data
            capacity: the max number of elements can be hold in this priority queue
    """

    def __init__(self):
        self.pq = ["DummyHead"]
        self.size = 0

    def empty(self):
        return self.size == 0

    def push(self, item):
        """
        Add a new item to the priority queue

        :param item: the new item to add
        :return: None
        """
        self.size += 1
        self.pq.append(item)
        self.swim(self.size)

    def pop(self):
        """
        Remove and return the minimum element in this priority queue

        :return: the minimum element
        """
        if self.empty():
            print("The priority is already empty! Illegal operation!")
        else:
            minimum = self.pq[1]
            self.swap(1, self.size)
            self.size -= 1
            self.sink(1)
            self.pq.pop()
            return minimum

    def swap(self, i, j):
        """
        Exchange two elements in this priority queue

        :param i: index of the first element
        :param j: index of the second element
        :return: None
        """
        swap = self.pq[i]
        self.pq[i] = self.pq[j]
        self.pq[j] = swap

    def swim(self, k):
        """
        Move an element up to the position where it should be

        :param k: the index of the element
        :return: None
        """
        while k > 1 and self.pq[k // 2][0] > self.pq[k][0]:
            self.swap(k, k // 2)
            k //= 2

    def sink(self, k):
        """
        Move an element down to the position where it should be

        :param k: the index of the element
        :return: None
        """
        while 2 * k <= self.size:
            j = 2 * k
            if j < self.size and self.pq[j][0] > self.pq[j + 1][0]:
                j += 1
            if self.pq[k][0] <= self.pq[j][0]:
                break
            self.swap(k, j)
            k = j

    def print(self):
        """
        print all elements in the priority queue

        :return: None
        """
        print(self.pq[1:])


class Pancakes:
    """ Abstraction for the stack of pancakes

    This class represents a stack of pancakes with size of 10.
    In order to define the problem as a search problem, we simply
    write the functions to get the forward cost and the backward
    cost in this class. Note that in this class, the top pancake is
    at the first position of the list.
    """

    def __init__(self, input_list):
        self.stack = input_list

    def flip(self, pos):
        """
        Reverse the first pos + 1 elements in the list

        :param pos: the position(index) to do the reverse
        :return: A new Pancake object with new pancake stack
        """
        return Pancakes(self.stack[pos::-1] + self.stack[pos + 1:])

    def backward_cost(self):
        """
        The backward cost in A* search

        :return: 1, which means every flip cost the same
        """
        return 1

    def forward_cost(self):
        """
        the forward cost in A* search

        :return: the forward cost measured by the "gap" in the stack
        """
        cost = 0
        for i in range(9):
            if abs(self.stack[i] - self.stack[i + 1])!= 1:
                cost += 1
        if self.stack[-1] != 10:
            cost += 1
        return cost

    def goal_checker(self):
        """
        Check if we reach the final state

        :return: Whether we reach the final state or not
        """
        return self.stack == [i for i in range(1, 11)]


def a_star_search(pancakes: Pancakes):
    """
    A* search algorithm for pancake sorting problem.

    :param pancakes: the pancake stack
    :return: a lise of flips we need to make in order to sort the pancakes in order
    """
    if pancakes.goal_checker():
        return []
    pq = MinPriorityQueue()
    for i in range(1, 10):
        new_pancakes = pancakes.flip(i)
        if new_pancakes.goal_checker():
            return [i]
        else:
            # the inserted item: (cost(priority), actual stack of pancakes, flips that have already been performed)
            pq.push((new_pancakes.backward_cost() + new_pancakes.forward_cost(), new_pancakes, [i]))
    while not pq.empty():
        expansion = pq.pop()
        cur_pancakes = expansion[1]
        done_flips = expansion[2]
        for f in range(1, 10):
            if f != done_flips[-1]:
                # never be back to the state we just came from
                new_pancakes = cur_pancakes.flip(f)
                if new_pancakes.goal_checker():
                    return done_flips + [f]
                else:
                    pq.push(
                        (1 + len(done_flips) + new_pancakes.forward_cost(), new_pancakes, done_flips + [f]))
    print("A star search failed!")
    return None


def ucs(pancakes: Pancakes):
    """
    The uniform cost search for pancake sorting problem. It is basically the same
    as the A* one, except that we just use the backward cost in UCS.
    :param pancakes:
    :return:
    """
    if pancakes.goal_checker():
        return []
    frontier = MinPriorityQueue()
    for i in range(1, 10):
        new_pancakes = pancakes.flip(i)
        if new_pancakes.goal_checker():
            return [i]
        else:
            frontier.push((new_pancakes.backward_cost(), new_pancakes, [i]))
    while not frontier.empty():
        expansion = frontier.pop()
        cur_pancakes = expansion[1]
        done_flips = expansion[2]
        for f in range(1, 10):
            if f != done_flips[-1]:
                # never be back to the state we just came from
                new_pancakes = cur_pancakes.flip(f)
                if new_pancakes.goal_checker():
                    return done_flips + [f]
                else:
                    frontier.push(
                        (1 + len(done_flips), new_pancakes, done_flips + [f]))
    print("A star search failed!")
    return None


def show_pancakes(stack):
    """
    Print the stack of pancakes

    :param stack: pancake stack
    :return: None
    """
    for sz in stack:
        print("_" * sz)


def visualization(stack, flips):
    """
    Show the origin stack of pancakes and the stack after every flip

    :param stack: pancake stack
    :param flips: a list of flips
    :return: None
    """
    print(f"{len(flips)} flip(s) in total")
    time.sleep(1)
    print("The origin stack of pancakes is: ")
    show_pancakes(stack)
    pc = Pancakes(stack)
    # sleep for 1.5secs, to make the process easier to validate
    time.sleep(1.5)
    for i in range(len(flips)):
        f = flips[i]
        pc = pc.flip(f)
        print(f"#{i + 1}, flip at position {f}")
        show_pancakes(pc.stack)
        time.sleep(1.5)


if __name__ == '__main__':
    # test_case1 = Pancakes([3, 4, 6, 2, 1, 7, 8, 9, 10, 5])
    # print(a_star_search(test_case1))
    # print(ucs(test_case1))
    # test_case2 = Pancakes([2, 7, 8, 9, 1, 4, 5, 3, 6, 10])
    # print(a_star_search(test_case2))
    # print(ucs(test_case2))
    print("INPUT EXAMPLE: 2,7,8,9,1,4,5,3,6,10")
    test_case = [int(i) for i in input("Please input the stack of pancakes: ").split(",")]
    start = time.time()
    flips = a_star_search(Pancakes(test_case))
    end = time.time()
    print(f"It takes {end - start:2.2f} to run the a* algorithm.")
    time.sleep(3)
    print("The process of flips display as follows: ")
    visualization(test_case, flips)
    # Check the ucs
    flips_ucs = ucs(Pancakes(test_case))
    assert(flips_ucs == flips)
