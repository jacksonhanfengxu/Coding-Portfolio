# Author: Hanfeng Xu
# Date: Sep 27, 2021
# Introduction: Implemented the provided behavior tree in the form of a robotic
#               Vacuum cleaner.The program evaluate the behavior of house
#               vacuum including charging, spot cleaning, dust cleaning,
#               general cleaning, etc.
import sys
import time
import random

SUCCESSED = 'Succeeded'
FAILED = 'Failed'
RUNNING = 'Running'

TRUE = 'True'
FALSE = 'False'

# root node type
class Node:
    children = None

    def __init__(self):
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def run(self, blackboard):
        return FAILED


# general node types
class Task(Node):
    func = None

    def __init__(self, func):
        self.func = func

    def run(self, blackboard):
        return self.func(blackboard)


class Condition(Node):
    key = None
    func = None

    def __init__(self, key, func):
        super().__init__()
        self.key = key
        self.func = func

    def run(self, blackboard):
        if self.func(blackboard[self.key]):
            return SUCCESSED
        else:
            return FAILED


class Sequence(Node):
    def run(self, blackboard):
        for child in self.children:
            result = child.run(blackboard)
            if result == FAILED:
                return FAILED
        return SUCCESSED


class Selector(Node):
    def run(self, blackboard):
        for child in self.children:
            result = child.run(blackboard)
            if result == SUCCESSED:
                return SUCCESSED
        return FAILED


class Priority(Node):
    def add_child(self, child, p):
        self.children.append((child, p))

    def run(self, blackboard):
        for child, p in sorted(self.children, key=lambda e: e[1]):
            result = child.run(blackboard)
            if result in (SUCCESSED, RUNNING):
                return result


class Decorator(Node):
    total = None

    def __init__(self, total):
        super().__init__()
        self.total = total


class UntilSucceeds(Decorator):
    def __init__(self):
        super().__init__(-1)

    def run(self, blackboard):
        while True:
            result = self.children[0].run(blackboard)
            if result == SUCCESSED:
                return SUCCESSED


class Timer(Decorator):
    def run(self, blackboard):
        counter = 0
        while counter < self.total:
            counter += 1
            result = self.children[0].run(blackboard)
            if result == SUCCESSED:
                return SUCCESSED


def print_wait(msg):
    print(msg)
    time.sleep(1)


def dock_func(blackboard):
    blackboard['BATTERY_LEVEL'] += 3
    if blackboard['BATTERY_LEVEL'] > 100:
        blackboard['BATTERY_LEVEL'] = 100
    print("DOCK")
    time.sleep(1)


def clean_floor_func():
    if random.randint(1, 10) < 2:
        print("NOTHING TO CLEAN")
        time.sleep(1)
        return FAILED
    else:
        print("CLEANING FLOOR")
        time.sleep(1)
        return SUCCESSED


def set_key(blackboard, key):
    blackboard[key] = TRUE


def unset_key(blackboard, key):
    blackboard[key] = FALSE


# Decison tree building with the given graph
def build_tree():
    root = Priority()

    do_nothing = Task(lambda e: print_wait("DO NOTHING"))

    charge_seq = Sequence()
    battery_cond = Condition('BATTERY_LEVEL', lambda e: e < 30)
    find_home = Task(lambda e: print_wait("STORE HOME PATH"))
    go_home = Task(lambda e: print_wait("RECALL HOME PATH"))
    dock = Task(dock_func)
    charge_seq.add_child(battery_cond)
    charge_seq.add_child(find_home)
    charge_seq.add_child(go_home)
    charge_seq.add_child(dock)

    clean_sel = Selector()

    spot_seq = Sequence()
    spot_cond = Condition('SPOT_CLEANING', lambda e: e == TRUE)
    spot_timer = Timer(20)
    clean_spot = Task(lambda e: print_wait("CLEAN SPOT"))
    spot_timer.add_child(clean_spot)
    done_spot = Task(lambda e: unset_key(e, 'SPOT_CLEANING'))
    spot_seq.add_child(spot_cond)
    spot_seq.add_child(spot_timer)
    spot_seq.add_child(done_spot)

    gen_seq = Sequence()
    gen_cond = Condition('GENERAL_CLEANING', lambda e: e == TRUE)

    gen_sub_seq = Sequence()

    dust_pri = Priority()

    dust_seq = Sequence()
    dust_cond = Condition('DUSTY_SPOT', lambda e: e == TRUE)
    dust_timer = Timer(35)
    dust_timer.add_child(clean_spot)
    dust_seq.add_child(dust_cond)
    dust_seq.add_child(dust_timer)

    clean_floor_timer = UntilSucceeds()
    clean_floor = Task(lambda e: clean_floor_func())
    clean_floor_timer.add_child(clean_floor)

    dust_pri.add_child(dust_seq, 1)
    dust_pri.add_child(clean_floor_timer, 2)

    done_general = Task(lambda e: unset_key(e, 'GENERAL_CLEANING'))

    gen_sub_seq.add_child(dust_pri)
    gen_sub_seq.add_child(done_general)

    gen_seq.add_child(gen_cond)
    gen_seq.add_child(gen_sub_seq)

    clean_sel.add_child(spot_seq)
    clean_sel.add_child(gen_seq)

    root.add_child(charge_seq, 1)
    root.add_child(clean_sel, 2)
    root.add_child(do_nothing, 3)

    return root

# To generate a default blackboard
def gen_blackboard():
    return {
        'BATTERY_LEVEL': random.randint(1, 100),
        'SPOT_CLEANING': FALSE,
        'GENERAL_CLEANING': FALSE,
        'DUSTY_SPOT': FALSE,
        'HOME_PATH': "/path/to/dock"
    }

# Print out round and all the parameters in the given blackboard
def print_blackboard(rnd, blackboard):
    print("\nRound: %5d" % rnd)
    print("====================blackboard====================")
    for k, v in blackboard.items():
        print("%20s: %10s" % (k, v))
    print("====================blackboard====================")

# Main class for user interface and testing
if __name__ == '__main__':
    root = build_tree()
    blackboard = gen_blackboard()
    msg = "Please type a number to select your option:\n"\
                       "1:SpotClean 2:General Cleaning 3: Both 4: None "\
                       "5: *TESTING USING RANDOM*\n"
    option = int(input(msg))
    if option == 5:
        msg2 = "How many rounds to run?"\
        "Please enter an integer to specify round: "
        rnd = int(input(msg2))
        for i in range(rnd):
            if random.randint(1, 10) < 3:
                blackboard['SPOT_CLEANING'] = TRUE
            if random.randint(1, 10) < 3:
                blackboard['GENERAL_CLEANING'] = TRUE
            if random.randint(1, 10) < 3:
                blackboard['DUSTY_SPOT'] = TRUE
            print_blackboard(i + 1, blackboard)
            root.run(blackboard)
            blackboard['BATTERY_LEVEL'] -= 1
    else:
      if option == 1:
        blackboard['SPOT_CLEANING'] = TRUE
      elif option == 2:
        blackboard['GENERAL_CLEANING'] = TRUE
      elif option == 3:
        blackboard['SPOT_CLEANING'] = TRUE
        blackboard['GENERAL_CLEANING'] = TRUE
      if random.randint(1, 10) < 3:
        blackboard['DUSTY_SPOT'] = TRUE
      print_blackboard(1, blackboard)
      root.run(blackboard)
