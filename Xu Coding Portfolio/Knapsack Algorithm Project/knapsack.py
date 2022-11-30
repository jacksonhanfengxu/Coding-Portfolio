# Python3 program to solve Knapsack Problem
# Name: Hanfeng Xu
# Date: Dec 10, 2021
# Introduction: Solved the classic knapsack problem with three algorithms and compared
#               their runtime in different scenarios :
#               1) Greedy 2) Brute Force 3) Dynamic Programming
class Item:

	"""Item Value DataClass"""
	def __init__(self, wt, val, ind):
		self.wt = wt
		self.val = val
		self.ind = ind
		self.cost = val // wt

	def __lt__(self, other):
		return self.cost < other.cost

# Greedy Approach
# :param Capacity: Weight restriction
# :param wt: Array of weights of items
# :param val: Array of value of items
# :return: the maximum value able to fit in knapsack
def GreedyFractionalKnapSack(wt, val, capacity):
	"""function to get maximum value """
	iVal = []
	for i in range(len(wt)):
		iVal.append(Item(wt[i], val[i], i))

		# sorting items by value
	iVal.sort(reverse=True)

	totalValue = 0
	for i in iVal:
		curWt = int(i.wt)
		curVal = int(i.val)
		if capacity - curWt >= 0:
			capacity -= curWt
			totalValue += curVal
		else:
			fraction = capacity / curWt
			totalValue += curVal * fraction
			capacity = int(capacity - (curWt * fraction))
			break
	return totalValue


# The brute force solution of 0-1 Knapsack Problem
# :param W: Weight restriction
# :param wt: Array of weights of items
# :param val: Array of value of items
# :param n: number of items
# :return: the maximum value able to fit in knapsack
def knapSackBruteForce(W, wt, val, n):
	# Base Case: terminate when we are out of space or items
	if n == 0 or W == 0:
		return 0

	# If weight of the nth item is more than Knapsack of capacity
	# W, then this item cannot be included in the optimal solution
	if (wt[n - 1] > W):
		return knapSackBruteForce(W, wt, val, n - 1)

	# return the maximum of nth item included or not included
	else:
		return max(val[n - 1] + knapSackBruteForce(W - wt[n - 1], wt, val, n - 1),
				   knapSackBruteForce(W, wt, val, n - 1))

# A Dynamic Programming based solution
# :param W: Weight restriction
# :param wt: Array of weights of items
# :param val: Array of value of items
# :param n: number of items
# :return: the maximum value able to fit in knapsack
def knapSackDP(W, wt, val, n):
	K = [[0 for x in range(W + 1)] for x in range(n + 1)]

	# Build table K[][] in bottom up manner
	for i in range(n + 1):
		for w in range(W + 1):
			if i == 0 or w == 0:
				K[i][w] = 0
			elif wt[i-1] <= w:
				K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]], K[i-1][w])
			else:
				K[i][w] = K[i-1][w]

	return K[n][W]



# Driver Code
if __name__ == "__main__":
	import time
	wt = [10, 40, 20, 30]
	val = [60, 40, 100, 120]
	capacity = 50
	maxValue = GreedyFractionalKnapSack(wt, val, capacity)
	print("Test for Greedy: \n Maximum value in Knapsack for greedy =", maxValue)

	print("Test for input of 3 items:")

	val0 = [60, 100, 120]
	wt0 = [10, 20, 30]
	W = 50
	n = len(val)
	start = time.time()
	print("Max value from brute force is: ", knapSackBruteForce(W, wt, val, n))
	end = time.time()
	print(f"It takes {end - start:f} to run the brute force algorithm.")
	start = time.time()
	print("Max value from DP is: ", knapSackDP(W, wt, val, n))
	end = time.time()
	print(f"It takes {end - start:f} to run the DP algorithm.")
	print()
	print("Test for input of 10 items:")
	# Correct answer 295
	w1 = [95,4,60,32,23,72,80,62,65,46]
	val1 = [55,10,47,5,4,50,8,61,85,87]
	capacity1 = 269
	n1 = len(val1)
	start = time.time()
	print("Max value from brute force is: ", knapSackBruteForce(capacity1, w1, val1, n1))
	end = time.time()
	print(f"It takes {end - start:f} to run the brute force algorithm.")
	start = time.time()
	print("Max value from DP is: ", knapSackDP(capacity1, w1, val1, n1))
	end = time.time()
	print(f"It takes {end - start:f} to run the DP algorithm.")
	print()
	print("Test for input of 20 items:")
	w2 = [92,4,43,83,84,68,92,82,6,44,32,18,56,83,25,96,70,48,14,58]
	val2 = [44,46,90,72,91,40,75,35,8,54,78,40,77,15,61,17,75,29,75,63]
	capacity2 = 878
	n2 = len(val2)
	start = time.time()
	print("Max value from brute force is: ", knapSackBruteForce(capacity2, w2, val2, n2))
	end = time.time()
	print(f"It takes {end - start:f} to run the brute force algorithm.")
	start = time.time()
	print("Max value from DP is: ", knapSackDP(capacity2, w2, val2, n2))
	end = time.time()
	print(f"It takes {end - start:f} to run the DP algorithm.")
	w3= [54, 95, 36, 18, 4, 71, 83, 16, 27, 84, 88, 45, 94, 64, 14, 80, 4, 23,75, 36, 90, 20, 77, 32, 58, 6, 14, 86, 84, 59, 71, 21, 30, 22, 96, 49, 81,48, 37, 28, 6, 84, 19, 55, 88, 38, 51, 52, 79, 55, 70, 53, 64, 99, 61, 86,1, 64, 32, 60, 42, 45, 34, 22, 49, 37, 33, 1, 78, 43, 85, 24, 96, 32, 99,57, 23, 8, 10, 74, 59, 89, 95, 40, 46, 65, 6, 89, 84, 83, 6, 19, 45, 59,26, 13, 8, 26, 5, 9]
	val3 = [297, 295, 293, 292, 291, 289, 284, 284, 283, 283, 281, 280, 279,277, 276, 275, 273,264, 260, 257, 250, 236, 236, 235, 235, 233, 232,232, 228, 218, 217, 214, 211, 208, 205, 204, 203, 201, 196, 194, 193,193, 192, 191, 190, 187, 187, 184, 184, 184, 181, 179, 176, 173, 172,171, 160, 128, 123, 114, 113, 107, 105, 101, 100, 100, 99, 98, 97, 94,94, 93, 91, 80, 74, 73, 72, 63, 63, 62, 61, 60, 56, 53, 52, 50, 48, 46,40, 40, 35, 28, 22, 22, 18, 15, 12, 11, 6, 5]
	capacity3 = 3818
	n3 = len(val3)
	print("Test for input of 100 items:")
	start = time.time()
	# This takes more than 10 minutes so I blocked it out
	# print("Max value from brute force is: ", knapSackBruteForce(capacity3, w3, val3, n3))
	end = time.time()
	print(f"It takes {end - start:f} to run the brute force algorithm.")
	start = time.time()
	print("Max value from DP is: ", knapSackDP(capacity3, w3, val3, n3))
	end = time.time()
	print(f"It takes {end - start:f} to run the DP algorithm.")

	# Function call

