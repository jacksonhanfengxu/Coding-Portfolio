# Author: Hanfeng Xu
# Date: Oct 27, 2021
# Introduction: Solved and Implemented the knapsack problem with genetic
#               algorithm. The goal is to fill the backpack to make it
#               as valuable as possible without exceeding the maximum
#               weight (250). Defined the problem as genetic algorithm,
#               provided the genome and defined all fringe operations. 
import random
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    """A GA implementation for knapsack problem

    This class contains most of the basic components of a genetic algorithm.
    The genotypes are defined as a list and ls[i] = 1 means we select the
    ith item, otherwise we do not select the ith item. In the following
    implementation of the algorithm, the probability of mutation is set to
    0.03 and we use the cull method to select  individuals for mating to
    produce the next generation population.

    """

    def __init__(self, population, weights, values, max_iter=150):
        """

        :param population: the origin population
        :param weights: the weights of boxes
        :param values: the values of boxes
        :param max_iter: the maximum rounds of iteration
        """
        self.pop_size = len(population)
        self.population = population
        self.weights = weights
        self.values = values
        self.max_iter = max_iter
        self.fit_values = []

    def fitness(self, idx):
        """
        The fitness is defined as: SUM(VALUES) - ABS(SUM(WEIGHTS) - MAX_WEIGHT)

        :param idx: the idx-th individual in the population
        :return: the fitness
        """
        a = self.population[idx]
        total_weight = 0
        total_value = 0
        for i in range(len(a)):
            if self.population[idx][i] == 1:
                total_weight += self.weights[i]
                total_value += self.values[i]
        return total_value - abs(total_weight - 250)

    def sort_by_fitness(self):
        """
        Sort the population by their corresponding fitness

        :return: the sorted population, individual with higher fitness is at the beginning of the list
        """
        self.fit_values.clear()
        for i in range(self.pop_size):
            self.fit_values.append(self.fitness(i))
        z = zip(self.fit_values, self.population)
        z = sorted(z, reverse=True)
        self.fit_values, self.population = zip(*z)
        self.fit_values = list(self.fit_values)
        self.population = list(self.population)

    def crossover(self, a, b):
        """
        Get the offspring of two individuals

        :param a: individual a
        :param b: individual b
        :return: the offspring that a and b reproduce
        """
        point = random.randint(1, len(a) - 1)
        ret_a, ret_b = [_ for _ in range(len(a))], [_ for _ in range(len(a))]
        for i in range(len(a)):
            if i < point:
                ret_a[i] = a[i]
                ret_b[i] = b[i]
            else:
                ret_a[i] = b[i]
                ret_b[i] = a[i]
        return ret_a, ret_b

    def mutate(self, a):
        """
        A gene has been mutated.

        :param a: individual a
        :return: a with possible mutate on one gene in the genotype
        """
        point = random.randint(0, len(a) - 1)
        a[point] ^= 1

    def reproduce(self, a, b):
        """
        The process of reproduce a and b, note that the probability of mutation
        is 0.03

        :param a: individual a
        :param b: individual b
        :return: the offspring that a and b reproduce with probable mutation
        """
        ret_a, ret_b = self.crossover(a, b)
        if random.random() < 0.03:
            self.mutate(ret_a)
        if random.random() < 0.03:
            self.mutate(ret_b)
        return ret_a, ret_b

    def terminate_check(self):
        """
        Check if we reach at the final state.(i.e. 98% of the population has the same fitness)

        :return: Whether we reach at the final state of not
        """
        cnt = 0
        for i in range(len(self.fit_values) - 1):
            if abs(self.fit_values[i] - self.fit_values[i + 1]) < 1e-5:
                cnt += 1
        if (cnt + 1) / self.pop_size > 0.98:
            return True
        return False

    def run(self):
        """
        Driven function to run this algorithm

        :return: The final solution, with visualized graph being plotted
        """
        arr = []
        for i in range(self.max_iter):
            self.sort_by_fitness()
            arr.append(self.fit_values[0])
            if self.terminate_check():
                print(f"{i} generations in total")
                plt.ylabel('Best Fitness')
                plt.xlabel('Generation')
                plt.plot(arr)
                plt.show()
                return self.population[0]
            mating_pool = self.population[:self.pop_size // 2 + 1]
            n = len(mating_pool)
            new_pop = []
            for j in range(n // 2):
                x = mating_pool[j]
                y = mating_pool[n - 1 - j]
                child_a, child_b = self.reproduce(x, y)
                new_pop.append(child_a)
                new_pop.append(child_b)
            self.population = mating_pool + new_pop
            self.pop_size = len(self.population)
        plt.ylabel('Best Fitness')
        plt.xlabel('Generation')
        plt.plot(arr)
        plt.show()
        print(f"{self.max_iter} generations in total")
        self.sort_by_fitness()
        return self.population[0]


if __name__ == '__main__':
    weights = [20, 30, 60, 90, 50, 70, 30, 30, 70, 20, 20, 60]
    values = [6, 5, 8, 7, 6, 9, 4, 5, 4, 9, 2, 1]
    test_case1 = []
    for k in range(50):
        g = [random.randint(0, 1) for j in range(12)]
        test_case1.append(g)
    ga = GeneticAlgorithm(test_case1, weights, values)
    ls = ga.run()
    w = 0
    v = 0
    for idx in range(len(ls)):
        if ls[idx] == 1:
            w += weights[idx]
            v += values[idx]
    print(f"The output genotype is {ls}, g[i] = 1 means that we put the ith box into the knapsack")
    print(f"The total weight is {w}, and the total value is {v}!")
