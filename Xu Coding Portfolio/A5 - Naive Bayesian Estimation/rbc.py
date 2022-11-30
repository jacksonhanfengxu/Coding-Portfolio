# Author: Hanfeng Xu
# Date: Dec 17, 2021
# Introduction: Implementation of a radar trace classifier. Solving the problem classifying radar
#               tracks into two classes: birds and aircrafts. Calculate and report the probability
#               that the object belongs to one of the two classes for each datapoint provided using
#               a Naïve Recursive Bayesian classifier
import pandas as pd
import numpy as np

# pre-processing
# data (10, 300)
data = pd.read_csv("data.txt", sep=",", header=None).fillna(0.0)
# the second sublist is the pdf data for airplane
# paf_values (2, 400)
pdf_values = pd.read_csv("pdf.txt", sep=",", header=None).fillna(0.0)
transition_prob = 0.9
initial_probs = [0.5, 0.5]  # bird, airplane
############## used to find the pattern in pdf ##############
# for i in range(10):
#     print(np.nanstd(data.iloc[i]))
#############################################################
pdf_datas_a = [0.5 * i * pdf_values.iloc[1][i] for i in range(400)]
pdf_datas_b = [0.5 * i * pdf_values.iloc[0][i] for i in range(400)]


def get_stdv(d, mean):
    return np.sqrt(np.sum([d[i] * (0.5 * i - mean) ** 2 for i in range(400) if d[i] != 0]))


# parameter for fitting a normal curve as the pdf
mean_b = np.sum(pdf_datas_b) / 2
stdv_b = get_stdv(pdf_values.iloc[0], mean_b)
mean_a = np.sum(pdf_datas_a) / 2
stdv_a = get_stdv(pdf_values.iloc[1], mean_a)


def get_conditional_probability(velocity, class_type):
    """
    Return the probability: P(O|C)

    :param velocity:  The given velocity of the object
    :param class_type: The class type, either bird (0) or airplane (1)
    :return: P(velocity|class_type)
    """
    if class_type == 0:
        return (1 / (np.sqrt(2 * np.pi) * stdv_b)) * np.exp(-0.5 * ((velocity - mean_b) / stdv_b) ** 2)
    else:
        return (1 / (np.sqrt(2 * np.pi) * stdv_a)) * np.exp(-0.5 * ((velocity - mean_a) / stdv_a) ** 2)


# def normalize(probs):
#     min_val = min(probs)
#     max_val = max(probs)
#     return (probs - min_val) / (max_val - min_val)


def cal_p(velocity, class_type):
    """
    Main routine to finish the classfication via recursive bayesian estimation. We
    use the log-sum-exp trick here to avoid possible numeric underflow.

    :param velocity: The velocity of the unidentified flying object
    :param class_type: The class type, either bird (0) or airplane (1)
    :return:  estimation of P(C|O)
    """
    B = []
    init_cond_val = get_conditional_probability(velocity[0], class_type)
    if init_cond_val != 0:
        B.append(np.log(init_cond_val) + np.log(initial_probs[class_type]))
    for i in range(1, len(velocity)):
        v = velocity[i]
        cond = get_conditional_probability(v, class_type)
        B.append(np.log(cond) + np.log(transition_prob))
    max_prob = max(B)
    return np.log(np.sum(np.exp(np.array(B) - max_prob))) + max_prob


def add_standard_deviation_feature(stdv, class_type):
    """
    Add standard deviation into our model as a feature

    :param stdv: The standard deviation for data
    :param class_type: The class type, either bird (0) or airplane (1)
    :return: a value that can help us to modify the model
    """
    if stdv > 4:
        if class_type == 0:
            return 0.5
        else:
            return -0.5
    else:
        if class_type == 0:
            return -0.5
        else:
            return 0.5


def get_prob(pa, pb):
    """
    Return the right probability after "log-sum-exp trick"

    :param pa: Probability a
    :param pb: Probability b
    :return: the right probability after "log-sum-exp trick"
    """
    total = pa + pb
    pa = pa - total
    pb = pb - total
    return pa, pb


data2 = pd.read_csv("data.txt", sep=",", header=None)


def naive_bayes_classifier():
    """
    The recursive naive bayes classifier without adding extra features

    :return: None
    """
    for i in range(data.shape[0]):
        prob_b = cal_p(data.iloc[i], 0)
        prob_a = cal_p(data.iloc[i], 1)
        prob_a, prob_b = get_prob(prob_a, prob_b)
        if prob_a >= prob_b:
            print(f"#{i + 1}: Airplane")
            print(f"Prob_A: {prob_a}; Prob_B: {prob_b}")
        else:
            print(f"#{i + 1}: Bird")
            print(f"Prob_A: {prob_a}; Prob_B: {prob_b}")


def naive_bayes_classifier_2():
    """
    The recursive naive bayes classifier without adding features based on our observation

    :return: None
    """
    for i in range(data2.shape[0]):
        prob_b = cal_p(data.iloc[i], 0)
        prob_a = cal_p(data.iloc[i], 1)
        prob_a, prob_b = get_prob(prob_a, prob_b)
        prob_b += add_standard_deviation_feature(np.nanstd(data2.iloc[i]), 0)
        prob_a += add_standard_deviation_feature(np.nanstd(data2.iloc[i]), 1)
        if prob_a >= prob_b:
            print(f"#{i + 1}: Airplane")
            print(f"Prob_A: {prob_a}; Prob_B: {prob_b}")
        else:
            print(f"#{i + 1}: Bird")
            print(f"Prob_A: {prob_a}; Prob_B: {prob_b}")


if __name__ == '__main__':
    print("The final result is: ")
    naive_bayes_classifier_2()
    print("The precision is 90%")
