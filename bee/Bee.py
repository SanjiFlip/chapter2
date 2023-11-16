# -*- coding: UTF-8 -*-
"""
@Project ：AI 
@File    ：Bee.py
@Author  ：wangyi
@Date    ：2023/11/7 17:47 
"""
import numpy as np
import random
import math
from random import choice
import matplotlib.pyplot as plt

# Parameters
# D、 N：维度以及组数
D = 5
N = 10
# ub、lb：区域上下界。
ub = 5
lb = -5
# Ub、 Lb：全域上下界。
Ub = 5 * np.ones(D)
Lb = -5 * np.ones(D)
# max_iter：迭代次数。
max_iter = 10000
# limit：尝试上限，最终决定是否放弃该蜜源。
limit = 50


# Rastrigin function
def fun(X):
    funsum = 0
    for i in range(D):
        x = X[:, i]
        funsum += x ** 2 - 10 * np.cos(2 * np.pi * x)
    funsum += 10 * D
    return funsum


# Styblinski-Tang function, (-5,5)
def fun_(X):
    funsum = 0
    for i in range(D):
        x = X[:, i]
        funsum += x ** 4 - 16 * (x ** 2) + 5 * x
    funsum *= 0.5
    return funsum


# 1D to 2D with calculate 'fun' function
def transform1Dto2D_fv(num, D, fun):
    substitute = np.zeros(D)
    toCalculate = np.row_stack((num, substitute))
    testOut = fun(toCalculate)
    result = np.delete(testOut, -1, axis=0)
    return result


# Fitness function 蜜源丰富程度（适应度）
def fitness_machine(num):
    fitness = 1 / (1 + num) if num >= 0 else 1 + abs(num)
    return fitness


# Build probability list 观察蜂阶段
def probability_list(fs, fun):
    """
        观察蜂根据轮盘法（轮盘赌选择）选择一个蜜源，在该蜜源附近更新此蜜源。
        针对每个组数依序生成一0~1之间的乱数r，透过和其组数对应的机率p比大小，
        如果r比p小，则进行到观察蜂阶段，若没有则保持不变，且试验增加1；
        由上述可知，如果适应度的机器率p越大，则可能需要替换的机器率就会越大。
    """
    out = fun(fs)
    fvList = []
    Probability_list = []
    for count_fitness in out:
        fitValue = fitness_machine(count_fitness)
        fvList.append(fitValue)
    sum_fv = sum(fvList)
    for temp in fvList:
        each_pro = temp / sum_fv
        Probability_list.append(each_pro)
    return Probability_list


# Employed Bee phase  雇佣蜂阶段
def Employed_update_fs(num_, fs, fitness_list, trial):
    """
        f(x)越小，代表适应度越大；f(x)越大，代表其适应度较小，故为了只保留简单，
        若生成的f(x)没有比较，则保留原值，必须试验加1，反如果生成的f(x)有比较小，
        则将其替换且试验保持不变。
    """
    fs_row = fs[num_].tolist()
    num = choice(fs_row)
    cvIndex = fs_row.index(num)

    waiting_area = np.delete(fs[:, cvIndex], num_)

    challenge_num = choice(waiting_area)
    Xnew = num + (random.uniform(-1, 1)) * (num - challenge_num)
    if Xnew > ub:
        Xnew = ub
    elif Xnew < lb:
        Xnew = lb
    fs_list = []
    for i in range(D):
        fs_list.append(fs[num_][i])
    fs_list[cvIndex] = Xnew
    fs_element = np.asarray(fs_list)

    # Judgment elements
    out_ = transform1Dto2D_fv(fs_element, D, fun)
    new_fitness = fitness_machine(out_)

    out_compare = transform1Dto2D_fv(fs[num_], D, fun)
    fitness_compare = fitness_machine(out_compare)

    # Judgment
    if new_fitness < fitness_compare:
        fs[num_][cvIndex] = fs[num_][cvIndex]
        trial[num_] += 1
    else:
        fs[num_][cvIndex] = Xnew
        trial[num_] = 0

    out = fun(fs)
    fitness = []
    for i in range(N):
        fv = fitness_machine(out[i])
        fitness.append(fv)
    return fs, out, fitness, trial, num, Xnew


# On-looker Bee phase 观察要点阶段
def Onlooker_update_fs_step2(num_, fs_step2):
    """
        其步骤与雇佣蜂阶段大致相同，首先针对需替换组的随机选择一个值（公式中为X），
        其后根据所选的值所在的列随机选择一个值为其更新的伙伴（公式中为Xp） )，其中phi为-1~1间的乱数，
        将以上数字带入Xnew公式中求出新的值，并Xnew暂时替代近似的值X，求出新的f(x)和fitness，借用通过和原健身比较，
        决定是否留下新的值以及试验次数(Trial)是否需要加1。
    """
    # choose vairable Xn
    numOnlooker = choice(fs_step2[num_])
    # choose partner
    fs_step2_row = fs_step2[num_].tolist()
    partnerIndex = fs_step2_row.index(numOnlooker)
    waiting_area_ = np.delete(fs_step2[:, partnerIndex], num_)
    partner = choice(waiting_area_)
    Xnew_ = numOnlooker + (random.uniform(-1, 1)) * (numOnlooker - partner)
    if Xnew_ > ub:
        Xnew_ = ub
    elif Xnew_ < lb:
        Xnew_ = lb
    fs_list_onlooker = []
    for i in range(D):
        fs_list_onlooker.append(fs_step2[num_][i])
    fs_list_onlooker[partnerIndex] = Xnew_
    fs_element_ = np.asarray(fs_list_onlooker)

    # Judgment elements
    test_out_onlooker = transform1Dto2D_fv(fs_element_, D, fun)
    new_fitness_ = fitness_machine(test_out_onlooker)

    out_compare = transform1Dto2D_fv(fs_step2[num_], D, fun)
    fitness_compare = fitness_machine(out_compare)

    # Judgment
    if new_fitness_ < fitness_compare:
        fs_step2[num_][partnerIndex] = fs_step2[num_][partnerIndex]
        fv_afe[num_] += 1
    else:
        fs_step2[num_][partnerIndex] = Xnew_
        fv_afe[num_] = 0
    return fs_step2, fv_afe


# Generate Food Source
fs = np.zeros((N, D))
for i in range(N):
    for j in range(D):
        fs[i, j] = np.random.uniform(-5, 5)

# Count f(x)
out = fun(fs)

# Count Fitness value
fitness_list = []
for i in range(N):
    fv = fitness_machine(out[i])
    fitness_list.append(fv)

# Show the Trial
trial = np.zeros(N).astype(int)

# Set up the best_volumn
best_list = []
Best_Food_Source_list = []

it = 0
while it <= max_iter:
    # Start the while
    # Start the part of Employed Bee
    for i in range(N):
        after_emoloyed = Employed_update_fs(i, fs, fitness_list, trial)

    # Give new variable names
    fs_1 = after_emoloyed[0]
    fv_afe = after_emoloyed[3]

    # Build the probability list
    Probability_list = probability_list(fs_1, fun)

    # Onlooker Bee Phase
    fs_step2 = after_emoloyed[0]

    # 判断是否进入观察高峰阶段
    for i in range(N):
        if random.uniform(0, 1) < Probability_list[i]:
            onlooker_out = Onlooker_update_fs_step2(i, fs_step2)
            fs_step2_2 = onlooker_out[0]
            fv_afe = onlooker_out[1]
        else:
            fs_step2_2 = fs_step2
            fv_afe[i] += 1

    # Scout bee phase
    for j in range(N):
        if fv_afe[j] > limit:
            for i in range(D):
                fs_step2_2[j][i] = random.uniform(-5, 5)
                fv_afe[j] = 0

    # Store the best answer
    final_out = fun(fs_step2)
    best_ = min(final_out)
    best_list.append(best_)

    # Cohesion
    fs = fs_step2_2
    trial = fv_afe

    index_best = np.where(best_)
    index_best = list(final_out).index(best_)
    Food_Source_best = fs_step2[index_best]
    Best_Food_Source_list.append(Food_Source_best)
    it += 1

min_best_num = min(best_list)
# arr = list(best_list).index(max_best_num)
arr = best_list.index(min_best_num)
Best_FS_ = Best_Food_Source_list[arr]

# print('')
np.set_printoptions(suppress=True)
print('Best F(X): ', min_best_num, 'Best food source: ', Best_FS_)

plt.figure(figsize=(15, 8))
plt.xlabel("Iteration", fontsize=15)
plt.ylabel("Value", fontsize=15)

plt.plot(best_list, linewidth=2, label="History Value", color='r')
# plt.plot(every_time_value,linewidth = 2, label = "Best value so far", color = 'b')
plt.legend()
plt.show()
