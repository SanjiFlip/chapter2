# -*- coding: UTF-8 -*-
"""
@Project ：chapter2 
@File    ：Salp_Swarm_Algorithm_Two.py
@Author  ：wangyi
@Date    ：2023/11/8 11:10 
"""

# Required Libraries
import numpy as np
import math
import test_fun_new
import matplotlib.pyplot as plt
import test_function


class salp_swarm_algorithm():
    def __init__(self, pop_size=50, n_dim=2, max_iter=150, lb=[-5, -5], ub=[5, 5], func=None):
        # 樽海鞘种群数量
        self.pop = pop_size
        # 为求解下边界
        self.lb = lb
        # 为求解上边界
        self.ub = ub
        # 测试函数
        self.func = func
        # 为求解维度
        self.n_dim = n_dim
        # 最大迭代次数
        self.max_iter = max_iter
        """
            生成种群 low 参数指定均匀分布的下限，而 high 参数指定上限 
            size=(self.pop, self.n_dim): 这是指定生成的随机数组的形状的参数。
            在这里，它创建一个二维数组，其中包含 self.pop 行和 self.n_dim 列，
            这将是一个矩阵，包含 self.pop 个样本，每个样本有 self.n_dim 个特征。
        """
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = [np.inf for i in range(self.pop)]  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_pbest()
        self.update_gbest()

    def update_pbest(self):
        '''
        personal best 更新个人最优解
        :return:
        '''
        for i in range(len(self.Y)):
            if self.pbest_y[i] > self.Y[i]:
                self.pbest_x[i] = self.X[i]
                self.pbest_y[i] = self.Y[i]

    def update_gbest(self):
        '''
        global best 更新全局最优解
        :return:
        '''
        idx_min = self.pbest_y.index(min(self.pbest_y))
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    # 获取衰减的影响因子
    def getAlpha(self, t):
        a = np.exp(-30 * (t / self.max_iter))
        return a

    # k是服从参数为0.5的指数分布随机数
    def getK(self):
        # 指数分布的参数 λ
        lambda_value = 0.5
        # 生成一个指数分布的随机数
        random_number = np.random.exponential(1 / lambda_value)
        return random_number

    # Function: Updtade Position
    def update_position(self, c1):
        for i in range(0, self.pop):
            if (i <= self.pop / 2):  # 领导者比例
                for j in range(0, self.n_dim):
                    # c2和c3为区间[0, 1]内产生的随机数
                    c2 = np.random.random()
                    c3 = np.random.random()
                    if (c3 >= 0.5):  # c3 < 0.5
                        try:
                            """
                                Xij = Fj + c1 * ((bu - b1) * c2 + b1)
                                np.clip 是 NumPy 中用于将数组中的值限制在指定范围内的函数
                                np.clip(a, a_min, a_max, out=None)
                            """
                            self.X[i, j] = np.clip(
                                (self.gbest_x[0][j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])), self.lb[j],
                                self.ub[j])
                        except:
                            self.X[i, j] = np.clip(
                                (self.gbest_x[j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])), self.lb[j],
                                self.ub[j])
                    else:
                        """
                           Xij = Fj - c1 * ((bu - b1) * c2 + b1)
                        """
                        try:
                            self.X[i, j] = np.clip(
                                (self.gbest_x[0][j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])), self.lb[j],
                                self.ub[j])
                        except:
                            self.X[i, j] = np.clip(
                                (self.gbest_x[j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])), self.lb[j],
                                self.ub[j])
            else:  # 追随者比例
                for j in range(0, self.n_dim):
                    self.X[i, j] = np.clip(((self.X[i - 1, j] + self.X[i, j]) / 2), self.lb[j], self.ub[j])
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles

    # Function: Updtade Position
    def update_position_two(self, c1, t):
        for i in range(0, self.pop):
            if (i <= self.pop / 2):  # 领导者比例
                for j in range(0, self.n_dim):
                    # c2和c3为区间[0, 1]内产生的随机数
                    c2 = np.random.random()
                    c3 = np.random.random()
                    if (c3 >= 0.5):  # c3 < 0.5
                        try:
                            self.X[i, j] = np.clip(
                                self.getAlpha(t) * (
                                        self.gbest_x[0][j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
                        except:
                            self.X[i, j] = np.clip(
                                self.getAlpha(t) * (
                                        self.gbest_x[j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
                    else:
                        try:
                            self.X[i, j] = np.clip(
                                self.getAlpha(t) * (
                                        self.gbest_x[0][j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
                        except:
                            self.X[i, j] = np.clip(
                                self.getAlpha(t) * (
                                        self.gbest_x[j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
            else:  # 追随者比例
                for j in range(0, self.n_dim):
                    self.X[i, j] = np.clip(((self.X[i - 1, j] + self.X[i, j]) / 2), self.lb[j], self.ub[j])
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles

    def update_position_three(self, c1):
        for i in range(0, self.pop):
            if (i <= self.pop / 2):  # 领导者比例
                for j in range(0, self.n_dim):
                    # c2和c3为区间[0, 1]内产生的随机数
                    c2 = np.random.random()
                    c3 = np.random.random()
                    if (c3 >= 0.5):  # c3 < 0.5
                        try:
                            """
                                Xij = Fj + c1 * ((bu - b1) * c2 + b1)
                                np.clip 是 NumPy 中用于将数组中的值限制在指定范围内的函数
                                np.clip(a, a_min, a_max, out=None)
                            """
                            self.X[i, j] = np.clip(
                                (self.gbest_x[0][j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])), self.lb[j],
                                self.ub[j])
                        except:
                            self.X[i, j] = np.clip(
                                (self.gbest_x[j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])), self.lb[j],
                                self.ub[j])
                    else:
                        """
                           Xij = Fj - c1 * ((bu - b1) * c2 + b1)
                        """
                        try:
                            self.X[i, j] = np.clip(
                                (self.gbest_x[0][j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])), self.lb[j],
                                self.ub[j])
                        except:
                            self.X[i, j] = np.clip(
                                (self.gbest_x[j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])), self.lb[j],
                                self.ub[j])
            else:  # 追随者比例
                for j in range(0, self.n_dim):
                    k = self.getK()
                    if i >= 2:
                        if self.func(self.X[i, j]) > self.func(self.X[i - 1, j]):
                            self.X[i, j] = np.clip(((self.X[i - 1, j] + k * self.X[i, j]) / 2), self.lb[j], self.ub[j])
                        elif self.func(self.X[i, j]) < self.func(self.X[i - 1, j]):
                            self.X[i, j] = np.clip(((k * self.X[i - 1, j] + self.X[i, j]) / 2), self.lb[j], self.ub[j])
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles

    def update_position_four(self, c1, t):
        for i in range(0, self.pop):
            if (i <= self.pop / 2):  # 领导者比例
                for j in range(0, self.n_dim):
                    # c2和c3为区间[0, 1]内产生的随机数
                    c2 = np.random.random()
                    c3 = np.random.random()
                    if (c3 >= 0.5):  # c3 < 0.5
                        try:
                            self.X[i, j] = np.clip(
                                self.getAlpha(t) * (
                                        self.gbest_x[0][j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
                        except:
                            self.X[i, j] = np.clip(
                                self.getAlpha(t) * (
                                        self.gbest_x[j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
                    else:
                        try:
                            self.X[i, j] = np.clip(
                                self.getAlpha(t) * (
                                        self.gbest_x[0][j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
                        except:
                            self.X[i, j] = np.clip(
                                self.getAlpha(t) * (
                                        self.gbest_x[j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
            else:  # 追随者比例
                for j in range(0, self.n_dim):
                    k = self.getK()
                    if i >= 2:
                        if self.func(self.X[i, j]) > self.func(self.X[i - 1, j]):
                            self.X[i, j] = np.clip(((self.X[i - 1, j] + k * self.X[i, j]) / 2), self.lb[j], self.ub[j])
                        elif self.func(self.X[i, j]) < self.func(self.X[i - 1, j]):
                            self.X[i, j] = np.clip(((k * self.X[i - 1, j] + self.X[i, j]) / 2), self.lb[j], self.ub[j])
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles

    # 差分策略
    def update_position_five(self, c1, t):
        for i in range(0, self.pop):
            if (i <= self.pop / 2):  # 领导者比例
                for j in range(0, self.n_dim):
                    # c2和c3为区间[0, 1]内产生的随机数
                    c2 = np.random.random()
                    c3 = np.random.random()
                    if (c3 >= 0.5):  # c3 < 0.5
                        try:
                            self.X[i, j] = np.clip(
                                self.getAlpha(t) * (
                                        self.gbest_x[0][j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
                        except:
                            self.X[i, j] = np.clip(
                                self.getAlpha(t) * (
                                        self.gbest_x[j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
                    else:
                        try:
                            self.X[i, j] = np.clip(
                                self.getAlpha(t) * (
                                        self.gbest_x[0][j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
                        except:
                            self.X[i, j] = np.clip(
                                self.getAlpha(t) * (
                                        self.gbest_x[j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
            else:  # 追随者比例
                for j in range(0, self.n_dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    if i >= 2 and t + 1 <= self.max_iter:
                        now = self.X[i, j] + r1 * (self.X[i - 1, j] - self.X[i, j]) + r2 * (
                                self.X[i - 2, j] - self.X[i, j])
                        self.getNextItemX(t, i, j, now)
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles

    # 差分策略
    def update_position_six(self, c1, t):
        for i in range(0, self.pop):
            if (i <= self.pop / 2):  # 领导者比例
                for j in range(0, self.n_dim):
                    # c2和c3为区间[0, 1]内产生的随机数
                    c2 = np.random.random()
                    c3 = np.random.random()
                    if (c3 >= 0.5):  # c3 < 0.5
                        try:
                            self.X[i, j] = np.clip(
                                (
                                        self.gbest_x[0][j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
                        except:
                            self.X[i, j] = np.clip(
                                (
                                        self.gbest_x[j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
                    else:
                        try:
                            self.X[i, j] = np.clip(
                                (
                                        self.gbest_x[0][j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
                        except:
                            self.X[i, j] = np.clip(
                                (
                                        self.gbest_x[j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])),
                                self.lb[j],
                                self.ub[j])
            else:  # 追随者比例
                for j in range(0, self.n_dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    if i >= 2:
                        now = self.X[i, j] + r1 * (self.X[i - 1, j] - self.X[i, j]) + r2 * (
                                self.X[i - 2, j] - self.X[i, j])
                        self.getNextItemX(t, i, j, now)
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles

    def getNextItemX(self, t, last_i, last_j, now):
        for count in range(self.max_iter):
            if count == t + 1:
                for i in range(0, self.pop):
                    if (i <= self.pop / 2):  # 领导者比例
                        pass
                    else:  # 追随者比例
                        for j in range(0, self.n_dim):
                            if last_i == i and last_j == j:
                                self.X[i, j] = now
                                return

    def run(self):
        for i in range(self.max_iter):
            """
                c1为平衡开发与探索的系数，减函数 c1 = 2*exp(-(4t/T)**2)
            """
            c1 = 2 * math.exp(-(4 * ((i + 1) / self.max_iter)) ** 2)
            self.update_position(c1)
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    def run2(self):
        for i in range(self.max_iter):
            """
                c1为平衡开发与探索的系数，减函数 c1 = 2*exp(-(4t/T)**2)
            """
            c1 = 2 * math.exp(-(4 * ((i + 1) / self.max_iter)) ** 2)
            self.update_position_two(c1, i)
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    def run3(self):
        for i in range(self.max_iter):
            """
                c1为平衡开发与探索的系数，减函数 c1 = 2*exp(-(4t/T)**2)
            """
            c1 = 2 * math.exp(-(4 * ((i + 1) / self.max_iter)) ** 2)
            self.update_position_three(c1)
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    def run4(self):
        for i in range(self.max_iter):
            """
                c1为平衡开发与探索的系数，减函数 c1 = 2*exp(-(4t/T)**2)
            """
            c1 = 2 * math.exp(-(4 * ((i + 1) / self.max_iter)) ** 2)
            self.update_position_four(c1, i)
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    def run5(self):
        for i in range(self.max_iter):
            """
                c1为平衡开发与探索的系数，减函数 c1 = 2*exp(-(4t/T)**2)
            """
            c1 = 2 * math.exp(-(4 * ((i + 1) / self.max_iter)) ** 2)
            self.update_position_five(c1, i)
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    def run6(self):
        for i in range(self.max_iter):
            """
                c1为平衡开发与探索的系数，减函数 c1 = 2*exp(-(4t/T)**2)
            """
            c1 = 2 * math.exp(-(4 * ((i + 1) / self.max_iter)) ** 2)
            self.update_position_six(c1, i)
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y


if __name__ == '__main__':
    # np.random.seed(1)
    n_dim = 30
    pop_size = 50
    max_iter = 500
    lb = [-100 for i in range(n_dim)]
    ub = [100 for i in range(n_dim)]
    # demo_func = test_fun_new.F2
    # demo_func = test_fun_new.F18
    # demo_func = test_fun_new.F10
    # demo_func = test_fun_new.F9
    """
        针对于 多峰函数优化较好
        test_fun_new F4(非多峰) F8 F9 F10 F11(一般) F12
    """
    demo_func = test_fun_new.F8
    ssa1 = salp_swarm_algorithm(pop_size=pop_size, n_dim=n_dim, max_iter=max_iter, lb=lb, ub=ub, func=demo_func)
    ssa2 = salp_swarm_algorithm(pop_size=pop_size, n_dim=n_dim, max_iter=max_iter, lb=lb, ub=ub, func=demo_func)
    # ssa3 = salp_swarm_algorithm(pop_size=pop_size, n_dim=n_dim, max_iter=max_iter, lb=lb, ub=ub, func=demo_func)
    # ssa4 = salp_swarm_algorithm(pop_size=pop_size, n_dim=n_dim, max_iter=max_iter, lb=lb, ub=ub, func=demo_func)
    ssa5 = salp_swarm_algorithm(pop_size=pop_size, n_dim=n_dim, max_iter=max_iter, lb=lb, ub=ub, func=demo_func)
    ssa6 = salp_swarm_algorithm(pop_size=pop_size, n_dim=n_dim, max_iter=max_iter, lb=lb, ub=ub, func=demo_func)
    """
        未改进之前 ssa1.run()
        改进之后 ssa2.run2()  添加了衰减因子
        改进之后 ssa5.run5() 差分策略 + 衰减因子
        改进之后 ssa6.run6() 差分策略
    """
    ssa1.run()
    ssa2.run2()
    # ssa3.run3()
    # ssa4.run4()
    ssa5.run5()
    ssa6.run6()
    # print('best_x is ', ssa1.gbest_x, 'best_y is', ssa1.gbest_y)
    # print(f'{demo_func(ssa1.gbest_x)}\t{ssa1.gbest_x}')
    plt.plot(ssa1.gbest_y_hist, linestyle='-', color='b', label="original_SSA")
    plt.plot(ssa2.gbest_y_hist, linestyle='-', color='r', label="new_SSA1")
    # plt.plot(ssa3.gbest_y_hist, linestyle='-', color='g', label="new_SSA2")
    # plt.plot(ssa4.gbest_y_hist, linestyle='-', color='y', label="new_SSA3")
    plt.plot(ssa5.gbest_y_hist, linestyle='-', color='m', label="new_SSA4")
    plt.plot(ssa6.gbest_y_hist, linestyle='-', color='g', label="new_SSA5")
    plt.grid(True)
    plt.legend()
    plt.show()
