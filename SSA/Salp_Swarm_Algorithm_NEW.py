# -*- coding: UTF-8 -*-
"""
@Project ：chapter2 
@File    ：Salp_Swarm_Algorithm_NEW.py
@Author  ：wangyi
@Date    ：2023/11/8 11:10 
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import test_function


class salp_swarm_algorithm():
    # 默认会填充一些参数
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
            黄金正弦算子变异策略 所需参数
        """
        # 黄金分割比
        t = (np.sqrt(5) - 1) / 2
        self.a = -np.pi + 2 * np.pi * t
        self.b = np.pi - 2 * np.pi * t
        self.R1 = np.random.uniform(0, 2 * np.pi)

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

    # 获取精英种群数量_new
    def getM(self, t):

        return math.ceil((t / self.pop) * (self.pop * 2))

    # 获取反向学习机制阈值 a
    def getAlpha(self, t):
        return 2 - 2 * t / self.max_iter

    # 该值确定该个体位置是否接近食物源 差值
    def poor(self, xj, fj):
        dj = abs(xj - fj)
        return dj

    # 是否开始反向学习
    def reverseLearning(self, t, xj, fj, j):
        a = self.getAlpha(t)
        dj = self.poor(xj, fj)
        if dj > a:
            xj_new = self.ub[j] + self.lb[j] - xj
        pass

    # Function: Updtade Position
    def update_position(self, c1):
        """
            更新位置
        """
        for i in range(0, self.pop):
            if (i <= self.pop / 2):  # 领导者比例
                for j in range(0, self.n_dim):
                    # c2和c3为区间[0, 1]内产生的随机数
                    c2 = np.random.random()
                    c3 = np.random.random()
                    # c2 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
                    # c3 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
                    """
                        领导者更新
                    """
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
                        try:
                            """
                                Xij = Fj - c1 * ((bu - b1) * c2 + b1)
                            """
                            self.X[i, j] = np.clip(
                                (self.gbest_x[0][j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])), self.lb[j],
                                self.ub[j])
                        except:
                            self.X[i, j] = np.clip(
                                (self.gbest_x[j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])), self.lb[j],
                                self.ub[j])
            else:  # 追随者比例
                """
                    更新追随者的位置
                """
                for j in range(0, self.n_dim):
                    self.X[i, j] = np.clip(((self.X[i - 1, j] + self.X[i, j]) / 2), self.lb[j], self.ub[j])
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles

    def update_position_new(self):
        pass

    def run(self):
        for i in range(self.max_iter):
            """
                c1为平衡开发与探索的系数，减函数 c1 = 2*exp(-(4t/T)**2)
            """
            c1 = 2 * np.exp(-(4 * ((i + 1) / self.max_iter)) ** 2)
            self.update_position(c1)
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y


if __name__ == '__main__':
    # np.random.seed(1)
    pop_size = 100
    max_iter = 150
    n_dim = 30
    lb = [-100 for i in range(n_dim)]
    ub = [100 for i in range(n_dim)]
    demo_func = test_function.fu1
    ssa = salp_swarm_algorithm(pop_size=pop_size, n_dim=n_dim, max_iter=max_iter, lb=lb, ub=ub, func=demo_func)
    ssa.run()
    print('best_x is ', ssa.gbest_x, 'best_y is', ssa.gbest_y)
    print(f'{demo_func(ssa.gbest_x)}\t{ssa.gbest_x}')
    plt.plot(ssa.gbest_y_hist, linestyle='--', color='b', label="old_SSA")
    plt.grid(True)
    plt.legend()
    plt.show()
