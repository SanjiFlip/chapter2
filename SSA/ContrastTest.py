# -*- coding: UTF-8 -*-
"""
@Project ：chapter2 
@File    ：ContrastTest.py
@Author  ：wangyi
@Date    ：2023/11/8 11:24 
"""
import test_function
from matplotlib import pyplot as plt
import copy
from SSA.SSA_new import SSA
from SSA.Salp_Swarm_Algorithm import salp_swarm_algorithm

# 全局参数
n_dim = 30
lb = [-100 for i in range(n_dim)]
ub = [100 for i in range(n_dim)]
pop_size = 100
max_iter = 100


def SSA_Old(demo_func):
    ssa = salp_swarm_algorithm(pop_size=pop_size, n_dim=n_dim, max_iter=max_iter, lb=lb, ub=ub, func=demo_func)
    ssa.run()
    print("SSA__OLD")
    print('best_x is ', ssa.gbest_x, 'best_y is', ssa.gbest_y)
    print(f'{demo_func(ssa.gbest_x)}\t{ssa.gbest_x}')
    print("SSA__OLD")
    return ssa


def SSA_New(demo_func):
    ssa = SSA(demo_func, n_dim=n_dim, pop_size=pop_size, max_iter=max_iter, lb=lb, ub=ub)
    ssa.run()
    print("SSA__NEW")
    print('best_x is ', ssa.gbest_x, 'best_y is', ssa.gbest_y)
    # print(f'{demo_func(ssa.gbest_x)}\t{ssa.gbest_x}')
    print("SSA__NEW")
    return ssa


if __name__ == '__main__':
    # 公共测试函数 fu1-fu7,fm1-fm5,f21-f27
    demo_func = test_function.fu5
    ssa_old = SSA_Old(demo_func)
    ssa_new = SSA_New(demo_func)

    plt.plot(ssa_old.gbest_y_hist, "r-")
    plt.plot(ssa_new.gbest_y_hist, "b-")
    plt.show()
