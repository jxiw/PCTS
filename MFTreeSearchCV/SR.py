import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from concurrent.futures import wait, ALL_COMPLETED

executor = ThreadPoolExecutor(max_workers=40)

def get_value(mfobject, cell, fidel):
    # get the middle point of the selected cell
    x = np.array([(s[0] + s[1]) / 2.0 for s in list(cell)])
    return mfobject.eval_at_fidel_single_point_normalised([fidel], x)

def callback_fun(result_future, values, invoke_node, invoke_time):
    values[invoke_node][invoke_time] = result_future.result()

def Subroutine(mfobject, delta, alpha, budget):
    d = mfobject.domain_dim
    cell = tuple([(0, 1)] * d)
    active_cells = []
    active_cells.append(cell)
    l = 0
    start_time = time.time()
    current_time = time.time()
    best_times = []
    best_values = []
    futures = []
    while (current_time - start_time) <= budget:
        M_l = -100
        f_mean = []
        # print(np.power(1.0 / 2.0, l * (d + 1)))
        delta_l = delta * np.power(1.0 / 2.0, l * (d + 1))
        b_l_alpha = np.power(np.power(1.0 / 2.0, l), alpha)
        print("level_l", l)
        print("delta_l", delta_l)
        print("b_l_alpha", b_l_alpha)
        t_l_alpha = int(0.5 * np.log(1.0 / delta_l) * np.power(1.0 / b_l_alpha, 2))
        print("t_l_alpha:", t_l_alpha)
        nr_active_cells = len(active_cells)
        values = np.zeros((nr_active_cells, t_l_alpha))
        for i in range(nr_active_cells):
            active_cell = active_cells[i]
            # cell_values = np.zeros(t_l_alpha)
            for j in range(0, t_l_alpha):
                # current_time = time.time()
                # if (current_time - start_time) >= budget:
                #     print("times:", best_times)
                #     print("values:", best_values)
                #     return M_l
                future = executor.submit(get_value, mfobject, active_cell, 1)
                future.add_done_callback(partial(callback_fun, values=values, invoke_node=i, invoke_time=j))
                futures.append(future)
        # wait until all results are returned
        wait(futures, timeout=(budget - (current_time - start_time)), return_when=ALL_COMPLETED)
        print("wait complete")
        for i in range(nr_active_cells):
            cell_values = values[i]
            mean_value = np.mean(cell_values)
            f_mean.append(mean_value)
            M_l = max(M_l, mean_value)
        active_cells_next_level = []
        for i in range(nr_active_cells):
            active_cell = active_cells[i]
            cell_mean = f_mean[i]
            B_l_alpha = 2 * (np.sqrt(np.log(1 / delta_l) / (2 * t_l_alpha)) + b_l_alpha)
            if M_l - cell_mean <= B_l_alpha:
                span = [abs(active_cell[i][1] - active_cell[i][0]) for i in range(len(active_cell))]
                dimension = np.argmax(span)
                children_cell = np.linspace(active_cell[dimension][0], active_cell[dimension][1], 3)
                for k in range(len(children_cell) - 1):
                    cell = []
                    for j in range(len(active_cell)):
                        # if j is not equal to selected dimension
                        if j != dimension:
                            # do not change the other cells
                            cell = cell + [active_cell[j]]
                        else:
                            cell = cell + [(children_cell[k], children_cell[k + 1])]
                    cell = tuple(cell)
                    active_cells_next_level.append(cell)
        l += 1
        current_time = time.time()
        print("M_l", M_l)
        print("time", (current_time - start_time))
        active_cells = active_cells_next_level
        best_times.append((current_time - start_time))
        best_values.append(M_l)
    print("times:", best_times)
    print("values:", best_values)
    return M_l
    # print("max value:", M_l)

from experiments import synthetic_functions
from mf.mf_func import get_noisy_mfof_from_mfof
from mf.mf_sum_delay_func import MFSumDelayFunction

import sys

if __name__ == '__main__':

    test_function = 'Hartmann6'
        # sys.argv[1]
    noisy_factor = 1
    if test_function == 'Hartmann3':
        mfof = synthetic_functions.get_mf_hartmann_as_mfof(1, 3)
        noise_var = noisy_factor * 0.01
        sigma = np.sqrt(noise_var)
    elif test_function == 'Hartmann6':
        mfof = synthetic_functions.get_mf_hartmann_as_mfof(1, 6)
        noise_var = noisy_factor * 0.05
        sigma = np.sqrt(noise_var)
    elif test_function == 'CurrinExp':
        mfof = synthetic_functions.get_mf_currin_exp_as_mfof()
        noise_var = noisy_factor * 0.5
        sigma = np.sqrt(noise_var)
    elif test_function == 'Branin':
        mfof = synthetic_functions.get_mf_branin_as_mfof(1)
        noise_var = noisy_factor * 0.05
        sigma = np.sqrt(noise_var)
    elif test_function == 'Borehole':
        mfof = synthetic_functions.get_mf_borehole_as_mfof()
        noise_var = noisy_factor * 5
        sigma = np.sqrt(noise_var)

    times = [600]
    mfobject = get_noisy_mfof_from_mfof(mfof, noise_var)

    delay_functions = [mfobject]
    delay_times = [4]
    delay_mfobject = MFSumDelayFunction(delay_functions, delay_times)

    delta = 0.1
    alpha = 2
    for budget in times:
        v = Subroutine(delay_mfobject, delta, alpha, budget)
        print("budget:", budget, "value:", v)



