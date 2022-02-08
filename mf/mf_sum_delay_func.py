import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor


class MFSumDelayFunction(object):
    """ This is for synthetic function under delay feedback """

    def __init__(self, mf_funcs, delays, is_stochastic_delay=False, seed_stochastic_delay=42, nr_executor=30):
        """ Constructor.
              mf_funcs: list of MFFunction objects
              delays: delay of each MFFunction
        """

        super(MFSumDelayFunction, self).__init__()
        self.mf_funcs = mf_funcs
        self.delays = delays
        self.max_avg_delay = max(self.delays)
        # for stochastic delay
        np.random.seed(seed_stochastic_delay)
        self.is_stochastic_delay = is_stochastic_delay
        self.stochastic_delays = [np.random.geometric(p=1 / float(delay), size=20000) for delay in delays]
        self.delay_cnt = 0
        # print("generated stochastic delays:", self.stochastic_delays)
        self.domain_dim = mf_funcs[0].domain_dim
        self.nr_executor = nr_executor
        self.opt_fidel = [mf_func.opt_fidel for mf_func in mf_funcs]
        self.executor = ThreadPoolExecutor(max_workers=self.nr_executor)

    def _eval_at_fidel_single_point(self, mf_compose):
        mf_fun = mf_compose[0]
        mf_fun_Z = mf_compose[1]
        mf_fun_X = mf_compose[2]
        delay = mf_compose[3]
        time.sleep(delay)
        return mf_fun.eval_at_fidel_single_point(mf_fun_Z, mf_fun_X)

    def _evaluate_at_fidel_single_point_normalised(self, mf_compose):
        mf_fun = mf_compose[0]
        mf_fun_Z = mf_compose[1]
        mf_fun_X = mf_compose[2]
        delay = mf_compose[3]
        time.sleep(delay)
        return mf_fun.eval_at_fidel_single_point_normalised(mf_fun_Z, mf_fun_X)

    def _eval_fidel_cost_single_point_normalised(self, mf_compose):
        mf_fun = mf_compose[0]
        mf_fun_Z = mf_compose[1]
        delay = mf_compose[2]
        time.sleep(delay)
        return mf_fun.eval_fidel_cost_single_point_normalised(mf_fun_Z)

    def _get_unnormalised_coords(self, mf_compose):
        mf_fun = mf_compose[0]
        mf_fun_Z = mf_compose[1]
        mf_fun_X = mf_compose[2]
        delay = mf_compose[3]
        time.sleep(delay)
        return mf_fun.get_unnormalised_coords(mf_fun_Z, mf_fun_X)

    def eval_at_fidel_single_point(self, Z, X):
        if self.is_stochastic_delay:
            delays = [delay[self.delay_cnt] for delay in self.stochastic_delays]
            self.delay_cnt += 1
        else:
            delays = self.delays
        # print("eval_at_fidel_single_point delays:", delays)
        results = self.executor.map(self._eval_at_fidel_single_point,
                                    zip(self.mf_funcs, [Z] * self.nr_executor, [X] * self.nr_executor, delays))
        # print(list(results))
        return sum(results)

    def eval_at_fidel_single_point_normalised(self, Z, X):
        if self.is_stochastic_delay:
            delays = [delay[self.delay_cnt] for delay in self.stochastic_delays]
            self.delay_cnt += 1
        else:
            delays = self.delays
        # print("eval_at_fidel_single_point_normalised delays:", delays)
        results = self.executor.map(self._evaluate_at_fidel_single_point_normalised,
                                    zip(self.mf_funcs, [Z] * self.nr_executor, [X] * self.nr_executor, delays))
        return sum(results)

    def eval_fidel_cost_single_point_normalised(self, Z):
        if self.is_stochastic_delay:
            delays = [delay[self.delay_cnt] for delay in self.stochastic_delays]
            self.delay_cnt += 1
        else:
            delays = self.delays
        # print("eval_fidel_cost_single_point_normalised delays:", delays)
        results = self.executor.map(self._eval_fidel_cost_single_point_normalised,
                                    zip(self.mf_funcs, [Z] * self.nr_executor, delays))
        return sum(results)

    def get_unnormalised_coords(self, Z, X):
        if self.is_stochastic_delay:
            delays = [delay[self.delay_time] for delay in self.stochastic_delays]
            self.delay_time += 1
        else:
            delays = self.delays
        results = self.executor.map(self._get_unnormalised_coords,
                                    zip(self.mf_funcs, [Z] * self.nr_executor, [X] * self.nr_executor, delays))
        print (results)
        return

    def eval_single_noiseless(self, Z, X):
        mf_fun_values = []
        for mf_fun in self.mf_funcs:
            mf_fun_values.append(mf_fun.eval_single_noiseless(Z, X))
        return sum(mf_fun_values)

    def eval_single_opt_fidel_noiseless(self, X):
        mf_fun_values = []
        for mf_fun in self.mf_funcs:
            mf_fun_values.append(mf_fun.eval_single_noiseless(mf_fun.opt_fidel, X))
        return sum(mf_fun_values)
