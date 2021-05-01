import time

from concurrent.futures import ThreadPoolExecutor


class MFSumDelayFunction(object):
    """ This just creates a wrapper to call the function by appropriately creating bounds
        and querying appropriately. """

    def __init__(self, mf_funcs, delays):
        """ Constructor.
              mf_funcs: list of MFFunction objects
              delays: delay of each MFFunction
        """

        super(MFSumDelayFunction, self).__init__()
        self.mf_funcs = mf_funcs
        self.delays = delays
        self.nr_executor = len(mf_funcs)
        self.executor = ThreadPoolExecutor(max_workers=self.nr_executor)

    def eval_single(self, mf_compose):
        mf_fun = mf_compose[0]
        mf_fun_Z = mf_compose[1]
        mf_fun_X = mf_compose[2]
        delay = mf_compose[3]
        time.sleep(delay)
        return mf_fun.eval_single(mf_fun_Z, mf_fun_X)

    def eval_sum(self, Z, X):
        # for (mf_fun, delay) in zip(self.mf_funcs, self.delays):
        #     self.executor.submit(mf_fun.eval_single, Z, X)
        print(Z * self.nr_executor)
        results = self.executor.map(self.eval_single, zip(self.mf_funcs, [Z] * self.nr_executor, [X] * self.nr_executor, self.delays))
        print(list(results))


