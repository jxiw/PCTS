from MFTreeSearchCV.MFHOO import *

from experiments import synthetic_functions
from mf.mf_func import get_noisy_mfof_from_mfof
from mf.mf_sum_delay_func import MFSumDelayFunction

NUM_EXP = 1

# mfobject is the function to optimize
# nu is the initial nu in HOO
# rho is the initial rho in HOO
# sigma is the sigma for HOO-UCB1
# bound is the bound for HOO-UCBV
# constant is the constant for HOO-UCBV
# policy is the exploration policy, UCBV or UCB1
# delay type is the HOO or DHOO
def run_one_experiment(mfobject, nu, rho, times, sigma, C, t0, hoo_config):
    for budget in times:
        # total budget
        print("total budget:", budget)
        MP = MFPOO(mfobject=mfobject, nu_max=nu, rho_max=rho, total_budget=budget,
                   sigma=sigma, C=C, mult=0.5, hoo_config=hoo_config,
                   tol=1e-3, Randomize=False, Auto=True, unit_cost=None)
        MP.run_all_MFHOO()


if __name__ == '__main__':

    policy = sys.argv[1]
    test_function = sys.argv[2]
    delay_type = sys.argv[3]
    max_hoo = sys.argv[4]
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
    multi_delay_functions = MFSumDelayFunction(delay_functions, delay_times)
    hoo_config = {"policy": policy, "delay_type": delay_type, "max_hoo": int(max_hoo)}
    if hoo_config["policy"] == "UCBV":
        hoo_config["ucbv_bound"] = float(sys.argv[5])
        hoo_config["ucbv_const"] = float(sys.argv[6])

    nu = 1.0
    rho = 0.95
    C = 0.1
    t0 = mfobject.opt_fidel_cost
    print("test function", test_function)
    print("opt_fidel_cost:", t0)

    for i in range(0, NUM_EXP):
        print 'Running Experiment' + str(i + 1) + ': '
        run_one_experiment(multi_delay_functions, nu, rho, times, sigma, C, t0, hoo_config)
