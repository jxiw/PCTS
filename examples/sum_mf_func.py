from mf.mf_func import get_noisy_mfof_from_mfof
import time

from MFTreeSearchCV.MFHOO import *

from mf.mf_func import get_noisy_mfof_from_mfof
import synthetic_functions
from mf.mf_sum_delay_func import MFSumDelayFunction

if __name__ == '__main__':

  mf_hartmn1 = synthetic_functions.get_mf_hartmann_as_mfof(2, 6)
  mf_hartmn2 = synthetic_functions.get_mf_hartmann_as_mfof(2, 6)
  mf_hartmn3 = synthetic_functions.get_mf_hartmann_as_mfof(2, 6)
  mf_hartmn4 = synthetic_functions.get_mf_hartmann_as_mfof(2, 6)
  mf_hartmn5 = synthetic_functions.get_mf_hartmann_as_mfof(2, 6)
  mf_hartmn6 = synthetic_functions.get_mf_hartmann_as_mfof(2, 6)

  # mf_exp = synthetic_functions.get_mf_currin_exp_as_mfof()
  # mf_branin = synthetic_functions.get_mf_branin_as_mfof(1)
  # mf_borehole = synthetic_functions.get_mf_borehole_as_mfof()

  hartmn1_object = get_noisy_mfof_from_mfof(mf_hartmn1, noise_var=0.01)
  hartmn2_object = get_noisy_mfof_from_mfof(mf_hartmn2, noise_var=0.05)
  hartmn3_object = get_noisy_mfof_from_mfof(mf_hartmn3, noise_var=0.1)
  hartmn4_object = get_noisy_mfof_from_mfof(mf_hartmn4, noise_var=0.5)
  hartmn5_object = get_noisy_mfof_from_mfof(mf_hartmn5, noise_var=1)
  hartmn6_object = get_noisy_mfof_from_mfof(mf_hartmn6, noise_var=5)

  # exp_object = get_noisy_mfof_from_mfof(mf_exp, noise_var=0.5)
  # branin_object = get_noisy_mfof_from_mfof(mf_branin, noise_var=0.05)
  # borehole_object = get_noisy_mfof_from_mfof(mf_borehole, noise_var=5)
  # funs = [hartmn1_object, hartmn2_object, exp_object, branin_object, borehole_object]

  funs = [hartmn1_object, hartmn2_object, hartmn3_object, hartmn4_object, hartmn5_object, hartmn6_object]
  delays = [10, 4, 9, 40, 10]
  sum_mf = MFSumDelayFunction(funs, delays)

  # print(mf_hartmn1.domain_dim)
  # print(mf_hartmn2.domain_dim)
  # print(mf_exp.domain_dim)
  # print(mf_branin.domain_dim)
  # print(mf_borehole.domain_dim)
  Z_dim = 2
  X_dim = 6
  Z = np.random.randint(0, 9, (Z_dim))
  X = np.random.randint(0, 9, (X_dim))
  sum_mf.eval_sum(Z, X)
