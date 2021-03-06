"""
  Harness to run experiments and save results.

"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=relative-import
# pylint: disable=abstract-class-not-used

from argparse import Namespace
import random
from time import time
import numpy as np
from scipy.io import savemat as sio_savemat
# Local imports
from reporters import get_reporter


class BasicExperimenter(object):
  """ Base class for running experiments. """

  def __init__(self, experiment_name, num_experiments, save_file_name,
               save_file_extension='', reporter='default', random_seed_val='time'):
    """ Constructor.
        random_seed_val: If None we will not change the random seed. If it is
          'time' we will set it to a time based value. If it is an int, we will set
          the seed to that value.
    """
    self.experiment_name = experiment_name
    self.num_experiments = num_experiments
    self.save_file_name = save_file_name
    self.save_file_extension = (save_file_extension if save_file_extension else
                                self.save_file_name.split('.')[-1])
    if not self.save_file_name.endswith('.' + self.save_file_extension):
      self.save_file_name += '.' + self.save_file_extension
    self.reporter = get_reporter(reporter)
    self.to_be_saved = Namespace(experiment_name=self.experiment_name)
    # We will need these going forward.
    self.experiment_iter = 0
    # Set the random seed
    if random_seed_val is not None:
      if random_seed_val == 'time':
        random_seed_val = int(time() * 100) % 100000
      self.reporter.writeln('Setting random seed to %d.'%(random_seed_val))
      np.random.seed(random_seed_val)
      random.seed(random_seed_val)

  def save_results(self):
    """ Saves results in save_file_name. """
    self.reporter.write('Saving results (exp-iter:%d) to %s ...  '%(self.experiment_iter,
                         self.save_file_name))
    try:
      if self.save_file_extension == 'mat':
        sio_savemat(self.save_file_name, mdict=vars(self.to_be_saved))
      else:
        raise NotImplementedError('Only implemented saving mat files so far.')
      save_successful = True
    except IOError:
      save_successful = False
    # Report saving status
    if save_successful:
      self.reporter.writeln('successful.')
    else:
      self.reporter.writeln('unsuccessful!!')

  def terminate_now(self):
    """ Returns true if we should terminate now. Can be overridden in a child class. """
    return self.experiment_iter >= self.num_experiments

  def run_experiments(self):
    """ This runs the experiments. """
    self.reporter.writeln(self.get_experiment_header())
    while not self.terminate_now():
      # Prelims
      self.experiment_iter += 1
      iter_header = ('\nExperiment %d/%d:: '%(self.experiment_iter, self.num_experiments)
                     + self.get_iteration_header())
      iter_header += '\n' + '=' * len(iter_header)
      self.reporter.write(iter_header)
      # R experiment iteration.
      self.run_experiment_iteration()
      # Save results
      self.save_results()
    # Wrap up the experiments
    self.wrapup_experiments()

  def get_experiment_header(self):
    """ Something to pring before running all the experiments. Can be overridden in a
        child class."""
    # pylint: disable=no-self-use
    return ''

  def get_iteration_header(self):
    """ A header for the particular iteration. """
    # pylint: disable=no-self-use
    return ''

  def run_experiment_iteration(self):
    """ Implements the current iteration of the exeperiment. """
    raise NotImplementedError('Implement this in a child class.')

  def wrapup_experiments(self):
    """ Any code to wrap up the experiments goes here. """
    # pylint: disable=no-self-use
    pass

