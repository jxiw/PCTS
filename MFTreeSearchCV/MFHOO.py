from __future__ import division
from __future__ import print_function

import os
import sys
from functools import partial

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

nu_mult = 1.0  # multiplier to the nu parameter

executor = ThreadPoolExecutor(max_workers=40)


def flip(p):
    return True if random.random() < p else False


class MF_node(object):
    def __init__(self, cell, value, fidel, upp_bound, height, dimension, num, hoo_config):
        '''This is a node of the MFTREE
        cell: tuple denoting the bounding boxes of the partition
        m_value: mean value of the observations in the cell and its children
        value: value in the cell
        fidelity: the last fidelity that the cell was queried with
        upp_bound: B_{i,t} in the paper
        t_bound: upper bound with the t dependent term
        height: height of the cell (sometimes can be referred to as depth in the tree)
        dimension: the dimension of the parent that was halved in order to obtain this cell
        num: number of queries inside this partition so far
        left,right,parent: pointers to left, right and parent
        hoo_config: hoo config
        '''
        # cell is the range of partition
        self.cell = cell
        # mean value of the observation
        self.m_value = value
        self.value = value
        self.fidelity = fidel
        self.upp_bound = upp_bound
        self.height = height
        self.dimension = dimension
        self.num = num
        self.t_bound = upp_bound
        # init delay type
        self.policy = hoo_config["policy"]
        self.delay_type = hoo_config["delay_type"]
        if self.policy == "UCBV":
            self.ucbv_bound = hoo_config["ucbv_bound"]
            self.ucbv_const = hoo_config["ucbv_const"]
        self.second_moment_value = value ** 2
        self.variance = 0
        self.left = None
        self.right = None
        self.parent = None

    def __cmp__(self, other):
        return cmp(other.t_bound, self.t_bound)

    def total_children(self):
        if self.left is None and self.right is None:
            return 1
        elif self.left is not None and self.right is None:
            return 1 + self.left.total_children()
        elif self.left is None and self.right is not None:
            return 1 + self.right.total_children()
        else:
            return 1 + self.left.total_children() + self.right.total_children()

    def max_height(self):
        if self.left is None and self.right is not None:
            return 1 + self.right.max_height()
        elif self.left is not None and self.right is None:
            return 1 + self.left.max_height()
        elif self.left is not None and self.right is not None:
            return 1 + max(self.left.max_height(), self.right.max_height())
        else:
            return 1

    def opt_value(self):
        if self.left is None and self.right is not None:
            return max(self.value, self.right.opt_value())
        elif self.left is not None and self.right is None:
            return max(self.value, self.left.opt_value())
        elif self.left is not None and self.right is not None:
            return max(self.value, self.left.opt_value(), self.right.opt_value())
        else:
            return self.value


def serializable_MF_node(node):
    "Recurse into tree to build a serializable object, make use of select_related"
    # print(str(node.cell))
    # print(node.dimension, node.m_value, node.t_bound)
    if node.left is None and node.right is None:
        return {'name': "%s,%.1f,%.1f" % (str(node.cell), node.m_value, node.t_bound),
                'size': '%d' % (node.value * 1000)}
    else:
        obj = {'name': "%s,%.1f,%.1f" % (str(node.cell), node.m_value, node.t_bound), 'children': []}
        if node.left is not None:
            obj['children'].append(serializable_MF_node(node.left))
        if node.right is not None:
            obj['children'].append(serializable_MF_node(node.right))
        return obj


def in_cell(node, parent):
    '''
    Check if 'node' is a subset of 'parent'
    node can either be a MF_node or just a tuple denoting its cell
    '''
    try:
        ncell = list(node.cell)
    except:
        ncell = list(node)
    pcell = list(parent.cell)
    flag = 0

    for i in range(len(ncell)):
        if ncell[i][0] >= pcell[i][0] and ncell[i][1] <= pcell[i][1]:
            flag = 0
        else:
            flag = 1
            break
    if flag == 0:
        return True
    else:
        return False


class MF_tree(object):
    '''
    MF_tree class that maintains the multi-fidelity tree
    nu: nu parameter in the paper
    rho: rho parameter in the paper
    sigma: noise variance, ususally a hyperparameter for the whole process
    C: parameter for the bias function as defined in the paper
    root: can initialize a root node, when this parameter is supplied by a MF_node object instance
    '''

    def __init__(self, nu, rho, sigma, C, root=None):
        self.nu = nu
        self.rho = rho
        self.sigma = sigma
        self.root = root
        self.C = C
        self.root = root
        self.mheight = 0
        self.maxi = float(-sys.maxsize - 1)
        self.current_best = root

    def insert_node(self, root, node):
        '''
        insert a node in the tree in the appropriate position
        '''
        if self.root is None:
            node.height = 0
            if self.mheight < node.height:
                self.mheight = node.height
            self.root = node
            self.root.parent = None
            return self.root
        if root is None:
            node.height = 0
            if self.mheight < node.height:
                self.mheight = node.height
            root = node
            root.parent = None
            return root
        if root.left is None and root.right is None:
            node.height = root.height + 1
            if self.mheight < node.height:
                self.mheight = node.height
            root.left = node
            root.left.parent = root
            return root.left
        elif root.left is not None:
            # recursively insert the node
            if in_cell(node, root.left):
                return self.insert_node(root.left, node)
            elif root.right is not None:
                # recursively insert the node
                if in_cell(node, root.right):
                    return self.insert_node(root.right, node)
            else:
                # insert the node to right
                node.height = root.height + 1
                if self.mheight < node.height:
                    self.mheight = node.height
                root.right = node
                root.right.parent = root
                return root.right

    def update_parents(self, node, val):
        '''
        update the upperbound and mean value of a parent node, once a new child is inserted in its child tree. This process proceeds recursively up the tree
        '''
        if node.parent is None:
            return
        else:
            # update the value of parent from bottom to up
            parent = node.parent
            # m_value is the mean reward
            parent.m_value = (parent.num * parent.m_value + val) / (1.0 + parent.num)
            # for UCB-V policy
            # update the second moment
            parent.second_moment_value = (parent.num * parent.second_moment_value + val ** 2) / (1.0 + parent.num)
            parent.variance = parent.second_moment_value - parent.m_value ** 2
            parent.num = parent.num + 1.0
            # get the up bound which also consider the exploration
            # why we ignore the exploration term here.
            parent.upp_bound = parent.m_value + 2 * ((self.rho) ** (parent.height)) * self.nu
            self.update_parents(parent, val)

    # update the bound score of the current node
    def update_tbounds(self, root, t):
        '''
        updating the tbounds of every node recursively
        '''
        if root is None:
            return
        # update the t-bound recursively.
        self.update_tbounds(root.left, t)
        self.update_tbounds(root.right, t)
        # update the bound of t
        # add the exploration term
        if root.policy == 'UCB1':
            root.t_bound = root.upp_bound + np.sqrt(2 * (self.sigma ** 2) * np.log(t) / root.num)
        elif root.policy == 'UCBV':
            # print("root.num:", root.num)
            # print("ucbv_const:", root.ucbv_const)
            # root.t_bound = root.upp_bound + np.sqrt(root.ucbv_const * root.variance * np.log(t) / root.num) + (
            #         root.ucbv_bound * np.log(t) / root.num)
            # print("ucbv_bound", root.ucbv_bound)
            # print("variance", root.variance)
            root.t_bound = root.upp_bound + np.sqrt(2 * root.variance * np.log(t) / root.num) + (
                    3 * root.ucbv_const * root.ucbv_bound * np.log(t) / root.num)
        else:
            raise Exception("policy is not found")
        maxi = None
        if root.left:
            maxi = root.left.t_bound
        if root.right:
            if maxi:
                if maxi < root.right.t_bound:
                    maxi = root.right.t_bound
            else:
                maxi = root.right.t_bound
        if maxi:
            root.t_bound = min(root.t_bound, maxi)

    def print_given_height(self, root, height):
        if root is None:
            return
        if root.height == height:
            print(root.cell, root.num, root.upp_bound, root.t_bound),
        elif root.height < height:
            if root.left:
                self.print_given_height(root.left, height)
            if root.right:
                self.print_given_height(root.right, height)
        else:
            return

    def levelorder_print(self):
        '''
        levelorder print
        '''
        for i in range(self.mheight + 1):
            self.print_given_height(self.root, i)
            print('\n')

    def search_cell(self, root, cell):
        '''
        check if a cell is present in the tree
        '''
        if root is None:
            return False, None, None
        if root.left is None and root.right is None:
            if root.cell == cell:
                return True, root, root.parent
            else:
                return False, None, root
        if root.left:
            if in_cell(cell, root.left):
                return self.search_cell(root.left, cell)
        if root.right:
            if in_cell(cell, root.right):
                return self.search_cell(root.right, cell)

    def get_next_node(self, root):
        '''
        getting the next node to be queried or broken, see the algorithm in the paper
        '''
        if root is None:
            print('Could not find next node. Check Tree.')
        if root.left is None and root.right is None:
            return root
        if root.left is None:
            return self.get_next_node(root.right)
        if root.right is None:
            return self.get_next_node(root.left)

        # # select the next node according to the softmax of t bound
        # prob = np.exp(root.left.t_bound) / (np.exp(root.left.t_bound) + np.exp(root.right.t_bound))
        # bit = flip(prob)
        # if bit:
        #     return self.get_next_node(root.left)
        # else:
        #     return self.get_next_node(root.right)

        # select next nod according to the t bound ...
        if root.left.t_bound > root.right.t_bound:
            return self.get_next_node(root.left)
        elif root.left.t_bound < root.right.t_bound:
            return self.get_next_node(root.right)
        else:
            bit = flip(0.5)
            if bit:
                return self.get_next_node(root.left)
            else:
                return self.get_next_node(root.right)

    def get_current_best(self, root, option=0):
        '''
        get current best cell from the tree
        '''
        if root.policy == "HOO":
            if root is None:
                return
            if root.right is None and root.left is None:
                val = root.m_value - self.nu * ((self.rho) ** (root.height))
                if self.maxi < val:
                    self.maxi = val
                    cell = list(root.cell)
                    self.current_best = np.array([(s[0] + s[1]) / 2.0 for s in cell])
                return
            if root.left:
                self.get_current_best(root.left)
            if root.right:
                self.get_current_best(root.right)
        else:
            if root is None:
                return
            if root.right is None and root.left is None:
                if option == 0:
                    val = root.m_value - self.nu * ((self.rho) ** (root.height))
                elif option == 1:
                    val = root.value
                elif option == 2:
                    val = root.m_value
                if self.maxi < val:
                    self.maxi = val
                    cell = list(root.cell)
                    self.current_best = np.array([(s[0] + s[1]) / 2.0 for s in cell])
                return
            if root.left:
                self.get_current_best(root.left, option)
            if root.right:
                self.get_current_best(root.right, option)


class MFHOO(object):
    '''
    MFHOO algorithm, given a fixed nu and rho
    mfobject: multi-fidelity noisy function object
    nu: nu parameter
    rho: rho parameter
    budget: total budget provided either in units or time in seconds
    sigma: noise parameter
    C: bias function parameter
    tol: default parameter to decide whether a new fidelity query is required for a cell
    Randomize: True implies that the leaf is split on a randomly chosen dimension, False means the scheme in DIRECT algorithm is used. We recommend using False.
    Auto: Select C automatically, which is recommended for real data experiments
    CAPITAL: 'Time' mean time in seconds is used as cost unit, while 'Actual' means unit cost used in synthetic experiments
    debug: If true then more messages are printed
    '''

    def __init__(self, mfobject, nu, rho, budget, sigma, C,
                 hoo_config, tol=1e-3, \
                 Randomize=False, Auto=False, value_dict={}, \
                 CAPITAL='Time', debug='True'):
        self.mfobject = mfobject
        self.nu = nu
        self.rho = rho
        self.budget = budget
        self.C = C
        self.t = 0
        self.sigma = sigma
        self.tol = tol
        self.Randomize = Randomize
        self.cost = 0
        self.cflag = False
        self.value_dict = value_dict
        self.CAPITAL = CAPITAL
        self.debug = debug
        self.hoo_config = hoo_config
        self.policy = hoo_config["policy"]
        self.delay_type = hoo_config["delay_type"]
        print("rho", self.rho)
        print("nu", self.nu)
        if Auto:
            z1 = 0.8
            z2 = 0.2
            d = self.mfobject.domain_dim
            x = np.array([0.5] * d)
            t1 = time.time()
            v1 = self.mfobject.eval_at_fidel_single_point_normalised([z1], x)
            v2 = self.mfobject.eval_at_fidel_single_point_normalised([z2], x)
            t2 = time.time()
            self.C = np.sqrt(2) * np.abs(v1 - v2) / np.abs(z1 - z2)
            self.nu = nu_mult * self.C
            if self.debug:
                print('Auto Init: ')
                print('C: ' + str(self.C))
                print('nu: ' + str(self.nu))
            c1 = self.mfobject.eval_fidel_cost_single_point_normalised([z1])
            c2 = self.mfobject.eval_fidel_cost_single_point_normalised([z2])
            self.cost = c1 + c2
            if self.CAPITAL == 'Time':
                self.cost = t2 - t1
        d = self.mfobject.domain_dim
        cell = tuple([(0, 1)] * d)
        height = 0
        dimension = 0
        root, cost = self.querie(cell, height, self.rho, self.nu, dimension, option=1)
        self.t = self.t + 1
        self.Tree = MF_tree(nu, rho, self.sigma, C, root)
        self.Tree.update_tbounds(self.Tree.root, self.t)
        self.cost = self.cost + cost
        # delay information
        self.node_under_evaluation = []
        # self.ts = 0
        # self.under_evaluate_ts = []
        # self.max_delay = 100

    # key function to change, need to max_delay the execution, get the value of the current point.
    def get_value(self, cell, fidel):
        '''cell: tuple'''
        # get the middle point of the selected cell
        x = np.array([(s[0] + s[1]) / 2.0 for s in list(cell)])
        # get random value between s[0] and s[1]
        # x = np.array([random.uniform(s[0], s[1]) for s in list(cell)])
        return self.mfobject.eval_at_fidel_single_point_normalised([fidel], x)

    def querie(self, cell, height, rho, nu, dimension, option=1):
        # get the diam of the current cell
        # nu and rho are for smoothness
        diam = nu * (rho ** height)
        # if option == 1:
        #     # z is the ferdelity
        #     z = min(max(1 - diam / self.C, self.tol), 1.0)
        # else:
        #     z = 1.0
        z = 1.0
        if cell in self.value_dict:
            # the selected cell is cached
            current = self.value_dict[cell]
            if abs(current.fidelity - z) <= self.tol:
                value = current.value
                cost = 0
                # print("value:", value)
            else:
                t1 = time.time()
                value = self.get_value(cell, z)
                t2 = time.time()
                if abs(value - current.value) > self.C * abs(current.fidelity - z):
                    self.cflag = True
                current.value = value
                current.m_value = value
                # set the fidelity
                current.fidelity = z
                self.value_dict[cell] = current
                cost = t2 - t1
        else:
            t1 = time.time()
            # get the value of the selected cell
            value = self.get_value(cell, z)
            t2 = time.time()
            bhi = 2 * diam + value
            # create the MF cache node
            self.value_dict[cell] = MF_node(cell, value, z, bhi, height, dimension, 1, self.hoo_config)
            cost = t2 - t1
            # print (cell)
            # print (value)

        bhi = 2 * diam + value
        # print('value:', value)
        current_object = MF_node(cell, value, z, bhi, height, dimension, 1, self.hoo_config)
        return current_object, cost

    # split children for the current node
    def split_children(self, current, rho, nu, option=1):
        # the child cells of parent cells
        pcell = list(current.cell)
        # get the range of the span
        span = [abs(pcell[i][1] - pcell[i][0]) for i in range(len(pcell))]
        if self.Randomize:
            dimension = np.random.choice(range(len(pcell)))
        else:
            dimension = np.argmax(span)
        dd = len(pcell)
        if dimension == current.dimension:
            dimension = (current.dimension - 1) % dd
        cost = 0
        h = current.height + 1
        # split the parent cell range into 2 sub-intervals
        # only for the selected interval
        l = np.linspace(pcell[dimension][0], pcell[dimension][1], 3)
        # we only have two children
        children = []
        for i in range(len(l) - 1):
            cell = []
            for j in range(len(pcell)):
                # if j is not equal to selected dimension
                if j != dimension:
                    # do not change the other cells
                    cell = cell + [pcell[j]]
                else:
                    cell = cell + [(l[i], l[i + 1])]
            cell = tuple(cell)
            # invoke the query.
            # dimension is the selected interval
            child, c = self.querie(cell, h, rho, nu, dimension, option)
            children = children + [child]
            cost = cost + c

        return children, cost

    def take_DHOO_step(self):

        # define back function to avoid delay waiting
        def callback_fun(result_future, invoke_node, child_num):
            # print("callback function")
            children, cost = result_future.result()
            # after get two chilren
            self.cost = self.cost + cost
            if child_num == 0:
                lnode = self.Tree.insert_node(self.Tree.root, children)
                self.Tree.update_parents(lnode, lnode.value)
            else:
                rnode = self.Tree.insert_node(self.Tree.root, children)
                self.Tree.update_parents(rnode, rnode.value)
            # remove the useless information
            if invoke_node in self.node_under_evaluation:
                self.node_under_evaluation.remove(invoke_node)

        # wait(lambda: len(self.under_evaluate_ts) == 0 or self.under_evaluate_ts[0] + self.max_delay < self.ts)

        # run HOO at one iteration
        current = self.Tree.get_next_node(self.Tree.root)
        if current not in self.node_under_evaluation:
            # self.under_evaluate_ts.append(self.ts)
            self.node_under_evaluation.append(current)
            self.t = self.t + 2
            self.Tree.update_tbounds(self.Tree.root, self.t)
            # if current node is not in the pending evaluations.
            # then split the children from the current node

            # the child cells of parent cells
            pcell = list(current.cell)
            # get the range of the span
            span = [abs(pcell[i][1] - pcell[i][0]) for i in range(len(pcell))]
            if self.Randomize:
                dimension = np.random.choice(range(len(pcell)))
            else:
                dimension = np.argmax(span)
            dd = len(pcell)
            if dimension == current.dimension:
                dimension = (current.dimension - 1) % dd
            h = current.height + 1
            # split the parent cell range into 2 sub-intervals
            # only for the selected interval
            l = np.linspace(pcell[dimension][0], pcell[dimension][1], 3)
            # we only have two children
            for i in range(len(l) - 1):
                cell = []
                for j in range(len(pcell)):
                    # if j is not equal to selected dimension
                    if j != dimension:
                        # do not change the other cells
                        cell = cell + [pcell[j]]
                    else:
                        cell = cell + [(l[i], l[i + 1])]
                cell = tuple(cell)
                # invoke the query.
                # dimension is the selected interval
                future = executor.submit(self.querie, cell, h, self.rho, self.nu, dimension)
                future.add_done_callback(partial(callback_fun, invoke_node=current, child_num=i))
            return True
        else:
            return False

    def take_HOO_step(self):
        current = self.Tree.get_next_node(self.Tree.root)
        children, cost = self.split_children(current, self.rho, self.nu, 1)
        self.t = self.t + 2
        self.cost = self.cost + cost
        rnode = self.Tree.insert_node(self.Tree.root, children[0])
        self.Tree.update_parents(rnode, rnode.value)
        rnode = self.Tree.insert_node(self.Tree.root, children[1])
        self.Tree.update_parents(rnode, rnode.value)
        self.Tree.update_tbounds(self.Tree.root, self.t)
        return True

    def run(self):
        # total number of updates, total number of nodes, depth of the tree
        # update
        iter_num = 0
        print("delay_type:", self.delay_type)
        print("policy:", self.policy)
        time_durations = []
        noise_values = []
        best_points = []
        best_points1 = []
        best_points2 = []
        start_time = time.time()
        current_time = time.time()
        while (current_time - start_time) <= self.budget:
            if self.delay_type == "HOO":
                ret = self.take_HOO_step()
            elif self.delay_type == "DHOO":
                ret = self.take_DHOO_step()
            else:
                raise Exception('unknown method')
            current_time = time.time()
            if ret:
                iter_num += 1
                # print("iter_num:", iter_num)
                # print("current iterations:%d" % iter_num)
                # print("current HOO number of nodes:%d" % self.Tree.root.total_children())
                # print("HOO height:%d" % self.Tree.root.max_height())
                # print("duration:%d" % (current_time - start_time))
                current_opt = self.Tree.root.opt_value()
                # print("optimal value:%f " % current_opt)
                current_best_point = self.get_point()
                # print("best point:", current_best_point)
                best_points.append(current_best_point)
                time_durations.append((current_time - start_time))
                noise_values.append(current_opt)
                best_points1.append(self.get_point(option=1))
                best_points2.append(self.get_point(option=2))

        end_time = time.time()
        print(self.get_point())
        print("final iterations:%d" % iter_num)
        print("final HOO number of nodes:%d" % self.Tree.root.total_children())
        print("final HOO height:%d" % self.Tree.root.max_height())
        print("final duration:%d" % (end_time - start_time))
        print("final optimal value:%f " % self.Tree.root.opt_value())
        print("time durations:", time_durations)
        print("noisy values:", noise_values)
        true_values = []
        point_value_dict = dict()
        print("data points:")
        ## evaluate the true value without noisy of selected points at each timestamp
        for point in best_points:
            point_str_hash = str(point)
            if point_str_hash in point_value_dict:
                true_value = point_value_dict[point_str_hash]
                true_values.append(true_value)
            else:
                true_value = self.mfobject.eval_single_opt_fidel_noiseless(point)
                true_values.append(true_value)
                point_value_dict[point_str_hash] = true_value
        print("values:", true_values)
        return true_values[-1]
        # if self.delay_type == "DHOO":
        #     true_values1 = []
        #     for point in best_points1:
        #         point_str_hash = str(point)
        #         if point_str_hash in point_value_dict:
        #             true_value = point_value_dict[point_str_hash]
        #             true_values1.append(true_value)
        #         else:
        #             true_value = self.mfobject.eval_single_noiseless(point)
        #             true_values1.append(true_value)
        #             point_value_dict[point_str_hash] = true_value
        #     print("real values1:", true_values1)
        #     true_values2 = []
        #     for point in best_points2:
        #         point_str_hash = str(point)
        #         if point_str_hash in point_value_dict:
        #             true_value = point_value_dict[point_str_hash]
        #             true_values2.append(true_value)
        #         else:
        #             true_value = self.mfobject.eval_single_noiseless(point)
        #             true_values2.append(true_value)
        #             point_value_dict[point_str_hash] = true_value
        #     print("real values2:", true_values2)

    def get_point(self, option=0):
        self.Tree.get_current_best(self.Tree.root, option=option)
        return self.Tree.current_best


class MFPOO(object):
    '''
    MFPOO object that spawns multiple MFHOO instances
    '''

    def __init__(self, mfobject, nu_max, rho_max, total_budget,
                 sigma, C, mult, hoo_config,
                 tol=1e-3, Randomize=False, Auto=False, debug=True,
                 unit_cost=1.0):
        # we only support time as cost unit
        self.mfobject = mfobject
        self.nu_max = nu_max
        self.rho_max = rho_max
        # total budget is the real time
        self.total_budget = total_budget
        self.C = C
        self.t = 0
        self.sigma = sigma
        self.tol = tol
        self.Randomize = Randomize
        self.cost = 0
        self.value_dict = {}
        self.MH_arr = []
        self.debug = debug
        # default CAPITAL is time
        self.hoo_config = hoo_config
        if Auto:
            if unit_cost is None:
                z1 = 1.0
                if self.debug:
                    print('Setting unit cost automatically as None was supplied')
            else:
                z1 = 0.8
            z2 = 0.2
            d = self.mfobject.domain_dim
            x = np.array([0.5] * d)
            t1 = time.time()
            v1 = self.mfobject.eval_at_fidel_single_point_normalised([z1], x)
            t3 = time.time()
            v2 = self.mfobject.eval_at_fidel_single_point_normalised([z2], x)
            t2 = time.time()
            if self.C is None:
                self.C = np.sqrt(2) * np.abs(v1 - v2) / np.abs(z1 - z2)
            self.nu_max = nu_mult * self.C
            if unit_cost is None:
                unit_cost = t3 - t1
                # if self.debug:
                print('Unit Cost: ', unit_cost)
            if self.debug:
                print('Auto Init: ')
                print('C: ' + str(self.C))
                print('nu: ' + str(self.nu_max))
            self.total_budget = self.total_budget - (t2 - t1)
            if self.debug:
                print('Budget Remaining: ' + str(self.total_budget))

        self.unit_cost = unit_cost
        n = max(self.total_budget / self.unit_cost, 1)
        Dm = int(np.log(2.0) / np.log(1 / self.rho_max))
        nHOO = int(mult * Dm * np.log(n / np.log(n + 1)))
        if hoo_config['max_hoo'] < 0:
            self.nHOO = max(1, int(min(max(1, nHOO), n / 2 + 1)))
        else:
            self.nHOO = min(max(1, int(min(max(1, nHOO), n / 2 + 1))), hoo_config['max_hoo'])
        self.budget = (self.total_budget - self.nHOO * self.unit_cost) / float(self.nHOO)
        if self.debug:
            print('Number of MFHOO Instances: ' + str(self.nHOO))
            print('Budget per MFHOO Instance:' + str(self.budget))

    def run_all_MFHOO(self):
        nu = self.nu_max
        opt_val = -100000
        print("number of HOO:", self.nHOO)
        for i in range(self.nHOO):
            # run HOO with different rho and nu.
            rho = self.rho_max ** (float(self.nHOO) / (self.nHOO - i))
            MH = MFHOO(mfobject=self.mfobject, nu=nu, rho=rho, budget=self.budget, sigma=self.sigma, C=self.C,
                       hoo_config=self.hoo_config, tol=1e-3, Randomize=False, Auto=False, value_dict=self.value_dict,
                       debug=self.debug)
            print('Running SOO number: ' + str(i + 1) + ' rho: ' + str(rho) + ' nu: ' + str(nu))
            opt_val = max(opt_val, MH.run())
            print('Done!')
            self.cost = self.cost + MH.cost
            if MH.cflag:
                self.C = 1.4 * self.C
                nu = nu_mult * self.C
                self.nu_max = nu_mult * self.C
                if self.debug:
                    print('Updating C')
                    print('C: ' + str(self.C))
                    print('nu_max: ' + str(nu))
            self.value_dict = MH.value_dict
            self.MH_arr = self.MH_arr + [MH]
        print("opt_val:", opt_val)

    def get_point(self):
        points = [H.get_point() for H in self.MH_arr]
        for H in self.MH_arr:
            self.t = self.t + H.t
        evals = [self.mfobject.eval_at_fidel_single_point_normalised([1.0], x) for x in points]
        if self.CAPITAL == 'Actual':
            self.cost = self.cost + self.nHOO * self.mfobject.eval_fidel_cost_single_point_normalised([1.0])
        else:
            self.cost = self.cost + self.nHOO * self.unit_cost

        index = np.argmax(evals)
        newp = []
        for p in points:
            _, npoint = self.mfobject.get_unnormalised_coords(None, p)
            newp = newp + [npoint]

        return newp, evals
