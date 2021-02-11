# -*- coding: utf-8 -*-
"""
@author: mblonsky
"""

import numpy as np
import pandas as pd
from scipy import linalg

try:
    import sympy  # Optional package - only for generating abstract matrices
except ImportError:
    sympy = None


class ModelException(Exception):
    pass


class RCModel:
    """
    Discrete Time State Space RC Model

    Generates an model based on RC parameters provided as a dictionary. The naming convention is as follows:
     - Resistors: "R_{node1}_{node2}" (order of nodes doesn't matter)
     - Capacitors: "C_{node}"

    From the RC parameter dictionary, the model generates collects all internal and external system nodes.
    Nodes are internal if there is a capacitance associated with it; otherwise it is external.
    The initialization process is as follows:
     - Load RC parameter dictionary
     - Create state names from internal nodes: "T_{node_internal}"
     - Create input names from internal and external nodes: "T_{node_external}" and "H_{node_internal}"
     - Create A and B continuous-time matrices from RC values
     - Discretize A and B matrices using datetime.timedelta parameter 'time_res' (required)
     - Creates initial state vector using paramter 'initial_states' (required)
     - Creates default input vector using paramter 'default_inputs' (optional, default sets all inputs to 0)
    """
    name = 'Generic RC'

    def __init__(self, time_res, rc_params=None, ext_node_names=None, **kwargs):
        self.time_res = time_res

        # Load RC parameters
        if rc_params is None:
            rc_params = self.load_rc_data(**kwargs)
        if not rc_params:
            raise ModelException('No RC Parameters found for {} Model'.format(self.name))

        # Load A and B abstract matrices and update with RC parameters
        A_c, B_c, state_names, input_names = self.create_matrices(rc_params, ext_node_names)
        # A_c, B_c, state_names, input_names = self.create_matrices(rc_params, print_abstract=True)
        self.A_c = A_c
        self.B_c = B_c

        # Convert matrices to discrete time
        A, B = self.to_discrete(A_c, B_c, time_res)
        self.A = A
        self.B = B

        # Create default input vector
        self.input_names = input_names
        self.default_u = self.load_default_inputs(**kwargs)
        self.u = self.default_u.copy()

        # Create initial state vector
        self.state_names = state_names
        self.x = self.load_initial_state(**kwargs)

        # remove unused inputs
        self.remove_unused_inputs(**kwargs)

    def load_rc_data(self, rc_filename, name_col='Name', val_col='Value', **kwargs):
        # Load file
        df = pd.read_csv(rc_filename, index_col=name_col)

        # Convert to dict of {Parameter Name: Parameter Value}
        return df[val_col].to_dict()

    def create_matrices(self, rc_params, ext_node_names_check=None, return_abstract=False):
        # uses RC parameter names to get list of internal/external nodes
        # C names should be 'C_{node}'; R names should be 'R_{node1}_{node2}'
        if sympy is None:
            return_abstract = False

        # parse RC names
        all_cap = {'_'.join(name.split('_')[1:]).upper(): val for name, val in rc_params.items() if name[0] == 'C'}
        all_res = {'_'.join(name.split('_')[1:]).upper(): val for name, val in rc_params.items() if name[0] == 'R'}

        # get all internal and external nodes (internal nodes have a C)
        internal_nodes = list(all_cap.keys())
        res_nodes = [node for name in all_res.keys() for node in name.split('_')]
        all_nodes = sorted(set(res_nodes), key=res_nodes.index)
        external_nodes = [node for node in all_nodes if node not in internal_nodes]

        bad = [node for node in internal_nodes if node not in all_nodes]
        if bad:
            raise ModelException(
                'Some nodes have capacitors but no connected resistors for {}: {}'.format(self.name, bad))
        if ext_node_names_check is not None:
            bad = [node for node in external_nodes if node not in ext_node_names_check]
            if bad:
                raise ModelException('Undefined external nodes for {}: {}'.format(self.name, bad))

        # Define states and inputs
        state_names = ['T_' + node for node in internal_nodes]
        input_names = ['T_' + node for node in external_nodes] + ['H_' + node for node in internal_nodes]
        n = len(state_names)
        m = len(input_names)

        # Create A, B matrices
        A = np.zeros((n, n))
        b_diag = [1 / all_cap[node] for node in internal_nodes]
        B = np.concatenate((np.zeros((n, m - n)), np.diag(b_diag)), axis=1)

        # Create A and B abstract matrices
        if return_abstract:
            cap_abstract = {name: sympy.Symbol('C_' + name) for name in all_cap.keys()}
            res_abstract = {name: sympy.Symbol('R_' + name) for name in all_res.keys()}
            A_abstract = sympy.zeros(n, n)
            b_diag = [1 / c for c in cap_abstract.values()]
            B_abstract = np.concatenate((sympy.zeros(n, m - n), np.diag(b_diag)), axis=1)
        else:
            A_abstract = None
            B_abstract = None

        def add_matrix_values(node1, node2, r_val, res_name):
            # add 1/RC term to A and B matrices (R is between node1 and node2)
            if node1 in internal_nodes and node2 in internal_nodes:
                # both are internal nodes - only update A
                i1 = internal_nodes.index(node1)
                c1 = all_cap[node1]
                i2 = internal_nodes.index(node2)
                c2 = all_cap[node2]
                A[i1, i1] -= 1 / c1 / r_val
                A[i2, i2] -= 1 / c2 / r_val
                A[i1, i2] += 1 / c1 / r_val
                A[i2, i1] += 1 / c2 / r_val
                if return_abstract:
                    r = res_abstract[res_name]
                    c1 = cap_abstract[node1]
                    c2 = cap_abstract[node2]
                    A_abstract[i1, i1] -= 1 / c1 / r
                    A_abstract[i2, i2] -= 1 / c2 / r
                    A_abstract[i1, i2] += 1 / c1 / r
                    A_abstract[i2, i1] += 1 / c2 / r
            else:
                if node1 not in internal_nodes:
                    # node1 is external, update A and B
                    i_ext = external_nodes.index(node1)
                    i_int = internal_nodes.index(node2)
                    c = all_cap[node2]
                else:
                    # node2 is external, update A and B
                    i_ext = external_nodes.index(node2)
                    i_int = internal_nodes.index(node1)
                    c = all_cap[node1]
                A[i_int, i_int] -= 1 / c / r_val
                B[i_int, i_ext] += 1 / c / r_val
                if return_abstract:
                    r = res_abstract[res_name]
                    c = cap_abstract[node1] if node1 in internal_nodes else cap_abstract[node2]
                    A_abstract[i_int, i_int] -= 1 / c / r
                    B_abstract[i_int, i_ext] += 1 / c / r

        # Iterate through resistances to build A, B matrices
        for res_name, res in all_res.items():
            n1, n2 = tuple(res_name.split('_'))
            if n1 not in all_nodes:
                raise ModelException('Error parsing resistor {}. {} not in {}.'.format(res_name, n1, self.name))
            if n2 not in all_nodes:
                raise ModelException('Error parsing resistor {}. {} not in {}.'.format(res_name, n2, self.name))

            add_matrix_values(n1, n2, res, res_name)

        if return_abstract:
            return A_abstract, B_abstract
        else:
            return A, B, state_names, input_names

    @staticmethod
    def to_discrete(A, B, time_res):
        # 2 options for discretization, see https://en.wikipedia.org/wiki/Discretization
        n, m = B.shape

        # first option
        A_d = linalg.expm(A * time_res.total_seconds())
        B_d = np.dot(np.dot(linalg.inv(A), A_d - np.eye(n)), B)

        # second option, without inverse
        # M = np.block([[A, B], [np.zeros((m, n + m))]])
        # M_exp = linalg.expm(M * time_res.total_seconds())
        # A_d = M_exp[:n, :n]
        # B_d = M_exp[:n, n:]
        return A_d, B_d

    def load_initial_state(self, initial_states=None, **kwargs):
        # can take initial states as a dict, list, or number
        # if initial_states is a number, all states are equal to that number
        if isinstance(initial_states, dict) and all([state in initial_states.keys() for state in self.state_names]):
            x0 = [initial_states[state] for state in self.state_names]
        elif isinstance(initial_states, list) and len(initial_states) == len(self.state_names):
            x0 = initial_states
        elif initial_states is not None:
            x0 = [initial_states] * len(self.state_names)
        else:
            raise ModelException('Initial state cannot be loaded from: {}'.format(initial_states))
        return np.array(x0, dtype=float)

    def load_default_inputs(self, default_inputs=None, **kwargs):
        # can take default inputs as a dict or list
        # if default_inputs is a dict, inputs not included in the dict are set to 0.
        if isinstance(default_inputs, dict) and all([u in self.input_names for u in default_inputs.keys()]):
            u0 = [default_inputs.get(u, 0) for u in self.input_names]
        elif isinstance(default_inputs, list) and len(default_inputs) == len(self.input_names):
            u0 = default_inputs
        elif default_inputs is None:
            u0 = [0] * len(self.input_names)
        else:
            raise ModelException('Default inputs cannot be loaded from: {}'.format(default_inputs))
        return np.array(u0, dtype=float)

    def remove_unused_inputs(self, unused_inputs=None, **kwargs):
        if unused_inputs is not None:
            keep_input_idx = [i for i, name in enumerate(self.input_names) if name not in unused_inputs]
            self.input_names = [name for name in self.input_names if name not in unused_inputs]
            self.default_u = self.default_u[keep_input_idx]
            self.u = self.u[keep_input_idx]
            self.B = self.B[:, keep_input_idx]
            self.B_c = self.B_c[:, keep_input_idx]

    @staticmethod
    def par(*args):
        return 1 / sum([1 / a for a in args])

    def update_input(self, input_name, new_value):
        if np.isnan(new_value):
            raise ModelException('NaN for {} input {}'.format(self.name, input_name))
        idx = self.input_names.index(input_name)
        self.u[idx] = new_value

    def update_inputs(self, inputs):
        # note: if an input isn't in kwargs, it is set to the default input value
        self.u = self.default_u.copy()
        for input_name, new_val in inputs.items():
            if input_name not in self.input_names:
                raise ModelException('Input {} not in {} Model'.format(input_name, self.name))

            self.update_input(input_name, new_val)

    def update(self, inputs):
        self.update_inputs(inputs)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        if np.isnan(self.x).any():
            raise ModelException('Error solving {}, states are NaN'.format(self.name))
        return self.get_states()

    def get_states(self):
        # convert states to dictionary
        return {name: val for name, val in zip(self.state_names, self.x)}

    def get_state(self, state_name):
        idx = self.state_names.index(state_name)
        return self.x[idx]

    def get_inputs(self):
        # convert inputs to dictionary
        return {name: val for name, val in zip(self.input_names, self.u)}

    def get_input(self, input_name):
        idx = self.input_names.index(input_name)
        return self.u[idx]

    def solve_for_input(self, x_name, u_name, x_desired):
        # if 1 state is known, solve for 1 unknown input
        x_idx = self.state_names.index(x_name)
        u_idx = self.input_names.index(u_name)

        a_i = self.A[x_idx, :]
        b_ij = self.B[x_idx, u_idx]
        # Note: inputs remain the same as they were (not set to defaults)
        b_times_u = sum([self.B[x_idx, k] * self.u[k] for k in range(len(self.u)) if k != u_idx])
        u_desired = (x_desired - np.dot(a_i, self.x) - b_times_u) / b_ij
        return u_desired
