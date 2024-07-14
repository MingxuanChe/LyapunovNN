
from collections.abc import Sequence
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Add the configuration settings
class Configuration(object):
    """Configuration class."""

    def __init__(self):
        """Initialization."""
        super(Configuration, self).__init__()

        # Dtype for computations
        self.dtype = torch.float32
        #######################################################################
        # Batch size for stability verification
        self.gp_batch_size = 10000
        #######################################################################

    @property
    def np_dtype(self):
        """Return the numpy dtype."""
        return np.float32

    def __repr__(self):
        """Print the parameters."""
        params = ['Configuration parameters:', '']
        for param, value in self.__dict__.items():
            params.append('{}: {}'.format(param, value.__repr__()))

        return '\n'.join(params)

config = Configuration()
del Configuration
_EPS = np.finfo(config.np_dtype).eps

class DimensionError(Exception):
    pass

class GridWorld(object):
    """Base class for function approximators on a regular grid.

    Parameters
    ----------
    limits: 2d array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d array-like
        The number of points with which to grid each dimension.

    NOTE: in original Lyapunov NN, the grid is defined in a normalized 
          fashion (i.e. [-1, 1] for each dimension)
    """

    def __init__(self, limits, num_points):
        """Initialization, see `GridWorld`."""
        super(GridWorld, self).__init__()

        self.limits = np.atleast_2d(limits).astype(config.np_dtype)
        num_points = np.broadcast_to(num_points, len(self.limits))
        self.num_points = num_points.astype(np.int16, copy=False)
        self.state_dim = len(self.limits)


        if np.any(self.num_points < 2):
            raise DimensionError('There must be at least 2 points in each '
                                 'dimension.')

        # Compute offset and unit hyperrectangle
        self.offset = self.limits[:, 0]
        self.unit_maxes = ((self.limits[:, 1] - self.offset)
                           / (self.num_points - 1)).astype(config.np_dtype)
        self.offset_limits = np.stack((np.zeros_like(self.limits[:, 0]),
                                       self.limits[:, 1] - self.offset),
                                      axis=1)

        # Statistics about the grid
        self.discrete_points = [np.linspace(low, up, n, dtype=config.np_dtype)
                                for (low, up), n in zip(self.limits,
                                                        self.num_points)]

        self.nrectangles = np.prod(self.num_points - 1)
        self.nindex = np.prod(self.num_points)

        self.ndim = len(self.limits)
        self._all_points = None

    @property
    def all_points(self):
        """Return all the discrete points of the discretization.

        Returns
        -------
        points : ndarray
            An array with all the discrete points with size
            (self.nindex, self.ndim).

        """
        if self._all_points is None:
            # my own implementation
            mesh = np.stack(np.meshgrid(*self.discrete_points),-1).reshape(-1,self.state_dim)
            self._all_points = mesh.astype(config.np_dtype)
            # if self.all_points.shape[1] == 2:
                # swap the first two columns
                # self._all_points[:,[0,1]] = self._all_points[:,[1,0]]

            # original implementation
            # mesh = np.meshgrid(*self.discrete_points, indexing='ij')
            # points = np.column_stack(col.ravel() for col in mesh)
            # each row of the mesh is a point in the stat space
            # self._all_points = points.astype(config.np_dtype)

        return self._all_points

    def __len__(self):
        """Return the number of points in the discretization."""
        return self.nindex

    def sample_continuous(self, num_samples):
        """Sample uniformly at random from the continuous domain.

        Parameters
        ----------
        num_samples : int

        Returns
        -------
        points : ndarray
            Random points on the continuous rectangle.

        """
        limits = self.limits
        rand = np.random.uniform(0, 1, size=(num_samples, self.ndim))
        return rand * np.diff(limits, axis=1).T + self.offset

    def sample_discrete(self, num_samples, replace=False):
        """Sample uniformly at random from the discrete domain.

        Parameters
        ----------
        num_samples : int
        replace : bool, optional
            Whether to sample with replacement.

        Returns
        -------
        points : ndarray
            Random points on the continuous rectangle.

        """
        idx = np.random.choice(self.nindex, size=num_samples, replace=replace)
        return self.index_to_state(idx)

    def _check_dimensions(self, states):
        """Raise an error if the states have the wrong dimension.

        Parameters
        ----------
        states : ndarray

        """
        if not states.shape[1] == self.ndim:
            raise DimensionError('the input argument has the wrong '
                                 'dimensions.')

    def _center_states(self, states, clip=True):
        """Center the states to the interval [0, x].

        Parameters
        ----------
        states : np.array
        clip : bool, optinal
            If False the data is not clipped to lie within the limits.

        Returns
        -------
        offset_states : ndarray

        """
        states = np.atleast_2d(states).astype(config.np_dtype)
        states = states - self.offset[None, :]
        if clip:
            np.clip(states,
                    self.offset_limits[:, 0] + 2 * _EPS,
                    self.offset_limits[:, 1] - 2 * _EPS,
                    out=states)
        return states

    def index_to_state(self, indices):
        """Convert indices to physical states.

        Parameters
        ----------
        indices : ndarray (int)
            The indices of points on the discretization.

        Returns
        -------
        states : ndarray
            The states with physical units that correspond to the indices.

        """
        indices = np.atleast_1d(indices)
        ijk_index = np.vstack(np.unravel_index(indices, self.num_points)).T
        ijk_index = ijk_index.astype(config.np_dtype)
        return ijk_index * self.unit_maxes + self.offset

    def state_to_index(self, states):
        """Convert physical states to indices.

        Parameters
        ----------
        states: ndarray
            Physical states on the discretization.

        Returns
        -------
        indices: ndarray (int)
            The indices that correspond to the physical states.

        """
        states = np.atleast_2d(states)
        self._check_dimensions(states)
        states = np.clip(states, self.limits[:, 0], self.limits[:, 1])
        states = (states - self.offset) * (1. / self.unit_maxes)
        ijk_index = np.rint(states).astype(np.int32)
        return np.ravel_multi_index(ijk_index.T, self.num_points)

    def state_to_rectangle(self, states):
        """Convert physical states to its closest rectangle index.

        Parameters
        ----------
        states : ndarray
            Physical states on the discretization.

        Returns
        -------
        rectangles : ndarray (int)
            The indices that correspond to rectangles of the physical states.

        """
        ind = []
        for i, (discrete, num_points) in enumerate(zip(self.discrete_points,
                                                       self.num_points)):
            idx = np.digitize(states[:, i], discrete)
            idx -= 1
            np.clip(idx, 0, num_points - 2, out=idx)

            ind.append(idx)
        return np.ravel_multi_index(ind, self.num_points - 1)

    def rectangle_to_state(self, rectangles):
        """
        Convert rectangle indices to the states of the bottem-left corners.

        Parameters
        ----------
        rectangles : ndarray (int)
            The indices of the rectangles

        Returns
        -------
        states : ndarray
            The states that correspond to the bottom-left corners of the
            corresponding rectangles.

        """
        rectangles = np.atleast_1d(rectangles)
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        ijk_index = ijk_index.astype(config.np_dtype)
        return (ijk_index.T * self.unit_maxes) + self.offset

    def rectangle_corner_index(self, rectangles):
        """Return the index of the bottom-left corner of the rectangle.

        Parameters
        ----------
        rectangles: ndarray
            The indices of the rectangles.

        Returns
        -------
        corners : ndarray (int)
            The indices of the bottom-left corners of the rectangles.

        """
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        return np.ravel_multi_index(np.atleast_2d(ijk_index),
                                    self.num_points)

class QuadraticFunction(object):
    """A quadratic function.

    values(x) = x.T P x

    Parameters
    ----------
    matrix : np.array
        2d cost matrix for lyapunov function.

    """
    def __init__(self, matrix):
        """Initialization, see `QuadraticLyapunovFunction`."""
        super(QuadraticFunction, self).__init__()

        self.matrix = np.atleast_2d(matrix).astype(config.np_dtype)
        self.ndim = self.matrix.shape[0]

    def __call__(self, *args, **kwargs):
        """Evaluate the function using the template to ensure variable sharing.

        Parameters
        ----------
        args : list
            The input arguments to the function.
        kwargs : dict, optional
            The keyword arguments to the function.

        Returns
        -------
        outputs : list
            The output arguments of the function as given by evaluate.

        """
        
        outputs = self.forward(*args, **kwargs)
        return outputs
    
    def forward(self, points):
        """Like evaluate, but returns a tensor instead."""
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
        # convert points to np array
        if isinstance(points, torch.Tensor):
            # if the tensor is on GPU, convert it to CPU first
            if points.is_cuda:
                points = points.cpu()
            points = points.detach().numpy()
            points = np.reshape(points, (-1, self.ndim))
        linear_form = points @ self.matrix # (N , n) @ (n, n) = (N, n)
        quadratic = np.sum(linear_form * points, axis=1)
        return torch.tensor(quadratic)

    def gradient(self, points):
        """Return the gradient of the function."""
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
        return torch.matmul(torch.tensor(points, dtype=config.dtype), \
                            torch.tensor(self.matrix + self.matrix.T, dtype=config.dtype))

class LyapunovNN(torch.nn.Module):
    def __init__(self, input_dim, layer_dims, activations, eps=1e-6, device='cpu'):
        super(LyapunovNN, self).__init__()
        # network layers
        self.input_dim = input_dim
        self.num_layers = len(layer_dims)
        self.activations = activations
        self.eps = eps
        self.layers_params = torch.nn.ModuleList()
        self.kernel = []
        self.device = device

        if layer_dims[0] < input_dim:
            raise ValueError('The first layer dimension must be at \
                             least the input dimension!')

        if np.all(np.diff(layer_dims) >= 0):
            self.output_dims = layer_dims
        else:
            raise ValueError('Each layer must maintain or increase \
                             the dimension of its input!')

        self.hidden_dims = np.zeros(self.num_layers, dtype=int)
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]
            self.hidden_dims[i] = np.ceil((layer_input_dim + 1) / 2).astype(int)

        # build the nn structure
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]
            self.layers_params.append(\
                        torch.nn.Linear(layer_input_dim, self.hidden_dims[i], bias=False))
            dim_diff = self.output_dims[i] - layer_input_dim
            if dim_diff > 0:
                self.layers_params.append(torch.nn.Linear(layer_input_dim, dim_diff, bias=False))
        self.update_kernel()

    def forward(self, x):
        # the input should have a shape of (batch_size, input_dim)
        # namely, each row is a state
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        # put the input to the device
        x = x.to(self.device)
        
        for i in range(self.num_layers):
            layer_output = torch.matmul(x, self.kernel[i].T)
            x = self.activations[i](layer_output)
        # sum over the rows
        values = torch.sum(torch.square(x), dim = 1)
        return values
    
    def update_kernel(self):
        self.kernel = [] # clear the kernel
        param_idx = 0 # for skipping the extra layer parameters
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]
            # build the positive definite part of the kernel
            W = self.layers_params[i + param_idx].weight
            weight = W.clone()
            kernel = torch.matmul(weight.T, weight) + self.eps * torch.eye(W.shape[1])
            eigvals, _ = np.linalg.eig(kernel.detach().numpy())
            # check whether all eigenvalues are positive
            assert np.all(eigvals > 0)
            # if the kernel need extra part, append the parameters of the next layer
            dim_diff = self.output_dims[i] - layer_input_dim
            if dim_diff > 0:
                kernel = torch.cat((kernel, self.layers_params[i+1].weight), dim=0)
                param_idx += 1
            self.kernel.append(kernel.to(self.device))

    def print_params(self):
        offset = 0
        for i, dim_diff in enumerate(np.diff(np.concatenate([[self.input_dim], self.output_dims]))):
            print('Layer weights {}:'.format(i))
            print('dim_diff: ', dim_diff)
            if dim_diff > 0:
                # cut-off the last dim_diff rows
                num_rows = self.kernel[i].shape[0] - dim_diff
                # print('num_rows: ', num_rows)
                kernel = self.kernel[i][0:num_rows, :]
            else:
                kernel = self.kernel[i]
            assert kernel.shape[0] == kernel.shape[1]
            # NOTE: eigenvalues might have imaginary parts because of numerical errors
            eigvals, _ = np.linalg.eig(kernel.detach().numpy())
            # check whether all eigenvalues are positive
            assert np.all(eigvals > 0)
            
            print('Eigenvalues of (W0.T*W0 + eps*I):', eigvals, '\n')

class Symmetric(nn.Module):
    # parametrize a symmetric matrix
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)
    
class MatrixExponential(nn.Module):
    # parametrize a matrix exponential function
    def forward(self, X):
        return torch.matrix_exp(X)

class SymmetricPositiveDefinite(nn.Module):
    # parametrize a symmetric positive definite matrix
    def forward(self, X):
        return torch.matrix_exp(X.triu() + X.triu(1).transpose(-1, -2))

class Lyapunov(object):
    """A class for general Lyapunov functions.

    Parameters
    ----------
    discretization : ndarray
        A discrete grid on which to evaluate the Lyapunov function.
    lyapunov_function : callable or instance of `DeterministicFunction`
        The lyapunov function. Can be called with states and returns the
        corresponding values of the Lyapunov function.
    dynamics : a callable or an instance of `Function`
        The dynamics model. Can be either a deterministic function or something
        uncertain that includes error bounds.
    lipschitz_dynamics : ndarray or float
        The Lipschitz constant of the dynamics. Either globally, or locally
        for each point in the discretization (within a radius given by the
        discretization constant. This is the closed-loop Lipschitz constant
        including the policy!
    lipschitz_lyapunov : ndarray or float
        The Lipschitz constant of the lyapunov function. Either globally, or
        locally for each point in the discretization (within a radius given by
        the discretization constant.
    tau : float
        The discretization constant.
    policy : ndarray, optional
        The control policy used at each state (Same number of rows as the
        discretization).
    initial_set : ndarray, optional
        A boolean array of states that are known to be safe a priori.
    adaptive : bool, optional
        A boolean determining whether an adaptive discretization is used for
        stability verification.

    """

    def __init__(self, discretization, lyapunov_function, dynamics,
                 lipschitz_dynamics, lipschitz_lyapunov,
                 tau, policy, initial_set=None, adaptive=False):
        """Initialization, see `Lyapunov` for details."""
        super(Lyapunov, self).__init__()

        self.discretization = discretization
        self.policy = policy

        # Keep track of the safe sets
        self.safe_set = np.zeros(np.prod(discretization.num_points),
                                 dtype=bool)

        self.initial_safe_set = initial_set
        if initial_set is not None:
            self.safe_set[initial_set] = True

        # Discretization constant
        self.tau = tau

        # Make sure dynamics are of standard framework
        self.dynamics = dynamics

        # Make sure Lyapunov fits into standard framework
        self.lyapunov_function = lyapunov_function

        # Storage for graph
        self._storage = dict()
        # self.feed_dict = get_feed_dict(tf.get_default_graph())

        # Lyapunov values
        self.values = None

        # self.c_max = tf.placeholder(config.dtype, shape=())
        self.c_max = None
        # self.feed_dict[self.c_max] = 0.

        self._lipschitz_dynamics = lipschitz_dynamics
        self._lipschitz_lyapunov = lipschitz_lyapunov

        self.update_values()

        self.adaptive = adaptive

        # Keep track of the refinement `N(x)` used around each state `x` in
        # the adaptive discretization; `N(x) = 0` by convention if `x` is
        # unsafe
        self._refinement = np.zeros(discretization.nindex, dtype=int)
        if initial_set is not None:
            self._refinement[initial_set] = 1

    def update_values(self):
        """Update the discretized values when the Lyapunov function changes."""

        values = self.lyapunov_function(self.discretization.all_points)
        self.values = values.cpu().detach().numpy()

    def update_safe_set(self, can_shrink=True, max_refinement=1,
                        safety_factor=1., parallel_iterations=1):
        """Compute and update the safe set.

        Parameters
        ----------
        can_shrink : bool, optional
            A boolean determining whether previously safe states other than the
            initial safe set must be verified again (i.e., can the safe set
            shrink in volume?)
        max_refinement : int, optional
            The maximum integer divisor used for adaptive discretization.
        safety_factor : float, optional
            A multiplicative factor greater than 1 used to conservatively
            estimate the required adaptive discretization.
        parallel_iterations : int, optional
            The number of parallel iterations to use for safety verification in
            the adaptive case. Passed to `tf.map_fn`.

        """
        safety_factor = np.maximum(safety_factor, 1.)

        np_states = lambda x: np.array(x, dtype=config.dtype)
        # decrease = lambda x: self.v_decrease_bound(x, self.dynamics(x, self.policy(x)))
        decrease = lambda x: self.v_decrease_bound(x, self.dynamics(x)).reshape(-1, 1)
        threshold = lambda x: self.threshold(x, self.tau)
        np_negative = lambda x: np.squeeze(decrease(x) < threshold(x), axis=0)

        if can_shrink:
            # Reset the safe set and adaptive discretization
            safe_set = np.zeros_like(self.safe_set, dtype=bool)
            refinement = np.zeros_like(self._refinement, dtype=int)
            if self.initial_safe_set is not None:
                safe_set[self.initial_safe_set] = True
                refinement[self.initial_safe_set] = 1
        else:
            # Assume safe set cannot shrink
            safe_set = self.safe_set
            refinement = self._refinement

        value_order = np.argsort(self.values)
        safe_set = safe_set[value_order]
        refinement = refinement[value_order]

        # Verify safety in batches
        batch_size = config.gp_batch_size
        batch_generator = batchify((value_order, safe_set, refinement),
                                   batch_size)
        index_to_state = self.discretization.index_to_state

        #######################################################################

        for i, (indices, safe_batch, refine_batch) in batch_generator:
            states = index_to_state(indices)
            np_state = np.squeeze(states)

            # Update the safety with the safe_batch result
            negative = np_negative(np_state)
            # convert negative to np array
            negative = np.squeeze(np.array(negative, dtype=bool))
            safe_batch |= negative
            refine_batch[negative] = 1
            # Boolean array: argmin returns first element that is False
            # If all are safe then it returns 0
            bound = np.argmin(safe_batch)
            refine_bound = 0

            # Check if there are unsafe elements in the batch
            if bound > 0 or not safe_batch[0]:
                    safe_batch[bound:] = False
                    refine_batch[bound:] = 0
                    break

        # The largest index of a safe value
        max_index = i + bound + refine_bound - 1

        #######################################################################

        # Set placeholder for c_max to the corresponding value
        self.c_max = self.values[value_order[max_index]]

        # Restore the order of the safe set and adaptive refinement
        safe_nodes = value_order[safe_set]
        self.safe_set[:] = False
        self.safe_set[safe_nodes] = True
        self._refinement[value_order] = refinement

        # Ensure the initial safe set is kept
        if self.initial_safe_set is not None:
            self.safe_set[self.initial_safe_set] = True
            self._refinement[self.initial_safe_set] = 1
        
    def threshold(self, states, tau=None):
        """Return the safety threshold for the Lyapunov condition.

        Parameters
        ----------
        states : ndarray or Tensor

        tau : float or Tensor, optional
            Discretization constant to consider.

        Returns
        -------
        lipschitz : float, ndarray or Tensor
            Either the scalar threshold or local thresholds, depending on
            whether lipschitz_lyapunov and lipschitz_dynamics are local or not.

        """
        if tau is None:
            tau = self.tau
        # if state is not a tensor, convert it to a tensor
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=config.dtype, requires_grad=True)
            states = states.float()
        lv = self._lipschitz_lyapunov(states)
        # convert states to np array
        if states.is_cuda:
            states = states.cpu()
        states = states.detach().numpy()
        lf = self._lipschitz_dynamics(states)
        return - lv * (1. + lf) * tau
    
    def v_decrease_bound(self, states, next_states):
        """Compute confidence intervals for the decrease along Lyapunov function.

        Parameters
        ----------
        states : np.array
            The states at which to start (could be equal to discretization).
        next_states : np.array or tuple
            The dynamics evaluated at each point on the discretization. If
            the dynamics are uncertain then next_states is a tuple with mean
            and error bounds.

        Returns
        -------
        upper_bound : np.array
            The upper bound on the change in values at each grid point.

        """
        v_dot, v_dot_error = self.v_decrease_confidence(states, next_states)

        return v_dot + v_dot_error
    
    def v_decrease_confidence(self, states, next_states):
        """Compute confidence intervals for the decrease along Lyapunov function.

        Parameters
        ----------
        states : np.array
            The states at which to start (could be equal to discretization).
        next_states : np.array
            The dynamics evaluated at each point on the discretization. If
            the dynamics are uncertain then next_states is a tuple with mean
            and error bounds.

        Returns
        -------
        mean : np.array
            The expected decrease in values at each grid point.
        error_bounds : np.array
            The error bounds for the decrease at each grid point

        """
        if isinstance(next_states, Sequence):
            next_states, error_bounds = next_states
            lv = self._lipschitz_lyapunov(next_states)
            bound = np.sum(lv * error_bounds, axis=1, keepdims=True)
        else:
            bound = torch.tensor(0., dtype=config.dtype)
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float64)
            states = states.float() # avoid feedforward data type error
        # convert the next_states first to numpy array, then to torch tensor
        if not isinstance(next_states, torch.Tensor):
            next_states = torch.tensor(np.array(next_states), dtype=torch.float64)
            next_states = next_states.float() # avoid feedforward data type error
        # if the state and next state are 1d array, convert them to 2d array
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        if len(next_states.shape) == 1:
            next_states = next_states.unsqueeze(0)
        v_decrease = (self.lyapunov_function(next_states)
            - self.lyapunov_function(states))

        return v_decrease, bound

def batchify(arrays, batch_size):
    """Yield the arrays in batches and in order.

    The last batch might be smaller than batch_size.

    Parameters
    ----------
    arrays : list of ndarray
        The arrays that we want to convert to batches.
    batch_size : int
        The size of each individual batch.
    """
    if not isinstance(arrays, (list, tuple)):
        arrays = (arrays,)

    # Iterate over array in batches
    for i, i_next in zip(itertools.count(start=0, step=batch_size),
                         itertools.count(start=batch_size, step=batch_size)):

        batches = [array[i:i_next] for array in arrays]

        # Break if there are no points left
        if batches[0].size:
            yield i, batches
        else:
            break

class GridWorld_pendulum(object):
    """Base class for function approximators on a regular grid.

    Parameters
    ----------
    limits: 2d array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d array-like
        The number of points with which to grid each dimension.

    NOTE: in original Lyapunov NN, the grid is defined in a normalized 
          fashion (i.e. [-1, 1] for each dimension)
    """

    def __init__(self, limits, num_points):
        """Initialization, see `GridWorld`."""
        super(GridWorld_pendulum, self).__init__()

        self.limits = np.atleast_2d(limits).astype(config.np_dtype)
        num_points = np.broadcast_to(num_points, len(self.limits))
        self.num_points = num_points.astype(np.int16, copy=False)
        self.state_dim = len(self.limits)

        if np.any(self.num_points < 2):
            raise DimensionError('There must be at least 2 points in each '
                                 'dimension.')

        # Compute offset and unit hyperrectangle
        self.offset = self.limits[:, 0]
        self.unit_maxes = ((self.limits[:, 1] - self.offset)
                           / (self.num_points - 1)).astype(config.np_dtype)
        self.offset_limits = np.stack((np.zeros_like(self.limits[:, 0]),
                                       self.limits[:, 1] - self.offset),
                                      axis=1)

        # Statistics about the grid
        self.discrete_points = [np.linspace(low, up, n, dtype=config.np_dtype)
                                for (low, up), n in zip(self.limits,
                                                        self.num_points)]

        self.nrectangles = np.prod(self.num_points - 1)
        self.nindex = np.prod(self.num_points)

        self.ndim = len(self.limits)
        self._all_points = None

    @property
    def all_points(self):
        """Return all the discrete points of the discretization.

        Returns
        -------
        points : ndarray
            An array with all the discrete points with size
            (self.nindex, self.ndim).

        """
        if self._all_points is None:
            # my own implementation
            mesh = np.stack(np.meshgrid(*self.discrete_points),-1).reshape(-1,self.state_dim)
            self._all_points = mesh.astype(config.np_dtype)
            if self.all_points.shape[1] == 2:
                # swap the first two columns
                self._all_points[:,[0,1]] = self._all_points[:,[1,0]]

            # original implementation
            # mesh = np.meshgrid(*self.discrete_points, indexing='ij')
            # points = np.column_stack(col.ravel() for col in mesh)
            # each row of the mesh is a point in the stat space
            # self._all_points = points.astype(config.np_dtype)

        return self._all_points

    def __len__(self):
        """Return the number of points in the discretization."""
        return self.nindex

    def sample_continuous(self, num_samples):
        """Sample uniformly at random from the continuous domain.

        Parameters
        ----------
        num_samples : int

        Returns
        -------
        points : ndarray
            Random points on the continuous rectangle.

        """
        limits = self.limits
        rand = np.random.uniform(0, 1, size=(num_samples, self.ndim))
        return rand * np.diff(limits, axis=1).T + self.offset

    def sample_discrete(self, num_samples, replace=False):
        """Sample uniformly at random from the discrete domain.

        Parameters
        ----------
        num_samples : int
        replace : bool, optional
            Whether to sample with replacement.

        Returns
        -------
        points : ndarray
            Random points on the continuous rectangle.

        """
        idx = np.random.choice(self.nindex, size=num_samples, replace=replace)
        return self.index_to_state(idx)

    def _check_dimensions(self, states):
        """Raise an error if the states have the wrong dimension.

        Parameters
        ----------
        states : ndarray

        """
        if not states.shape[1] == self.ndim:
            raise DimensionError('the input argument has the wrong '
                                 'dimensions.')

    def _center_states(self, states, clip=True):
        """Center the states to the interval [0, x].

        Parameters
        ----------
        states : np.array
        clip : bool, optinal
            If False the data is not clipped to lie within the limits.

        Returns
        -------
        offset_states : ndarray

        """
        states = np.atleast_2d(states).astype(config.np_dtype)
        states = states - self.offset[None, :]
        if clip:
            np.clip(states,
                    self.offset_limits[:, 0] + 2 * _EPS,
                    self.offset_limits[:, 1] - 2 * _EPS,
                    out=states)
        return states

    def index_to_state(self, indices):
        """Convert indices to physical states.

        Parameters
        ----------
        indices : ndarray (int)
            The indices of points on the discretization.

        Returns
        -------
        states : ndarray
            The states with physical units that correspond to the indices.

        """
        indices = np.atleast_1d(indices)
        ijk_index = np.vstack(np.unravel_index(indices, self.num_points)).T
        ijk_index = ijk_index.astype(config.np_dtype)
        return ijk_index * self.unit_maxes + self.offset

    def state_to_index(self, states):
        """Convert physical states to indices.

        Parameters
        ----------
        states: ndarray
            Physical states on the discretization.

        Returns
        -------
        indices: ndarray (int)
            The indices that correspond to the physical states.

        """
        states = np.atleast_2d(states)
        self._check_dimensions(states)
        states = np.clip(states, self.limits[:, 0], self.limits[:, 1])
        states = (states - self.offset) * (1. / self.unit_maxes)
        ijk_index = np.rint(states).astype(np.int32)
        return np.ravel_multi_index(ijk_index.T, self.num_points)

    def state_to_rectangle(self, states):
        """Convert physical states to its closest rectangle index.

        Parameters
        ----------
        states : ndarray
            Physical states on the discretization.

        Returns
        -------
        rectangles : ndarray (int)
            The indices that correspond to rectangles of the physical states.

        """
        ind = []
        for i, (discrete, num_points) in enumerate(zip(self.discrete_points,
                                                       self.num_points)):
            idx = np.digitize(states[:, i], discrete)
            idx -= 1
            np.clip(idx, 0, num_points - 2, out=idx)

            ind.append(idx)
        return np.ravel_multi_index(ind, self.num_points - 1)

    def rectangle_to_state(self, rectangles):
        """
        Convert rectangle indices to the states of the bottem-left corners.

        Parameters
        ----------
        rectangles : ndarray (int)
            The indices of the rectangles

        Returns
        -------
        states : ndarray
            The states that correspond to the bottom-left corners of the
            corresponding rectangles.

        """
        rectangles = np.atleast_1d(rectangles)
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        ijk_index = ijk_index.astype(config.np_dtype)
        return (ijk_index.T * self.unit_maxes) + self.offset

    def rectangle_corner_index(self, rectangles):
        """Return the index of the bottom-left corner of the rectangle.

        Parameters
        ----------
        rectangles: ndarray
            The indices of the rectangles.

        Returns
        -------
        corners : ndarray (int)
            The indices of the bottom-left corners of the rectangles.

        """
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        return np.ravel_multi_index(np.atleast_2d(ijk_index),
                                    self.num_points)
