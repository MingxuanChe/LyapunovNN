
# import itertools # for batchify (now in lyapnov.py)

import numpy as np
from matplotlib.colors import ListedColormap
import scipy.linalg
from scipy import signal
import torch
# from parfor import pmap
import multiprocessing as mp
import casadi as cs

from lyapunov import GridWorld
from lyapunov import config


NP_DTYPE = config.np_dtype
TF_DTYPE = config.dtype

def gridding(state_dim, state_constraints, num_states = 251, use_zero_threshold = True):
    ''' evenly discretize the state space

    Args:
        state_dim (int): The dimension of the state space.
        state_constraints (np array): The constraints of the state space.
        num_state (int): The number of states along each dimension.
        use_zero_threshold (bool): Whether to use zero threshold.
                                   False: the grid is infinitesimal
    '''
    
    # State grid
    if state_constraints is None:
        state_constraints = np.array([[-1., 1.], ] * state_dim)
    grid_limits = state_constraints
    state_discretization = GridWorld(grid_limits, num_states)

    # Discretization constant
    if use_zero_threshold:
        tau = 0.0 # assume the grid is infinitesimal
    else:
        tau = np.sum(state_discretization.unit_maxes) / 2

    print('Grid size: {}'.format(state_discretization.nindex))
    print('Discretization constant (tau): {}'.format(tau))
    return state_discretization
    
def binary_cmap(color='red', alpha=1.):
    """Construct a binary colormap."""
    if color == 'red':
        color_code = (1., 0., 0., alpha)
    elif color == 'green':
        color_code = (0., 1., 0., alpha)
    elif color == 'blue':
        color_code = (0., 0., 1., alpha)
    else:
        color_code = color
    transparent_code = (1., 1., 1., 0.)
    return ListedColormap([transparent_code, color_code])

def balanced_class_weights(y_true, scale_by_total=True):
    """Compute class weights from class label counts."""
    y = y_true.astype(np.bool_)
    nP = y.sum()
    nN = y.size - y.sum()
    class_counts = np.array([nN, nP])

    weights = np.ones_like(y, dtype=float)
    weights[ y] /= nP
    weights[~y] /= nN
    if scale_by_total:
        weights *= y.size

    return weights, class_counts

def dlqr(a, b, q, r):
    """Compute the discrete-time LQR controller.

    The optimal control input is `u = -k.dot(x)`.

    Parameters
    ----------
    a : np.array
    b : np.array
    q : np.array
    r : np.array

    Returns
    -------
    k : np.array
        Controller matrix
    p : np.array
        Cost to go matrix
    """
    a, b, q, r = map(np.atleast_2d, (a, b, q, r))
    p = scipy.linalg.solve_discrete_are(a, b, q, r)

    # LQR gain
    # k = (b.T * p * b + r)^-1 * (b.T * p * a)
    bp = b.T.dot(p)
    tmp1 = bp.dot(b)
    tmp1 += r
    tmp2 = bp.dot(a)
    k = np.linalg.solve(tmp1, tmp2)

    return k, p

def discretize_linear_system(A, B, dt, exact=False):
    '''Discretization of a linear system

    dx/dt = A x + B u
    --> xd[k+1] = Ad xd[k] + Bd ud[k] where xd[k] = x(k*dt)

    Args:
        A (ndarray): System transition matrix.
        B (ndarray): Input matrix.
        dt (scalar): Step time interval.
        exact (bool): If to use exact discretization.

    Returns:
        Ad (ndarray): The discrete linear state matrix A.
        Bd (ndarray): The discrete linear input matrix B.
    '''

    state_dim, input_dim = A.shape[1], B.shape[1]

    if exact:
        M = np.zeros((state_dim + input_dim, state_dim + input_dim))
        M[:state_dim, :state_dim] = A
        M[:state_dim, state_dim:] = B

        Md = scipy.linalg.expm(M * dt)
        Ad = Md[:state_dim, :state_dim]
        Bd = Md[:state_dim, state_dim:]
    else:
        Identity = np.eye(state_dim)
        Ad = Identity + A * dt
        Bd = B * dt

    return Ad, Bd

class InvertedPendulum(object):
    """Inverted Pendulum.

    Parameters
    ----------
    mass : float
    length : float
    friction : float, optional
    dt : float, optional
        The sampling time.
    normalization : tuple, optional
        A tuple (Tx, Tu) of arrays used to normalize the state and actions. It
        is so that diag(Tx) *x_norm = x and diag(Tu) * u_norm = u.

    """

    def __init__(self, mass, length, friction=0, dt=1 / 80,
                 normalization=None):
        """Initialization; see `InvertedPendulum`."""
        super(InvertedPendulum, self).__init__()
        self.mass = mass
        self.length = length
        self.gravity = 9.81
        self.friction = friction
        self.dt = dt
        self.nx = 2
        self.nu = 1
        self.symbolic = None

        self.normalization = normalization
        if normalization is not None:
            self.normalization = [np.array(norm, dtype=config.np_dtype)
                                  for norm in normalization]
            self.inv_norm = [norm ** -1 for norm in self.normalization]

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

    @property
    def inertia(self):
        """Return inertia of the pendulum."""
        return self.mass * self.length ** 2

    def normalize(self, state, action):
        """Normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx_inv, Tu_inv = map(np.diag, self.inv_norm)
        state = np.matmul(state, Tx_inv)

        if action is not None:
            # action = tf.matmul(action, Tu_inv)
            # action = torch.matmul(action, Tu_inv)
            action = np.matmul(action, Tu_inv)

        return state, action

    def denormalize(self, state, action):
        """De-normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx, Tu = map(np.diag, self.normalization)
        state = np.matmul(state, Tx)
        if action is not None:
            action = np.matmul(action, Tu)

        return state, action

    def linearize(self):
        """Return the linearized system.

        Returns
        -------
        a : ndarray
            The state matrix.
        b : ndarray
            The action matrix.

        """
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        A = np.array([[0, 1],
                      [gravity / length, -friction / inertia]],
                     dtype=config.np_dtype)

        B = np.array([[0],
                      [1 / inertia]],
                     dtype=config.np_dtype)

        if self.normalization is not None:
            Tx, Tu = map(np.diag, self.normalization)
            Tx_inv, Tu_inv = map(np.diag, self.inv_norm)

            A = np.linalg.multi_dot((Tx_inv, A, Tx))
            B = np.linalg.multi_dot((Tx_inv, B, Tu))

        sys = signal.StateSpace(A, B, np.eye(2), np.zeros((2, 1)))
        sysd = sys.to_discrete(self.dt)
        return sysd.A, sysd.B

    def forward(self, state_action):
        """Evaluate the dynamics."""
        # Denormalize
        # split the state_action into state and action, 
        # the first two column are state, the last column is action
        state, action = np.split(state_action, [2], axis=1)

        # state, action = np.split(state_action, [2], axis=0) 
        
        state, action = self.denormalize(state, action)

        n_inner = 10
        dt = self.dt / n_inner
        for i in range(n_inner):
            state_derivative = self.ode(state, action)
            state = state + dt * state_derivative

        return self.normalize(state, None)[0]

    def ode(self, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        states: ndarray or Tensor
            Unnormalized states.
        actions: ndarray or Tensor
            Unnormalized actions.

        Returns
        -------
        x_dot: Tensor
            The normalized derivative of the dynamics

        """
        # Physical dynamics
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        angle, angular_velocity = np.split(state, [1], axis=-1)
        x_ddot = gravity / length * np.sin(angle) + action / inertia

        if friction > 0:
            x_ddot -= friction / inertia * angular_velocity

        state_derivative = np.concatenate((angular_velocity, x_ddot), axis=-1)

        # Normalize
        return state_derivative
    
def compute_roa_pendulum(grid, closed_loop_dynamics, horizon=100, tol=1e-3, equilibrium=None, no_traj=True):
    """Compute the largest ROA as a set of states in a discretization."""
    if isinstance(grid, np.ndarray):
        all_points = grid
        nindex = grid.shape[0]
        ndim = grid.shape[1]
    else: # grid is a GridWorld instance
        all_points = grid.all_points
        nindex = grid.nindex
        ndim = grid.ndim

    # Forward-simulate all trajectories from initial points in the discretization
    if no_traj:
        end_states = all_points
        for t in range(1, horizon):
            end_states = closed_loop_dynamics(end_states)
    else:
        trajectories = np.empty((nindex, ndim, horizon))
        trajectories[:, :, 0] = all_points
        for t in range(1, horizon):
            # simulate all states in the grid
            trajectories[:, :, t] = closed_loop_dynamics(trajectories[:, :, t - 1])
               
        end_states = trajectories[:, :, -1]

    if equilibrium is None:
        equilibrium = np.zeros((1, ndim))

    # Compute an approximate ROA as all states that end up "close" to 0
    dists = np.linalg.norm(end_states - equilibrium, ord=2, axis=1, keepdims=True).ravel()
    roa = (dists <= tol)
    if no_traj:
        return roa
    else:
        return roa, trajectories