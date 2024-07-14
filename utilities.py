
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

def compute_roa(grid, env_func, ctrl ,equilibrium=None, no_traj=True):
    """Compute the largest ROA as a set of states in a discretization."""
    if isinstance(grid, np.ndarray):
        all_points = grid
        nindex = grid.shape[0]
        ndim = grid.shape[1]
    else: # grid is a GridWorld instance
        all_points = grid.all_points
        nindex = grid.nindex # number of points in the discretization
        ndim = grid.ndim  # dimension of the state space

    # Forward-simulate all trajectories from initial points in the discretization
    # if no_traj:
    #     end_states = all_points
    #     for t in range(1, horizon):
    #         end_states = closed_loop_dynamics(end_states)
    # else:
    #     trajectories = np.empty((nindex, ndim, horizon))
    #     trajectories[:, :, 0] = all_points
    #     for t in range(1, horizon):
    #         trajectories[:, :, t] = closed_loop_dynamics(trajectories[:, :, t - 1])
    #     end_states = trajectories[:, :, -1]
    random_env = env_func(gui=False)

    roa = np.zeros((nindex))
    trajectories = [{} for _ in range(nindex)]

    for state_index in range(nindex):
        # for all initial state in the grid
        # print('state_index', state_index)
        init_state = grid.all_points[state_index]
        init_state_dict = {'init_x': init_state[0], 'init_x_dot': init_state[1], \
                            'init_theta': init_state[2], 'init_theta_dot': init_state[3]}
        init_state, _ = random_env.reset(init_state = init_state_dict)
        # print('init_state', init_state)
        static_env = env_func(gui=False, random_state=False, init_state=init_state)
        static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)
        # Create experiment, train, and run evaluation
        experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
        
        try:
            trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1, verbose=False)
            roa[state_index] = trajs_data['info'][-1][-1]['goal_reached']
            input_traj = trajs_data['action'][0]
            state_traj = trajs_data['obs'][0]
            trajectories[state_index]['state_traj'] = state_traj
            trajectories[state_index]['input_traj'] = input_traj
            print('trajectory[state_index]', trajectories[state_index])
            
            print('goal reached', trajs_data['info'][-1][-1]['goal_reached'])
            # exit()
            # close environments
            static_env.close()
            static_train_env.close()
        except RuntimeError:
            print('RuntimeError: possibly infeasible initial state')
            roa[state_index] = False
            # print(ctrl.model.__dir__())
            # print(ctrl.model.nx)
            # exit()
            trajectories[state_index]['state_traj'] = np.zeros((2, ctrl.model.nx))
            trajectories[state_index]['input_traj'] = np.zeros((1, ctrl.model.nu))
            # close environments
            static_env.close()
            static_train_env.close()
            continue
        # trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1, verbose=False)
        # print('obs\n', trajs_data['obs'])
        # print('trajs_data\n', trajs_data['info'][-1][-1])
        # print('\n')
        # print('trajs_data[\'info\']\n', trajs_data['info'][-1][-1]['goal_reached'])
        # input('press enter to continue')
        # print('\n')
        # exit()
        # print('goal reached', trajs_data['info'][-1][1]['goal_reached'])
        

    # if equilibrium is None:
    #     equilibrium = np.zeros((1, ndim))
    random_env.close()
    # # Compute an approximate ROA as all states that end up "close" to 0
    # dists = np.linalg.norm(end_states - equilibrium, ord=2, axis=1, keepdims=True).ravel()
    # roa = (dists <= tol)
    if no_traj:
        return roa
    else:
        return roa, trajectories



def compute_roa_fix(grid, env_func, ctrl ,equilibrium=None, no_traj=True):
    """Compute the largest ROA as a set of states in a discretization."""
    if isinstance(grid, np.ndarray):
        all_points = grid
        nindex = grid.shape[0]
        ndim = grid.shape[1]
    else: # grid is a GridWorld instance
        all_points = grid.all_points
        nindex = grid.nindex # number of points in the discretization
        ndim = grid.ndim  # dimension of the state space

    random_env = env_func(gui=False)

    roa = np.zeros((nindex))
    
    for state_index in range(nindex):
        # for all initial state in the grid
        # print('state_index', state_index)
        init_state = grid.all_points[state_index]
        init_state_dict = {'init_x': 0.0, 'init_x_dot': init_state[0], \
                            'init_theta': init_state[1], 'init_theta_dot': init_state[2]}
        init_state, _ = random_env.reset(init_state = init_state_dict)
        # print('init_state', init_state)
        static_env = env_func(gui=False, random_state=False, init_state=init_state)
        static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)
        # Create experiment, train, and run evaluation
        experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)

        try:
            trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1, verbose=False)
            roa[state_index] = trajs_data['info'][-1][-1]['goal_reached']
            # close environments
            static_env.close()
            static_train_env.close()
        except RuntimeError:
            print('RuntimeError: possibly infeasible initial state')
            roa[state_index] = False
            # close environments
            static_env.close()
            static_train_env.close()
            continue    

    # if equilibrium is None:
    #     equilibrium = np.zeros((1, ndim))
    random_env.close()
    # # Compute an approximate ROA as all states that end up "close" to 0
    # dists = np.linalg.norm(end_states - equilibrium, ord=2, axis=1, keepdims=True).ravel()
    # roa = (dists <= tol)
    if no_traj:
        return roa
    else:
        return roa, trajectories


# define the function to be parallelized
def simulate_at_index(state_index, grid, env_func, ctrl):
    random_env = env_func(gui=False)
    init_state = grid.all_points[state_index]
    init_state_dict = {'init_x': init_state[0], 'init_x_dot': init_state[1], \
                        'init_theta': init_state[2], 'init_theta_dot': init_state[3]}
    init_state, _ = random_env.reset(init_state = init_state_dict)
    # print('init_state', init_state)
    static_env = env_func(gui=False, random_state=False, init_state=init_state)
    static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)
    # Create experiment, train, and run evaluation
    experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)

    # # if infeasible initial state, return False
    # try:
    #     trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1, verbose=False)
    #     static_env.close()
    #     static_train_env.close()
    #     print('goal reached', trajs_data['info'][-1][-1]['goal_reached'])
    #     # return result
    #     if trajs_data['info'][-1][-1]['goal_reached']:
    #         return True
    #     else:
    #         return False 
    # except RuntimeError:
    #     print('RuntimeError: possibly infeasible initial state')
    #     # close environments
    #     static_env.close()
    #     static_train_env.close()
    #     return False
    # # close the env
    trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1, verbose=False)
    static_env.close()
    static_train_env.close()
    random_env.close()

    return trajs_data['info'][-1][-1]['goal_reached']
               

def compute_roa_par(grid, env_func, ctrl, equilibrium=None, no_traj=True):
    """Compute the largest ROA as a set of states in a discretization."""
    if isinstance(grid, np.ndarray):
        all_points = grid
        nindex = grid.shape[0]
        ndim = grid.shape[1]
    else: # grid is a GridWorld instance
        all_points = grid.all_points
        nindex = grid.nindex # number of points in the discretization
        ndim = grid.ndim  # dimension of the state space

    # Forward-simulate all trajectories from initial points in the discretization
    # random_env = env_func(gui=False)
    roa = [False] * nindex

    # # init multiprocessing pool
    # pool = mp.Pool(mp.cpu_count())
    # # pool apply the 'simulate_at_index' function to all state indices
    # roa = [pool.apply(simulate_at_index, \
    #                   args=(state_idx, grid, random_env, env_func, ctrl)) for state_idx in range(nindex)]
    # # close the pool
    # pool.close()
    # roa = pmap(simulate_at_index, range(nindex), (grid, random_env, env_func, ctrl))
    roa = pmap(simulate_at_index, range(nindex), (grid, env_func, ctrl))
    
    # convert list to np array
    roa = np.array(roa)

    if no_traj:
        return roa
    else:
        return roa, trajectories

# define the function to be parallelized
def simulate_at_index_fix(state_index, grid, env_func, ctrl):
    random_env = env_func(gui=False)
    init_state = grid.all_points[state_index]
    init_state_dict = {'init_x': 0.0, 'init_x_dot': init_state[0], \
                        'init_theta': init_state[1], 'init_theta_dot': init_state[2]}
    init_state, _ = random_env.reset(init_state = init_state_dict)
    # print('init_state', init_state)
    static_env = env_func(gui=False, random_state=False, init_state=init_state)
    static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)
    # Create experiment, train, and run evaluation
    experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)

    trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1, verbose=False)
    static_env.close()
    static_train_env.close()
    random_env.close()

    return trajs_data['info'][-1][-1]['goal_reached']
               

def compute_roa_fix_par(grid, env_func, ctrl, equilibrium=None, no_traj=True):
    """Compute the largest ROA as a set of states in a discretization."""
    if isinstance(grid, np.ndarray):
        all_points = grid
        nindex = grid.shape[0]
        ndim = grid.shape[1]
    else: # grid is a GridWorld instance
        all_points = grid.all_points
        nindex = grid.nindex # number of points in the discretization
        ndim = grid.ndim  # dimension of the state space

    # Forward-simulate all trajectories from initial points in the discretization
    roa = [False] * nindex
    roa = pmap(simulate_at_index_fix, range(nindex), (grid, env_func, ctrl))
    # convert list to np array
    roa = np.array(roa)

    if no_traj:
        return roa
    else:
        return roa, trajectories
    
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

def get_discrete_linear_system_matrices(model, x_0, u_0):
    '''Get discrete linear system matrices for a given model.

    Args:
        model (ctrl.model)
        x_0 (ndarray): The initial state.
        u_0 (ndarray): The initial input.

    Returns:
        A (ndarray): The discrete linear state matrix A.
        B (ndarray): The discrete linear input matrix B.
    '''

    # Linearization.
    df = model.df_func(x_0, u_0)
    A, B = df[0].toarray(), df[1].toarray()

    # Discretize.
    A, B = discretize_linear_system(A, B, model.dt)

    return A, B

def onestep_dynamics(x, env_func, ctrl):
    ''' one-step forward dynamics '''
    # get the format of the initial state
    random_env = env_func(gui=False)
    init_state_dict = {'init_x': x[0], 'init_x_dot': x[1], \
                        'init_theta': x[2], 'init_theta_dot': x[3]}
    init_state, _ = random_env.reset(init_state = init_state_dict)
    static_env = env_func(gui=False, random_state=False, init_state=init_state)
    static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)
    experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
    trajs_data, _ = experiment.run_evaluation(training=False, n_steps=1, verbose=False)
    x = trajs_data['obs'][0][-1]
    static_env.close()
    static_train_env.close()
    random_env.close()

    return x  



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
        # if isinstance(Tx_inv, np.ndarray):
        #     Tx_inv = torch.from_numpy(Tx_inv)
        # if isinstance(Tu_inv, np.ndarray):
        #     Tu_inv = torch.from_numpy(Tu_inv)
        # state = tf.matmul(state, Tx_inv)
        # state = torch.matmul(state, Tx_inv)
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

        # state = tf.matmul(state, Tx)
        # convert to torch
        # if isinstance(Tx, np.ndarray):
        #     Tx = torch.from_numpy(Tx)
        # if isinstance(Tu, np.ndarray):
        #     Tu = torch.from_numpy(Tu)

        # state = torch.matmul(state, Tx)
        state = np.matmul(state, Tx)
        if action is not None:
            # action = tf.matmul(action, Tu)
            # action = torch.matmul(action, Tu)
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

    # @concatenate_inputs(start=1)
    def forward(self, state_action):
        """Evaluate the dynamics."""
        # Denormalize
        # state, action = tf.split(state_action, [2, 1], axis=1)
        # state, action = torch.split(state_action, [2, 1], dim=0)
        # print('np.split(state_action, [2, 1], axis=0)', np.split(state_action, [2], axis=0))
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

        # angle, angular_velocity = tf.split(state, 2, axis=1)
        # print('state', state)
        # print('split result', torch.split(state, 1, dim=0))
        # print('np.split(state, [1], axis=0)', np.split(state, [1], axis=-1))
        # angle, angular_velocity = torch.split(state, 1, dim=-1)
        angle, angular_velocity = np.split(state, [1], axis=-1)

        # x_ddot = gravity / length * tf.sin(angle) + action / inertia
        # x_ddot = gravity / length * torch.sin(angle) + action / inertia
        x_ddot = gravity / length * np.sin(angle) + action / inertia

        if friction > 0:
            x_ddot -= friction / inertia * angular_velocity

        # state_derivative = tf.concat((angular_velocity, x_ddot), axis=1)
        # state_derivative = torch.cat((angular_velocity, x_ddot), dim=-1)
        state_derivative = np.concatenate((angular_velocity, x_ddot), axis=-1)

        # Normalize
        return state_derivative
    
    def _setup_symbolic(self, prior_prop={}, **kwargs):
        """Setup the casadi symbolic dynamics."""
        length = self.length
        gravity = self.gravity
        mass = self.mass
        friction = self.friction
        inertia = self.inertia # mass * length ** 2
        dt = self.dt
        # Input variables.
        theta = cs.MX.sym('theta')
        theta_dot = cs.MX.sym('theta_dot')
        X = cs.vertcat(theta, theta_dot)
        U = cs.MX.sym('u')
        nx = 2
        nu = 1
        # Dynamics.
        theta_ddot = gravity / length * cs.sin(theta) + U / inertia
        if friction > 0:
            theta_ddot -= friction / inertia * theta_dot
        X_dot = cs.vertcat(theta_dot, theta_ddot)
        # Observation.
        Y = cs.vertcat(theta, theta_dot)
        # Define cost (quandratic form).
        Q = cs.MX.sym('Q', nx, nx)
        R = cs.MX.sym('R', nu, nu)
        Xr = cs.MX.sym('Xr', nx, 1)
        Ur = cs.MX.sym('Ur', nu, 1)
        cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        # Define dynamics and cost dictionaries.
        dynamics = {'dyn_eqn': X_dot, 'obs_eqn': Y, 'vars': {'X': X, 'U': U}}
        cost = {'cost_func': cost_func, 'vars': {'X': X, 'U': U, 'Xr': Xr, 'Ur': Ur, 'Q': Q, 'R': R}}
        params = {
            # prior inertial properties
            'pole_length': length,
            'pole_mass': mass,
            # equilibrium point for linearization
            'X_EQ': np.zeros(self.nx),
            'U_EQ': np.atleast_2d(Ur)[0, :],
        }
         # Setup symbolic model.
        self.symbolic = SymbolicModel(dynamics=dynamics, cost=cost, dt=dt, params=params)

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
            # print('trajectories[:, :, t - 1]', trajectories[1, :, t - 1])
            # print('trajectories[:, :, t - 1].shape', trajectories[1, :, t - 1].shape)
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