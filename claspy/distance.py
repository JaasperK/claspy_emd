import numpy as np
from numba import jit, njit, objmode, types

from scipy.optimize import linprog


@njit(fastmath=True, cache=True)
def sliding_mean_std(time_series, window_size):
    """
    Calculate sliding mean and standard deviation of a time series.
    The sliding mean and standard deviation are calculated by computing
    the mean and standard deviation over a sliding window of fixed size,
    which is moved over the time series with a stride of one element at
    a time.

    Parameters
    ----------
    time_series : array-like of shape (n,)
        The time series sequence.
    window_size : int
        The size of the sliding window.

    Returns
    -------
    movmean : ndarray of shape (n - window_size + 1,)
        The sliding mean of the time series.
    movstd : ndarray of shape (n - window_size + 1,)
        The sliding standard deviation of the time series.

    Notes
    -----
    This function calculates the sliding mean and standard deviation of
    the input time series using a sliding window approach. It first computes
    the cumulative sum and cumulative sum of squares of the time series, then
    computes the window sum and window sum of squares for each sliding window.
    Finally, it computes the mean and standard deviation over each window and
    returns the results.

    Examples
    --------
    >>> time_series = [1, 2, 3, 4, 5, 6, 7]
    >>> window_size = 3
    >>> movmean, movstd = sliding_mean_std(time_series, window_size)
    """
    s = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(time_series)))
    sSq = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(time_series ** 2)))

    segSum = s[window_size:] - s[:-window_size]
    segSumSq = sSq[window_size:] - sSq[:-window_size]

    movmean = segSum / window_size

    movstd = np.sqrt(np.clip(segSumSq / window_size - (segSum / window_size) ** 2, 0, None))
    movstd = np.where(np.abs(movstd) < 1e-3, 1, movstd)

    return [movmean, movstd]


@njit(fastmath=True, cache=True)
def znormed_euclidean_distance(idx, dot, window_size, preprocessing, squared=True):
    """
    Computes the z-normalized Euclidean distance between a time series subsequence at index `idx`
    and all other ones (of length window size) using the `dot` product.

    Parameters:
    -----------
    idx: int
        The index of the subsequence.
    dot: int
        The dot products between the subsequence at `idx` and all other ones (of length window size).
    window_size: int
        The window size of the subsequences.
    preprocessing: tuple
        A tuple of two NumPy arrays (means and stds), containing the means and standard deviations of
        the subsequences used for normalization.
    squared: bool, default=True
        If True, the squared distance is returned. If False, the non-squared distance is returned.

    Returns:
    --------
    dist : ndarray
        The z-normalized Euclidean distances between the subsequence at index `idx` and all other subsequences.
    """
    means, stds = preprocessing
    dist = 2 * window_size * (1 - (dot - window_size * means * means[idx]) / (window_size * stds * stds[idx]))
    if squared is True: return dist
    return np.sqrt(dist)


@njit(fastmath=True, cache=True)
def sliding_csum(time_series, window_size):
    """
    Computes the sliding cumulative sum of squares of a time series with a specified window size.

    Parameters:
    -----------
    time_series: numpy.ndarray
        A 1-dimensional numpy array containing the time series data.
    window_size: int
        The size of the sliding window.

    Returns:
    --------
    csumsq_diff: numpy.ndarray
        A 1-dimensional numpy array containing the difference between the sliding cumulative sum of
        squares of the time series with the current window and that with the previous window.
    """
    csumsq = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(time_series ** 2)))
    return csumsq[window_size:] - csumsq[:-window_size]


@njit(fastmath=True, cache=True)
def euclidean_distance(idx, dot, window_size, csumsq, squared=True):
    """
    Computes the Euclidean distance between a time series subsequence at index `idx` and all other ones
    (of length window size) using the `dot` product and the precomputed cumulative sum of squares `csumsq`.

    Parameters:
    -----------
    idx: int
        The index of the subsequence.
    dot: float
        The dot products between the subsequence at `idx` and all other ones (of length window size).
    window_size: int
        The window size of the subsequences.
    csumsq: numpy.ndarray
        A 1-D NumPy array containing the cumulative sum of squares of the time series.
    squared: bool, default=True
        If True, the squared distance is returned. If False, the non-squared distance is returned.

    Returns:
    --------
    dist : ndarray
        The Euclidean distances between the subsequence at index `idx` and all other subsequences.
    """
    dist = -2 * dot + csumsq + csumsq[idx]
    if squared is True: return dist
    return np.sqrt(dist)


@njit(fastmath=True, cache=True)
def sliding_csum_dcsum(time_series, window_size):
    """
    Computes the sliding cumulative sum of squares, the sliding cumulative sum of squared
    differences between consecutive elements, and the sliding mean and standard deviation of the time series.

    Parameters
    ----------
    time_series : array-like of shape (n,)
        The input time series sequence.
    window_size : int
        The size of the sliding window.

    Returns
    -------
    csumsq_diff : ndarray of shape (n - window_size + 1,)
        The sliding sum of squares over the window.
    dcsumsq_diff : ndarray of shape (n - window_size + 1,)
        The sliding sum of squared differences between consecutive elements over the window,
        with a small constant (1e-5) added to prevent division by zero.
    means : ndarray of shape (n - window_size + 1,)
        The sliding mean of the time series over the window.
    stds : ndarray of shape (n - window_size + 1,)
        The sliding standard deviation of the time series over the window.
    """
    means, stds = sliding_mean_std(time_series, window_size)
    csumsq = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(time_series ** 2)))
    dcsumsq = np.concatenate((np.zeros(2, dtype=np.float64), np.cumsum((time_series[1:] - time_series[:-1]) ** 2)))
    return [csumsq[window_size:] - csumsq[:-window_size], dcsumsq[window_size:] - dcsumsq[:-window_size] + 1e-5, means,
            stds]


@njit(fastmath=True, cache=True)
def cinvariant_euclidean_distance(idx, dot, window_size, preprocessing, squared=True):
    """
    Computes the complexity-invariant Euclidean distance between a time series subsequence at index `idx`
    and all other subsequences (of length `window_size`) using the dot product and precomputed statistics.

    Parameters
    ----------
    idx : int
        The index of the subsequence in the time series.
    dot : numpy.ndarray
        The dot products between the subsequence at index `idx` and all other subsequences.
    window_size : int
        The length of the subsequences.
    preprocessing : tuple
        A tuple containing precomputed arrays required for the calculation.
    squared : bool, default=True
        If `True`, returns the squared distance. If `False`, returns the non-squared distance.

    Returns
    -------
    dist : ndarray
        The complexity-invariant Euclidean distances between the subsequence at index `idx` and all other subsequences.
    """
    csumsq, ce, means, stds = preprocessing

    ed = -2 * dot + csumsq + csumsq[idx]
    # zed = 2 * window_size * (1 - (dot - window_size * means * means[idx]) / (window_size * stds * stds[idx]))
    curr_ce = np.repeat(ce[idx], ce.shape[0])

    with objmode(cf="float64[:]"):
        cf = (np.max(np.dstack((ce, curr_ce)), axis=2) / np.min(np.dstack((ce, curr_ce)), axis=2))[0]

    if squared is True: return ed * cf
    return np.sqrt(ed) * np.sqrt(cf)


@njit(fastmath=True, cache=True)
def sliding_csum_abs(time_series, window_size):
    """
    Computes the sliding cumulative sum of absolute values of each time series subsequence
    with a specified window size.

    Parameters
    ----------
    time_series: numpy.ndarray
        A 1-dimensional numpy array containing the time series data.
    window_size: int
        The size of the sliding window.

    Returns
    -------
    time_series : numpy.ndarray
        A 1-dimensional numpy array containing the time series data.
    csum_abs : numpy.ndarray
        A 1-dimensional numpy array containing the cumulative sum of absolute values of each
        time series subsequence.
    """
    csum = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(np.abs(time_series))))
    return (time_series, csum[window_size:] - csum[:-window_size])


@njit(cache=True)
def cost_vector(window_size, supplier_idx, consumer_idx):
    """
    Setup the cost vector (the vector of coefficients of the objective function) of the
    linear programming problem.

    Parameters
    ----------
    window_size : int
        Size of the sliding window.
    supplier_idx : int
        Index of the subsequence that 
    """
    c = np.zeros((window_size * window_size) + window_size, dtype=np.float64)
    for i in range(window_size):
        for j in range(window_size):
            c[i * window_size + j] = np.abs(supplier_idx + i - (consumer_idx + j))
    return c


@njit("float64[:,:](int64)", cache=True)
def constraint_matrix(window_size):
    """
    Setup the constraint matrix of the linear programming problem. 

    Parameters
    ----------
    window_size : int
        Size of the sliding window.
    
    Returns
    -------
    A : numpy.ndarray
        The constraint matrix of the linear programming problem.
    """
    A_eq = np.zeros((window_size, window_size * window_size + window_size), dtype=np.float64)
    for j in range(window_size):
        for i in range(window_size):
            A_eq[i, j * window_size + i] = 1.0

    A_ub = np.zeros((window_size, window_size * window_size + window_size), dtype=np.float64)
    for i in range(window_size):
        for j in range(window_size):
            A_ub[i, i * window_size + j] = 1.0
        A_ub[i, window_size * window_size + i] = 1.0
    
    return np.vstack((np.ascontiguousarray(A_eq), np.ascontiguousarray(A_ub)))  # numba hack


@njit(cache=True)
def lp_params(idx, jdx, window_size, time_series, csum_abs):
    """
    Setup the cost and solution vectors of the linear programming problem.

    Parameters
    ----------
    idx, jdx : int
        Indices of subsequences in the time series. 
    window_size : int
        Size of the sliding window.
    time_series : numpy.ndarray
        The 1-D array containing the time series data.
    csum_abs : numpy.ndarray
        The 1-D array containing the cumulative sums of absolute values for all
        subsequences in the time series.
    
    Returns
    -------
    c : numpy.ndarray
        The coefficients of the objective function for the linear programming problem.
    b : numpy.ndarray
        The right-hand side of the (in-)equalities for the linear programming problem.
    """
    if csum_abs[idx] >= csum_abs[jdx]:
        supplier_idx = idx  # s1
        consumer_idx = jdx  # s2
    else:
        supplier_idx = jdx  # s1
        consumer_idx = idx  # s2

    c = cost_vector(window_size, supplier_idx, consumer_idx)

    b_eq = np.abs(time_series[consumer_idx : consumer_idx + window_size])
    b_ub = np.abs(time_series[supplier_idx : supplier_idx + window_size])
    b = np.concatenate((b_eq, b_ub))

    return c, b


@njit("float64[:,:](float64[:], float64[:,:], float64[:])", cache=True)
def initial_tableau(c, A, b):
    """
    Setup the initial tableau for the tableau simplex.

    Parameters
    ----------
    c : numpy.ndarray
        Coefficients of the objective function, a.k.a. the cost vector.
    A : numpy.ndarray
        Constraint matrix of the linear programming problem.
    b : numpy.ndarray
        Right-hand side of the constraints for the linear programming problem.
    """
    t = np.vstack((-np.ascontiguousarray(c).reshape(1, c.size), A))  # numba hack
    b = np.concatenate((np.zeros(1), b))
    return np.column_stack((t, b.reshape(b.size, 1)))


@njit("float64(float64[:], float64[:,:], float64[:])", fastmath=True, cache=True)
def simplex(c, A, b):
    """
    Implementation of the tableau simplex algorithm. Solves linear programming
    problems in standard form, meaning all inequalities have been converted to
    equalities and c and A contain the logical/slack variables. The problem can
    be formalized like this:
    ```
    max z = c @ x
    s.t. Ax = b
    where x >= 0
    ``` 
    
    Parameters
    ----------
    c : np.ndarray
        Coefficients of the objective function.
    A : np.ndarray
        Constraint matrix.
    b : np.ndarray
        Right-hand side of constraints.

    Returns
    -------
    z : float
        Value of the objective function.
    """
    tab = initial_tableau(c, A, b)
    tab_rows = np.arange(tab.shape[0])

    while np.any(tab[0] < 0.0):  # solution is not optimal
        pivot_col = np.argmin(tab[0, :-1])
        r1 = tab[1:, pivot_col]
        r1 = np.where(r1 <= 0, -1, r1)  # avoid division by 0
        r2 = tab[1:, -1]
        ratio = np.where(r1 > 0, r2 / r1, np.inf)
        pivot_row = np.argmin(ratio) + 1  # +1 to account for the z-row that was not included in r1 and r2

        # update tableau, probably not necessary since pivot element is always 1 in our special case
        # tab[pivot_row] = tab[pivot_row] / tab[pivot_row, pivot_col]
        
        for i in tab_rows[tab_rows != pivot_row]:
            tab[i] = tab[i] - tab[i, pivot_col] * tab[pivot_row]
    
    return tab[0][-1]


@njit(fastmath=True, cache=True)
def earth_movers_distance(idx, dot, window_size, preprocessing):
    """
    Computes the earth movers distance between a time series subsequence at index `idx` and
    all other subsequences (of length window_size) by solving a transportation problem, a
    form of a linear programming problem.

    Parameters
    ----------
    idx : int
        The index of the subsequence in the time series.
    dot : numpy.ndarray
        The dot products between the subsequence at index `idx` and all other subsequences.
    window_size : int
        The length of the subsequences.
    preprocessing: tuple
        A tuple containing the 1-D NumPy array of the time series data and
        a 1-D NumPy array containing the cumulative sum of absolute values of the time series.

    Returns
    -------
    dist : numpy.ndarray
        The earth movers distances between the subsequence at index `idx` and all other subsequences.
    """
    time_series, csum_abs = preprocessing

    dist = np.zeros(csum_abs.shape[0])
    A = constraint_matrix(window_size)  # the same for each run
    for jdx in range(len(csum_abs)):
        if idx == jdx:
            continue
        c, b = lp_params(idx, jdx, window_size, time_series, csum_abs)
        z = simplex(c, A, b)
        dist[jdx] = z
    return dist

_DISTANCE_MAPPING = {
    "znormed_euclidean_distance": (sliding_mean_std, znormed_euclidean_distance),
    "euclidean_distance": (sliding_csum, euclidean_distance),
    "cinvariant_euclidean_distance": (sliding_csum_dcsum, cinvariant_euclidean_distance),
    "earth_movers_distance" : (sliding_csum_abs, earth_movers_distance)
}


def map_distances(distance_name):
    """
    Computes and returns the distance function and its corresponding preprocessing function, given a distance name.

    Parameters:
    -----------
    distance_name: str
        The name of the distance function to be computed. Available options are "znormed_euclidean_distance"
        and "euclidean_distance".

    Returns:
    --------
    tuple:
        A tuple containing two functions - the preprocessing function and the distance function.
        The preprocessing function takes in a time series and the window size. The distance function takes in
        the index of the subsequence, the dot product between the subsequence and all other subsequences,
        the window size, the preprocessing output, and a boolean flag indicating whether to compute the
        squared distance. It returns the distance between the two subsequences.

    Raises:
    -------
    ValueError:
        If `distance_name` is not a valid distance function name. Valid options are "znormed_euclidean_distance"
        and "euclidean_distance".
    """
    if distance_name not in _DISTANCE_MAPPING:
        raise ValueError(
            f"{distance_name} is not a valid distance. Implementations include: {', '.join(_DISTANCE_MAPPING.keys())}")

    return _DISTANCE_MAPPING[distance_name]
