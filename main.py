from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 6 * np.sin(np.pi * x) * np.cos(4 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    prod = np.outer(x_i,x_j) #np.outer allows us to compute the outer product of x_i and x_j
    kernel = (prod+1)**d #computing the rest of the kernel
    return kernel
    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    diff = np.subtract.outer(x_i, x_j) #np.subtract.outer again allows to compute outer subtractions
    exp_term = -gamma*((diff)**2) #computing the term that will be raised to exponential function
    kernel = np.exp(exp_term) #obtaining the kernel by raising the exp_term to exp()
    return kernel
    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    #we know alpha = argmin_alpha ||K*alpha - Y||^2 + lambda* alpha^T *K*alpha
    #thus alpha = (K^T * K + lambda*K)^(-1)* K^T*y
    kernel = kernel_function(x,x, kernel_param) #using kernel_function to compute K
    #computing terms of equation for alpha below
    kt = np.transpose(kernel)
    ktk = np.matmul(kt, kernel)
    lk = _lambda*kernel
    kty = np.matmul(kt, y)

    alpha = np.matmul(np.linalg.inv(ktk + lk),kty) #computing alpha
    return alpha

    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across all folds.
    """
    #we are performing LOO CV

    fold_size = len(x) // num_folds #creating a variable for fold size which will be useful for CV (taken from last homework)
    losses = np.zeros(num_folds) #creating a numpy array of zeros

    for fold in range(num_folds): #even though we are using LOO CV, just to ensure, we have no issues
        vals1 = fold*fold_size #the start index for the validation set in each fold
        vals2 = vals1 + fold_size #the end index for the validation set in each fold

        x_val = x[vals1:vals2] #obtaining the validation x data for each fold
        y_val = y[vals1:vals2] #obtaining the validation y data for each fold
        x_train = np.concatenate([x[:vals1], x[vals2:]], axis=0) #the rest of the data excluding the validation x will be training set using np.concatenate
        y_train = np.concatenate([y[:vals1], y[vals2:]], axis=0) #the rest of the targets excluding the validation y will be training target

        alpha = train(x_train, y_train, kernel_function, kernel_param, _lambda) #training our model with the kernel function and lambda
        kernel = kernel_function(x_train,x_val,kernel_param) #obtaining the kernel from this data too
        y_pred = alpha @ kernel #making predictions with alpha and kernel according to PDF

        mse = np.mean((y_pred - y_val)**2) #computing mse
        losses[fold] = mse #appending ith mse to losses array

    mean_loss = np.mean(losses)
    return mean_loss

    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be len(x) for LOO.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / (median(dist(x_i, x_j)^2) for all unique pairs x_i, x_j in x
            should be sufficient for this problem (where unique means i != j). That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    #we will use grid search to find the optimal hyperparameters
    lambdas = np.linspace(-5,-1) #creating a linspace array for lambdas we will be finding optimal hyperparameter of
    lam_losses = np.zeros(lambdas.size) #creating an array of losses with each lambda
    
    # we now obtain the gamma according to the formula given above
    n = len(x)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):  #i+1 ensures i != j and avoids duplicates
            sq_dist = (x[i] - x[j]) ** 2
            dists.append(sq_dist)

    gamma = 1/np.median(dists)

    for i in range(lambdas.size): #now searching for optimal lambda with grid search and LOO CV
        lam = 10**lambdas[i]
        loss = cross_validation(x,y, rbf_kernel, gamma, lam, num_folds)
        lam_losses[i] = loss

    i_star = np.argmin(lam_losses) #using np.argmin to obtain which index for lambda yielded the lowest loss
    lam_star = lambdas[i_star] #retrieving that lambda
    return(lam_star,gamma) #returning lambda and gamma in a tuple

    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution [5, 6, ..., 24, 25]
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [5, 6, ..., 24, 25]
    """
    #we again will use grid search 
    lambdas = np.linspace(-5,-1) #creating a linspace array for lambdas we will be finding optimal hyperparameter of
    ds = list(range(5,26)) #creating a list of ds we will be finding optimal hyperparameter of

    losses = np.zeros((lambdas.size,len(ds))) #creating a 2d array of zeros based on number of lambdas and ds

    for i in range(lambdas.size):  #now searching for optimal lambda and d with grid search and LOO CV
        lam = 10**lambdas[i]
        for j in range(len(ds)):
            d = ds[j]
            loss = cross_validation(x,y, poly_kernel, d, lam, num_folds)
            losses[i,j] = loss

    i1,i2 = np.unravel_index(np.argmin(losses), losses.shape) #using np.unravel_index to obtain both indices of the lowest loss 
    #np.unravel_index source: https://numpy.org/devdocs/reference/generated/numpy.unravel_index.html
    lam_star = 10**lambdas[i1] #obtaining best lambda
    d_star = ds[i2] #obtaining best d

    return (lam_star,d_star) #returning best lambda and d pair
    raise NotImplementedError("Your Code Goes Here")

@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
            Note that x_30, y_30 has been loaded in for you. You do not need to use (x_300, y_300) or (x_1000, y_1000).
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")

    #part (a)
    #now performing hyperparameter search for the RBF kernel to find its best lambda and gamma
    rbf_hyp = rbf_param_search(x_30,y_30,len(x_30))
    #print(f'RBF Optimal Hyperparamters: {rbf_hyp}')
    print(f'RBF Optimal Lambda =  {rbf_hyp[0]}')
    print(f'RBF Optimal Gamma = {rbf_hyp[1]}')
    lam_rbf = rbf_hyp[0]
    gamma_rbf = rbf_hyp[1]

    #now performing hyperparameter search for the Poly kernel to find its best lambda and d
    poly_hyp = poly_param_search(x_30,y_30,len(x_30))
    #print(f'Poly Optimal Hyperparamters: {poly_hyp}')
    print(f'Poly Optimal Lambda = {poly_hyp[0]}')
    print(f'Poly Optimal d = {poly_hyp[1]}')
    lam_poly = poly_hyp[0]
    d_poly = poly_hyp[1]

    #part (b)
    #training the optimal rbf kernel model with the x_30, y_30 data
    rbf_alpha30 = train(x_30,y_30,rbf_kernel,gamma_rbf,lam_rbf)
    #obtaining the specific rbf kernel for this data set
    rbf_kernel30 = rbf_kernel(x_30,x_30,gamma_rbf)
    #computing predictions with the training RBF kernel model
    rbf_pred30 = rbf_alpha30 @ rbf_kernel30

    #training the optimal rbf kernel model with the x_30, y_30 data
    poly_alpha30 = train(x_30,y_30,poly_kernel,d_poly,lam_poly)
    #obtaining the specific rbf kernel for this data set
    poly_kernel30 = poly_kernel(x_30,x_30,d_poly)
    #computing predictions with the training RBF kernel model
    poly_pred30 = poly_alpha30 @ poly_kernel30

    #plotting the the true function f along with the RBF kernel functions predictions for x_30,y_30
    plt.figure(1, figsize = (10,5))
    plt.scatter(x_30,y_30)
    plt.plot(np.sort(x_30),np.sort(f_true(x_30)), label = "true f")
    plt.plot(np.sort(x_30),np.sort(rbf_pred30), label = 'RBF kernel')
    plt.ylim(-6,6)
    plt.legend()

    #plotting the the true function f along with the Poly kernel functions predictions for x_30,y_30
    plt.figure(2, figsize = (10,5))
    plt.scatter(x_30,y_30)
    plt.plot(np.sort(x_30),np.sort(f_true(x_30)), label = "true f")
    plt.plot(np.sort(x_30),np.sort(poly_pred30), label = 'Poly Kernel')
    plt.ylim(-6,6)
    plt.legend()
    plt.show()

    return None
    raise NotImplementedError("Your Code Goes Here")

if __name__ == "__main__":
    main()
