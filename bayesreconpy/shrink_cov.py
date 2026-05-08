import numpy as np

def _cov2cor(cov):
    d = np.sqrt(np.diag(cov))
    eps = 1e-6  # Small regularization value
    d = np.where(d == 0, eps, d)  # Replace zero standard deviations with a small value
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def _schafer_strimmer_cov(x):
    n, p = x.shape

    sMat = np.dot(x.T, x) / n
    tMat = np.diag(np.diag(sMat))
    std_devs = np.sqrt(np.diag(sMat))
    eps = 1e-6  # Regularization value to avoid division by zero
    std_devs = np.where(std_devs == 0, eps, std_devs)  # Regularize std_devs to avoid zero
    xscale = x / std_devs

    rSmat = _cov2cor(sMat)
    rTmat = _cov2cor(tMat)

    xscale_sq = xscale ** 2
    crossprod_xscale_sq = np.dot(xscale_sq.T, xscale_sq)
    crossprod_xscale = np.dot(xscale.T, xscale)
    varSij = (1 / (n * (n - 1))) * (crossprod_xscale_sq - (1 / n) * (crossprod_xscale ** 2))
    np.fill_diagonal(varSij, 0)

    sqSij = (rSmat - rTmat) ** 2
    lambda_star = np.sum(varSij) / np.sum(sqSij)
    lambda_star = np.clip(lambda_star, 0, 1)

    shrink_cov = lambda_star * tMat + (1 - lambda_star) * sMat
    return {'shrink_cov': shrink_cov, 'lambda_star': lambda_star}



# Generate synthetic data
    #np.random.seed(0)  # For reproducibility
    #n = 100  # Number of samples
    #p = 5  # Number of variables
    #x = np.random.randn(n, p)

    # Compute covariance and shrinkage using the function
    #result = schafer_strimmer_cov(x)

    #print("Shrinkage Covariance Matrix:")
    #print(result['shrink_cov'])

    #print("\nLambda Star:")
    #print(result['lambda_star'])
