import numpy as np

def _cov2cor(cov):
    """
    Convert a covariance matrix to a correlation matrix.
    """
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def _schafer_strimmer_cov(x):
    n, p = x.shape

    # Compute the sample covariance matrix
    sMat = np.dot(x.T, x) / n

    # Compute the diagonal matrix of variances
    tMat = np.diag(np.diag(sMat))

    # Standardize x
    std_devs = np.sqrt(np.diag(sMat))
    xscale = x / std_devs

    # Compute correlation matrices
    rSmat = _cov2cor(sMat)
    rTmat = _cov2cor(tMat)

    # Compute numerator: varSij
    xscale_sq = xscale ** 2
    crossprod_xscale_sq = np.dot(xscale_sq.T, xscale_sq)
    crossprod_xscale = np.dot(xscale.T, xscale)
    varSij = (1 / (n * (n - 1))) * (crossprod_xscale_sq - (1 / n) * (crossprod_xscale ** 2))

    # Set diagonal to 0
    np.fill_diagonal(varSij, 0)

    # Compute denominator: sqSij
    sqSij = (rSmat - rTmat) ** 2

    # Compute lambda star
    lambda_star = np.sum(varSij) / np.sum(sqSij)
    lambda_star = np.clip(lambda_star, 0, 1)

    # Compute shrinkage covariance matrix
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
