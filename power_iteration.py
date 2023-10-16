import numpy as np


def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    b = np.random.rand(data.shape[0])
    b = b / np.linalg.norm(b)

    for i in range(num_steps):
        b = data @ b
        b = b / np.linalg.norm(b)

    eigenvalue = (b @ data @ b) / (b @ b)

    return float(eigenvalue), b