import numpy as np

def Recursive_SVD_U(H_old, H_new, C_old):
    """
        The recursive SVD (RSVD) process.

        :param H_old: numpy.ndarray, Old_Hankel_Matrix
        :param H_old: numpy.ndarray, New_Hankel_Matrix, almost have some lines that are different with the Old_Hankel_Matrix

        :return: U_new: the left singular vector of H_new,
                 V_new: the singular matrix of H_new
    """
    row, column = H_old.shape

    # C_old = H_old @ H_old.T

    # u_old = H_old @ np.ones(shape=[column, 1]) / column
    # u_new = u_old + (H_new[:, -1].reshape([row, 1]) - H_old[:, 0].reshape([row, 1])) / column
    #
    # C_old_normalized = (H_old.T - np.ones(shape=[column, 1]) @ u_old.T).T @ (H_old.T - np.ones(shape=[column, 1]) @ u_old.T) / column
    #
    # C_new = C_old_normalized + u_old @ u_old.T - u_new @ u_new.T - (H_old[:, 0].reshape(row, 1)) @ (H_old[:, 0].reshape(1, row)) / column + (H_new[:, -1].reshape(row, 1)) @ (H_new[:, -1].reshape(1, row)) / column
    #
    # C_new_1 = H_new @ H_new.T

    C_new = C_old - (H_old[:, 0].reshape(row, 1)) @ (H_old[:, 0].reshape(1, row)) + (H_new[:, -1].reshape(row, 1)) @ (H_new[:, -1].reshape(1, row))

    V_new, U_new = np.linalg.eig(C_new)
    sorted_indices = np.argsort(V_new)[::-1]  # 降序排列
    sorted_eigenvalues = V_new[sorted_indices]
    sorted_eigenvectors = U_new[:, sorted_indices]

    return sorted_eigenvectors, sorted_eigenvalues, C_new

