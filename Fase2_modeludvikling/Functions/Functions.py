import numpy as np

class DataUtils:
    @staticmethod
    def get_filtered_rows_percent_point(Y_pred_proba, Y_test, x):
        """
        Extracts rows where at least one predicted probability is higher than the true probability
        by at least x percentage points.

        :param Y_pred_proba: numpy array of predicted probabilities (n x 3)
        :param Y_test: numpy array of actual probabilities (n x 3)
        :param x: percent point deviation threshold (e.g., 0.10 for 10% points)
        :return: boolean mask (True for rows to extract)
        """
        deviation = Y_pred_proba - Y_test  # Absolute percent point difference
        mask = np.any(deviation >= x, axis=1)  # Keep rows where at least one entry deviates >= x percent points
        return mask
    
    @staticmethod
    def get_filtered_rows_percentwise(Y_pred_proba, Y_test, x):
        """
        Extracts rows where at least one predicted probability is higher than the true probability
        by at least x percent deviation.

        :param Y_pred_proba: numpy array of predicted probabilities (n x 3)
        :param Y_test: numpy array of actual probabilities (n x 3)
        :param x: percent deviation threshold (e.g., 0.50 for 50%)
        :return: boolean mask (True for rows to extract)
        """
        epsilon = 1e-10  # To avoid division by zero
        deviation = (Y_pred_proba - Y_test) / (Y_test + epsilon)  # Relative deviation
        mask = np.any(deviation >= x, axis=1)  # Keep rows where at least one entry deviates >= x%
        return mask

