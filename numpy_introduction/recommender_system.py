import numpy as np

class MatrixFactorization(object):
    """Matrix factorization for movie recommendations.

    Parameters:
        R (ndarray): ratings matrix (0 for no ratings, 1-5 if a rating exists)
        factors (int): number of factors for matrix factorization
        steps (int): number of steps to perform during training
        lr (float): learning rate
    """
    def __init__(self, R, factors=5, steps=1000, lr=1e-4):
        self.R = R
        self.factors = factors
        self.steps = steps
        self.lr = lr

        # Generate mask for known entries (non-zero elements), split the mask into a train and test mask
        self.mask = # TODO (TASK 1)
        self.split = # TODO (TASK 1)
        self.mask_train = # TODO (TASK 1)
        self.mask_test = # TODO (TASK 1)
        print(f"Known entries: {self.mask.sum()}, {self.mask_train.sum()} used for training and {self.mask_test.sum()} used for testing.")

        # Initialize low-rank user and movie matrix uniformly between 0 and 1
        self.U = np.random.rand(self.R.shape[0], self.factors).astype(dtype='float32')
        self.V = np.random.rand(self.R.shape[1], self.factors).astype(dtype='float32')

        # Compute total amount of parameters that have to be estimated
        total_parameters = self.U.reshape(-1).size + self.V.reshape(-1).size
        print(f"User matrix shape: {self.U.shape}, movie matrix shape: {self.V.shape}, total parameters: {total_parameters}")

    def gradient_user_matrix(self, error):
        return # TODO (TASK 2)

    def gradient_movie_matrix(self, error):
        return # TODO (TASK 2)

    def update_user_matrix(self, u_grad):
        self.U = # TODO (TASK 3)

    def update_movie_matrix(self, v_grad):
        self.V = # TODO (TASK 3)

    def rmse(self, split='all'):
        if split == 'train':
            rmse = np.sqrt(np.sum(self.mask_train * (self.R - np.matmul(self.U, self.V.T)) ** 2) / np.sum(self.mask_train))
        elif split == 'test':
            rmse = np.sqrt(np.sum(self.mask_test * (self.R - np.matmul(self.U, self.V.T)) ** 2) / np.sum(self.mask_test))
        else:
            rmse = np.sqrt(np.sum(self.mask * (self.R - np.matmul(self.U, self.V.T)) ** 2) / np.sum(self.mask))
        return rmse

    def fit(self):
        for i in range(self.steps):

            if i % 100 == 0:
                print(f"Step {i}/{self.steps}, RMSE (train): {self.rmse('train'):.4f}, RMSE (test): {self.rmse('test'):.4f}")

            error = # TODO (TASK 2) # Compute the error outside the gradient computation, so we don't have to do it twice
            u_grad = self.gradient_user_matrix(error)
            v_grad = self.gradient_movie_matrix(error)
            self.update_user_matrix(u_grad)
            self.update_movie_matrix(v_grad)

        print(f"Step {self.steps}/{self.steps}, RMSE (train): {self.rmse('train'):.4f}, RMSE (test): {self.rmse('test'):.4f}")

        return np.matmul(self.U, self.V.T)
