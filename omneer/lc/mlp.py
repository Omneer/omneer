import torch
import torch.nn as nn
import numpy as np
import tqdm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from .dataset import Data

__all__ = ['MLPClassifier']

class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * torch.sigmoid(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigmoid_input = torch.sigmoid(input)
        return grad_output * (sigmoid_input * (1 + input * (1 - sigmoid_input)))

class Swish(nn.Module):
    def forward(self, input):
        return SwishFunction.apply(input)

class MLPClassifier(BaseEstimator):
    ''' ... '''
    def __init__(
        self,
        hidden_dims = (32,),
        num_epochs = 32,
        batch_size = 8,
        lr = 1e-2,
        lambda_l1 = 1e-3,     # weight for L1 regularization
        lambda_l2 = 1e-2,     # weight for L2 regularization
        device = 'cpu',
        verbose = False,
    ):  
        ''' ... '''
        # training parameters
        params = locals()
        params.pop('self')
        for k, v in params.items():
            setattr(self, k, v)

    def fit(self, X, y):
        ''' ... '''
        # input validation
        X, y = self._validate_inputs(X, y)

        # for PyTorch computational efficiency
        torch.set_num_threads(1)

        # initialize model
        self.net_ = MLP(X.shape[1], self.hidden_dims, 1)
        self.net_.to(self.device)

        # initialize dataset and dataloader
        ldr = torch.utils.data.DataLoader(
            HelperDataset(X, y),
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = True,
        )

        # initialize optimizer
        optimizer = torch.optim.Adam(
            self.net_.parameters(),
            lr = self.lr,
            weight_decay = self.lambda_l2
        )
        
        # initialize loss function (binary cross entropy)
        loss_fn = torch.nn.BCELoss()

        # set model to train mode
        torch.set_grad_enabled(True)
        self.net_.train()

        # training loop
        iter_ = tqdm.tqdm(range(self.num_epochs)) if self.verbose else range(self.num_epochs)
        for e in iter_:        
            for X, y in ldr:
                # convert inputs to proper dtype and device
                X = X.to(torch.float).to(self.device)
                y = y.to(torch.float).to(self.device)

                # feed inputs to model, forward
                outputs = self.net_(X)

                # calculate train loss and add 
                loss = loss_fn(outputs, y)
                if self.lambda_l1 > 0:
                    loss += l1_regularizer(self.net_, lambda_l1=self.lambda_l1)            
            
                # backward and update parameters
                optimizer.zero_grad() 
                loss.backward()        
                optimizer.step()

        return self

    def predict(self, X):
        ''' ... '''
        proba = self.predict_proba(X)
        return (proba[:, 1] > proba[:, 0]).astype(np.int32)
        
    def predict_proba(self, X):
        ''' ... '''
        # input validation
        check_is_fitted(self)
        self._validate_inputs(X)
        
        # for PyTorch computational efficiency
        torch.set_num_threads(1)

        # convert to PyTorch tensor
        X = torch.tensor(X, dtype = torch.float, device = self.device)

        # set model to eval mode
        torch.set_grad_enabled(False)
        self.net_.eval()

        # collect model outputs/probabilities 
        # and make them consistence with the outputs of sklearn models
        proba_pos = self.net_(X).detach().cpu().numpy()
        proba_neg = 1 - proba_pos
        return np.array(list(zip(proba_neg, proba_pos)))

    def load(self, filepath):
        ''' ... '''
        state_dict = torch.load(filepath, map_location='cpu')
        num_features = len(state_dict['module.0.running_mean'])
        self.net_ = MLP(num_features, self.hidden_dims, 1)
        self.net_.load_state_dict(state_dict)
        self.net_.to(self.device)

    def save(self, filepath):
        ''' ... '''
        check_is_fitted(self)
        torch.save(self.net_.state_dict(), filepath)

    def estimate_shap(self, X):
        ''' ... '''
        check_is_fitted(self)
        self._validate_inputs(X)
        raise NotImplementedError

    def _validate_inputs(self, X, y=None):
        ''' ... '''
        if y is None:
            check_array(X)

        else:
            X, y = check_X_y(X, y)

            # y shall always be 0 or 1
            for label in unique_labels(y):
                assert label in (0, 1), 'Label must be 0 or 1. {} is detected.'.format(label)

        return X, y


class MLP(nn.Module):
    ''' ... '''
    def __init__(self, in_dim, hidden_dims, out_dim):
        ''' ... '''
        super(MLP, self).__init__()

        # initialize hidden layers
        hidden_layers = []
        hidden_dims = [n for n in hidden_dims]
        hidden_dims.insert(0, in_dim)
        for _in, _out in zip(hidden_dims, hidden_dims[1:]):
            hidden_layers.append(nn.Linear(_in, _out))
            hidden_layers.append(Swish())  # Use custom Swish activation function

        # initialize the entire network
        self.module = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            *hidden_layers,
            nn.Linear(hidden_dims[-1], out_dim),
        )

    def forward(self, x):
        ''' ... '''
        x = self.module(x)
        x = torch.sigmoid(x)
        x = x.squeeze()
        return x

    
class HelperDataset(torch.utils.data.Dataset):
    ''' ... '''
    def __init__(self, x, y = None):
        ''' ... '''
        self.x, self.y = x, y

    def __len__(self):
        ''' ... '''
        return len(self.x)

    def __getitem__(self, idx):
        ''' ... '''
        return self.x[idx], self.y[idx]

def l1_regularizer(model, lambda_l1=0.01):
    ''' LASSO '''
    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.metrics import accuracy_score
    from scipy.stats import randint as sp_randint

    # Load the data from Final.csv
    data = pd.read_csv('Final.csv')

    # Extract the input feature (X) and target variable (y)
    X = data.drop('PD', axis=1).values
    y = data['PD'].values

    # Reshape X to a 2D array if needed
    X = X.reshape(-1, 1)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter distribution for randomized search
    param_dist = {
        'hidden_dims': [(32,), (64,), (32, 32)],
        'num_epochs': sp_randint(32, 128),
        'batch_size': sp_randint(8, 32),
        'lr': [1e-2, 1e-3, 1e-4],
        'lambda_l1': [1e-3, 1e-4, 0],
        'lambda_l2': [1e-2, 1e-3, 1e-4],
    }

    # Create an instance of MLPClassifier
    cls = MLPClassifier()

    # Create a RandomizedSearchCV object
    randomized_search = RandomizedSearchCV(cls, param_distributions=param_dist, n_iter=100, cv=3, random_state=42)

    # Fit the randomized search to the training data
    randomized_search.fit(X_train, y_train)

    # Get the best estimator and its corresponding parameters
    best_estimator = randomized_search.best_estimator_
    best_params = randomized_search.best_params_

    # Make predictions on the test set using the best estimator
    y_pred = best_estimator.predict(X_test)

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Save the best estimator
    best_estimator.save('./best_model.pt')

    # Print the best parameters found by the randomized search
    print("Best Parameters: ", best_params)