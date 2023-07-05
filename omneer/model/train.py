import pandas as pd
import numpy as np
import torch
from omneer.model.deeplearning.mlp import MLPClassifier as MLP
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def train(
    model_name,
    dataset, train_size, valid_size, save_dir,
    seed = 0,
):

    assert model_name in ['mlp', 'xgb', 'ensemble', 'rf', 'lr', 'svm', 'lda', 'lgbm', 'tabnet']

    # initialize models
    if model_name == 'mlp':
        model = MLP(
            hidden_dims = (32,), # Add more layers and increase the number of hidden units
            num_epochs = 64, # Increase the number of epochs
            batch_size = 16, # Increase the batch size
            lambda_l1 = 0.0011697870951761014,
            lambda_l2 = 0.0004719190005714674, 
            device = 'cpu'
            )

    elif model_name == 'xgb':
        model = XGBClassifier(
            booster = 'gbtree',
            learning_rate = .3,
            subsample = 1,
            colsample_bytree = .4,
            max_depth = 1,
            reg_alpha = .5,
            reg_lambda = .05,
            use_label_encoder = False,
            eval_metric = 'logloss',
            n_estimators = 50
        )

    elif model_name == 'ensemble':
        base_models = [
            ('xgb', XGBClassifier(booster='gbtree', learning_rate=0.3, subsample=1, colsample_bytree=0.4,
                                max_depth=1, reg_alpha=0.5, reg_lambda=0.05,
                                eval_metric='logloss', n_estimators=50)),
            ('mlp', MLP(hidden_dims=(32,), num_epochs=64, batch_size=16, lambda_l1=0.0011697870951761014,
                        lambda_l2=0.0004719190005714674, device='cpu'))
        ]
        
        final_model = LogisticRegression()  # Change this to experiment with different final estimators
    
        model = StackingClassifier(estimators=base_models, final_estimator=final_model, cv=10)

        """

        voting_model = VotingClassifier(estimators=base_models, voting='soft')
        model = voting_model

        # Use GradientBoostingClassifier as the final estimator
        final_model = GradientBoostingClassifier(random_state=42)
        
        stacking_model = StackingClassifier(estimators=base_models, final_estimator=final_model, cv=5)

        # Define the hyperparameters you want to tune
        param_grid = {
            'final_estimator__learning_rate': [0.1, 0.01, 0.001],
            'final_estimator__n_estimators': [50, 100, 200]
        }
        
        # Use GridSearchCV for hyperparameter tuning
        model = GridSearchCV(stacking_model, param_grid, cv=5)

        """

    elif model_name == 'rf':
        model = RandomForestClassifier(
            criterion='entropy',
            n_estimators = 500,
            max_depth = None,
            max_features = 'sqrt',
            min_samples_split = 5,
            random_state = seed
        )

    elif model_name == 'lr':
        model = LogisticRegression(
            C = 241,
            penalty = 'l1',
            solver = 'liblinear',
            random_state = seed
        )

    elif model_name == 'svm':
        model = SVC(
            kernel = 'linear',
            C = 0.1,
            gamma = 0.0001,
            random_state = seed
        )

    elif model_name == 'lda':
        model = LinearDiscriminantAnalysis(
            solver = 'lsqr',
            shrinkage = 'auto'
        )

    elif model_name == 'lgbm':
        model = LGBMClassifier(
            boosting_type='gbdt',
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            random_state=seed
        )

    elif model_name == 'tabnet':
        model = TabNetClassifier(
            n_d=64,  # Number of features in the input data
            n_a=64,  # Number of features in the output of each attention step
            n_steps=5,  # Number of sequential attention steps
            gamma=1.5,  # Float >= 1.0, control the gap between training accuracy and valid accuracy
            lambda_sparse=0.001,  # Float, sparsity penalty
            optimizer_fn=torch.optim.Adam,  # Optimizer type
            optimizer_params=dict(lr=2e-2),  # Optimizer hyperparameters
            mask_type='entmax',  # Either "sparsemax" or "entmax" - the masking function to use
            device_name='auto',  # 'auto' or specific cuda device
            verbose=1,  # Controls the verbosity level
        )

    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        *dataset.all,
        train_size = train_size,
        test_size = valid_size,
        random_state = seed,
    )

    # normalize features
    norm_obj = StandardScaler().fit(x_train)
    x_train = norm_obj.transform(x_train)
    x_test = norm_obj.transform(x_test)

    # store the normalization object for future use
    model.norm_obj_ = norm_obj

    # convert labels to integers
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    # train model
    model.fit(x_train, y_train)
    
    # save model checkpoint
    joblib.dump(model, '{}/checkpoints/test_{}.ckpt'.format(save_dir, seed))

    # evaluate and save results
    s_test = model.predict_proba(x_test)[:, 1] if model_name != 'svm' else model.decision_function(x_test)
    p_test = model.predict(x_test)
    _save_results(y_test, p_test, s_test, save_dir, seed)
    

def _save_results(labels, predictions, scores, save_dir, seed):

    df_results = pd.DataFrame()
    df_results['Y Label'] = labels
    df_results['Y Predicted'] = predictions
    df_results['Predicted score'] = scores
    df_results.to_csv('{}/results/test_{}.csv'.format(save_dir, seed), index=False)     
