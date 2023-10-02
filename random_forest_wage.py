#%%
import pandas as pd 
import numpy as np
import lightgbm as lgb
import sklearn
import optuna 
from ydata_profiling import ProfileReport
# %%

df = pd.read_excel('/home/fast-pc-2023/Téléchargements/wagesmicrodata.xls',sheet_name='Data',header=0,index_col=0, skiprows=[1])
df.WAGE = np.log(df.WAGE)
#%%
profile = ProfileReport(df)
profile.to_file('/home/fast-pc-2023/wagesdata.html', silent=False)

# AGE and EXPERIENCE are highly correlated 
# median education is 12
# WAGE have a few outliers

# %%
# FEATURE ENGINEERING : you can add features using this kind of methods and compare the model final results
print(df[df.columns[df.dtypes == int]]**2)

df['exp_div_edu'] = df.EXPERIENCE / df.EDUCATION
df['exp_div_age'] = df.EXPERIENCE / df.AGE
df['exp_squared'] = df.EXPERIENCE**2

df
#%%
# FEATURE SELECTION : (not needed on such a small number of features but at least you can see what it looks like)
import featurewiz as fwiz

output = fwiz.featurewiz(df.iloc[:-50], 'WAGE', 0.7, 2, test_data=df.iloc[-50:], feature_engg='')
selected_col = output[1].columns.to_list()
selected_col

#%%
def objective(trial):
    
    # IN THE FOLLOWING DICT WE PASS A RANGE OF PARAMETER VALUES WE WANT THE MODEL TO ITERATE OVER
        
    param = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "rf",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 0.1, log=True),
       # "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 0.1, log=True),
       # "min_split_gain": trial.suggest_float("min_split_gain", 1e-10, 0.01),
        "num_leaves": trial.suggest_int("num_leaves", 5, 50),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 20),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1,0.8),
        'n_estimators': trial.suggest_int('n_estimators', 5, 40),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.8)
    }

    # Add a callback for pruning -> given the fact we iterate multiple times to average out final results and thus keep the model that generalizes best we don't need pruning
    # But if your computer is slow or the problem more complex we wouldn't have the following loop and would use pruning 

    #pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")
    
    n_trials = 20
    accuracy = 0
    for i in range(n_trials): 
        temp_train = df.sample(frac=0.8)   # We randomly select a subset of the dataframe for training 
        temp_test = df.loc[df.index.difference(temp_train.index)]    # We select the inverse of the training dataset so as to form the test set
        
        dtrain = lgb.Dataset(temp_train[selected_col],temp_train.WAGE)   # Use the Dataset class from lightgbm for faster processing but this is optional
        dvalid = lgb.Dataset(temp_test[selected_col], temp_test.WAGE)        
        
        gbm = lgb.train(param, dtrain, valid_sets=[dvalid], categorical_feature='auto')  #callbacks=[pruning_callback]
        preds = gbm.predict(temp_test[selected_col])
        accuracy += sklearn.metrics.mean_squared_error(temp_test.WAGE, preds, squared=True)
    
    return accuracy /n_trials 

if 1:
    if __name__ == "__main__":
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="minimize", 
            sampler=optuna.samplers.NSGAIISampler(),
            study_name='reg_pct'
        )
        study.optimize(objective, n_trials=1000, n_jobs=20)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
            


# %%

model = lgb.LGBMRegressor('rf', metric='rmse', **trial.params, n_jobs=25)    
print(trial.params)

n_tries = 10
loss = 0

for i in range(n_tries):
    temp_train = df.sample(frac=0.8)
    temp_test = df.loc[df.index.difference(temp_train.index)]
    model_fitted = model.fit(temp_train[selected_col],temp_train.WAGE, eval_set=(temp_test[selected_col], temp_test.WAGE))
    loss += model_fitted.best_score_['valid_0']['rmse']
    
print(loss/n_tries)
#%%
import matplotlib.pyplot as plt

plt.barh(df[selected_col].columns, model.feature_importances_)

# %%
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut, LeavePOut, ShuffleSplit, KFold # you can test different cv methods, watch out -> LeavePOut requires high computation power

X = df[selected_col]
y = df.WAGE
scores = cross_val_score(model, X, y, cv=ShuffleSplit(n_splits=10, test_size=50), scoring='neg_root_mean_squared_error', n_jobs=20) 
scores.mean(), scores.std(), scores
# %%

# Set of good params for columns = ['AGE', 'EDUCATION', 'OCCUPATION', 'SEX', 'RACE', 'SOUTH']
# {'lambda_l1': 5.05930394412948e-08, 'num_leaves': 24, 'bagging_freq': 3, 'min_child_samples': 10, 'max_depth': 14, 'feature_fraction': 0.7603375303243309, 'n_estimators': 19, 'bagging_fraction': 0.6700136485341264}



from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

regr = RandomForestRegressor(max_depth=20, random_state=0, criterion='squared_error')
scores = cross_val_score(regr, X, y, cv=ShuffleSplit(n_splits=10, test_size=50), scoring='neg_root_mean_squared_error', n_jobs=20)
scores.mean(), scores.std(), scores
