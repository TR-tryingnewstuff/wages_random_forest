#%%
import pandas as pd 
import numpy as np
from ydata_profiling import ProfileReport
import seaborn as sns
# %%

df = pd.read_excel('/home/fast-pc-2023/Téléchargements/wagesmicrodata.xls',sheet_name='Data',header=0,index_col=0, skiprows=[1])
df.WAGE = np.log(df.WAGE)

sns.heatmap(df.corr())

#%%
profile = ProfileReport(df)
profile.to_file('/home/fast-pc-2023/wagesdata.html', silent=False)

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

# %%

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut, LeavePOut, ShuffleSplit, KFold # you can test different cv methods, watch out -> LeavePOut requires high computation power
from sklearn.ensemble import RandomForestRegressor

X = df[selected_col]
y = df.WAGE

regr = RandomForestRegressor(max_depth=20, random_state=0)
scores = cross_val_score(regr, X, y, cv=ShuffleSplit(n_splits=10, test_size=50), scoring='neg_root_mean_squared_error', n_jobs=20)
scores.mean(), scores.std(), scores
# %%
