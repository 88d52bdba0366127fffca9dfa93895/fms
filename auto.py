import pandas as pd
import numpy as np
from subprocess import Popen, PIPE
from scipy.stats import norm
from pandas.tools.plotting import autocorrelation_plot
import scipy.stats.stats as st
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def f_autocorr(x, t=1):
    return np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))


i = 0
result_df = pd.DataFrame()
total_number = 10000
for iteration in range(10):
    for herding_pct in np.arange(0.05, 1, 0.05):
        for threshold_pct in np.arange(0.05, 1 - herding_pct, 0.05):
            i += 1
            zero_pct = 1 - herding_pct - threshold_pct
            zero_number = str(int(zero_pct * total_number))
            herding_number = str(int(herding_pct * total_number))
            threshold_number = str(int(threshold_pct * total_number))
            print(i, zero_number, herding_number, threshold_number)
            with open('config.yml.origin', 'r') as f:
                config = f.read()
            #
            config = config.replace('{seed}', str(np.random.randint(99999)))
            config = config.replace('{zero_number}', zero_number)
            config = config.replace('{herding_number}', herding_number)
            config = config.replace('{threshold_number}', threshold_number)
            with open('config.yml', 'w') as f:
                f.write(config)
            #
            #
            with open('zerointelligencetrader.py.origin', 'r') as f:
                zero_agent = f.read()
            #
            zero_agent = zero_agent.replace('{mu}', '0.000620456917145')
            zero_agent = zero_agent.replace('{sigma}', '0.0154018847092')
            #
            zero_file = 'fms/agents/zerointelligencetrader.py'
            with open(zero_file, 'w') as f:
                f.write(zero_agent)
            #
            process = Popen(['python2', 'startfms.py', 'run', 'config.yml'], stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            if len(stdout) != 0:
                print('STDOUT', stdout)
            if len(stderr) != 0:
                print('STDERR', stderr)
            #
            df = pd.read_csv('output.csv', skiprows=[0], sep=';')
            df['return'] = df['price'] / 100000 - 1
            #
            mu, sigma = norm.fit(df['return'])
            skew, kurtosis = st.skew(df['return']), st.kurtosis(df['return'])
            autocorr = f_autocorr(df['return'].dropna().abs())[0, 1]
            print('{},{},{},{},{}'.format(
                mu, sigma, skew, kurtosis, autocorr))
            result_df = result_df.append({
                'zero_pct': zero_pct,
                'herding_pct': herding_pct,
                'threshold_pct': threshold_pct,
                'mu': mu,
                'sigma': sigma,
                'skew': skew,
                'kurtosis': kurtosis,
                'autocorr': autocorr,
            }, ignore_index=True)
            result_df.to_csv('result.csv.10times.2.csv', index=False)


"""
>>> df = pd.read_csv('data/vnindex.csv')
>>> df[['mu', 'sigma', 'skew', 'kurtosis']].dropna().describe()
                mu        sigma         skew     kurtosis
count  4019.000000  4019.000000  4019.000000  4019.000000
mean      0.000555     0.013249     0.034758     0.919012
std       0.003585     0.007116     0.584378     1.887081
min      -0.010235     0.002856    -2.066065    -1.471308
25%      -0.001183     0.008470    -0.282480    -0.245252
50%       0.000185     0.012062     0.038870     0.344189
75%       0.001673     0.015984     0.340954     1.402114
max       0.014084     0.049296     2.663322    16.195090

>>> df = pd.read_csv('result.csv.10times')
>>> df = df[(df['mu'] < 0.014084) & (df['mu'] > -0.010235)]
>>> df = df[(df['sigma'] < 0.049296) & (df['sigma'] > 0.002856)]
>>> df = df[(df['skew'] < 2.663322) & (df['skew'] > -2.066065)]
>>> df = df[(df['kurtosis'] < 16.195090) & (df['kurtosis'] > -1.471308)]
>>> df.describe()
         autocorr  herding_pct    kurtosis          mu       sigma  \
count  331.000000   331.000000  331.000000  331.000000  331.000000
mean     0.670958     0.189275    1.229543    0.001463    0.029809
std      0.046336     0.126544    1.949039    0.006482    0.011782
min      0.539005     0.050000   -1.108329   -0.010175    0.008367
25%      0.651189     0.100000   -0.365963   -0.003450    0.020813
50%      0.682286     0.150000    0.666032    0.000975    0.030041
75%      0.703035     0.250000    2.426190    0.006356    0.040355
max      0.756809     0.600000    8.155131    0.014042    0.049233

             skew  threshold_pct    zero_pct
count  331.000000     331.000000  331.000000
mean     0.018155       0.167069    0.643656
std      0.638313       0.112561    0.134183
min     -1.872183       0.050000    0.350000
25%     -0.237098       0.100000    0.550000
50%     -0.027594       0.150000    0.650000
75%      0.191446       0.250000    0.750000
max      1.878884       0.550000    0.900000
"""


model = RandomForestRegressor(criterion='mse', n_estimators=500, random_state=0)
parameters = {'max_depth': [5]}
grid_cv = GridSearchCV(model, parameters, cv=3)

grid_cv.fit(df[['mu', 'sigma', 'skew', 'kurtosis']], df[['zero_pct']])
model = grid_cv.best_estimator_
print(pd.DataFrame(grid_cv.cv_results_))
print(grid_cv.best_score_)
print(grid_cv.best_params_)
print(model.feature_importances_)
print(model.predict([[0.0006, 0.0154, -0.1101, 2.9780]]))


"""
# zero_pct
>>> print(model.feature_importances_)
[ 0.0018494   0.874976    0.00309786  0.12007675]
>>> print(model.predict([[0.0006, 0.0154, -0.1101, 2.9780]]))
[ 0.79433169]

# herding_pct
>>> print(model.feature_importances_)
[ 0.14512855  0.52619939  0.14064758  0.18802448]
>>> print(model.predict([[0.0006, 0.0154, -0.1101, 2.9780]]))
[ 0.11269233]

# zero & herding
>>> df[df['zero_pct'] == 0.8][df['herding_pct'] == 0.1]
      autocorr  herding_pct  kurtosis        mu     sigma      skew  \
19    0.633572          0.1  4.837001  0.000461  0.015402  1.144850
190   0.661086          0.1  3.851944  0.000742  0.013510  0.930163
361   0.702124          0.1  5.804982  0.001795  0.015159  0.760474
532   0.579471          0.1  3.912982 -0.000576  0.013744 -0.181222
703   0.588676          0.1  4.942518 -0.001887  0.014840 -1.176365
874   0.673145          0.1  6.478805 -0.002929  0.015912 -0.771877
1045  0.699790          0.1  6.267179  0.003834  0.015426  1.878884
1216  0.586950          0.1  0.416697 -0.000314  0.012365 -0.065568
1387  0.677319          0.1  5.328986  0.004262  0.014839  1.503461
1558  0.656592          0.1  3.466932  0.001370  0.012800  0.451323
>>> df[df['zero_pct'] == 0.8][df['herding_pct'] == 0.1].describe()
        autocorr   herding_pct   kurtosis         mu      sigma       skew  \
count  10.000000  1.000000e+01  10.000000  10.000000  10.000000  10.000000
mean    0.645872  1.000000e-01   4.530803   0.000676   0.014400   0.447412
std     0.046493  1.462847e-17   1.771184   0.002279   0.001212   0.986773
min     0.579471  1.000000e-01   0.416697  -0.002929   0.012365  -1.176365
25%     0.599900  1.000000e-01   3.867204  -0.000510   0.013568  -0.152308
50%     0.658839  1.000000e-01   4.889759   0.000601   0.014840   0.605899
75%     0.676275  1.000000e-01   5.685983   0.001689   0.015342   1.091178
max     0.702124  1.000000e-01   6.478805   0.004262   0.015912   1.878884


>>> print('{},{},{},{},{}'.format(
    mu, sigma, skew, kurtosis, autocorr))
-0.0017510445159560168,0.014134209940860214,-1.5310030014628917,5.805701671510782,0.6337583736943383
>>> (-0.0017510445159560168 - 0.0006)**2 / 0.0006 + \
(0.014134209940860214 - 0.0154)**2 / 0.0154 + \
(-1.5310030014628917 - -0.1101)**2 / 0.1101 + \
(5.805701671510782 - 2.9780)**2 / 2.9780
== 21.0318


df = pd.read_csv('data/vnindex.csv')
print(
    ((mu - df['mu'].mean()) / df['mu'].std())**2 + \
    ((sigma - df['sigma'].mean()) / df['sigma'].std())**2 + \
    ((skew - df['skew'].mean()) / df['skew'].std())**2 + \
    ((kurtosis - df['kurtosis'].mean()) / df['kurtosis'].std())**2
)
==
a = [14.314110380084895, 95.4985923449, 400.391537609, 181.996937943, 113.65467784]
>>> print(np.mean(a), np.std(a))
(161.17117122339698, 130.99450166527186)
"""