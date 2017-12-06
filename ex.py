"""
rsync -avz --exclude ".*" --exclude "*.pyc" --exclude "*.csv" --exclude "*.log" --exclude "*.png" ~/Dropbox/JVN/Capstone\ project/fms tuanta@tp:/home/tuanta/Dropbox/ && ssh tp 'cd /home/tuanta/Dropbox/fms && date && python2 startfms.py run config.yml && date' && scp tuanta@tp:/home/tuanta/Dropbox/fms/output.csv tuanta@tp:/home/tuanta/Dropbox/fms/log.log ~/Downloads/ && python ~/Dropbox/JVN/Capstone\ project/fms/ex.py


df = pd.read_csv('/opt/xquant_crawler/data/cophieu68_history.csv')
info = pd.read_csv('/opt/xquant_crawler/data/infor.csv')

for sector in info['sector'].unique():
    means = dict()
    stds = dict()
    symbols = info[info['sector'] == sector]['symbol']
    for symbol in symbols:
        if symbol[0] == '^': continue
        returns = df[df['symbol'] == symbol]['close'].pct_change().dropna()
        mean, std = norm.fit(returns)
        if np.abs(mean) == np.inf or np.isnan(mean): continue
        if np.abs(std) == np.inf or np.isnan(std): continue
        means[symbol] = mean
        stds[symbol] = std
    #
    means = list(means.values())
    stds = list(stds.values())
    print(
        sector,
        'mean of means', round(np.mean(means), 4),
        'std of means', round(np.std(means), 4),
        'mean of stds', round(np.mean(stds), 4),
        'std of stds', round(np.std(stds), 4),
        'skew', st.skew(heights),
        'kurtosis', st.kurtosis(heights),
    )

# toan bo VN
0.0027, 0.0303

# exchange
hose, 0.0020, 0.0311
upcom, 0.0045, 0.0428
hnx, 0.0008, 0.0011

# industry
nguyen vat lieu, 0.0012, 0.0018
nong nghiep, 0.0075, 0.06
vien thong, 0.0024, 0.0017
tai chinh, 0.0005, 0.0015
bat dong san va xay dung, 0.0026, 0.0315
cong nghiep, 0.0008, 0.0016
dich vu, 0.0008, 0.0012
hang tieu dung, 0.0008, 0.0014
cong nghe, 0.0202, 0.1046
y te, 0.0004, 0.0015
nang luong, 0.0085, 0.064





df = pd.read_csv('/home/tuanta/Dropbox/X/xquant_crawler/data/cophieu68_history.csv')
returns = df[['symbol', 'close']].groupby('symbol').pct_change().dropna()
returns = returns[returns < 0.25]
returns = returns[returns > -0.25]
volatility = pd.rolling_std(returns, window=2) * np.sqrt(2)
plt.hist(returns['close'].dropna(), bins=50); plt.show()
sns.kdeplot(returns['close'].dropna()); plt.show()
volatility.plot(subplots=True, color='blue',figsize=(8, 6)); plt.show()





df = pd.read_csv('/home/tuanta/Dropbox/X/xquant_crawler/data/vnindex.csv')
df['log-price'] = np.log(df['close'])
df['log-return'] = df['log-price'] - df['log-price'].shift(1)
df['volatility'] = pd.rolling_std(df['log-return'], window=100) * np.sqrt(100)
df['return']= df['log-return']


df['vol'] = df['volatility'].round(2)
df.groupby('vol')['volume'].mean()

plt.hist(df['log-return'].dropna(), bins=50); plt.show()
sns.kdeplot(df['log-return'].dropna()); plt.show()
df[['log-return', 'volatility']].plot(subplots=True, color='blue',figsize=(8, 6)); plt.show()

>>> df.groupby('vol')['volume'].mean()
vol
0.00    34448900
0.01    50819056
0.02    45011640
0.03    34379544
0.04    34272044
0.05    33827976
0.06    25261555
0.07    26020606
0.08    18176741
0.09    20903785
0.10     9720710
0.11      265650
0.12      213987
0.13      241800





df = pd.read_csv('~/Downloads/hose.csv')
returns = df[['symbol', 'close']].dropna().groupby('symbol').pct_change().dropna()
returns = returns[returns != np.inf]
returns = returns[returns < 0.08]
returns = returns[returns > -0.08]
volatility = pd.rolling_std(returns, window=2) * np.sqrt(2)
plt.hist(returns['close'].dropna(), bins=50); plt.show()
sns.kdeplot(returns['close'].dropna()); plt.show()
volatility.plot(subplots=True, color='blue',figsize=(8, 6)); plt.show()










df = pd.read_csv('~/Downloads/AAPL.csv')
returns = df[['Adj Close']].pct_change().dropna()
returns = returns[returns != np.inf]
returns = returns[returns < 0.08]
returns = returns[returns > -0.08]
volatility = pd.rolling_std(returns, window=2) * np.sqrt(2)
plt.hist(returns['Adj Close'].dropna(), bins=50); plt.show()
sns.kdeplot(returns['Adj Close'].dropna()); plt.show()
volatility.plot(subplots=True, color='blue',figsize=(8, 6)); plt.show()


"""
import random
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
import powerlaw
from scipy.stats import norm
from pandas.tools.plotting import autocorrelation_plot
import scipy.stats.stats as st

df = pd.read_csv('~/Downloads/output.csv', skiprows=[0], sep=';')
df['return'] = df['price'] / 100000 - 1
df['volatility'] = pd.rolling_std(df['return'], window=2) * np.sqrt(2)

# df = df[df['return'] < 0.08]
# df = df[df['return'] > -0.08]
plt.hist(df['return'].dropna(), bins=50); plt.show()
sns.kdeplot(df['return'].dropna()); plt.show()
df[['return', 'volatility']].dropna().plot(subplots=True, color='blue',figsize=(8, 6)); plt.show()


autocorrelation_plot(df['return'].dropna()); plt.show()
autocorrelation_plot(df['return'].dropna().abs()); plt.show()



df = pd.read_csv('~/Downloads/log.log', skiprows=[0,1,2], sep=';', header=None)
df.columns = ['direction', 'price', 'volume', 'agent']

a = df.groupby('agent')
print(a.count())
print(a['volume'].sum())
print(a['volume'].sum() / df['volume'].sum())

#
# data = df['volume']
# results = powerlaw.Fit(data)
# print(results.power_law.alpha)
# print(results.power_law.xmin)
# R, p = results.distribution_compare('power_law', 'lognormal')