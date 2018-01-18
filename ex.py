"""
rsync -avz --exclude ".*" --exclude "*.pyc" --exclude "*.csv" --exclude "*.log" --exclude "*.png" ~/Dropbox/JVN/Capstone\ project/fms tuanta@tp:/home/tuanta/Dropbox/ && ssh tp 'cd /home/tuanta/Dropbox/fms && date && python2 startfms.py run config.yml && date' && scp tuanta@tp:/home/tuanta/Dropbox/fms/output.csv tuanta@tp:/home/tuanta/Dropbox/fms/log.log ~/Downloads/ && python ~/Dropbox/JVN/Capstone\ project/fms/ex.py



^vnindex,0.0006204569171448406,0.015401884709210997,-0.11011288573765722,2.978097875540062,0.5179668944065984


df1 = pd.read_csv('returns1.csv', header=None)
df5 = df.groupby('symbol')['close'].pct_change(5).dropna()
df10 = pd.read_csv('returns5.csv', header=None)
df22 = pd.read_csv('returns22.csv', header=None)

df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()
df5 = df5.replace([np.inf, -np.inf], np.nan).dropna()
df10 = df10.replace([np.inf, -np.inf], np.nan).dropna()
df22 = df22.replace([np.inf, -np.inf], np.nan).dropna()

df1.hist(bins=25, range=(-0.1, 0.1)); plt.show()
df5.hist(bins=25, range=(-1, 1)); plt.show()
df10.hist(bins=25, range=(-1, 1)); plt.show()
df22.hist(bins=25, range=(-1, 1)); plt.show()

st.skew(list(df1[0].values)), st.kurtosis(list(df1[0].values))
st.skew(list(df5[0].values)), st.kurtosis(list(df5[0].values))
st.skew(list(df10[0].values)), st.kurtosis(list(df10[0].values))
st.skew(list(df22[0].values)), st.kurtosis(list(df22[0].values))


for symbol in valid_syms:
    if symbol[0] == '^' or symbol == '000001.ss': continue
    if info[info['symbol'] == symbol]['exchange'].values[0] == 'upcom': continue
    returns = df[df['symbol'] == symbol]['close'].pct_change(1).dropna()
    mean, std = norm.fit(returns)
    if np.abs(mean) == np.inf or np.isnan(mean): continue
    if np.abs(std) == np.inf or np.isnan(std) or std == 0: continue
    print('{},{},{},{},{}'.format(
        symbol, mean, std, st.skew(returns), st.kurtosis(returns)))
#
means = list(means.values())
stds = list(stds.values())
skews = list(skews.values())
kurtosiss = list(kurtosiss.values())



for sector in info['sector'].unique():
    means = dict()
    stds = dict()
    skews = dict()
    kurtosiss = dict()
    symbols = df.symbol.unique()#info[info['sector'] == sector]['symbol']
    for symbol in symbols:
        print(symbol)
        if symbol[0] == '^': continue
        returns = df[df['symbol'] == symbol]['close'].pct_change().dropna()
        mean, std = norm.fit(returns)
        if np.abs(mean) == np.inf or np.isnan(mean): continue
        if np.abs(std) == np.inf or np.isnan(std): continue
        means[symbol] = mean
        stds[symbol] = std
        skews[symbol] = st.skew(returns)
        kurtosiss[symbol] = st.kurtosis(returns)
    #
    means = list(means.values())
    stds = list(stds.values())
    skews = list(skews.values())
    kurtosiss = list(kurtosiss.values())
    print(means, stds, skews, kurtosiss)
    print(
        sector, '\n',
        'mean of means', round(np.mean(means), 4), '\n',
        'std of means', round(np.std(means), 4), '\n',
        'mean of stds', round(np.mean(stds), 4), '\n',
        'std of stds', round(np.std(stds), 4), '\n',
        'mean of skews', round(np.mean(skews), 4), '\n',
        'std of skews', round(np.std(skews), 4), '\n',
        'mean of kurtosiss', round(np.mean(kurtosiss), 4), '\n',
        'std of kurtosiss', round(np.std(kurtosiss), 4), '\n',
    )
    break

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


info = pd.read_csv('/opt/xquant_crawler/data/infor.csv')
df = pd.read_csv('/opt/xquant_crawler/data/cophieu68_history.csv', parse_dates=['date'])

valid_syms = info[info['exchange'] != 'upcom']['symbol']
valid_syms = [s for s in valid_syms if s[0] != '^' and s != '000001.ss']

df = df[df['symbol'].isin(valid_syms)]
df = df.sort_values('date', ascending=True)
df.index = df['date']

vn30 = ['bid', 'bmp', 'bvh', 'cii', 'ctd', 'ctg', 'dhg', 'dpm', 'fpt', 'gas', 'gmd', 'hpg', 'hsg', 'kbc', 'kdc', 'mbb', 'msn', 'mwg', 'nt2', 'nvl', 'pvd', 'ree', 'ros', 'sab', 'sbt', 'ssi', 'stb', 'vcb', 'vic', 'vnm']

"""
running 60
"""
for symbol in vn30:
    returns = df[df['symbol'] == symbol]['close'].pct_change(20).dropna()
    mess = ''
    for i in range(60, len(returns)):
        sub_returns = returns[i-60:i]
        mean, std = norm.fit(sub_returns)
        # if np.abs(mean) == np.inf or np.isnan(mean): continue
        # if np.abs(std) == np.inf or np.isnan(std) or std == 0: continue
        mes = '{},{},{},{},{},{}\n'.format(
            symbol, returns.index[i], mean, std, st.skew(sub_returns), st.kurtosis(sub_returns))
        print(mes)
        mess += mes
    #
    with open("{}_60.csv".format(symbol), "w") as text_file:
        text_file.write(mess)


df_60 = pd.DataFrame()
for s in vn30:
    a = pd.read_csv('{}_60.csv'.format(s), header=None, parse_dates=[1])
    df_60 = df_60.append(a)






"""
groupby quarter
"""
for symbol in vn30:
    subdf = df[df['symbol'] == symbol]
    subdf['quarter'] = subdf.index.month / 4
    subdf['quarter'] = subdf['quarter'].astype(int) / 4
    subdf['time'] = subdf.index.year
    subdf['time'] = subdf['time'] + subdf['quarter']
    subdf['return'] = subdf['close'].pct_change(20)
    mess = ''
    for time in subdf['time'].unique():
        sub_returns = subdf[subdf['time'] == time]['return'].dropna()
        mean, std = norm.fit(sub_returns)
        # if np.abs(mean) == np.inf or np.isnan(mean): continue
        # if np.abs(std) == np.inf or np.isnan(std) or std == 0: continue
        mes = '{},{},{},{},{},{}\n'.format(
            symbol, time, mean, std, st.skew(sub_returns), st.kurtosis(sub_returns))
        print(mes)
        mess += mes
    #
    with open("{}_60.csv".format(symbol), "w") as text_file:
        text_file.write(mess)


df_60 = pd.DataFrame()
for s in vn30:
    a = pd.read_csv('{}_60.csv'.format(s), header=None)
    df_60 = df_60.append(a)






def autocorr(x, t=1):
    return np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))





df = pd.read_csv('output.csv', skiprows=[0], sep=';')
df['return'] = df['price'] / 100000 - 1
df['volatility'] = pd.rolling_std(df['return'], window=2) * (252**0.5)

plt.hist(df['return'].dropna(), bins=50); plt.show()
sns.kdeplot(df['return'].dropna()); plt.show()
df[['return', 'volatility']].dropna().plot(subplots=True, color='blue',figsize=(8, 6)); plt.show()



df = df2[df2['symbol'] == 'ree']
df = df.sort_values('close', ascending=True)
df['return'] = df['close'].pct_change()

autocorrelation_plot(df['return'].dropna()); plt.show()
autocorrelation_plot(df['return'].dropna().abs()); plt.show()
np.corrcoef(df['return'].dropna())
autocorr(df['return'].dropna())

df = pd.read_csv('~/Downloads/log.log', skiprows=[0,1,2], sep=';', header=None)
df.columns = ['direction', 'price', 'volume', 'agent']

a = df.groupby('agent')
print(a.count())
print(a['volume'].sum())
print(a['volume'].sum() / df['volume'].sum())




















"""
Reinforcement learning

OLD approach 4k:
0.000542147059442,0.0186895839792,-0.129869704818,4.26498080145,0.620492911377
-0.0136987345847,0.0288804907061,-0.975666049639,-0.158491970843,0.687489770351
-0.00867309972961,0.0288286957933,-0.675912855616,0.929047170831,0.690605942964
-0.00947280720261,0.0297052752834,-0.633180910744,0.778102680118,0.699554059884
-0.00526735727203,0.0304362119351,-0.411678744565,1.03462481597,0.694325118728

=> AVERAGE
-0.0073139703	0.0273080515	-0.5652616531	1.3696526995	0.6784935607
=> STD
0.0053193099	0.0048629974	0.3156114468	1.6863021149	0.0327333919

REAL 4k:
0.000620456917145 0.0154018847092 -0.11011288573765728 2.978097875540062 0.517966894407



NEW approach 2k (RL):
SIMULATION
-0.000277807598039 0.0402289371151 -0.006413492511516556 -0.23311368168814273 0.125531749516
-0.00220247846154 0.0537145142654 0.04425527337792079 -1.367323722640098 -0.0139785869991


REAL 2k
0.00051683557164625398, 0.01261776758749274, -0.1741695242531113, 1.9535601832828702, 0.27439821005000697



1-day:
20-day:
0.000410,0.007503,-0.009279,-0.612845,0.607084
0.008226,0.092008,0.011332,-0.293914,0.984391
tuanta@10:~/Dropbox/fms$ python2 startfms.py run config.yml && python3 test.py
0.000113,0.008420,-0.090618,-0.706241,0.541735
0.002642,0.098837,0.071333,-0.473571,0.981236
tuanta@10:~/Dropbox/fms$ python2 startfms.py run config.yml && python3 test.py
-0.000321,0.009135,-0.339067,1.581689,0.584450
-0.005765,0.109026,-0.255905,0.021265,0.984755
tuanta@10:~/Dropbox/fms$ python2 startfms.py run config.yml && python3 test.py
-0.000876,0.009367,-0.316163,3.272976,0.612644
-0.016671,0.107179,-0.280839,0.109596,0.983543
tuanta@10:~/Dropbox/fms$ python2 startfms.py run config.yml && python3 test.py
0.001083,0.008474,0.054584,-0.615163,0.582597
0.021367,0.096131,-0.235227,0.009618,0.981848

"""
# 0.001445952769607868, 0.048534114293451079, -0.025804067643815237, -1.0271377648059368, 0.27923043426126049
sim = pd.read_csv('simulation_rl.csv')
# 0.000620456917145 0.0154018847092 -0.11011288573765728 2.978097875540062 0.517966894407
df = pd.read_csv('data/vnindex.csv')


data = df['return'].dropna()
mu, sigma = norm.fit(data)
skew, kurtosis = st.skew(data), st.kurtosis(data)
auto_corr = autocorr(data.abs())[0, 1]
print(mu, sigma, skew, kurtosis, auto_corr)
