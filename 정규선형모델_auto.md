```python
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.formula.api as smf
import statsmodels.api as sm
```


```python
df = pd.read_pickle('./dataset/auto-mpg.pkl')

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
      <th>car name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>




```python
ndf = df[['mpg','weight']]
```


```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(ndf)
df_ms = scaler.transform(ndf)
df_ms_ndf = pd.DataFrame(data=df_ms,columns=ndf.columns)
df_ms_ndf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.239362</td>
      <td>0.536150</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.159574</td>
      <td>0.589736</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.239362</td>
      <td>0.516870</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.186170</td>
      <td>0.516019</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.212766</td>
      <td>0.520556</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 모델 구축 
lm_model = smf.ols(formula = "mpg ~ weight",
                  data = df_ms_ndf).fit()
```


```python
lm_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.693</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.692</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   878.8</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 21 Nov 2020</td> <th>  Prob (F-statistic):</th> <td>6.02e-102</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:48:46</td>     <th>  Log-Likelihood:    </th> <td>  291.82</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>  -579.6</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   390</td>      <th>  BIC:               </th> <td>  -571.7</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    0.6617</td> <td>    0.011</td> <td>   60.029</td> <td> 0.000</td> <td>    0.640</td> <td>    0.683</td>
</tr>
<tr>
  <th>weight</th>    <td>   -0.7173</td> <td>    0.024</td> <td>  -29.645</td> <td> 0.000</td> <td>   -0.765</td> <td>   -0.670</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>41.682</td> <th>  Durbin-Watson:     </th> <td>   0.808</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  60.039</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.727</td> <th>  Prob(JB):          </th> <td>9.18e-14</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.251</td> <th>  Cond. No.          </th> <td>    4.81</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
# Prob(Omnibus), Prob(JB)는 잔차의 정규성에 대한 검정결과이며, p값이 0.05보다 
# 작으므로 잔차가 정규분포와 다르다
# 왜도(Skew)가 0보다 크며 오른쪽 자락이 길어진 형태이며 첨도(Kurtosis)도 
# 정규분포의 첨도 3가 차이가 있음
# Durbin-Watson 통계량이 2와 차이가 많이 나며 일반화 제곱법 등의 사용 검토 필요
```


```python
# 잔차 계산(실제값 - 예측값)
resid = lm_model.resid
resid.head(3)
```


```python
# 잔차 그래프
# 그래프를 보면 좌우대칭으로 정규분포를 따르는 것으로 보임
import warnings
warnings.filterwarnings('ignore')
sns.distplot(resid, color = 'blue')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-14-77651d6f2707> in <module>
          3 import warnings
          4 warnings.filterwarnings('ignore')
    ----> 5 sns.distplot(resid, color = 'blue')
    

    NameError: name 'resid' is not defined

