import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('saveecobot_17112.csv', on_bad_lines='skip')
print(df) 

df = df.drop("value_text", axis=1)
print(df) 


for column in df.columns:
    unique_count = df[column].nunique()
    unique_values = df[column].unique()
    print(f"Column '{column}' has {unique_count} unique values:")
    print(unique_values)
   
    df = df.dropna(subset=[column])

df_pm1 = df[df["phenomenon"].isin(["pm1"])]
df_pm25 = df[df["phenomenon"].isin(["pm25"])]
df_pm10 = df[df["phenomenon"].isin(["pm10"])]
df_temperature = df[df["phenomenon"].isin(["temperature"])]
df_humidity = df[df["phenomenon"].isin(["humidity"])]
df_pressure_pa = df[df["phenomenon"].isin(["pressure_pa"])]
df_no2_ppb = df[df["phenomenon"].isin(["no2_ppb"])]
df_o3_ppb = df[df["phenomenon"].isin(["o3_ppb"])]


df_pm25['logged_at'] = pd.to_datetime(df_pm25['logged_at'])
df_pm25['hour'] = df_pm25['logged_at'].dt.hour
print(df_pm25)


df_morning = df_pm25.query("hour> 4 & hour < 10")  
df_night = df_pm25.query("hour> 16 & hour < 24")
print(df_morning)
print(df_night)

feature_df = df.groupby(['phenomenon', 'logged_at'], as_index=False).aggregate('mean')
phenomenon_time_df = feature_df.pivot_table(index=['logged_at'], columns='phenomenon', values=['value',])
phenomenon_time_df.reset_index(inplace=True)
phenomenon_time_df.columns = [col[1] if col[1]!='' else col[0] for col in phenomenon_time_df.columns.values]
print('Columns:  ', phenomenon_time_df.columns)
phenomenon_time_df['logged_at'] = pd.to_datetime(phenomenon_time_df['logged_at'])
phenomenon_time_df['hour'] = phenomenon_time_df['logged_at'].dt.hour
phenomenon_time_df['pm1'] = pd.to_numeric(phenomenon_time_df['pm1'], errors='coerce')
phenomenon_time_df['pm25'] = pd.to_numeric(phenomenon_time_df['pm25'], errors='coerce')
phenomenon_time_df['pm10'] = pd.to_numeric(phenomenon_time_df['pm10'], errors='coerce')
phenomenon_time_df['temperature'] = pd.to_numeric(phenomenon_time_df['temperature'], errors='coerce')
phenomenon_time_df['humidity'] = pd.to_numeric(phenomenon_time_df['humidity'], errors='coerce')
phenomenon_time_df['pressure_pa'] = pd.to_numeric(phenomenon_time_df['pressure_pa'], errors='coerce')
phenomenon_time_df['no2_ppb'] = pd.to_numeric(phenomenon_time_df['no2_ppb'], errors='coerce')
phenomenon_time_df['o3_ug'] = pd.to_numeric(phenomenon_time_df['o3_ug'], errors='coerce')
phenomenon_time_df = phenomenon_time_df.dropna()
print(phenomenon_time_df)


corr_matrix = phenomenon_time_df.corr().abs()
corr = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool)).stack().sort_values(ascending=False))[:16]
print(corr)
corr_items = list(corr.index)


X = pd.DataFrame(phenomenon_time_df['pm1'])
Y = pd.DataFrame(phenomenon_time_df['pm25'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train) 
print(regressor.intercept_, regressor.coef_)
sns.regplot(x='pm1', y='pm25', data=phenomenon_time_df, ci=None, scatter_kws={'s':100, 'facecolor':'purple'})
plt.show()
Y_pred = regressor.predict(X_test)
Y_pred = Y_pred.flatten()
Y_test['pm25_pred'] = Y_pred
print(Y_test)

print(f"r2_score  ", r2_score(X_test, Y_pred))
print(f"RMSE  ", mean_squared_error(X_test, Y_pred, squared=True), '\n\n')



X = pd.DataFrame(phenomenon_time_df['pm10'])
Y = pd.DataFrame(phenomenon_time_df['pm25'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train) 
print(regressor.intercept_, regressor.coef_)
sns.regplot(x='pm10', y='pm25', data=phenomenon_time_df, ci=None, scatter_kws={'s':100, 'facecolor':'green'})
plt.show()
Y_pred = regressor.predict(X_test)
Y_pred = Y_pred.flatten()
Y_test['pm25_pred'] = Y_pred
print(Y_test)

print(f"r2_score  ", r2_score(X_test, Y_pred))
print(f"RMSE  ", mean_squared_error(X_test, Y_pred, squared=True), '\n\n')



X = pd.DataFrame(phenomenon_time_df['pm1'])
Y = pd.DataFrame(phenomenon_time_df['pm10'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train) 
print(regressor.intercept_, regressor.coef_)
sns.regplot(x='pm1', y='pm10', data=phenomenon_time_df, ci=None, scatter_kws={'s':100, 'facecolor':'red'})
plt.show()
Y_pred = regressor.predict(X_test)
Y_pred = Y_pred.flatten()
Y_test['pm10_pred'] = Y_pred
print(Y_test)

print(f"r2_score  ", r2_score(X_test, Y_pred))
print(f"RMSE  ", mean_squared_error(X_test, Y_pred, squared=True), '\n\n')



X = pd.DataFrame(phenomenon_time_df['no2_ppb'])
Y = pd.DataFrame(phenomenon_time_df['temperature'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train) 
print(regressor.intercept_, regressor.coef_)
sns.regplot(x='no2_ppb', y='temperature', data=phenomenon_time_df, ci=None, scatter_kws={'s':100, 'facecolor':'pink'})
plt.show()
Y_pred = regressor.predict(X_test)
Y_pred = Y_pred.flatten()
Y_test['temperature_pred'] = Y_pred
print(Y_test)

print(f"r2_score  ", r2_score(X_test, Y_pred))
print(f"RMSE  ", mean_squared_error(X_test, Y_pred, squared=True), '\n\n')