import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('weather_mood_data.csv')

# grafics
sns.countplot(data=df, x='mood_score')
plt.show()

sns.countplot(data=df, x='weather_type')
plt.show()

plt.scatter(df['temperature'], df['mood_score'])
plt.show()

# tests
sunny = df[df['weather_type'] == 'Sunny']['mood_score']
other = df[df['weather_type'] != 'Sunny']['mood_score']

t_val, p_val1 = stats.ttest_ind(sunny, other, equal_var=False)
print("T-test p-value:", p_val1)

corr, p_val2 = stats.pearsonr(df['temperature'], df['mood_score'])
print("Korelasyon r:", corr, "p-value:", p_val2)

# model
df_ml = pd.get_dummies(df, columns=['weather_type'], drop_first=True)
X = df_ml.drop(['date', 'mood_score'], axis=1)
y = df_ml['mood_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("r2 score:", r2_score(y_test, preds))
print("mse:", mean_squared_error(y_test, preds))