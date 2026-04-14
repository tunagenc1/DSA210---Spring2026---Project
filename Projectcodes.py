import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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