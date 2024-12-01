#Data Cleaning and Preprocessing:

import pandas as pd

# Load the dataset
df = pd.read_csv("recruitment_data.csv")

# Check for missing values
print(df.isnull().sum())

# Fill or drop missing values if necessary (optional)
# For example, filling missing Years of Experience with the median Status:
df['Years of Experience'].fillna(df['Years of Experience'].median(), inplace=True)

# Convert categorical columns to numerical ones if necessary
df['Hiring_Decision'] = df['Status'].map({'Offered': 1, 'Rejected': 0, 'Interviewing' : 2})

# Check data types and convert them if needed
print(df.dtypes)

# Clean up text columns like Gender or Education Level if necessary
df['Gender'] = df['Education Level'].str.lower()

# Preview the cleaned data
print(df.head())


# 2. Exploratory Data Analysis (EDA):

# Descriptive statistics
print(df.describe())

# Correlation matrix
correlation = df.corr()
print(correlation)

import matplotlib.pyplot as plt
import seaborn as sns

# Hiring Decision distribution
sns.countplot(x='Hiring_Decision', data=df)
plt.title('Hiring Decision Distribution')
plt.xlabel('Status (1 = Offered, 0 = Rejected), 2 = Interviewing')
plt.ylabel('Count')
plt.show()


# Create a bar plot to analyze the skills and their impact on hiring
skills_counts = df['Skills'].value_counts()

plt.figure(figsize=(10,6))
sns.barplot(x=skills_counts.index, y=skills_counts.values)
plt.title('Skills Distribution')
plt.xlabel('Skills')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# Experience vs Hiring Decision
sns.boxplot(x='Hiring_Decision', y='Experience', data=df)
plt.title('Experience vs Hiring Decision')
plt.show()


#Hypothesis Testing:

from scipy import stats

# T-test for interview scores of hired vs rejected candidates
hired_scores = df[df['Hiring_Decision'] == 1]['Interview_Score']
rejected_scores = df[df['Hiring_Decision'] == 0]['Interview_Score']

t_stat, p_val = stats.ttest_ind(hired_scores, rejected_scores)
print(f"T-statistic: {t_stat}, P-value: {p_val}")


 #Visualization of Insights:

# Hiring decision by education level
sns.countplot(x='Education', hue='Hiring_Decision', data=df)
plt.title('Hiring Decision by Education Level')
plt.show()


# Impact of Interview Score on Hiring Decision
sns.histplot(data=df, x='Interview_Score', hue='Hiring_Decision', kde=True)
plt.title('Interview Score Distribution')
plt.show()
