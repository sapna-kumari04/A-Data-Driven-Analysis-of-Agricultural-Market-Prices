import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:\\datatoolbox.py\\pythonproject.csv")
print(df)
df.columns = df.columns.str.replace("x0020_", "", regex=False).str.replace(" ", "_")
print(df.columns)
print("1. Info:")
print(df.info())

#2 Summary statistics
print("\n2. Summary:")
print(df.describe(include='all'))
#3 Total rows and columns
print(f"\n3. Total Rows: {df.shape[0]}, Columns: {df.shape[1]}")

#4 Max values
print("\n4. Max values:")
print(df[['Min_Price', 'Max_Price', 'Modal_Price']].max())

#5 Min values
print("\n5. Min values:")
print(df[['Min_Price', 'Max_Price', 'Modal_Price']].min())

#6 Mode values
print("\n6. Mode:")
print(df.mode().iloc[0])
#7 Head and Tail
print("\n7. Head:")
print(df.head())

print("\n8. Tail:")
print(df.tail())

#8 Sort data
print("\n9. Sorted by Modal_Price:")
print(df.sort_values(by='Modal_Price', ascending=False).head())

#9 Use loc
print("\n10. Data for Banana:")
print(df.loc[df['Commodity'] == 'Banana'].head())

#10 Check for missing values
print("\n11. Missing values:")
print(df.isnull().sum())

#11 Fill or drop missing values (data cleaning)
df_cleaned = df.dropna()
print("\n12. After dropping NA rows:")
print(df_cleaned.shape)
# 13 Outlier detection using IQR
print("\n13. Outlier Detection:")
Q1 = df[['Min_Price', 'Max_Price', 'Modal_Price']].quantile(0.25)
Q3 = df[['Min_Price', 'Max_Price', 'Modal_Price']].quantile(0.75)
IQR = Q3 - Q1
outliers = df[
    (df['Min_Price'] < (Q1['Min_Price'] - 1.5 * IQR['Min_Price'])) | 
    (df['Min_Price'] > (Q3['Min_Price'] + 1.5 * IQR['Min_Price'])) |
    (df['Max_Price'] < (Q1['Max_Price'] - 1.5 * IQR['Max_Price'])) | 
    (df['Max_Price'] > (Q3['Max_Price'] + 1.5 * IQR['Max_Price'])) |
    (df['Modal_Price'] < (Q1['Modal_Price'] - 1.5 * IQR['Modal_Price'])) | 
    (df['Modal_Price'] > (Q3['Modal_Price'] + 1.5 * IQR['Modal_Price']))
]
print(f"Outliers found: {len(outliers)}")
# 14. Correlation matrix
print("\n14. Correlation Matrix:")
corr = df[['Min_Price', 'Max_Price', 'Modal_Price']].corr()
print(corr)


# 15. Pairplot
sns.pairplot(df[['Min_Price', 'Max_Price', 'Modal_Price']])
plt.suptitle("16. Pairplot of Prices", y=1.02)
plt.tight_layout()
plt.show()


# 17. Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap="YlGnBu")
plt.title("Correlation Heatmap")
plt.show()


# 18. Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['Modal_Price'], bins=50, kde=True)
plt.title("Histogram of Modal Price")
plt.xlabel("Modal Price")
plt.ylabel("Frequency")
plt.show()


# 19. Countplot of top commodities
top_commodities = df['Commodity'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_commodities.index, y=top_commodities.values)
plt.title("Top 10 Commodities by Count")
plt.xticks(rotation=45)
plt.ylabel("Count")
plt.show()


# 20. Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['Min_Price', 'Max_Price', 'Modal_Price']])
plt.title("Boxplot of Prices")
plt.show()


# 21. Line plot for Modal_Price
plt.figure(figsize=(10, 4))
plt.plot(df['Modal_Price'].head(100), marker='o')
plt.title("Modal Price (First 100 Records)")
plt.xlabel("Index")
plt.ylabel("Modal Price")
plt.grid(True)
plt.show()

# 22. Group by Commodity and find average price
avg_price_by_commodity = df.groupby('Commodity')['Modal_Price'].mean().sort_values(ascending=False).head()
print("\nAverage Modal Price by Commodity:")
print(avg_price_by_commodity)

# 23. Value counts for Market
print("\nMarket Frequency:")
print(df['Market'].value_counts().head())

# 24. Check unique districts
print(f"\nNumber of Unique Districts: {df['District'].nunique()}")

# 25. Plot bar for one market vs modal price (example)
market_avg = df.groupby('Market')['Modal_Price'].mean().sort_values(ascending=False).head(10)
market_avg.plot(kind='bar', figsize=(10, 5), title="Top 10 Markets by Average Modal Price")
plt.ylabel("Average Modal Price")
plt.show()


# 25. Bar Plot for Max Price per Commodity
commodity_max = df.groupby('Commodity')['Max_Price'].max().sort_values(ascending=False)
commodity_max.plot(kind='bar', figsize=(10, 5), title='Max_Price per Commodity')
plt.ylabel('Max_Price')
plt.xlabel('Commodity')
plt.tight_layout()
plt.show()

# 26.Pie Chart of Top 5 Commodities
top_commodities = df['Commodity'].value_counts().nlargest(5)
plt.figure(figsize=(7, 7))
plt.pie(top_commodities, labels=top_commodities.index, autopct='%1.1f%%', startangle=140)
plt.title("Top 5 Commodities")
plt.axis('equal')
plt.show()

# 27. Fill missing numeric values with median
df_filled = df.fillna(df.median(numeric_only=True))
print("Filled Missing Values with Median")

# 28. Scatter Plot - Max vs Modal
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Max_Price', y='Modal_Price', hue='Commodity', legend=False)
plt.title("Scatter Plot: Max Price vs Modal Price")
plt.xlabel("Max Price")
plt.ylabel("Modal Price")
plt.show()


# 29. Swarm Plot (for small dataset only)
plt.figure(figsize=(10, 6))
sampled = df.sample(n=200) if len(df) > 200 else df
sns.swarmplot(data=sampled, x='Commodity', y='Modal_Price')
plt.title("Swarm Plot of Modal Price by Commodity")
plt.xticks(rotation=45)
plt.show()

#30. Bar Plot - Avg Modal by State
avg_modal_by_state = df.groupby('State')['Modal_Price'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=avg_modal_by_state.index, y=avg_modal_by_state.values)
plt.title("Top 10 States by Avg Modal Price")
plt.xticks(rotation=45)
plt.ylabel("Average Modal Price")
plt.show()

 





