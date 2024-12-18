import pandas as pd
import numpy as np

print("deep KRAI")
df = pd.read_csv("sales_data_updated.csv")

print("First five rows : ")
print(df.head(5))

print("Total sales of each product : ")
df['total_sales'] = df['Price'] * df['Units']
grouped_df = df.groupby('Product')['total_sales'].sum()
print(grouped_df)

print("Average price of each product : ")
grouped_df = df.groupby('Product')['Price'].mean()
print(grouped_df)

print("Average price of all the products : ")
print(np.round(df['Price'].mean(), 2))

print("Country with highest sales : ")
country_sales = df.groupby('Country')['Total Sales'].sum()
max_sales_country = country_sales.idxmax()
max_sales_amount = country_sales.max()
print(f"{max_sales_country}: {max_sales_amount}")

df['Total Revenue'] = df['Price'] * df['Units']
print("New Column Added : ")
print(df['Total Revenue'])

print("Sort Dataframe for total revenue descending : ")
print(df['Total Revenue'].sort_values(ascending=False))

print("Saving this dataframe : ")
df.to_csv('deep_Dataframe.csv', index=False)







