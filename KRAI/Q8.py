import pandas as pd
import numpy as np

print("deep KRAI")
subjects = ['Maths', 'Science', 'Hindi', 'English']
students = ['Rohit', 'Omkar', 'Anuj', 'Sanket', 'Vishal', 'Prashant']
data = [
    [34, 47, 25, 26, 50, 39],
    [45, 48, 37, 32, 25, 40],
    [30, 25, 28, 37, 42, 45],
    [40, 30, 29, 28, 40, 46]
]

df = pd.DataFrame(data, columns=students, index=subjects)
print(df)

print("Rows : ", len(df.axes[0]))
print("Columns : ", len(df.axes[1]))

print("Rohit Mathematics Marks : ")
print(df['Arnab']['Maths'])

print("Rohit Subjects Score less than 35 : ")
print(df['Arnab'][df.iloc[:, 0] < 35])

mysum = df.sum()
print("Percentages : ")
print(mysum / 200 * 100)

mymean = df.mean(axis=0)

print("Average marks subject : ")
df['AverageMarks'] = df.mean(axis=1)
print(df[df['AverageMarks'] > 35].index.tolist())