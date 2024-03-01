import pandas as pd

df1 = pd.read_csv("CIC_run_result.csv")
df2 =  pd.read_csv("IoT_2020_run_result.csv")

averages = df1.groupby('name').mean().reset_index()
sorted_averages = averages.sort_values('f1', ascending=False).reset_index(drop=True).drop(columns=['Unnamed: 0'])

print(sorted_averages)

averages = df2.groupby('name').mean().reset_index()
sorted_averages = averages.sort_values('f1', ascending=False).reset_index(drop=True).drop(columns=['Unnamed: 0'])

print(sorted_averages)