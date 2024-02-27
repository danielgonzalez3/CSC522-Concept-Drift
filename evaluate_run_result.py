import pandas as pd

df1 = pd.read_csv("run_result_iot.csv")
df2 =  pd.read_csv("run_result_cic.csv")

averages = df1.groupby('name').mean().reset_index()
sorted_averages = averages.sort_values('f1', ascending=False).reset_index(drop=True)

print(sorted_averages)

averages = df2.groupby('name').mean().reset_index()
sorted_averages = averages.sort_values('f1', ascending=False).reset_index(drop=True)

print(sorted_averages)