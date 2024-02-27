import pandas as pd

df = pd.read_csv("run_result.csv")

averages = df.groupby('name').mean().reset_index()
sorted_averages = averages.sort_values('f1', ascending=False).reset_index(drop=True)

print(sorted_averages)