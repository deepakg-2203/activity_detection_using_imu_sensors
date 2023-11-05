import pandas as pd

# Load the three CSV files into separate DataFrames
file1_path = "E:/GEETHA/sit_test.csv"
file2_path = "E:/GEETHA/walking_test.csv"
file3_path = "E:/GEETHA/stand_test.csv"
file4_path = "E:/GEETHA/sit_test.csv"
file5_path = "E:/GEETHA/stand_test.csv"
file6_path = "E:/GEETHA/walking_test.csv"
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)
df3 = pd.read_csv(file3_path)
df4 = pd.read_csv(file4_path)
df5 = pd.read_csv(file5_path)
df6 = pd.read_csv(file6_path)
# Concatenate the DataFrames into one and reindex
combined_df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)


# Save the combined DataFrame to a new CSV file

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('E:/GEETHA/sit_stand_walk_test2.csv', index=False)