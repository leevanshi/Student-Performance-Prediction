import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

plt.scatter(df["StudyHours"], df["FinalMarks"])
plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.title("Study Hours vs Marks")
plt.show()