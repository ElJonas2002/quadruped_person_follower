import pandas as pd

OUTPUT_CSV = "dataset/HaGrid/landmarks.csv"

# Comprobar distribución train/val del dataset
df = pd.read_csv(OUTPUT_CSV)
print(df.groupby(["clase", "split"]).size())