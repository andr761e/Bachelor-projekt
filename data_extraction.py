import pandas as pd
import numpy as np
import warnings  # Import Warnings to suppress unnecessary warnings
import shap
import matplotlib.pyplot as plt

# Suppress warning messages
warnings.filterwarnings("ignore")

# Read the dataset into a Pandas DataFrame
df = pd.read_csv("games.csv")
item0 = df.shape[0]
df = df.drop_duplicates()
item1 = df.shape[0]
print(f"There are {item0-item1} duplicates found in the dataset")

# choose only games from Danish Superligaen (DK1)
competition_code = "DK1"
df = df[df['competition_id']==competition_code]
print(df)
print(df.shape)

df.to_excel("danske_kampe.xlsx", index=False)

# Assuming "date" is already in datetime format
df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
df.to_excel("danske_kampe2.xlsx", index=False)


def normalize_odds_to_probabilities(odds_matrix):
    """
    Convert an odds matrix to probabilities by taking the inverse of each entry 
    and normalizing each row to sum to 1.

    Args:
        odds_matrix: numpy array of shape (n, k), where each row contains odds for k outcomes.

    Returns:
        probabilities: numpy array of shape (n, k), where each row contains normalized probabilities.
    """
    # Step 1: Take the inverse of each entry
    probabilities = 1 / odds_matrix
    
    # Step 2: Normalize each row to sum to 1
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    
    return probabilities


# Load only column "Q" and rows from Q2 to Q2641
HomeWinProb = pd.read_excel("DNK.xlsx", usecols="Q", skiprows=0, nrows=2640)
DrawProb = pd.read_excel("DNK.xlsx", usecols="R", skiprows=0, nrows=2640)
AwayWinProb = pd.read_excel("DNK.xlsx", usecols="S", skiprows=0, nrows=2640)

#Concatenate data to matrix
y = np.hstack((HomeWinProb.to_numpy(), DrawProb.to_numpy(), AwayWinProb.to_numpy()))

print(y)
print(normalize_odds_to_probabilities(y))