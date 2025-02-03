import pandas as pd
import numpy as np
import warnings  # Import Warnings to suppress unnecessary warnings
import shap
import matplotlib.pyplot as plt

# Suppress warning messages
warnings.filterwarnings("ignore")

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
HomeWinProb = pd.read_excel("Fase1_Datamanipulation/engelske_kampe_scrapped.xlsx", usecols="AQ", skiprows=0, nrows=7981)
DrawProb = pd.read_excel("Fase1_Datamanipulation/engelske_kampe_scrapped.xlsx", usecols="AR", skiprows=0, nrows=7981)
AwayWinProb = pd.read_excel("Fase1_Datamanipulation/engelske_kampe_scrapped.xlsx", usecols="AS", skiprows=0, nrows=7981)

#Concatenate data to matrix
y = np.hstack((HomeWinProb.to_numpy(), DrawProb.to_numpy(), AwayWinProb.to_numpy()))

print(y)
print(normalize_odds_to_probabilities(y))