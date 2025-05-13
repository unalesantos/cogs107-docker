#Name: Una Santos
#Assignment: Cogs 107 Midterm
#ChatGPT was used to clarify coding issues, as well as to create the structure.


import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Load Data
def load_plant_data(filepath):
    """
    Loads the plant knowledge dataset and removes the 'Informant' column.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        np.array: N x M matrix of 0s and 1s (informants x questions)
    """
    df = pd.read_csv(filepath)
    data = df.drop(columns=["Informant"]).values
    return data


# Defining Model
def run_cct_model(data):
    """
    Runs the Cultural Consensus Theory model using PyMC.

    Args:
        data (np.array): Binary matrix of shape (N, M)

    Returns:
        trace: Posterior samples from the model
    """
    N, M = data.shape  # N = number of informants, M = number of items

    with pm.Model() as model:
        # Prior for informant competence (between 0 and 1)
        D = pm.Beta("D", alpha=2, beta=1, shape=N)

        # Prior for consensus answers (each item is 0 or 1)
        Z = pm.Bernoulli("Z", p=0.5, shape=M)

        # Reshape so we can compute pairwise probabilities
        D_matrix = D[:, None]      # Shape: (N, 1)
        Z_matrix = Z[None, :]      # Shape: (1, M)

        # Probability that person i answers item j correctly
        p = Z_matrix * D_matrix + (1 - Z_matrix) * (1 - D_matrix)

        # Likelihood
        pm.Bernoulli("X_obs", p=p, observed=data)

        # Sample from the model
        trace = pm.sample(draws=2000, chains=4, tune=1000, return_inferencedata=True)

    return trace

# Posterior Analysis
def plot_posterior_distributions(trace):
    """
    Plots posterior distributions for D and Z.
    """
    az.plot_posterior(trace, var_names=["D"])
    plt.title("Posterior of Informant Competence (D)")
    plt.show()

    az.plot_posterior(trace, var_names=["Z"])
    plt.title("Posterior of Consensus Answers (Z)")
    plt.show()

def print_model_summary(trace):
    """
    Prints summary statistics of D and Z posteriors.
    """
    summary = az.summary(trace, var_names=["D", "Z"])
    print(summary)

# Compare With Majority Vote
def compare_to_majority(data, trace):
    """
    Compares model consensus answers to simple majority vote answers.

    Args:
        data (np.array): Observed data
        trace: Posterior samples
    """
    # Posterior mean of each Z (consensus answer)
    z_mean = trace.posterior["Z"].mean(dim=["chain", "draw"]).values
    z_estimated = (z_mean >= 0.5).astype(int)

    # Simple majority vote per item
    majority_vote = (data.mean(axis=0) >= 0.5).astype(int)

    print("Consensus vs. Majority Vote:")
    print("Estimated Z:       ", z_estimated)
    print("Majority Vote:     ", majority_vote)
    print("Match (%):         ", (z_estimated == majority_vote).mean() * 100)

# Main
if __name__ == "__main__":
    # Loads dataset
    data = load_plant_data("/Users/unasantos/Documents/GitHub/cogs107-docker/plant_knowledge.csv")

    # Run CCT model
    trace = run_cct_model(data)

    # Analyze results
    print_model_summary(trace)
    plot_posterior_distributions(trace)
    compare_to_majority(data, trace)
