import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 1000

# Generate random values for features (1 to 5 scale)
data = {
    "Anxiety": np.random.randint(1, 6, num_samples),
    "Stress": np.random.randint(1, 6, num_samples),
    "Anger": np.random.randint(1, 6, num_samples),
    "Sleep Issues": np.random.randint(1, 6, num_samples),
    "Mood Swings": np.random.randint(1, 6, num_samples),
    "Depression": np.random.randint(1, 6, num_samples)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define target: Mental Health Status (0 = Low, 1 = Medium, 2 = High)
df["Mental_Health_Status"] = np.where(
    (df["Anxiety"] + df["Stress"] + df["Depression"]) > 12,  # If high in these
    2,  # High Risk
    np.where(
        (df["Anxiety"] + df["Stress"] + df["Depression"]) > 8,  # If moderate
        1,  # Medium Risk
        0  # Low Risk
    )
)

# Save to CSV
df.to_csv("mental_health_data.csv", index=False)
print("✅ Large dataset (1000+ rows) saved as mental_health_data.csv")
