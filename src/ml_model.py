"""
AI Financial Crime Investigator
Machine Learning Detection Layer

This script:

1. Loads the synthetic transaction dataset
2. Creates behavioral features for each account
3. Trains an Isolation Forest anomaly-detection model
4. Produces a risk score for every account
5. Saves the results for the investigation system

IMPORTANT:
    The `is_suspicious` column is NOT used as an input feature.

    It is ground truth that we created ourselves when generating
    the synthetic data. We keep it only so that we can evaluate
    the model later.
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# 2. LOAD THE TRANSACTION DATA
# ============================================================

INPUT_PATH = "data/transactions.csv"

print("Loading transaction data...")

df = pd.read_csv(INPUT_PATH)

# Convert timestamp from text into an actual datetime.
df["timestamp"] = pd.to_datetime(df["timestamp"])


print("\nDataset loaded successfully.")

print(f"Transactions: {len(df):,}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 3. CREATE A LIST OF ALL ACCOUNTS
# ============================================================

# An account can appear either as:
#     sender
# or
#     receiver
#
# We need every account in our final feature table.

sender_accounts = set(df["sender_id"])
receiver_accounts = set(df["receiver_id"])

all_accounts = sorted(
    sender_accounts.union(receiver_accounts)
)

print(f"Unique accounts: {len(all_accounts):,}")


# ============================================================
# 4. CREATE BASIC ACCOUNT FEATURES
# ============================================================

print("\nCreating behavioral features...")


# ------------------------------------------------------------
# 4.1 Total money sent
# ------------------------------------------------------------

money_sent = (
    df.groupby("sender_id")["amount"]
    .sum()
    .rename("total_money_sent")
)


# ------------------------------------------------------------
# 4.2 Total money received
# ------------------------------------------------------------

money_received = (
    df.groupby("receiver_id")["amount"]
    .sum()
    .rename("total_money_received")
)


# ------------------------------------------------------------
# 4.3 Number of transactions sent
# ------------------------------------------------------------

transactions_sent = (
    df.groupby("sender_id")
    .size()
    .rename("transactions_sent")
)


# ------------------------------------------------------------
# 4.4 Number of transactions received
# ------------------------------------------------------------

transactions_received = (
    df.groupby("receiver_id")
    .size()
    .rename("transactions_received")
)


# ------------------------------------------------------------
# 4.5 Average amount sent
# ------------------------------------------------------------

average_sent = (
    df.groupby("sender_id")["amount"]
    .mean()
    .rename("average_amount_sent")
)


# ------------------------------------------------------------
# 4.6 Average amount received
# ------------------------------------------------------------

average_received = (
    df.groupby("receiver_id")["amount"]
    .mean()
    .rename("average_amount_received")
)


# ============================================================
# 5. UNIQUE CONNECTION FEATURES
# ============================================================

# These features tell us how many different accounts
# an account interacts with.

# ------------------------------------------------------------
# 5.1 Number of unique recipients
# ------------------------------------------------------------

unique_recipients = (
    df.groupby("sender_id")["receiver_id"]
    .nunique()
    .rename("unique_recipients")
)


# ------------------------------------------------------------
# 5.2 Number of unique senders
# ------------------------------------------------------------

unique_senders = (
    df.groupby("receiver_id")["sender_id"]
    .nunique()
    .rename("unique_senders")
)


# ============================================================
# 6. COMBINE FEATURES INTO ONE ACCOUNT TABLE
# ============================================================

# Start with every account.

features = pd.DataFrame(
    index=all_accounts
)

features.index.name = "account_id"


# Add all previously calculated features.

feature_tables = [
    money_sent,
    money_received,
    transactions_sent,
    transactions_received,
    average_sent,
    average_received,
    unique_recipients,
    unique_senders
]


for feature in feature_tables:

    features = features.join(
        feature,
        how="left"
    )


# Accounts that don't have a particular type of transaction
# will have NaN values.
#
# For example:
# An account that never sent money has no "money_sent" value.
#
# We replace those missing values with 0.

features = features.fillna(0)


# ============================================================
# 7. TOTAL TRANSACTION FREQUENCY
# ============================================================

features["transaction_frequency"] = (
    features["transactions_sent"]
    +
    features["transactions_received"]
)


# ============================================================
# 8. INCOMING / OUTGOING RATIO
# ============================================================

# This measures how much money an account receives
# compared with how much it sends.

#
# Example:
#
# received = 100,000
# sent     = 95,000
#
# ratio ≈ 1.05
#
# A ratio close to 1 can indicate that money coming into
# an account is being moved onward.

features["incoming_outgoing_ratio"] = (
    features["total_money_received"]
    /
    (features["total_money_sent"] + 1)
)


# ============================================================
# 9. TRANSACTION AMOUNT VARIABILITY
# ============================================================

# Calculate the standard deviation of transaction amounts
# for each account.
#
# High variability means the account is handling transactions
# of very different sizes.

amount_std = (
    df.groupby("sender_id")["amount"]
    .std()
    .rename("amount_std")
)

features = features.join(
    amount_std,
    how="left"
)

features["amount_std"] = features["amount_std"].fillna(0)


# ============================================================
# 10. RAPID TRANSFER FEATURE
# ============================================================

"""
We want to identify accounts that make multiple transactions
very close together in time.

Example:

09:00 → transaction
09:02 → transaction
09:04 → transaction

This can represent rapid movement of funds.

For each sender, we calculate the time difference between
consecutive outgoing transactions.
"""

df_sorted = df.sort_values(
    ["sender_id", "timestamp"]
).copy()


# Calculate the difference between consecutive transactions
# for the same sender.

df_sorted["time_since_previous"] = (
    df_sorted
    .groupby("sender_id")["timestamp"]
    .diff()
)


# Convert the time difference to minutes.

df_sorted["minutes_since_previous"] = (
    df_sorted["time_since_previous"]
    .dt.total_seconds()
    / 60
)


# Count transactions that happened within 10 minutes
# of the previous transaction.

rapid_transfers = (
    df_sorted[
        df_sorted["minutes_since_previous"] <= 10
    ]
    .groupby("sender_id")
    .size()
    .rename("rapid_transfers")
)


features = features.join(
    rapid_transfers,
    how="left"
)

features["rapid_transfers"] = (
    features["rapid_transfers"]
    .fillna(0)
)


# ============================================================
# 11. NETWORK CONNECTIVITY
# ============================================================

"""
A simple network-connectivity feature:

unique recipients + unique senders

An account interacting with many different accounts
has a larger transaction network.
"""

features["network_connections"] = (
    features["unique_recipients"]
    +
    features["unique_senders"]
)


# ============================================================
# 12. CREATE THE ML FEATURE MATRIX
# ============================================================

# IMPORTANT:
#
# We deliberately DO NOT include:
#
#     is_suspicious
#     pattern_type
#
# Those columns reveal the answer to the model.
#
# The model should learn from behavior instead.

ml_features = [
    "total_money_sent",
    "total_money_received",
    "transactions_sent",
    "transactions_received",
    "average_amount_sent",
    "average_amount_received",
    "unique_recipients",
    "unique_senders",
    "transaction_frequency",
    "incoming_outgoing_ratio",
    "amount_std",
    "rapid_transfers",
    "network_connections"
]


X = features[ml_features]


print("\nML feature matrix created.")

print(f"Accounts: {X.shape[0]:,}")
print(f"Features: {X.shape[1]}")


# ============================================================
# 13. SCALE THE FEATURES
# ============================================================

"""
Different features have very different numerical ranges.

For example:

total_money_sent
    → potentially hundreds of thousands

unique_recipients
    → maybe 1–20

rapid_transfers
    → maybe 0–10

Scaling puts them onto a comparable numerical scale.
"""

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# 14. TRAIN THE ISOLATION FOREST
# ============================================================

print("\nTraining Isolation Forest...")


"""
Isolation Forest is an anomaly-detection algorithm.

The idea:

Normal behavior:
    Many accounts behave similarly.

Anomalous behavior:
    Some accounts behave very differently.

The model attempts to isolate unusual observations.

contamination:
    Our first estimate of the proportion of unusual accounts.

We start with 10%.

This is a parameter we can tune later.
"""

model = IsolationForest(
    n_estimators=200,
    contamination=0.10,
    random_state=42,
    n_jobs=-1
)


model.fit(X_scaled)


# ============================================================
# 15. GENERATE ANOMALY SCORES
# ============================================================

# Isolation Forest's decision_function produces
# larger values for more normal observations.
#
# We reverse the score so that:
#
# higher risk score = more anomalous

raw_scores = model.decision_function(X_scaled)

features["anomaly_score"] = -raw_scores


# ============================================================
# 16. CONVERT SCORES INTO 0–100 RISK SCORES
# ============================================================

"""
The anomaly score isn't naturally between 0 and 100.

We normalize it so that our dashboard can display:

    0   = lower anomaly
    100 = higher anomaly
"""

minimum_score = features["anomaly_score"].min()
maximum_score = features["anomaly_score"].max()


features["risk_score"] = (
    (
        features["anomaly_score"]
        -
        minimum_score
    )
    /
    (
        maximum_score
        -
        minimum_score
    )
) * 100


# Round the score for readability.

features["risk_score"] = (
    features["risk_score"]
    .round(2)
)


# ============================================================
# 17. CREATE RISK LEVELS
# ============================================================

def assign_risk_level(score):
    """
    Convert the numerical risk score into a readable category.
    """

    if score >= 75:
        return "High"

    elif score >= 50:
        return "Medium"

    else:
        return "Low"


features["risk_level"] = (
    features["risk_score"]
    .apply(assign_risk_level)
)


# ============================================================
# 18. SORT ACCOUNTS BY RISK
# ============================================================

results = features.sort_values(
    by="risk_score",
    ascending=False
)


# ============================================================
# 19. SAVE THE RESULTS
# ============================================================

OUTPUT_PATH = "data/account_risk_scores.csv"

results.to_csv(
    OUTPUT_PATH
)


# ============================================================
# 20. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("ML detection completed!")
print("========================================")

print("\nHighest-risk accounts:")

print(
    results[
        [
            "risk_score",
            "risk_level",
            "transaction_frequency",
            "total_money_sent",
            "total_money_received",
            "unique_recipients",
            "unique_senders",
            "rapid_transfers",
            "network_connections"
        ]
    ]
    .head(10)
    .to_string()
)


print("\nRisk-level distribution:")

print(
    results["risk_level"]
    .value_counts()
)


print(f"\nResults saved to: {OUTPUT_PATH}")
