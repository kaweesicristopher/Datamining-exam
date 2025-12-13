# Import libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# PART A: DATA PREPARATION

# Loading the dataset
data = {
    'Transaction_ID':[1,2,3,4,5,6,7,8,9,10],
    'Items':[
        ['Bread','Milk','Eggs'],
        ['Bread','Butter'],
        ['Milk','Diapers','Beer'],
        ['Bread','Milk','Butter'],
        ['Milk','Diapers','Bread'],
        ['Beer','Diapers'],
        ['Bread','Milk','Eggs','Butter'],
        ['Eggs','Milk'],
        ['Bread','Diapers','Beer'],
        ['Milk','Butter']
    ]
}

df = pd.DataFrame(data)

# Convert items into a transactions list
transactions = df['Items'].tolist()

# One-hot encoding
te = TransactionEncoder()
encoded_array = te.fit_transform(transactions)
df_encoded = pd.DataFrame(encoded_array, columns=te.columns_)

print("One-hot encoded dataset:")
print(df_encoded)
print("\n")

# ---------------------------
# PART B: APRIORI ALGORITHM
# ---------------------------

# Minimum support = 0.2
freq_items = apriori(df_encoded, min_support=0.2, use_colnames=True)

print("Frequent Itemsets:")
print(freq_items)
print("\n")

# Generate association rules (confidence ≥ 0.5)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)

# Display selected columns
rules_output = rules[['antecedents','consequents','support','confidence','lift']]

print("Association Rules:")
print(rules_output)
print("\n")

# ---------------------------
# PART C: INTERPRETATION (OPTIONAL PRINTING)
# ---------------------------

# Sort by highest lift
top_rules = rules_output.sort_values(by='lift', ascending=False).head(3)

print("Top 3 Rules Based on Lift:")
print(top_rules)
