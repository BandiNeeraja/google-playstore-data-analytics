import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# STEP 1: Load datasets
# -----------------------------
apps = pd.read_csv("apps.csv")
reviews = pd.read_csv("user_reviews.csv")

print("Datasets loaded successfully")

# -----------------------------
# STEP 2: Data Cleaning
# -----------------------------

# Drop missing values
apps.dropna(inplace=True)

# Clean Installs column
apps['Installs'] = apps['Installs'].str.replace('[+,]', '', regex=True)
apps['Installs'] = pd.to_numeric(apps['Installs'], errors='coerce')

# ✅ FINAL PRICE FIX (THIS NEVER FAILS)
apps['Price'] = apps['Price'].str.replace('$', '', regex=True)
apps['Price'] = apps['Price'].replace('Free', '0')
apps['Price'] = pd.to_numeric(apps['Price'], errors='coerce')

print("Data cleaning completed")

# -----------------------------
# STEP 3: Category Analysis
# -----------------------------
plt.figure(figsize=(10, 6))
sns.countplot(y=apps['Category'], order=apps['Category'].value_counts().index)
plt.title("Number of Apps in Each Category")
plt.show()

# -----------------------------
# STEP 4: Rating Distribution
# -----------------------------
plt.figure(figsize=(6, 4))
sns.histplot(apps['Rating'], bins=10)
plt.title("App Ratings Distribution")
plt.show()

# -----------------------------
# STEP 5: Free vs Paid Apps
# -----------------------------
apps['Type'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    title="Free vs Paid Apps"
)
plt.ylabel("")
plt.show()

# -----------------------------
# STEP 6: Sentiment Analysis
# -----------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x=reviews['Sentiment'])
plt.title("User Sentiment Analysis")
plt.show()

print("🎉 SUCCESS: Project completed without errors!")
