from os import error
import pandas as pd
import matplotlib.pyplot as plt #matplotlib-3.10.1
import seaborn as sns #seaborn-0.13.2

## Read the raw data
raw_df = pd.read_csv('data/AxionRay_assignment.csv')

########### Data Cleaning: starts ###########
## Check the first few rows
raw_df.head()
print(raw_df.shape)

## Check the data types & null values
raw_df.info()
print(raw_df.columns)

## Handle missing values in columns: CAUSAL_VERBATIM, CAUSAL_CD_DESC, IN_USE_DATE
raw_df[raw_df['CAUSAL_VERBATIM'].isna()] #Row 110 with spanish text in other cols
raw_df[raw_df['CAUSAL_CD_DESC'].isna()] #3 rows have nulls
raw_df.loc[:,['CAUSAL_VERBATIM','CAUSAL_CD_DESC']] = raw_df.loc[:,['CAUSAL_VERBATIM','CAUSAL_CD_DESC']].fillna('NA')
# Note: For CAUSAL_CD_DESC, we can also predict the category based on business rules if available, or predict it using a classifier like RF

## Check for unique entries
raw_df["Event id"].nunique() # check for duplicates
raw_df["MAKE"].unique() #['NovaSprint', 'NebulaCruiser', 'ThunderVolt', 'TurboFlux']]
raw_df["MODEL"].nunique() 
raw_df["MODEL"].unique() # has 23 unique models
raw_df["MODLYR"].unique() # 1 unique i.e. 2020
raw_df["BUILD_PLANT_DESC"].nunique() # 8 unique plants
raw_df["PLANT"].nunique() # acronyms for 8 unique plants
raw_df["COMPLAINT_CD_DESC"].unique() # has 2 unique plants
raw_df["CAUSAL_CD_DESC"].nunique() # has 38 unique plants


## Standardize the date column
raw_df['Opened date'] = pd.to_datetime(raw_df['Opened date'], errors='coerce')
raw_df['BUILD_DATE'] = pd.to_datetime(raw_df['BUILD_DATE'], errors='coerce')
raw_df['IN_USE_DATE'] = pd.to_datetime(raw_df['IN_USE_DATE'], errors='coerce')

# Create a flag column for missing values
raw_df['IN_USE_DATE_MISSING'] = raw_df['IN_USE_DATE'].isna()

## Impute IN_USE_DATE with BUILD_DATE+30 days (using business rule-assumed) if missing
raw_df['IN_USE_DATE'] = raw_df['IN_USE_DATE'].fillna(raw_df['BUILD_DATE'] + pd.Timedelta(days=30))
raw_df['IN_USE_DATE'].isna().sum()
raw_df.info()

complaints_df = raw_df.copy()
del(raw_df) # delete the raw_df to free up memory
complaints_df.to_csv('data/complaints_df.csv', index=False)

########### Data Exploration: starts ###########
complaints_df.head()

# Create a new column for time difference (days) between BUILD_DATE and IN_USE_DATE
complaints_df['Days_to_Use'] = (complaints_df['IN_USE_DATE'] - complaints_df['Opened date']).dt.days

# List of critical columns based on stakeholder insight requirements
critical_columns = ['Opened date', 'BUILD_DATE', 'IN_USE_DATE', 'Days_to_Use',
                    'IN_USE_DATE_MISSING', 'MAKE', 'MODEL', 'PLANT', 
                    'CAUSAL_CD_DESC', 'COMPLAINT_CD_DESC', 'Failure Component', 
                    'Failure Condition']

print("Summary statistics for numerical columns:")
print(complaints_df[['Days_to_Use']].describe())

# For date distributions, let’s plot histograms of dates
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.histplot(complaints_df['Opened date'].dropna(), bins=20, kde=True)
plt.title("Opened Date Distribution")
plt.xlabel("Opened Date", fontsize=10)
plt.xticks(rotation=45, fontsize=8)

plt.subplot(1, 3, 2)
sns.histplot(complaints_df['BUILD_DATE'].dropna(), bins=20, kde=True, color='green')
plt.title("Build Date Distribution")
plt.xlabel("Build Date", fontsize=10)
plt.xticks(rotation=45, fontsize=8)

plt.subplot(1, 3, 3)
sns.histplot(complaints_df['IN_USE_DATE'].dropna(), bins=20, kde=True, color='orange')
plt.title("In Use Date Distribution")
plt.xlabel("In Use Date", fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()
plt.show()

# Plot the distribution of the Days_to_Use column
plt.figure(figsize=(8, 4))
sns.boxplot(x=complaints_df['Days_to_Use'])
plt.title("Distribution of Days to Use (In Use - Opened)")
plt.xlabel("Days to Use", fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.show()

# Categorical variables: Distribution of MAKE, MODEL, and PLANT (only top 10 for clarity)
categorical_cols = ['MAKE', 'MODEL', 'BUILD_PLANT_DESC']
for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    order = complaints_df[col].value_counts().head(10).index
    sns.countplot(data=complaints_df, x=col, order=order, palette="viridis")
    plt.title(f"Top 10 counts for {col}")
    plt.xticks(rotation=45)
    plt.show()

