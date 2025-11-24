## Traffic Accident Prediction – Data Preprocessing & Exploratory Data Analysis

This project involves preparing a traffic accident dataset for future predictive modeling. The notebook performs extensive data cleaning, feature engineering, and exploratory data analysis to identify relationships between different traffic-related features and accident occurrences.

## NAME: PAVITRA J
## REG NO: 212224110043

## Objective

* Clean and organize the accident dataset
* Handle missing or duplicate records
* Fix inconsistent or unrealistic values
* Analyze influence of driver and road conditions on accidents
* Encode and scale features for model readiness
* Generate visual insights into patterns and correlations

## Dataset Information

* Original dataset: dataset_traffic_accident_prediction1.csv
* Cleaned dataset output: Cleaned_Accidents.csv
* Target feature: Accident
  (0 = No accident, 1 = Accident occurred)

## Features analyzed include:

Weather, Road Type, Road Condition, Time of Day, Vehicle Type, Speed Limit, Traffic Density, Driver Age and Experience, Road Light Condition
Number of Vehicles involved, Accident Severity.

## Tools and Libraries Used

Python, Pandas and NumPy for data manipulation, Matplotlib and Seaborn for visualizations
Label Encoding for categorical data transformation, StandardScaler for numerical scaling.

## PROGRAM AND OUTPUT
````
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("/content/dataset_traffic_accident_prediction1.csv")


df = df.drop_duplicates()
df


cat_cols = ['Weather', 'Road_Type', 'Time_of_Day', 'Road_Condition',
            'Vehicle_Type', 'Accident_Severity', 'Road_Light_Condition']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


num_cols = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles',
            'Driver_Alcohol', 'Driver_Age', 'Driver_Experience', 'Accident']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
df


df.loc[df['Speed_Limit'] > 150, 'Speed_Limit'] = 150
df


df['Accident'] = df['Accident'].astype(int)
df['Number_of_Vehicles'] = df['Number_of_Vehicles'].astype(int)
df['Driver_Age'] = df['Driver_Age'].astype(int)
df['Driver_Alcohol'] = df['Driver_Alcohol'].astype(int)
df


df = df.reset_index(drop=True)
df.to_csv("Cleaned_Accidents.csv", index=False)

df
````

<img width="1588" height="373" alt="image" src="https://github.com/user-attachments/assets/75a6c9bd-d9f0-474b-aa39-4414b3d605f6" />


````
# EDA

pd.set_option('display.max_columns', None)


df = pd.read_csv("Cleaned_Accidents.csv")


print("Shape of dataset:", df.shape)
print("\nColumn names:\n", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isna().sum())

print("\nStatistical Summary:\n", df.describe())


cat_cols = ['Weather', 'Road_Type', 'Time_of_Day', 'Road_Condition',
            'Vehicle_Type', 'Accident_Severity', 'Road_Light_Condition']

for col in cat_cols:
    print(f"\nValue counts for {col}:\n", df[col].value_counts())

    df[col].value_counts().plot(kind='bar')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()


num_cols = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles',
            'Driver_Alcohol', 'Driver_Age', 'Driver_Experience']

for col in num_cols:
    plt.hist(df[col])
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


corr = df.corr(numeric_only=True)
print("\nCorrelation Matrix:\n", corr)

plt.figure(figsize=(10, 6))
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title("Correlation Heatmap")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()


plt.figure()
df.groupby('Accident_Severity')['Accident'].count().plot(kind='bar')
plt.title("Severity vs Number of Accidents")
plt.xlabel("Severity Level")
plt.ylabel("Count")
plt.show()

````

<img width="1740" height="639" alt="image" src="https://github.com/user-attachments/assets/56a88cd8-9611-4fe0-b3e2-d3d610da0e37" />
<img width="626" height="491" alt="image" src="https://github.com/user-attachments/assets/f5bb1b8b-1ec2-4dba-b4f1-d906fa4fc6a3" />
<img width="633" height="595" alt="image" src="https://github.com/user-attachments/assets/6e4f2015-f6a0-45c4-b0db-a57924b56a1d" />
<img width="580" height="642" alt="image" src="https://github.com/user-attachments/assets/f6398555-0534-4b47-ac79-c4c07f2353d1" />
<img width="549" height="608" alt="image" src="https://github.com/user-attachments/assets/7727a248-93b0-4fde-8e77-0f7b9a1d4619" />
<img width="555" height="680" alt="image" src="https://github.com/user-attachments/assets/85ec22bd-0cce-491c-9d3e-dbe2e324c1ae" />
<img width="566" height="597" alt="image" src="https://github.com/user-attachments/assets/f35fa016-590f-46b0-963c-4492a682b8d4" />
<img width="624" height="317" alt="image" src="https://github.com/user-attachments/assets/a56bdebd-0a1f-4b11-b085-9dc944076ed3" />
<img width="719" height="574" alt="image" src="https://github.com/user-attachments/assets/b22a5683-fe0f-4e97-aa9d-c87a1be6b1a9" />


````
# FEATURE ENCODING AND SCALING
from sklearn.preprocessing import LabelEncoder, StandardScaler

label_enc = LabelEncoder()
scaler = StandardScaler()

cat_cols = ['Weather', 'Road_Type', 'Time_of_Day', 'Road_Condition',
            'Vehicle_Type', 'Accident_Severity', 'Road_Light_Condition']

for col in cat_cols:
    df[col] = label_enc.fit_transform(df[col])

num_cols = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles',
            'Driver_Alcohol', 'Driver_Age', 'Driver_Experience']

df[num_cols] = scaler.fit_transform(df[num_cols])

print("Feature Encoding & Transformation Completed Successfully ")
print(df.head())

````

<img width="595" height="360" alt="image" src="https://github.com/user-attachments/assets/3856386a-f365-4c07-9ff8-4b259f3052c5" />


````
# FEATURE SELECTION
from sklearn.feature_selection import SelectKBest, mutual_info_classif

X = df.drop('Accident', axis=1)  # Features
y = df['Accident']              # Target

selector = SelectKBest(score_func=mutual_info_classif, k=5)
selector.fit(X, y)

selected_columns = X.columns[selector.get_support()]

print("\n Selected Best Features for Prediction:")
print(list(selected_columns))

````

<img width="586" height="56" alt="image" src="https://github.com/user-attachments/assets/b06f1054-6a33-49d6-b969-4136301704b4" />


````
plt.figure(figsize=(6,4))
df['Accident'].value_counts().plot(kind='bar')
plt.title("Accident Target Variable Distribution")
plt.xlabel("Accident (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df['Driver_Age'], bins=20)
plt.title("Driver Age Distribution")
plt.xlabel("Driver Age (Scaled)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df['Number_of_Vehicles'], bins=15)
plt.title("Number of Vehicles Involved")
plt.xlabel("Vehicles (Scaled)")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(6,4))
df['Road_Type'].value_counts().plot(kind='bar')
plt.title("Distribution of Road Types")
plt.xlabel("Road Type (Encoded)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
df['Accident_Severity'].value_counts().plot(kind='bar')
plt.title("Accident Severity Levels")
plt.xlabel("Severity (Encoded)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df['Traffic_Density'], bins=20)
plt.title("Traffic Density Distribution")
plt.xlabel("Traffic Density (Scaled)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df['Speed_Limit'], bins=15)
plt.title("Speed Limit Distribution")
plt.xlabel("Speed Limit (Scaled)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
````

<img width="534" height="743" alt="image" src="https://github.com/user-attachments/assets/9a2c1b23-70cf-47f6-ab24-35c5bbf6041e" />
<img width="527" height="753" alt="image" src="https://github.com/user-attachments/assets/34df847d-0717-44f9-a094-3d53d8a90b5a" />
<img width="561" height="744" alt="image" src="https://github.com/user-attachments/assets/6e368c6b-d83f-42c4-8b63-519f8c87aae1" />
<img width="552" height="371" alt="image" src="https://github.com/user-attachments/assets/50362a3c-684f-4acd-90b0-eac27c1e05ac" />
<img width="878" height="639" alt="image" src="https://github.com/user-attachments/assets/f0eef0a1-947b-4555-8c32-485e278789d9" />


````
sns.set(style="whitegrid")
df = pd.read_csv("Cleaned_Accidents.csv")

plt.figure(figsize=(6,4))
sns.countplot(x='Accident', data=df)
plt.title("Accident Distribution")
plt.show()

plt.figure(figsize=(7,5))
sns.countplot(x='Road_Type', hue='Accident', data=df)
plt.title("Road Type vs Accident")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(7,5))
sns.histplot(df['Driver_Age'], kde=True, bins=20)
plt.title("Driver Age Distribution")
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x="Accident", y="Speed_Limit", data=df)
plt.title("Speed Limit vs Accident")
plt.show()

num_cols = ['Traffic_Density','Speed_Limit','Number_of_Vehicles',
            'Driver_Alcohol','Driver_Age','Driver_Experience','Accident']
plt.figure(figsize=(10,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Numerical Only)")
plt.show()

sns.pairplot(df[['Accident','Driver_Age','Speed_Limit','Traffic_Density']], hue='Accident')
plt.show()
````

<img width="549" height="364" alt="image" src="https://github.com/user-attachments/assets/924898a4-0f60-4e91-a591-66df28d65de0" />
<img width="614" height="534" alt="image" src="https://github.com/user-attachments/assets/8e29d6f9-f5ad-4bf6-b8cb-cdd956b1d46f" />
<img width="629" height="459" alt="image" src="https://github.com/user-attachments/assets/adce13c3-b6ce-4b8e-9e14-bbccf5b5271d" />
<img width="609" height="437" alt="image" src="https://github.com/user-attachments/assets/2839b108-8818-42a6-85d7-2af4673f9de9" />
<img width="907" height="616" alt="image" src="https://github.com/user-attachments/assets/ba320971-5ba0-45aa-98f3-dc0b6e6cf32a" />
<img width="821" height="700" alt="image" src="https://github.com/user-attachments/assets/178ca278-5665-4b1a-adac-f45051fadcb5" />


## Exploratory Data Analysis Performed

* Removed duplicate records
* Filled missing values using mode (categorical) and median (numerical)
* Corrected unrealistic values (example: speed limit capped at 150)
* Converted datatypes for model compatibility
* Reset index and saved cleaned dataset
* Generated descriptive statistics and dataset summary
* Created visualizations such as:
* Histograms for numerical features
* Count plots for categorical features
* Correlation heatmap
* Pairplots to observe relationships between key features
* Analysis of accident severity and accident counts

## Key Findings

* Higher traffic density and high speed values show stronger association with accidents
* Driver-related factors such as age and experience play an important role
* Accident rates vary based on lighting conditions and road type
* Correlation analysis reveals multiple interlinked features worth considering for prediction modeling


## Future Work

* Build and evaluate machine learning models to predict accident likelihood
* Perform hyperparameter tuning to improve model performance
* Conduct feature importance analysis for better insights
* Deploy the trained model as a user-friendly application

## Conclusion

The dataset was cleaned, analyzed, and transformed to understand key accident factors and prepare the data for machine learning.
The insights and visualizations highlight how traffic, driver behavior, and road conditions influence accident occurrence, providing a solid base for future prediction models.
