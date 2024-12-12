air_quality
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Read the CSV file from GitHub
url = 'https://raw.githubusercontent.com/kssandraeshwar/Data-Visualisation_course/refs/heads/main/AirQuality.csv'
df = pd.read_csv(url)

# Initial Data Exploration
print("Original Dataset Information:")
print(df.info())

# Identify numeric and categorical columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Handle Negative Values
def handle_negative_values(df, columns):
    for col in columns:
        # Check if column has negative values
        if (df[col] < 0).any():
            print(f"Negative values found in {col}")
            
            # Replace negative values with absolute values
            df[col] = np.abs(df[col])
    return df

# Apply negative value handling
df = handle_negative_values(df, numeric_columns)

# Detailed Missing Value Analysis
print("\nMissing Values:")
missing_values = df.isnull().sum()
missing_percentages = 100 * df.isnull().sum() / len(df)
missing_table = pd.concat([missing_values, missing_percentages], axis=1, keys=['Missing Values', 'Percentage'])
print(missing_table)

# Comprehensive Missing Value Handling
# For numeric columns
for col in numeric_columns:
    # Handle extreme outliers with the median
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Replace extreme values with median
    df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = df[col].median()
    
    # Fill remaining missing values with median
    df[col].fillna(df[col].median(), inplace=True)

# For categorical columns
for col in categorical_columns:
    # Fill missing values with mode
    df[col].fillna(df[col].mode()[0], inplace=True)

# Remove Duplicate Rows
df.drop_duplicates(inplace=True)

# Normalization
# Create a copy of numeric columns
numeric_df = df[numeric_columns]

# Apply Min-Max Scaling to ensure all values are between 0 and 1
min_max_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    min_max_scaler.fit_transform(numeric_df), 
    columns=numeric_columns,
    index=numeric_df.index
)

# Combine Normalized Numeric and Original Categorical Columns
df_final = pd.concat([df_normalized, df[categorical_columns]], axis=1)

# Final Dataset Information
print("\nFinal Preprocessed Dataset Information:")
print(df_final.info())

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(df_final.describe())

# Save Preprocessed Data
df_final.to_csv('preprocessed_air_quality.csv', index=False)
print("Preprocessing Complete. Preprocessed file saved as 'preprocessed_air_quality.csv'")

# Dataset dimensions
numeric_columns_count = len(numeric_columns)
numeric_rows = df.shape[0]
print('Number of numeric columns:', numeric_columns_count)
print('Number of rows:', numeric_rows)

# Dataset data types
data_types = df.dtypes
print('Data types:', data_types)

# Check null values
null_values = df.isnull().sum()
print('Null values present:', null_values)

#Boxplot to identify the pollutant Levels, by printing the numeric values of each


pollutant_columns = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)'
]

# Ensuring pollutants are absolute
df[pollutant_columns] = df[pollutant_columns].abs()

# Plotting the boxplot
plt.figure(figsize=(14, 7))
sns.boxplot(data=df[pollutant_columns], palette="pastel")

# Adding numeric values above each box
for col_index, column in enumerate(pollutant_columns):
    col_data = df[column].dropna()  # Exclude nulls for correct stats
    median_value = col_data.median()
    plt.text(x=col_index, y=median_value + (0.05 * median_value), 
             s=f"{median_value:.2f}", ha='center', color='black', fontsize=10)

plt.title("Boxplot of Pollutant Levels", fontsize=16)
plt.ylabel("Pollutant Levels", fontsize=12)
plt.xlabel("Pollutants", fontsize=12)
plt.xticks(ticks=range(len(pollutant_columns)), labels=pollutant_columns, rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# Contribution of pollutants pie chart
pollutants = ['CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)']  # Example pollutants
existing_pollutants = [pollutant for pollutant in pollutants if pollutant in df.columns]

if existing_pollutants:
    pollutant_contributions = df[existing_pollutants].sum()
    plt.figure(figsize=(8, 8))
    pollutant_contributions.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Contribution of Pollutants')
    plt.ylabel('')  # Remove default ylabel for better aesthetics
    plt.show()
else:
    print("None of the specified pollutants are found in the dataset.")

# Scatter Plot for NOx vs Temperature 
if 'NOx(GT)' in df.columns:
    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['NOx(GT)'].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of NOx')
    plt.xlabel('NOx Concentration')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
else:
    print("NOx column not found in the dataset.")
if 'NOx(GT)' in df.columns and 'T' in df.columns:
    # Plotting the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['T'], df['NOx(GT)'], color='blue', alpha=0.6)
    plt.title('Scatter Plot of NOx vs Temperature')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('NOx Concentration')
    plt.grid(True)
    plt.show()
else:
    print("One or both of the columns 'NOx' and 'Temperature' are not found in the dataset.")

# Line Chart for the trend of C6H6 over Time
# Ensure 'Start Time' and 'C6H6' columns exist
if 'Time' in df.columns and 'C6H6(GT)' in df.columns:
    # Convert 'Start Time' to datetime format
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

    # Drop rows where 'C6H6' or 'Start Time' is NaT or NaN
    df = df.dropna(subset=['Time', 'C6H6(GT)'])

    # Plotting the line chart
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time'], df['C6H6(GT)'], color='green', marker='o', linestyle='-', alpha=0.7)
    plt.title('Trend of C6H6 (Benzene) Over Time')
    plt.xlabel('Time')
    plt.ylabel('C6H6 Concentration (ppm)')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid()
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()
else:
   
    print("One or both of the columns 'Start Time' and 'C6H6' are not found in the dataset.")

#Stacked Bar Chart for Monthly Pollutant Levels
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

# Extract month and year from 'Start Time'
df['Month'] = df['Time'].dt.month
df['Year'] = df['Time'].dt.year

# Group by Year and Month and calculate the sum of each pollutant (assuming columns are named accordingly)
monthly_pollutants = df.groupby(['Year', 'Month'])[['NOx(GT)', 'C6H6(GT)', 'T']].sum().reset_index()
monthly_pollutants_pivot = monthly_pollutants.pivot(index='Month', columns='Year', values=['NOx(GT)', 'C6H6(GT)', 'T'])

# Plotting the stacked bar chart
monthly_pollutants_pivot.plot(kind='bar', stacked=True, figsize=(12, 7))

plt.title('Monthly Pollutant Levels')
plt.xlabel('Month')
plt.ylabel('Concentration')
plt.xticks(rotation=0)
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y')
plt.tight_layout()  # Adjust layout
plt.show()

#Draw the Bar Chart for hourly average NOx 
# Ensure 'Start Time' is in datetime format
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

# Extract hour from 'Start Time'
df['Hour'] = df['Time'].dt.hour

# Calculate hourly average of NOx (assuming NOx is in a column named 'NOx')
hourly_avg_nox = df.groupby('Hour')['NOx(GT)'].mean()

# Plotting the bar chart
plt.figure(figsize=(10, 6))
hourly_avg_nox.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Hourly Average NOx Levels')
plt.xlabel('Hour of Day')
plt.ylabel('Average NOx Concentration (ppm)')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

