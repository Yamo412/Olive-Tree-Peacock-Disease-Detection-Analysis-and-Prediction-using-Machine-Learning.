import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import calendar
import numpy as np



# Function to read and preprocess the data
def read_and_preprocess(filepath):
    # Reading the data from the Excel file
    df = pd.read_excel(filepath)

    # Converting the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # Handling missing data: Dropping rows where any of the essential data is missing
    essential_columns = ['Trip Num', 'Date', 'City', 'Species', 'Tree Num',
                         'Leaf Type', 'Leaf Num', 'OLS Before', 'OLS After', 'Non Visible Lesions']
    df = df.dropna(subset=essential_columns)

    # Returning the preprocessed dataframe
    return df

# Function to split the data into training, validation, and testing sets
def split_data(data, train_size=0.6, test_size=0.2):
    train_data, remaining_data = train_test_split(data, train_size=train_size, random_state=42)
    adjusted_test_size = test_size / (1 - train_size)
    validation_data, test_data = train_test_split(remaining_data, test_size=adjusted_test_size, random_state=42)
    return train_data, validation_data, test_data


# Helper Function to calculate treatment effectiveness
def create_effectiveness_feature(data):
    # Calculate the treatment effectiveness
    data['Treatment Effectiveness'] = 100 * (data['OLS Before'] - data['OLS After']) / data['OLS Before']

    # Replace infinities and NaNs with 0 or another suitable value
    data['Treatment Effectiveness'] = data['Treatment Effectiveness'].replace([float('inf'), -float('inf')], 0)
    data['Treatment Effectiveness'].fillna(0, inplace=True)  # Replace NaNs with 0

    return data


# Function to analyze treatment effectiveness over time
def analyze_treatment_effectiveness(train_data, validation_data):
    train_data = create_effectiveness_feature(train_data)
    validation_data = create_effectiveness_feature(validation_data)

    feature_cols = ['Trip Num', 'Tree Num', 'Leaf Num', 'OLS Before']
    target_col = 'Treatment Effectiveness'

    train_X = train_data[feature_cols]
    train_y = train_data[target_col]
    val_X = validation_data[feature_cols]
    val_y = validation_data[target_col]

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
    }

    model_results = {}

    for name, model in models.items():
        model.fit(train_X, train_y)
        predictions = model.predict(val_X)
        mse = mean_squared_error(val_y, predictions)
        r2 = r2_score(val_y, predictions)

        model_results[name] = {'MSE': mse, 'R2': r2}

        print(f"Model: {name}, MSE: {mse}, R2: {r2}")

    best_model_mse = min(model_results, key=lambda x: model_results[x]['MSE'])
    best_model_r2 = max(model_results, key=lambda x: model_results[x]['R2'])

    print(f"\nBest Model by MSE: {best_model_mse} (MSE: {model_results[best_model_mse]['MSE']})")
    print(f"Best Model by R2: {best_model_r2} (R2: {model_results[best_model_r2]['R2']})")

    # Calculate and print treatment metrics
    avg_reduction_ols = (train_data['OLS Before'] - train_data['OLS After']).mean()
    improvement_percentage = ((train_data['OLS After'] < train_data['OLS Before']).sum() / len(train_data)) * 100

    print(f"\nAverage Reduction in Olive Leaf Spot (OLS): {avg_reduction_ols}%")
    print(f"Percentage of Trees Showing Improvement: {improvement_percentage}%")

def predict_susceptible_months_to_OLS(train_data, validation_data):
    # Feature Engineering: Extract month from the date
    for data in [train_data, validation_data]:
        data['Month'] = data['Date'].dt.month

    # Selecting relevant features and target
    feature_cols = ['Month', 'Tree Num', 'Leaf Num']  # Add or remove features as needed
    target_col = 'OLS Before'

    # Preparing training and validation sets
    train_X = train_data[feature_cols]
    train_y = train_data[target_col]
    val_X = validation_data[feature_cols]
    val_y = validation_data[target_col]

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
    }

    model_results = {}

    # Train and evaluate models
    for name, model in models.items():
        model.fit(train_X, train_y)

        val_predictions = model.predict(val_X)
        mse = mean_squared_error(val_y, val_predictions)
        r2 = r2_score(val_y, val_predictions)

        model_results[name] = {'MSE': mse, 'R2': r2}

        print(f"Model: {name}, MSE: {mse}, R2: {r2}")

    # Compare models based on MSE and R2
    best_model_mse = min(model_results, key=lambda x: model_results[x]['MSE'])
    best_model_r2 = max(model_results, key=lambda x: model_results[x]['R2'])

    print(f"\nBest Model by MSE: {best_model_mse} (MSE: {model_results[best_model_mse]['MSE']})")
    print(f"Best Model by R2: {best_model_r2} (R2: {model_results[best_model_r2]['R2']})")

    # Predicting OLS for each month using all models and averaging
    avg_month_predictions = {}
    for month in range(1, 13):
        month_df = pd.DataFrame({feature_cols[0]: [month], feature_cols[1]: [0], feature_cols[2]: [0]})
        month_avg_prediction = np.mean([model.predict(month_df)[0] for model in models.values()])
        avg_month_predictions[calendar.month_name[month]] = month_avg_prediction

    # Normalize OLS predictions to a percentage scale using absolute maximum
    max_ols = max(avg_month_predictions.values(), key=abs)
    for month in avg_month_predictions:
        avg_month_predictions[month] = (avg_month_predictions[month] / max_ols) * 100

    # Sorting months by average prediction (percentage)
    sorted_months = sorted(avg_month_predictions.items(), key=lambda x: x[1], reverse=True)

    print("\nMonths Most Susceptible to Olive Leaf Spot (Ranked by Percentage):")
    for i, (month, prediction) in enumerate(sorted_months):
        print(f"{i + 1}. {month}: Average Predicted OLS Susceptibility = {prediction:.2f}%")


#helper function for treatment
def calculate_treatment_duration(data):
    # Sort by Tree Number and Date
    data = data.sort_values(by=['Tree Num', 'Date'])

    # Calculate the difference in days between consecutive entries for the same tree
    data['Next Entry Date'] = data.groupby('Tree Num')['Date'].shift(-1)
    data['Treatment Duration'] = (data['Next Entry Date'] - data['Date']).dt.days

    # Handle NaN values in 'Treatment Duration' (for the last entry of each tree)
    data['Treatment Duration'].fillna(0, inplace=True)  # Or any other logic that fits your dataset

    return data

#helper function for treatment
def format_duration(days):
    # Convert days to total minutes
    total_minutes = days * 24 * 60

    # Extract days, hours, and minutes
    days = int(total_minutes // (24 * 60))
    hours = int((total_minutes % (24 * 60)) // 60)
    minutes = int(total_minutes % 60)

    return f"{days} days, {hours} hours, {minutes} minutes"

def estimate_time_duration_for_effective_treatment(train_data, validation_data):
    # Assuming 'Treatment Duration' is a column in your data. Replace it with the correct column name.
    target_col = 'Treatment Duration'
    feature_cols = ['Month', 'Tree Num', 'Leaf Num']  # Modify as needed

    # Preparing the data
    train_data['Month'] = train_data['Date'].dt.month
    validation_data['Month'] = validation_data['Date'].dt.month

    train_X = train_data[feature_cols]
    train_y = train_data[target_col]
    val_X = validation_data[feature_cols]
    val_y = validation_data[target_col]

    # Define the models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
    }

    model_results = {}

    # Train and evaluate models
    for name, model in models.items():
        model.fit(train_X, train_y)
        predictions = model.predict(val_X)
        mse = mean_squared_error(val_y, predictions)
        r2 = r2_score(val_y, predictions)
        model_results[name] = {'MSE': mse, 'R2': r2}
        print(f"Model: {name}, MSE: {mse}, R2: {r2}")

    # Calculate and format average treatment duration for each month
    avg_month_durations = train_data.groupby('Month')[target_col].mean()

    print("\nAverage Treatment Duration for Each Month:")
    for month, avg_duration in avg_month_durations.items():
        formatted_avg_duration = format_duration(avg_duration)
        print(f"Month: {calendar.month_name[month]}, Average Duration: {formatted_avg_duration}")

# Main execution
if __name__ == "__main__":
    filepath = 'C:/Users/ACER/Desktop/ML/all trips.xlsx'
    data = read_and_preprocess(filepath)
    # Calculate the Treatment Duration
    data = calculate_treatment_duration(data)
    train_data, validation_data, test_data = split_data(data)

    while True:
        # Menu
        print("1. Analyze Treatment Effectiveness Over Time")
        print("2. Predict Most Susceptible Months to OLS")
        print("3. Estimate Time Duration For Effective Treatment")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            analyze_treatment_effectiveness(train_data, validation_data)
        elif choice == '2':
            predict_susceptible_months_to_OLS(train_data, validation_data)
        elif choice == '3':
            estimate_time_duration_for_effective_treatment(train_data, validation_data)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")