# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import category_encoders as ce

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv("fraud_oracle.csv")
    return data.copy()

# Data Preprocessing
def preprocess_data(data):
    data['Age'] = data['Age'].replace(0, float('nan'))
    nonfraud_median = data[data['FraudFound_P'] == 0]['Age'].median()
    data['Age'].fillna(nonfraud_median, inplace=True)
    data_clean = data
    return data_clean

# Exploratory Data Analysis
def eda(data):
    st.subheader('Fraud Rate by Gender')
    fraud_rate_gender = data.groupby('Sex')['FraudFound_P'].mean()
    fig, ax = plt.subplots()
    sns.barplot(x=fraud_rate_gender.index, y=fraud_rate_gender.values, palette='coolwarm', ax=ax)
    ax.set_title('Fraud Rate by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Fraud Rate')
    st.pyplot(fig)

    st.subheader('Distribution of Case Submissions by Day of the Week')
    fig, ax = plt.subplots()
    sns.countplot(x='DayOfWeek', data=data, palette='viridis', order=data['DayOfWeek'].value_counts().index, ax=ax)
    ax.set_title('Distribution of Case Submissions by Day of the Week')
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Number of Case Submissions')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    st.subheader('Distribution of Vehicle Prices')
    vehicle_price_counts = data['VehiclePrice'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=vehicle_price_counts.index, y=vehicle_price_counts.values, palette='viridis', ax=ax)
    ax.set_title('Distribution of Vehicle Prices')
    ax.set_xlabel('Vehicle Price')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    st.subheader('Distribution of Vehicle Makes')
    car_type_distribution = data['Make'].value_counts()
    threshold = 0.02 * len(data)
    car_type_simplified = car_type_distribution[car_type_distribution > threshold]
    car_type_simplified['Other'] = car_type_distribution[car_type_distribution <= threshold].sum()
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 7))
    plt.pie(car_type_simplified, labels=car_type_simplified.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', len(car_type_simplified)))
    plt.title('Distribution of Vehicle Makes')
    plt.axis('equal')
    st.pyplot(plt)

    st.subheader('Fraud Rate by Car Type')
    fraud_rate_make = data.groupby('Make')['FraudFound_P'].mean()
    plt.figure(figsize=(14, 8))
    sns.barplot(x=fraud_rate_make.index, y=fraud_rate_make.values, palette='coolwarm')
    plt.title('Fraud Rate by Car Type')
    plt.xlabel('Car Make')
    plt.ylabel('Fraud Rate')
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()
    st.pyplot(plt)

    st.subheader('Distribution of Age of Policy Holders')
    plt.figure(figsize=(12, 6))
    sns.countplot(x='AgeOfPolicyHolder', data=data, palette='viridis', order=data['AgeOfPolicyHolder'].value_counts().index)
    plt.title('Distribution of Age of Policy Holders')
    plt.xlabel('Age of Policy Holder')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    st.pyplot(plt)

    st.subheader('Distribution of Vehicle Prices')
    vehicle_price_counts = data['VehiclePrice'].value_counts()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=vehicle_price_counts.index, y=vehicle_price_counts.values, palette='viridis')
    plt.title('Distribution of Vehicle Prices')
    plt.xlabel('Vehicle Price')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    st.subheader('Distribution of Found Fraud Cases')
    fraud_found_counts = data['FraudFound_P'].value_counts()
    plt.figure(figsize=(8, 5))
    plt.pie(fraud_found_counts, labels=['No Fraud', 'Fraud'], autopct='%1.1f%%', colors=sns.color_palette('coolwarm', len(fraud_found_counts)))
    plt.title('Distribution of Found Fraud Cases')
    plt.axis('equal') 
    st.pyplot(plt)

    st.subheader('Correlation Matrix (Numeric Columns Only)')
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_columns].corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix (Numeric Columns Only)')
    st.pyplot(plt)

lr_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "Decision Tree": dt_model,
    "XGBoost": xgb_model,
    "Gradient Boosting": gb_model
}

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc, report

# Model Training and Evaluation
def train_models(data):
    categorical_cols = [
                'Month', 'DayOfWeek', 'Make', 'AccidentArea', 'DayOfWeekClaimed', 'MonthClaimed', 'Sex', 'MaritalStatus',
                'Fault', 'PolicyType', 'VehicleCategory', 'VehiclePrice', 'Days_Policy_Accident', 'Days_Policy_Claim',
                'PastNumberOfClaims', 'AgeOfVehicle', 'AgeOfPolicyHolder', 'PoliceReportFiled', 'WitnessPresent',
                'AgentType', 'NumberOfSuppliments', 'AddressChange_Claim', 'NumberOfCars', 'BasePolicy'
            ]
    encoder = ce.OrdinalEncoder(cols=categorical_cols)
    data = encoder.fit_transform(data)

    X = data.drop(['FraudFound_P'], axis=1)
    y = data['FraudFound_P']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    metrics = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader(name)
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        st.write(f"Precision: {precision_score(y_test, y_pred):.4f}")
        st.write(f"Recall: {recall_score(y_test, y_pred):.4f}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        st.write('---')
        metric = evaluate_model(model, X_train, X_test, y_train, y_test)
        metrics.append(metric)
    
    # Transform data for plotting
        accuracy = [m[0] for m in metrics]
        precision = [m[1] for m in metrics]
        recall = [m[2] for m in metrics]
        f1 = [m[3] for m in metrics]
        roc_auc = [m[4] for m in metrics]

    # Setting up drawing data
    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width*2, accuracy, width, label='Accuracy')
    rects2 = ax.bar(x - width, precision, width, label='Precision')
    rects3 = ax.bar(x, recall, width, label='Recall')
    rects4 = ax.bar(x + width, f1, width, label='F1-Score')
    rects5 = ax.bar(x + width*2, roc_auc, width, label='ROC-AUC')

    # Add some text labels
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    def autolabel(rects):
        """Add labels to each bar"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    fig.tight_layout()
    st.pyplot(plt)

def model_inference(data):
    
        st.subheader("Input features for prediction")
    
        # Create a form for user inputs
        with st.form(key='inference_form'):
            form_data = {}
            
            months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            dayofweek_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            vehicleprice_order = ["less than 20000", "20000 to 29000", "30000 to 39000", "40000 to 59000", "60000 to 69000", "more than 69000"]
            dayspolicyaccident_order = ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"]
            dayspolicyclaim_order = ["none", "8 to 15", "15 to 30", "more than 30"]
            ageofvehicle_order = ["new", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7"]
            ageofpolicyholder_order = ["16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35", "36 to 40", "41 to 50", "51 to 65", "over 65"]
            addresschangeclaim_order = ["no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years"]
            numberofcars_order = ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"]

            categorical_cols = [
                'Month', 'DayOfWeek', 'Make', 'AccidentArea', 'DayOfWeekClaimed', 'MonthClaimed', 'Sex', 'MaritalStatus',
                'Fault', 'PolicyType', 'VehicleCategory', 'VehiclePrice', 'Days_Policy_Accident', 'Days_Policy_Claim',
                'PastNumberOfClaims', 'AgeOfVehicle', 'AgeOfPolicyHolder', 'PoliceReportFiled', 'WitnessPresent',
                'AgentType', 'NumberOfSuppliments', 'AddressChange_Claim', 'NumberOfCars', 'BasePolicy'
            ]
            
            for col in categorical_cols:
                if col == 'Month' or col == 'MonthClaimed':
                    form_data[col] = st.selectbox(col, months_order)
                elif col == 'DayOfWeek' or col == 'DayOfWeekClaimed':
                    form_data[col] = st.selectbox(col, dayofweek_order)
                elif col == 'VehiclePrice':
                    form_data[col] = st.selectbox(col, vehicleprice_order)
                elif col == 'Days_Policy_Accident':
                    form_data[col] = st.selectbox(col, dayspolicyaccident_order)
                elif col == 'Days_Policy_Claim':
                    form_data[col] = st.selectbox(col, dayspolicyclaim_order)
                elif col == 'AgeOfVehicle':
                    form_data[col] = st.selectbox(col, ageofvehicle_order)
                elif col == 'AgeOfPolicyHolder':
                    form_data[col] = st.selectbox(col, ageofpolicyholder_order)
                elif col == 'AddressChange_Claim':
                    form_data[col] = st.selectbox(col, addresschangeclaim_order)
                elif col == 'NumberOfCars':
                    form_data[col] = st.selectbox(col, numberofcars_order)
                else:
                    form_data[col] = st.selectbox(col, data[col].unique())
            
            numerical_cols = data.drop(categorical_cols + ['FraudFound_P'], axis=1).columns
            for col in numerical_cols:
                form_data[col] = st.number_input(col, value=int(data[col].mean()), step=1)
                
            submit_button = st.form_submit_button(label='Predict')
        
        if submit_button:
            input_data = pd.DataFrame([form_data])
            encoder = ce.OrdinalEncoder(cols=categorical_cols)
            input_data = encoder.fit_transform(data)
            input_data.drop('FraudFound_P', axis=1, inplace=True)
            y = data['FraudFound_P']

            for name, model in models.items():
                st.subheader(f"Model: {name}")
                
                X_train, X_test, y_train, y_test = train_test_split(input_data, y, test_size=0.3, random_state=42)
                model.fit(X_train, y_train)
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0][1]                
                st.write(f"Prediction: {'Fraud' if prediction == 1 else 'No Fraud'}")
                st.write(f"Prediction Probability: {prediction_proba:.4f}")

# Streamlit App Layout
def main():
    st.title("Vehicle Insurance Claim Fraud Detection Analysis :car:")
    st.subheader("ML4DS Group Project")
    st.write("By: Kean Soh Zhe Herng, Chin Zhe Kang, Liang Kai Qi, Li Shi Yao, Siti Iesha Sariah Nor Azman")
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox("Choose an option", ["Data Preprocessing", "Exploratory Data Analysis", "Model Training & Evaluation", "Model Inference"])

    data = load_data()

    if option == "Data Preprocessing":
        st.header("Data Preprocessing")
        data_clean = preprocess_data(data)
        st.subheader("Preview of dataset")
        st.write(data_clean.head())        
        st.write('---')
        st.subheader("Descriptive statistics of dataset")
        st.write(data_clean.describe())

        numerical_cols = data_clean.select_dtypes(include=['int64', 'float']).columns

        for col in numerical_cols:
            Q1 = data_clean[col].quantile(0.25)
            Q3 = data_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data_clean[(data_clean[col] < lower_bound) | (data_clean[col] > upper_bound)]
            print(f"Outliers in {col}: {outliers.shape[0]} instances")

            plt.figure(figsize=(10, 6))
            plt.boxplot(data_clean[col].dropna())
            plt.title(f"Box plot of {col}")
            st.pyplot(plt)

    elif option == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        data_clean = preprocess_data(data)
        eda(data_clean)

    elif option == "Model Training & Evaluation":
        st.header("Model Training & Evaluation")
        data_clean = preprocess_data(data)
        train_models(data_clean)
    
    elif option == "Model Inference":
        data_clean = preprocess_data(data)
        model_inference(data_clean)

if __name__ == '__main__':
    main()