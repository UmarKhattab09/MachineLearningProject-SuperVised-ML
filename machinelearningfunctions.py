import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier




regression = {'Linear Regression': LinearRegression,
              'Ridge Regression': Ridge,
              'Lasso Regression': Lasso,
              'Random Forest Regressor': RandomForestRegressor,
              'XGBoost Regressor': XGBRegressor 
                     }

def regression_model(df, target_column):
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_results = {}
        for model_name, model_class in regression.items():
            model = model_class()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            model_results[model_name] = {'MSE': mse, 'R2': r2}
            st.write(f"Model: {model_name}")
            st.write("Mean Squared Error:", mse)
            st.write("R² Score:", r2)
            st.write("-----")
            print(model_name, "trained successfully.")

        return model_results
    except Exception as e:
        st.error(f"An error occurred while training regression models: {e}")
        return None





def random_forest_classifier(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    st.write("Accuracy:", accuracy)
    st.text("Classification Report:\n" + report)
    
    return model


def logistic_regression(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the model
    model = LogisticRegression(max_iter=200)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    st.write("Accuracy:", accuracy)
    st.text("Classification Report:\n" + report)
    
    return model

classification_models = {
    'Logistic Regression': logistic_regression,
    'Random Forest Classifier': random_forest_classifier,
    'Decision Tree Classifier': DecisionTreeClassifier,
    'XGBoost Classifier': XGBClassifier,
    'AdaBoost Classifier': AdaBoostClassifier,
    'Gradient Boosting Classifier': GradientBoostingClassifier
    
}

def classification_model(df, target_column):
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        model_results = {}
        for model_name, model_func in classification_models.items():
            st.write(f"Training {model_name}...")
            model = model_func(df, target_column)
            model_results[model_name] = model
            st.write("-----")
            print(model_name, "trained successfully.")
        return model_results
    except Exception as e:
        st.error(f"An error occurred while training classification models: {e}")
        return None