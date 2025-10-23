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
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error



regression = {'Linear Regression': LinearRegression,
              'Ridge Regression': Ridge,
              'Lasso Regression': Lasso,
              'Decision Tree Regressor': DecisionTreeRegressor,
              'Random Forest Regressor': RandomForestRegressor,
              'XGBoost Regressor': XGBRegressor 
                     }

    

def drop_non_numeric_columns(df):
    return df.drop(columns=df.select_dtypes(exclude=[np.number]).columns)


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
            mae = mean_absolute_error(y_test, y_pred)
            model_results[model_name] = {'MSE': mse, 'R2': r2, 'MAE': mae}
            st.write(f"Model: {model_name}")
            st.write("Mean Squared Error:", mse)
            st.write("R² Score:", r2)
            st.write("Mean Absolute Error:", mae)
            st.write("-----")
            print(model_name, "trained successfully.")

        return model_results
    except Exception as e:
        st.error(f"An error occurred while training regression models: {e}")
        return None
    

classification_models = {
    'Logistic Regression': LogisticRegression,
    'Random Forest Classifier': RandomForestClassifier,
    'Decision Tree Classifier': DecisionTreeClassifier,
    'XGBoost Classifier': XGBClassifier,
    'AdaBoost Classifier': AdaBoostClassifier,
    'Gradient Boosting Classifier': GradientBoostingClassifier
    
}

def classification_model(df, target_column):
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_results = {}
        for model_name, model_class in classification_models.items():
            model = model_class()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            # report = classification_report(y_test, y_pred)
            report_dict = classification_report(y_test, y_pred, output_dict=True)

            # model_results[model_name] = {'Accuracy': accuracy, 'Classification Report': report}
            model_results[model_name] = {'Accuracy': accuracy, 'Classification Report': report_dict}
            report_df = pd.DataFrame(report_dict).transpose()
            st.write(f"Model: {model_name}")
            st.write("Accuracy:", accuracy)
            # st.text("Classification Report:\n" + report)
            st.dataframe(report_df.style)
            roc_curve_plot(y_test, model.predict_proba(X_test)[:,1], model_name)
            st.write("-----")
            

            print(model_name, "trained successfully.")

        return model_results
    except Exception as e:
        st.error(f"An error occurred while training classification models: {e}")
        return None


def roc_curve_plot(y_test, y_pred_proba, model_name):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    st.pyplot(plt)
    st.write(f"**ROC AUC for {model_name}:** {roc_auc}")




import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def neural_network(df, target_column, n_hidden_layers=2, neurons_per_layer=64, learning_rate=0.001, epochs=100, batch_size=32):
    try:
        # -------------------
        # 1. Prepare data
        # -------------------
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values.reshape(-1, 1)

        # # Feature scaling
        # scaler_X = StandardScaler()
        # scaler_y = StandardScaler()
        # X = scaler_X.fit_transform(X)
        # y = scaler_y.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # -------------------
        # 2. Get user hyperparameters
        # -------------------
       

        # -------------------
        # 3. Build PyTorch Model dynamically
        # -------------------
        layers_list = []
        input_size = X_train.shape[1]

        for i in range(n_hidden_layers):
            layers_list.append(nn.Linear(input_size, neurons_per_layer))
            layers_list.append(nn.ReLU())
            input_size = neurons_per_layer

        # Output layer
        layers_list.append(nn.Linear(input_size, 1))

        model = nn.Sequential(*layers_list)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # -------------------
        # 4. Training Loop
        # -------------------
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                st.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        # -------------------
        # 5. Evaluate on test set
        # -------------------
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            mae = torch.mean(torch.abs(y_pred - y_test)).item()

        st.write("### Neural Network Model")
        st.write(f"Mean Absolute Error on Test Set: {mae:.4f}")

        

        return model

    except Exception as e:
        st.error(f"An error occurred while training the neural network model: {e}")
        return None

def neural_network_classifier(df, target_column, n_hidden_layers=2, neurons_per_layer=64, learning_rate=0.001, epochs=100, batch_size=32):
    try:
        # -------------------
        # 1. Prepare data
        # -------------------
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # -------------------
        # 2. Build PyTorch Model dynamically
        # -------------------
        layers_list = []
        input_size = X_train.shape[1]

        for i in range(n_hidden_layers):
            layers_list.append(nn.Linear(input_size, neurons_per_layer))
            layers_list.append(nn.ReLU())
            input_size = neurons_per_layer

        # Output layer
        layers_list.append(nn.Linear(input_size, len(torch.unique(y_train))))
        layers_list.append(nn.Softmax(dim=1))

        model = nn.Sequential(*layers_list)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # -------------------
        # 3. Training Loop
        # -------------------
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                st.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        # -------------------
        # 4. Evaluate on test set
        # -------------------
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            _, predicted = torch.max(y_pred, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        st.write("### Neural Network Classifier")
        st.write(f"Accuracy on Test Set: {accuracy:.4f}")
        return model
    except Exception as e:
        st.error(f"An error occurred while training the neural network classifier: {e}")
        return None
    





    # def random_forest_classifier(df, target_column):
#     X = df.drop(columns=[target_column])
#     y = df[target_column]
    
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Create the model
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     # Train the model
#     model.fit(X_train, y_train)
#     # Make predictions
#     y_pred = model.predict(X_test)
#     # Evaluate the model
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)

#     st.write("Accuracy:", accuracy)
#     st.text("Classification Report:\n" + report)
    
#     return model


# def logistic_regression(df, target_column):
#     X = df.drop(columns=[target_column])
#     y = df[target_column]
    
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Create the model
#     model = LogisticRegression(max_iter=200)
    
#     # Train the model
#     model.fit(X_train, y_train)
    
#     # Make predictions
#     y_pred = model.predict(X_test)
    
#     # Evaluate the model
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
    
#     st.write("Accuracy:", accuracy)
#     st.text("Classification Report:\n" + report)
    
#     return model