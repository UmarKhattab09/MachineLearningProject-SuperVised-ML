import streamlit as st
import pandas as pd
import numpy as np
import shap
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
from imblearn.over_sampling import SMOTE, RandomOverSampler


from imblearn.under_sampling import RandomUnderSampler

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


regression = {'Linear Regression': LinearRegression,
              'Ridge Regression': Ridge,
              'Lasso Regression': Lasso,
              'Decision Tree Regressor': DecisionTreeRegressor,
              'Random Forest Regressor': RandomForestRegressor,
              'XGBoost Regressor': XGBRegressor 
                     }



def drop_non_numeric_columns(df):
    return df.drop(columns=df.select_dtypes(exclude=[np.number]).columns)

    
def regression_model(df, target_column, selected_model_name, hyperparams, test_size=0.2, random_state=42):
    try:
        st.write("### Regression Model Training")
        print("Starting regression model training...")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model_class = regression[selected_model_name]
        model = model_class(**hyperparams)
        st.write(f"Training {selected_model_name} with hyperparameters: {hyperparams}")
        print(f"Training {selected_model_name} with hyperparameters: {hyperparams}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.write(f"**Model**: {selected_model_name}")
        st.write(f"**Mean Squared Error**: {mse}")
        st.write(f"**R² Score**: {r2}")
        st.write(f"**Mean Absolute Error**: {mae}")
        plot_actual_vs_predicted(y_test, y_pred, selected_model_name)
        shap_explain(model, X_train, X_test, )

        return {'MSE': mse, 'R2': r2, 'MAE': mae}

    except Exception as e:
        st.error(f"An error occurred while training regression model: {e}")
        return None


def plot_actual_vs_predicted(y_test, y_pred, model_name):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import streamlit as st

        y_test_arr = np.array(y_test).ravel()
        y_pred_arr = np.array(y_pred).ravel()

        fig, ax = plt.subplots(figsize=(6,6))

        # Ideal diagonal line
        min_val = min(y_test_arr.min(), y_pred_arr.min())
        max_val = max(y_test_arr.max(), y_pred_arr.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')

        # Scatter plots
        ax.scatter(y_test_arr, y_test_arr, alpha=0.6, c='tab:green', marker='o', label='Actual')
        ax.scatter(y_test_arr, y_pred_arr, alpha=0.7, c='tab:blue', marker='x', label='Predicted')

        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Actual vs Predicted - {model_name}')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        st.pyplot(fig)
        plt.close(fig)  # <- important to prevent figure spillover
        print(f"Plotted Actual vs Predicted for {model_name}")

    except Exception as e:
        st.warning(f"Could not plot Actual vs Predicted: {e}")


# def shap_explain(model, X_train, X_test):
#     """
#     Generate SHAP plots for regression or classification models.
#     Works with tree-based, linear, or sklearn models.
#     """

#     try:
#         st.subheader("🔹 SHAP Feature Importance")

#         # Determine if classification or regression
#         if hasattr(model, "predict_proba"):
#             # Classification: explain probabilities
#             explainer = shap.Explainer(model, X_train)
#             shap_values = explainer(X_test)
#         else:
#             # Regression: explain predictions
#             explainer = shap.Explainer(model, X_train)
#             shap_values = explainer(X_test)

#         # --- Summary bar plot ---
#         st.subheader("Feature Importance (Bar)")
#         fig1 = plt.figure()
#         shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
#         st.pyplot(fig1)
#         plt.close(fig1)

#         # --- Detailed summary plot ---
#         st.subheader("SHAP Summary (Detailed)")
#         fig2 = plt.figure()
#         shap.summary_plot(shap_values, X_test, show=False)
#         st.pyplot(fig2)
#         plt.close(fig2)

#     except Exception as e:
#         st.warning(f"SHAP could not be generated: {e}")


def shap_explain(model, X_train, X_test):
    """
    Generate SHAP plots for regression or classification models.
    Works with tree-based, linear, and sklearn-compatible models.
    """
    import shap
    import matplotlib.pyplot as plt

    try:
        st.subheader("🔹 SHAP Feature Importance")

        # --- Determine model type ---
        is_tree_based = any(
            keyword in type(model).__name__.lower()
            for keyword in ["forest", "xgb", "boost", "tree"]
        )
        print(f"Model type detected: {'Tree-based' if is_tree_based else 'Other'}")
        # --- Choose appropriate explainer ---
        if is_tree_based:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_train)

        shap_values = explainer(X_test)

        # --- Bar summary plot ---
        st.subheader("Feature Importance (Bar)")
        fig1, ax1 = plt.subplots()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(fig1)
        plt.close(fig1)

        # --- Detailed SHAP summary plot ---
        st.subheader("SHAP Summary (Detailed)")
        fig2, ax2 = plt.subplots()
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig2)
        plt.close(fig2)

    except Exception as e:
        st.warning(f"SHAP could not be generated: {e}")



classification_models = {
    'Logistic Regression': LogisticRegression,
    'Random Forest Classifier': RandomForestClassifier,
    'Decision Tree Classifier': DecisionTreeClassifier,
    'XGBoost Classifier': XGBClassifier,
    'AdaBoost Classifier': AdaBoostClassifier,
    'Gradient Boosting Classifier': GradientBoostingClassifier
    
}
def classification_model(df, target_column, selected_model_name, hyperparams, test_size=0.2, random_state=42):
    try:
        st.write("### Classification Model Training")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print("Training classification model...")

        model_class = classification_models[selected_model_name]
        model = model_class(**hyperparams)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        # --- Display metrics ---
        st.write(f"**Model:** {selected_model_name}")
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.dataframe(pd.DataFrame(report_dict).transpose())

        # --- ROC curve ---
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
            if y_pred_proba.ndim == 1:
                y_pred_proba = np.vstack([1 - y_pred_proba, y_pred_proba]).T
            roc_curve_plot(y_test, y_pred_proba, selected_model_name)

        # --- SHAP Explanation ---
        # shap_explain(model, X_train, X_test)

        return {'Accuracy': accuracy, 'Classification Report': report_dict}

    except Exception as e:
        st.error(f"Error during classification model training: {e}")
        return None



# def classification_model(df, target_column, selected_model_name, hyperparams, test_size=0.2, random_state=42):
#     try:
#         st.write("### Classification Model Training")
#         print("Starting classification model training...")
#         X = df.drop(columns=[target_column])
#         y = df[target_column]
            

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, random_state=random_state
#         )

#         model_class = classification_models[selected_model_name]

#         model = model_class(**hyperparams)
#         print(f"Training {selected_model_name} with hyperparameters: {hyperparams}")
#         st.write(f"Training {selected_model_name} with hyperparameters: {hyperparams}")
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         accuracy = accuracy_score(y_test, y_pred)
#         report_dict = classification_report(y_test, y_pred, output_dict=True)
#         # shap_explain(model, X_train, X_test)

#         st.write(f"Model: {selected_model_name}")
#         st.write(f"Accuracy: {accuracy}")
#         report_df = pd.DataFrame(report_dict).transpose()
#         st.dataframe(report_df.style)
#         y_pred_proba = model.predict_proba(X_test)
#         if y_pred_proba.ndim == 1:
#             # Convert 1D array to 2D by expanding; typical in binary classification
#             y_pred_proba = np.vstack([1 - y_pred_proba, y_pred_proba]).T

#         roc_curve_plot(y_test, y_pred_proba, selected_model_name)
#         # roc_curve_plot(y_test, model.predict_proba(X_test)[:,1], selected_model_name)

#         return {'Accuracy': accuracy, 'Classification Report': report_dict}

#     except Exception as e:
#         st.error(f"An error occurred while training classification model: {e}")
#         return None


def detect_class_imbalance(df, target_column):
    if target_column not in df.columns:
        st.error(f"Select A Target Column.")
        return None, None, False
    y = df[target_column]
    X = df.drop(columns=[target_column])
    class_counts = y.value_counts()
    st.write("#### Class Distribution")
    st.bar_chart(class_counts)

    imbalance_ratio = class_counts.min() / class_counts.max()
    imbalance_detected = False
    if imbalance_ratio < 0.5:
        
        imbalance_detected = True

    return X, y, imbalance_detected


def class_imbalance_handling(df, method,target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if method == "SMOTE Oversampling":
        sampler = SMOTE(random_state=42)
    elif method == "Random Oversampling":
        sampler = RandomOverSampler(random_state=42)
    elif method == "Random UnderSampling":
        sampler = RandomUnderSampler(random_state=42)
    else:
        st.error("Invalid sampling method selected.")
        return X, y

    X_res, y_res = sampler.fit_resample(X, y)
    st.write("#### Resampled Class Distribution")
    st.bar_chart(pd.Series(y_res).value_counts())

    return X_res, y_res

def roc_curve_plot(y_test, y_pred_proba, model_name):
    classes = np.unique(y_test)
    n_classes = len(classes)

    plt.figure()

    if n_classes == 2:
        # Binary classification ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {model_name}')
        plt.legend(loc="lower right")
        st.pyplot(plt)
        st.write(f"**ROC AUC for {model_name}:** {roc_auc}")
    else:
        # Multiclass ROC - One-vs-Rest
        from sklearn.preprocessing import label_binarize

        y_test_binarized = label_binarize(y_test, classes=classes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Multiclass Receiver Operating Characteristic - {model_name}')
        plt.legend(loc="lower right")
        st.pyplot(plt)
        st.write(f"**Micro-average ROC AUC for {model_name}:** {roc_auc['micro']}")





def neural_network(df, target_column, n_hidden_layers=2, neurons_per_layer=None, learning_rate=0.001, epochs=100, batch_size=32):
    try:
        st.write("Preparing Data for Neural Network...")
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
        st.write("Building Neural Network Model...")
        layers_list = []
        input_size = X_train.shape[1]

        for i in range(n_hidden_layers):
            layers_list.append(nn.Linear(input_size, neurons_per_layer[i]))
            layers_list.append(nn.ReLU())
            input_size = neurons_per_layer[i]

        # Output layer
        layers_list.append(nn.Linear(input_size, 1))

        model = nn.Sequential(*layers_list)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # -------------------
        # 4. Training Loop
        # -------------------
        st.write("Training the model")
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_loss = []

        for epoch in range(epochs):
            model.train()
            runningloss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                runningloss += loss.item()
            avg_loss = runningloss / len(dataloader)
            train_loss.append(avg_loss)
            # if (epoch + 1) % 2 == 0:
                # st.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
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
        print("Neural Network model trained successfully.")


        ### plotting History graph
        # import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(range(1, epochs + 1), train_loss, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        st.pyplot(plt.gcf())  # Pass the current figure to Streamlit


        return model
    


    except Exception as e:
        st.error(f"An error occurred while training the neural network model: {e}")
        return None

def neural_network_classifier(df, target_column, n_hidden_layers=2, neurons_per_layer=None, learning_rate=0.001, epochs=100, batch_size=32):

    print("Starting neural network classifier training...")
    st.write("### Neural Network Classifier Training")
    try:
        if neurons_per_layer is None:
            neurons_per_layer = [64] * n_hidden_layers
        elif isinstance(neurons_per_layer, int):
            neurons_per_layer = [neurons_per_layer] * n_hidden_layers
        elif len(neurons_per_layer) != n_hidden_layers:
            st.error("Length of neurons_per_layer list must equal n_hidden_layers")
            return None

        # Prepare data
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Build model dynamically
        layers_list = []
        input_size = X_train.shape[1]

        for i in range(n_hidden_layers):
            layers_list.append(nn.Linear(input_size, neurons_per_layer[i]))
            layers_list.append(nn.ReLU())
            input_size = neurons_per_layer[i]

        # Output layer without softmax (CrossEntropy expects logits)
        output_size = len(torch.unique(y_train))
        layers_list.append(nn.Linear(input_size, output_size))

        model = nn.Sequential(*layers_list)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_loss = []
        print("Beginning training loop...")
        st.write("Beginning training loop...")
        for epoch in range(epochs):
            model.train()
            runningloss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                runningloss += loss.item()
            avg_loss = runningloss / len(dataloader)
            train_loss.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                st.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            _, predicted = torch.max(y_pred, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)

        st.write("### Neural Network Classifier")
        st.write(f"Accuracy on Test Set: {accuracy:.4f}")
    
        plt.figure()
        plt.plot(range(1, epochs + 1), train_loss, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        st.pyplot(plt.gcf())  # Pass the current figure to Streamlit

    
        return model
    

    except Exception as e:
        st.error(f"An error occurred while training the neural network classifier: {e}")
        return None

# def neural_network_classifier(df, target_column, n_hidden_layers=2, neurons_per_layer=None, learning_rate=0.001, epochs=100, batch_size=32):
#     try:
#         # -------------------
#         # 1. Prepare data
#         # -------------------
#         X = df.drop(columns=[target_column]).values
#         y = df[target_column].values

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )

#         X_train = torch.tensor(X_train, dtype=torch.float32)
#         X_test = torch.tensor(X_test, dtype=torch.float32)
#         y_train = torch.tensor(y_train, dtype=torch.long)
#         y_test = torch.tensor(y_test, dtype=torch.long)

#         # -------------------
#         # 2. Build PyTorch Model dynamically
#         # -------------------
#         layers_list = []
#         input_size = X_train.shape[1]

#         for i in range(n_hidden_layers):
#             layers_list.append(nn.Linear(input_size, neurons_per_layer[i]))
#             layers_list.append(nn.ReLU())
#             input_size = neurons_per_layer[i]

#         # Output layer
#         layers_list.append(nn.Linear(input_size, len(torch.unique(y_train))))
#         layers_list.append(nn.Softmax(dim=1))

#         model = nn.Sequential(*layers_list)

#         # Loss and optimizer
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#         # -------------------
#         # 3. Training Loop
#         # -------------------
#         dataset = torch.utils.data.TensorDataset(X_train, y_train)
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#         for epoch in range(epochs):
#             model.train()
#             for batch_X, batch_y in dataloader:
#                 optimizer.zero_grad()
#                 outputs = model(batch_X)
#                 loss = criterion(outputs, batch_y)
#                 loss.backward()
#                 optimizer.step()
#             if (epoch + 1) % 10 == 0:
#                 st.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
#             print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
#         # -------------------
#         # 4. Evaluate on test set
#         # -------------------
#         model.eval()
#         with torch.no_grad():
#             y_pred = model(X_test)
#             _, predicted = torch.max(y_pred, 1)
#             accuracy = (predicted == y_test).sum().item() / y_test.size(0)
#         st.write("### Neural Network Classifier")
#         st.write(f"Accuracy on Test Set: {accuracy:.4f}")
#         return model
#     except Exception as e:
#         st.error(f"An error occurred while training the neural network classifier: {e}")
#         return None
    





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