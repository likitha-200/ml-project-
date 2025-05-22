import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# Load the dataset
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv('heart.csv')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Preprocess the dataset
@st.cache_data
def preprocess_data(data):
    # Encode categorical variables
    for column in data.columns:
        if data[column].dtype == 'object':  # Check if the column is categorical
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])
    # Handle missing values
    if data.isnull().sum().any():
        data = data.fillna(data.mean())  # Filling missing values with column means
    return data

# Hyperparameter tuning using GridSearchCV (before fitting the model)
def tune_model(X_train, y_train, model_name, kernel='rbf'):
    if model_name == 'SVM':
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': [kernel]
        }
        model = SVC(random_state=42, probability=True)
    elif model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'Decision Tree':
        param_grid = {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2']
        }
        model = LogisticRegression(random_state=42)
    
    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    return best_model, best_params

# Train multiple models and return their accuracies and y_test
@st.cache_data
def train_multiple_models(data, kernels=['rbf', 'linear', 'poly', 'sigmoid']):
    # Define features (X) and target (y)
    X = data.iloc[:, :-1]  # All columns except the last
    y = data.iloc[:, -1]   # The last column

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    accuracies = {}
    algorithm_results = {}

    best_kernel = None
    best_accuracy = 0

    for kernel in kernels:
        models = {
            "SVM": SVC(kernel=kernel, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42)
        }

        model_accuracies = []
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            model_accuracies.append(accuracy)

        accuracies[kernel] = model_accuracies
        algorithm_results[kernel] = pd.DataFrame(list(models.keys()), columns=["Model"])
        algorithm_results[kernel]["Accuracy"] = [f"{acc * 100:.2f}%" for acc in model_accuracies]

        # Track the best kernel based on SVM accuracy
        if accuracies[kernel][0] > best_accuracy:
            best_accuracy = accuracies[kernel][0]
            best_kernel = kernel

    return accuracies, scaler, algorithm_results, X_test, y_test, best_kernel

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["No Heart Failure", "Heart Failure"], 
                yticklabels=["No Heart Failure", "Heart Failure"], 
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

# Function to plot correlation matrix
def plot_correlation_matrix(data):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

# Main Streamlit App
def main():
    st.title("Heart Failure Prediction App")
    st.write("This app predicts the likelihood of heart failure based on input data.")

    # Load and preprocess data
    data = load_data('heart.csv')  # Specify the file path
    if data is None:
        st.stop()

    data = preprocess_data(data)

    # Train multiple models and get accuracies
    kernels = ['rbf', 'linear', 'poly', 'sigmoid']
    accuracies, scaler, algorithm_results, X_test, y_test, best_kernel = train_multiple_models(data, kernels)

    # Show the best kernel based on tuning results
    st.info(f"The best SVM kernel based on hyperparameter tuning is '{best_kernel}'.")

    # Sidebar input for user data
    st.sidebar.header("Input Features")
    user_input = {}
    for feature in data.columns[:-1]:  # Excluding target column
        user_input[feature] = st.sidebar.number_input(
            f"Enter {feature}", value=float(data[feature].mean()), min_value=0.0, step=0.01)

    # Convert user input to a DataFrame
    user_data = pd.DataFrame([user_input])

    # Scale the user input
    user_data_scaled = scaler.transform(user_data)

    # Display buttons side by side
    col1, col2, col3 = st.columns(3)

    # Button for Predictions
    with col1:
        if st.button("Predict"):
            model = SVC(kernel=best_kernel, random_state=42, probability=True)  # Best model for prediction
            model.fit(scaler.transform(data.iloc[:, :-1]), data.iloc[:, -1])
            prediction = model.predict(user_data_scaled)[0]
            probability = model.predict_proba(user_data_scaled)[0]

            st.write("### Prediction Result")
            if prediction == 1:
                st.error("Prediction: Heart Failure Risk Detected")
                st.warning("Take precautions as listed below:")
                st.write("""
                1. Consult a healthcare provider immediately.
                2. Follow a heart-healthy diet.
                3. Monitor symptoms and stay active (if permitted by your doctor).
                """)
            else:
                st.success("Prediction: No Heart Failure Risk Detected")

            st.write(f"Confidence: {probability[prediction] * 100:.2f}%")

    # Button for Algorithm Information
    with col2:
        if st.button("Algorithm Info"):
            st.write("### Algorithm Details")
            st.write(f"""
            This model uses a Support Vector Machine (SVM) with the kernel '{best_kernel}':
            - Kernel: {best_kernel} for classification.
            - Scaler: StandardScaler to normalize features for better SVM performance.
            - Train-Test Split: 80% training, 20% testing.
            """)

    # Button to show Kernel Performance (Accuracy)
    with col3:
        if st.button("Kernel Accuracies"):
            st.write("### SVM Kernel Performance (Accuracy)")
            
            # Reformat accuracies for display
            kernel_names = list(accuracies.keys())
            model_names = ["SVM", "Decision Tree", "Random Forest", "Logistic Regression"]
            accuracy_data = []

            for kernel in kernel_names:
                accuracy_data.append([kernel] + [f"{acc * 100:.2f}%" for acc in accuracies[kernel]])

            # Create DataFrame from the structured data
            accuracy_df = pd.DataFrame(accuracy_data, columns=["Kernel"] + model_names)
            st.table(accuracy_df)

    # Show ROC Curve
    if st.checkbox("Show ROC Curve"):
        st.write("### ROC Curve")
        model = SVC(kernel=best_kernel, random_state=42, probability=True)  # Best SVM model
        model.fit(scaler.transform(data.iloc[:, :-1]), data.iloc[:, -1])
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plot_roc_curve(fpr, tpr, roc_auc)

    # Show Confusion Matrix
    if st.checkbox("Show Confusion Matrix"):
        st.write("### Confusion Matrix")
        model = SVC(kernel=best_kernel, random_state=42)  # Best SVM model
        model.fit(scaler.transform(data.iloc[:, :-1]), data.iloc[:, -1])
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred)

    # Show Correlation Matrix at the end
    if st.checkbox("Show Correlation Matrix"):
        st.write("### Correlation Matrix")
        plot_correlation_matrix(data)

if __name__ == "__main__":
    main()
