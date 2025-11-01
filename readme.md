# Machine Learning Project - Supervised ML with AutoEDA

A comprehensive machine learning application that combines powerful supervised learning algorithms with automated Exploratory Data Analysis (EDA). This project provides an interactive web-based platform for data analysis, preprocessing, and model training using various classification and regression algorithms.

**[Live Website Link](https://machinelearningproject-supervised-ml.onrender.com/)**

**[Live Website Link](https://machinelearningproject-supervised-ml.onrender.com/)**

## Preview
![Alt text](images/image.png)

This is the home screen of the web application. From here, you can upload any dataset you want and perform Exploratory Data Analysis right through the web.



## Key Features

### Machine Learning Models

#### Classification Models
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- XGBoost Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier

#### Regression Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- Neural Network (TensorFlow/Keras)

### Data Analysis and Preprocessing
- Dataset Overview: Quick data preview and structure analysis
- Automated Data Preprocessing: Handle missing data, encode categorical variables, scale features
- Interactive Visualizations: Distribution plots, correlation heatmaps, scatter plots
- Feature Engineering Tools
- Outlier Detection and Handling

### Model Evaluation
- Comprehensive Performance Metrics
- ROC Curves for Classification
- Cross-validation Support
- Model Comparison Tools

## Why Choose This Project?

- **Complete ML Pipeline**: End-to-end solution from data preprocessing to model deployment
- **Multiple Algorithm Support**: Wide range of classification and regression algorithms
- **Interactive Interface**: User-friendly web interface powered by Streamlit
- **Automated Workflows**: Streamlined processes for routine ML tasks
- **Visualization Tools**: Rich set of data visualization capabilities

## Project Structure
```
├── data_analysis_functions.py    # Data analysis utilities
├── data_preprocessing_function.py # Data preprocessing functions
├── machinelearningfunctions.py   # Core ML model implementations
├── home_page.py                  # Main application interface
├── loginsystem.py               # User authentication system
├── requirements.txt             # Project dependencies
├── example_dataset/            
│   └── titanic.csv             # Sample dataset
└── pages/                      
    └── main.py                 # Additional application pages
```

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/UmarKhattab09/MachineLearningProject-SuperVised-ML.git
cd MachineLearningProject-SuperVised-ML
```

2. Create and activate a Python virtual environment:

For Windows:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Launch the application:
```bash
streamlit run loginsystem.py
```

Note: Make sure you have Python 3.7+ installed on your system before starting the installation process.


## Usage Guide

1. **Data Upload**: 
   - Launch the application and upload your dataset
   - Review the automatic data summary and statistics

2. **Data Preprocessing**:
   - Handle missing values
   - Encode categorical variables
   - Scale numerical features
   - Perform feature selection if needed

3. **Model Selection and Training**:
   - Choose between classification or regression based on your task
   - Select the target variable and features
   - Choose one or multiple models to train
   - Review performance metrics and visualizations

4. **Model Evaluation**:
   - Compare model performances
   - View detailed metrics and visualizations
   - Export results if needed

## Dependencies

Major dependencies include:
- streamlit
- pandas
- numpy
- scikit-learn
- tensorflow
- xgboost
- matplotlib

For a complete list of dependencies, see `requirements.txt`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Created by Umar Khattab,Jay Patel






