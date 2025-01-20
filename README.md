# Diabetes Prediction using Logistic Regression

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dataset](#dataset)
5. [Model](#model)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction

This project demonstrates the use of **Logistic Regression** to predict whether a patient has diabetes based on various health metrics. The model is trained on the **Pima Indians Diabetes Dataset**, which includes features such as the number of pregnancies, glucose levels, blood pressure, BMI, age, and others. The objective is to predict whether the patient has diabetes (`Outcome = 1`) or not (`Outcome = 0`).

## Installation

### Prerequisites

- Python 3.x
- Required libraries:
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn

To install the required libraries, run the following command:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Usage

### Step 1: Load and Preprocess the Dataset

The dataset is loaded from a URL and features are separated from the target variable (`Outcome`). The data is then split into training and testing sets, and standardized using **StandardScaler** to improve model performance.

```python
# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=column_names)

# Split the dataset into features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Step 2: Train the Logistic Regression Model

A **Logistic Regression** model is trained using the training data.

```python
# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

### Step 3: Evaluate the Model

The model's performance is evaluated using accuracy, a classification report, and a confusion matrix.

```python
# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

## Dataset

The dataset used in this project is the **Pima Indians Diabetes Dataset**, which can be accessed from various open datasets platforms such as [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes). It consists of the following columns:

- **Pregnancies**: Number of pregnancies the patient has had
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg / height in m²)
- **DiabetesPedigreeFunction**: A function that gives insight into the likelihood of diabetes based on family history
- **Age**: Age of the patient
- **Outcome**: Whether the patient has diabetes (`1`) or not (`0`)

## Model

The model used for prediction is **Logistic Regression**, a linear model for binary classification. It predicts the probability of the binary outcome (`Outcome = 1` for diabetes, `Outcome = 0` for no diabetes) based on the input features.

### Key Hyperparameters:
- `max_iter=1000`: The number of iterations for the optimization algorithm.

## Evaluation

The model is evaluated using the following metrics:
- **Accuracy**: Percentage of correct predictions made by the model.
- **Classification Report**: Includes precision, recall, and F1-score for both classes (diabetes and no diabetes).
- **Confusion Matrix**: A matrix showing the true positive, false positive, true negative, and false negative predictions.

The confusion matrix is plotted as a heatmap for better visualization.

## Results

After training, the accuracy of the model is printed, and the classification report and confusion matrix are displayed.

```bash
Accuracy: 75.32%

Classification Report:
              precision    recall  f1-score   support

           0       0.81      0.80      0.81        99
           1       0.65      0.67      0.66        55

    accuracy                           0.75       154
   macro avg       0.73      0.74      0.73       154
weighted avg       0.76      0.75      0.75       154

```

The confusion matrix is displayed as a heatmap with labels for `No Diabetes` and `Diabetes`.

## Contributing

If you have suggestions or improvements, feel free to fork the repository and submit a pull request. Please ensure that all contributions are well-documented. Contributions are welcome to enhance the model's accuracy or extend it to other classification tasks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Key Details:
- **Introduction**: Overview of the project and its objective.
- **Installation**: How to install the necessary libraries.
- **Usage**: Steps for running the code, including loading, preprocessing, training, and evaluating the model.
- **Dataset**: Description of the dataset used.
- **Model**: Details about the Logistic Regression model used.
- **Evaluation**: Explanation of the evaluation metrics and confusion matrix.
- **Results**: How to interpret the results, including accuracy, classification report, and confusion matrix.
- **Contributing**: Invitation for others to contribute to the project.
- **License**: Licensing terms for the project.
