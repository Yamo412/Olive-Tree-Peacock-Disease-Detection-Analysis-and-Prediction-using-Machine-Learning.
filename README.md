**Olive Tree Peacock Disease Prediction using Machine Learning**

This project aims to develop a Machine Learning model to predict and analyze the susceptibility of olive trees to the Peacock Disease in Palestine, West Bank. Utilizing over 44,000 data entries collected from various cities, this research contributes to agricultural sustainability by improving disease detection methods.

**Project Overview**

- Objective: To create an effective Machine Learning model to recognize Peacock Disease in olive trees, particularly focusing on detecting small circular lesions on leaves.
- Data Source: Data was collected from Prof. Mazen Salman's research at Al-Khadourie University, encompassing various regions including Nablus, Tulkarem, Qalqilya, etc.
- Technology Stack:
  - Programming Language: Python
  - Libraries: pandas, sklearn, numpy, calendar
- Machine Learning Models:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
 
**Installation**

Ensure Python 3.x is installed along with the necessary libraries. Clone this repository to your local machine and navigate into the project directory.

```git clone <repository-url>
cd <project-directory>```

**Usage**

To run the model:

```python MachineLearningProjectOLS.py```

Make sure to adjust the script with the correct paths to your data files.

**Project Outcomes**

- Disease Characteristics: Key findings include that the disease spreads mostly during winter and cold months but doesn't affect many trees. It can be significantly controlled using the methods detailed in our dataset.
- Technical Results: The Random Forest Regressor was found to be the most effective model, yielding the best results in terms of MSE and R2 score, indicating high accuracy and reduced overfitting.
