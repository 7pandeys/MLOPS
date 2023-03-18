"""
Date : 3-Mar-2023
Author:  pandesav
"""


# import libraries
import os
# from sklearn.metrics import plot_roc_curve,
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import normalize
# import shap
# import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df_churn = pd.read_csv(pth)
    return df_churn


def perform_eda(df_churn):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df_churn['Churn'] = df_churn['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df_churn['Churn'].hist()
    plt.savefig('images/Churn_hist.png')
    plt.figure(figsize=(20, 10))
    df_churn['Customer_Age'].hist()
    plt.savefig('images/Cust_age_hist.png')
    plt.figure(figsize=(20, 10))
    df_churn.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/marital_bar.png')
    plt.figure(figsize=(20, 10))
    sns.histplot(df_churn['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/Total_Trans_Ct.png')
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_churn.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/corr_heatmap.png')
# 	pass


def encoder_helper(df_churn, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: optional argument could be used for naming variables or index y column

    output:
            df: pandas dataframe with new columns for
    '''
    for i in category_lst:
        category_ls = []
        category_groups = df_churn.groupby(i).mean()['Churn']

        for val in df_churn[i]:
            category_ls.append(category_groups.loc[val])

        df_churn[f'{i}_Churn'] = category_ls

    return df_churn


def perform_feature_engineering(df_churn, response):
    '''
    input:
              df: pandas dataframe
              response: optional argument that could be used for naming variables or index y column

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y_label = df_churn['Churn']
    x_features = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    x_features[keep_cols] = df_churn[keep_cols]
    return train_test_split(
        x_features,
        y_label,
        test_size=0.3,
        random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # Generate the classification report
    report_test_rf = classification_report(y_test, y_test_preds_rf)
    report_rf = classification_report(y_train, y_train_preds_rf)
    report_test_lr = classification_report(y_test, y_test_preds_lr)
    report_lr = classification_report(y_train, y_train_preds_lr)

    for i in [report_test_rf, report_rf, report_test_lr, report_lr]:
        # Create a plot of the report
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        ax.table(cellText=i.split('\n'), loc='center')

        # Save the plot as an image file
        plt.savefig(f'{i}_classification_report.png')



def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    # store in outputpath
    plt.savefig(f'{output_pth}_feature_importance.png')


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)


if __name__ == '__main__':
    pass
