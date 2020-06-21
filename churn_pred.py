import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

dataset = pd.read_csv("/home/malay/PycharmProjects/ChurnPrediction/churn_prediction.csv")
x = dataset.iloc[:, 1:20]
x = x.drop(columns="branch_code")
x = x.drop(columns="city")
y = dataset.iloc[:, -1]
categorical_featue = ["dependent", "gender", "occupation", "customer_nw_category"]
numerical_feature = ["vintage", "age", "days_since_last_transaction", "current_balance",
                     "previous_month_end_balance", "average_monthly_balance_prevQ",
                     "average_monthly_balance_prevQ2", "current_month_credit", "previous_month_credit",
                     "current_month_debit", "previous_month_debit", "current_month_balance", "previous_month_balance"]


# categorising age
def categorise_Age(churn=0):
    catList = [0] * 4
    for i in range(len(x)):
        if 1 <= x["age"][i] <= 15:
            if y[i] == churn: catList[0] += 1
        if 16 <= x["age"][i] <= 30:
            if y[i] == churn: catList[1] += 1
        if 31 <= x["age"][i] <= 60:
            if y[i] == churn: catList[2] += 1
        if 61 <= x["age"][i] <= 90:
            if y[i] == churn: catList[3] += 1

    categories = ["1-15", "16-30", "31-60", "61-90"]
    plt.bar(categories, catList)


def occ_dis(churn=1):
    occList = [0] * 5
    for i in range(len(x)):
        if x["occupation"][i] == "self_employed":
            if y[i] == churn: occList[0] += 1
        if x["occupation"][i] == "salaried":
            if y[i] == churn: occList[1] += 1
        if x["occupation"][i] == "retired":
            if y[i] == churn: occList[2] += 1
        if x["occupation"][i] == "student":
            if y[i] == churn: occList[3] += 1
        if x["occupation"][i] == "company":
            if y[i] == churn: occList[4] += 1

    categorise = ["self_emp", "salaried", "retired", "student", "company"]
    plt.bar(categorise, occList)


def plot2feat(x, f1, f2):
    color = x[f1] / (max(x[f1]) - min(x[f1]))
    x1 = x[f1] / (max(x[f1]) - min(x[f1]))
    y1 = x[f2] / (max(x[f2]) - min(x[f2]))
    corr, _ = pearsonr(x[f1], x[f2])
    plt.figure()
    plt.title("Correlation is {}".format(corr))
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.scatter(x1, y1, c=color)
    plt.show()

def getOcc(x, index):
    if x["occupation"][index] == "self_employed":
        return 0
    if x["occupation"][index] == "salaried":
        return 1
    if x["occupation"][index] == "student":
        return 2
    if x["occupation"][index] == "retired":
        return 3
    if x["occupation"][index] == "company":
        return 4

def genderBased(x, y):
    cat = ["Male", "Female"]
    freq = [0, 0]
    new = x.dropna(axis=0, subset=["gender"])
    new = new.reset_index(drop=True)
    for i in range(len(new)):
        if new["gender"][i] == "Male" and y[i] == 1:
            freq[0] += 1
        elif new["gender"][i] == "Female" and y[i] == 1:
            freq[1] += 1
    plt.bar(cat, freq)

def dependentsCat():
    new = x.dropna(axis=0, subset=["dependents", "occupation"])
    new = new.reset_index(drop=True)
    depn = []
    for i in range(3):
        depn.append([0]*5)
    for i in range(len(new)):
        index = getOcc(new, i)
        if 0 <= new["dependents"][i] < 4:
            #print("0")
            depn[0][index] += 1
        elif 4 <= new["dependents"][i] <= 9:
            #print("1")
            depn[1][index] += 1
        elif new["dependents"][i] > 9:
            #print("1")
            depn[2][index] += 1

    return depn

def genDep():
    new = x.dropna(axis=0, subset=["dependents", "gender"])
    new = new.reset_index(drop=True)
    freq = [0, 0]
    for i in range(len(new)):
        if new["dependents"][i] > 0 and new["gender"][i] == "Male":
            freq[0] += 1
        elif new["dependents"][i] > 0 and new["gender"][i] == "Female":
            freq[1] += 1
    plt.bar(["dep on male", "dep on female"], freq)

def plot_DepOcc_chart():
    categorise = ["self_emp", "salaried", "retired", "student", "company"]
    cat2 = ["0-3", "4-9", ">9"]
    depn = dependentsCat()
    for i in range(1, len(depn)+1):
        plt.subplot(1, 3, i)
        plt.bar(categorise, depn[i-1])
    plt.show()

def show_corr():
    plot2feat(x, numerical_feature[3], numerical_feature[11])
    plot2feat(x, numerical_feature[9], numerical_feature[10])
    plot2feat(x, numerical_feature[5], numerical_feature[6])
    plot2feat(x, numerical_feature[7], numerical_feature[8])
    plot2feat(x, numerical_feature[4], numerical_feature[11])
    plot2feat(x, numerical_feature[4], numerical_feature[3])
    plot2feat(x, numerical_feature[4], numerical_feature[12])




categorise_Age(0)
categorise_Age(1)
occ_dis(0)
occ_dis(1)
genDep()
plot_DepOcc_chart()
genderBased(x, y)
genDep() # more peolpe are dependent on males. Empty gender columns can be filled
show_corr()






