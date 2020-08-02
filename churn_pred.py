import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import tkinter
import matplotlib

matplotlib.use('TkAgg')

dataset = pd.read_csv("/home/malay/Work/Pycharm Projects/ChurnPrediction/churn_prediction.csv")
x = dataset.iloc[:, 1:20]
x = x.drop(columns="branch_code")
x = x.drop(columns="city")
y = dataset.iloc[:, -1]
categorical_feature = ["dependents", "gender", "occupation", "customer_nw_category"]
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


def plot2feat(x, f1, f2, fig, i):
    color = x[f1] / (max(x[f1]) - min(x[f1]))
    x1 = x[f1] / (max(x[f1]) - min(x[f1]))
    y1 = x[f2] / (max(x[f2]) - min(x[f2]))
    corr, _ = pearsonr(x[f1], x[f2])
    fig.add_subplot(4, 4, i)
    plt.title("Correlation is {}".format(corr))
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.scatter(x1, y1, c=color)
    plt.tight_layout(3.0)
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


def dependentsCat(categorise):
    new = x.dropna(axis=0, subset=["dependents", "occupation"])
    new = new.reset_index(drop=True)
    depn = []
    total_occupation = [0] * len(categorise)
    for i in range(3):
        depn.append([0] * 5)
    for i in range(len(new)):
        index = categorise.index(new["occupation"][i])
        total_occupation[index] += 1
        if 1 <= new["dependents"][i] < 4:
            # print("0")
            depn[0][index] += 1
        elif 4 <= new["dependents"][i] <= 9:
            # print("1")
            depn[1][index] += 1
        elif new["dependents"][i] > 9:
            # print("1")
            depn[2][index] += 1
    print(total_occupation)
    for i in range(3):
        for j in range(len(depn[i])):
            depn[i][j] = depn[i][j] / total_occupation[j]
    return depn


def genDep():
    new = x.dropna(axis=0, subset=["dependents", "gender"])
    new = new.reset_index(drop=True)
    freq = [0, 0]
    gender = [0] * 2
    for i in range(len(new)):
        if new["gender"][i] == "Male":
            gender[0] += 1
        else:
            gender[1] += 1
        if new["dependents"][i] > 0 and new["gender"][i] == "Male":
            freq[0] += 1
        elif new["dependents"][i] > 0 and new["gender"][i] == "Female":
            freq[1] += 1
    freq[0] = freq[0] / gender[0]
    freq[1] = freq[1] / gender[1]
    plt.bar(["dep on male", "dep on female"], freq)


def check_families(x, y):
    new = x.dropna(axis=0, subset=["dependents", "gender"])
    new = new.reset_index(drop=True)
    count = 0
    for i in range(len(new)):
        if 1 <= new["dependents"][i] <= 3:
            count += 1
    families = ["family", "not_family"]
    color = ["blue", "red"]
    size = [count, len(new) - count]
    explode = [0.1, 0]
    plt.pie(size, explode=explode, labels=families, colors=color)


def plot_DepOcc_chart():
    categorise = ['self_employed', 'salaried', 'retired', 'student', 'company']
    depn = dependentsCat(categorise)
    for i in range(1, len(depn) + 1):
        plt.subplot(1, 3, i)
        plt.ylim(0, 1)
        if i == 1:
            plt.title("1-4")
        elif i == 2:
            plt.title("4-9")
        else:
            plt.title(">9")
        plt.bar(categorise, depn[i - 1])
    plt.show()


def show_corr():
    fig = plt.figure(figsize=(15, 15))
    plot2feat(x, numerical_feature[3], numerical_feature[11], fig, 1)
    plot2feat(x, numerical_feature[9], numerical_feature[10], fig, 2)
    plot2feat(x, numerical_feature[5], numerical_feature[6], fig, 3)
    plot2feat(x, numerical_feature[7], numerical_feature[8], fig, 4)
    plot2feat(x, numerical_feature[4], numerical_feature[11], fig, 5)
    plot2feat(x, numerical_feature[4], numerical_feature[3], fig, 6)
    plot2feat(x, numerical_feature[4], numerical_feature[12], fig, 7)


def dependencies_on_days(x, y, churn):
    days = x["days_since_last_transaction"]
    months = [0] * 3
    total_people = [0] * 3
    for i in range(len(days)):
        if 1 <= days[i] <= 90:
            total_people[0] += 1
            if y[i] == churn:
                months[0] += 1
        if 90 < days[i] <= 180:
            total_people[1] += 1
            if y[i] == churn:
                months[1] += 1
        if 180 < days[i] <= 240:
            total_people[2] += 1
            if y[i] == churn:
                months[2] += 1
    for i in range(3):
        months[i] /= total_people[i]
    plt.bar(["3 months", "3-6 months", "6-8 months"], months)


def dependency_on_networth(x, y):
    networth = x["customer_nw_category"]
    nw = [0] * 3
    for i in range(len(networth)):
        if networth[i] == 1 and y[i] == 1:
            nw[0] += 1
        elif networth[i] == 2 and y[i] == 1:
            nw[1] += 1
        elif networth[i] == 3 and y[i] == 1:
            nw[2] += 1
    plt.bar(["low", "medium", "high"], nw)


def fill_gender(x):
    deps = x["dependents"].isnull()
    gender = x["gender"].isnull()
    for i in range(len(x)):
        if gender[i] and ~deps[i]:
            if x["dependents"][i] > 0:
                x["gender"][i] = "Male"


def genderOcc(x):
    categories = ['self_employed', 'salaried', 'retired', 'student', 'company']
    male = [0] * len(categories)
    female = [0] * len(categories)
    totMale = 0
    totFemale = 0
    new = x.dropna(axis=0, subset=["occupation", "gender"])
    new = new.reset_index(drop=True)
    for i in range(len(new)):
        if new["gender"][i] == "Male":
            totMale += 1
            male[categories.index(new["occupation"][i])] += 1
        else:
            totFemale += 1
            female[categories.index(new["occupation"][i])] += 1
    fig = plt.figure(figsize=(12, 12))
    fig.add_subplot(1, 2, 1)
    plt.title("Males")
    plt.ylim(0, 1)
    plt.bar(categories, np.array(male) / totMale)
    fig.add_subplot(1, 2, 2)
    plt.ylim(0, 1)
    plt.title("Females")
    plt.bar(categories, np.array(female) / totFemale)


def dependentsChart(x, y):
    new = x.dropna(axis=0, subset=["dependents"])
    new = new.reset_index(drop=True)
    churn = [0] * 2
    for i in range(len(new)):
        if new["dependents"][i] > 0:
            churn[y[i]] += 1

    plt.bar(["not likely", "likely"], churn)


def fill_dependents(x, y):
    dependents = x["dependents"].isnull()
    for i in range(len(x)):
        if y[i] == 0 and dependents[i]:
            x["dependents"][i] = 1.0
        elif y[i] == 1 and dependents[i]:
            x["dependents"][i] = 0.0


def categoriseDependents(x):
    dependents = []
    for i in range(len(x)):
        if x["dependents"][i] == 0.0:
            dependents.append(0)
        elif 1.0 <= x["dependents"][i] <= 4.0:
            dependents.append(1)
        else:
            dependents.append(3)
    x = x.drop(columns="dependents")
    x["dependents"] = dependents
    return x


def categoriseOccupation(x):
    occupation = []
    for i in range(len(x)):
        if x["occupation"][i] == "student" or x["occupation"][i] == "retired":
            occupation.append(1)
        elif x["occupation"][i] == "salarie" or x["occupation"][i] == "self_employed":
            occupation.append(2)
        else:
            occupation.append(3)
    x = x.drop(columns="occupation")
    x["occupation"] = occupation
    return x


def drop_row(x, y):
    occ = x["occupation"].isnull()
    gender = x["gender"].isnull()
    rmidx = []
    for i in range(len(x)):
        if occ[i] or gender[i]:
            rmidx.append(i)
    x = x.drop(x.index[rmidx])
    y = y.drop(y.index[rmidx])
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return x, y


categorise_Age(0)
categorise_Age(1)  # 31-60 are likely to churn
occ_dis(0)
occ_dis(1)  # self-employed are likely to churn
plot_DepOcc_chart()
genderBased(x, y)  # males and females are almost equally likely
genDep()  # more people are dependent on males. Empty gender columns can be filled
show_corr()  # remove highly correlated features
dependencies_on_days(x, y, 0)  # not much information
dependencies_on_days(x, y, 1)
dependency_on_networth(x, y)  # people with net worth of mid and high likely to churn
check_families(x, y)  # not many families
genderOcc(x)
dependentsChart(x, y)  # if dependents are more than 0, people are not likely to churn

# cleaning data
x.isnull().sum()
fill_dependents(x, y)
fill_gender(x)
# x = x.drop(columns=["days_since_last_transaction", "current_balance", "previous_month_end_balance", "previous_month_balance"])
x = x.drop(columns=["days_since_last_transaction"])
x = x.reset_index(drop=True)
x, y = drop_row(x, y)
x = x.reset_index(drop=True)
y = y.reset_index(drop=True)
x = categoriseDependents(x)
x = categoriseOccupation(x)

# preprocessing

labelencoder = LabelEncoder()
x["gender"] = labelencoder.fit_transform(x["gender"])
ct = ColumnTransformer([('onehotencoder', OneHotEncoder(categories='auto'), categorical_feature)],
                       remainder='passthrough')
x = ct.fit_transform(x)

# applying models

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)

# Logistic Regression
logReg = LogisticRegression(max_iter=1000)
logReg.fit(xTrain, np.array(yTrain))
yPred = logReg.predict(xTest)
print(accuracy_score(yPred, yTest))

# SVM
svm = SVC(C=10000, kernel='rbf', probability=True)
svm.fit(xTrain, yTrain)
yPred = svm.predict(xTest)
print(accuracy_score(yPred, yTest))

# Decision Tree

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(xTrain, yTrain)
yPred = dt.predict(xTest)
print(accuracy_score(yPred, yTest))

# Random Forest

rndmFrst = RandomForestClassifier(n_estimators=1000, max_depth=15,
                                  oob_score=True)
rndmFrst.fit(xTrain, yTrain)
yPred = rndmFrst.predict(xTest)
print(accuracy_score(yPred, yTest))

# Ensemble Model

ensbl = VotingClassifier(estimators=[('logReg', logReg), ('svm', svm), ('dt', dt)],
                         voting='soft', weights=[1, 2, 2])
ensbl.fit(xTrain, yTrain)
yPred = ensbl.predict(xTest)
print(accuracy_score(yPred, yTest))




