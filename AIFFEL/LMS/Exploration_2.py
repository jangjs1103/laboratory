# 필요한 모둘 import 하기
from sklearn.datasets import load_digits # 손글씨 분류
from sklearn.datasets import load_wine   # 와인 종류
from sklearn.datasets import load_breast_cancer  # 유방암 진단
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

''' 진행 순서
1. 손글씨 분류 : load_digits
2. 와인 분류 : load_wine
3. 유방암 여부 진단: load_breast_cancer
4. 총평
'''

# 1. 손글씨 분류 시작
print('----------------- Classifying Handwritten Digits -----------------\n')

# 데이터 준비
digits = load_digits()

# 데이터 이해하기
digits_data = digits.data
digits_label = digits.target
print(f'Target Names: {digits.target_names}')
print(digits.DESCR)
print('------------------------------------------------------------------\n') # DESCR과 결과값 구분

# train, test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(digits_data, digits_label, test_size=0.2, random_state = 10)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
print('\t\t    [[ Model : Decision Tree ]]')
print(classification_report(y_test, y_pred))

# Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print('\t\t    [[ Model : Random Forest ]]')
print(classification_report(y_test, y_pred))

# SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print('\t\t\t[[ Model : SVM ]]')
print(classification_report(y_test, y_pred))

# SGD Classifier
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)
print('\t\t\t[[ Model : SGD ]]')
print(classification_report(y_test, y_pred))

# Logistic Regression
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
print('\t\t  [[ Model : Logistic Regression ]]')
print(classification_report(y_test, y_pred))

# 손글씨 분류 종료
print('------------------------------------------------------------------\n')


# 2. 와인 분류 시작
print('------------------ Classifying The Kind Of Wine ------------------\n')

# 데이터 준비
wine = load_wine()

# 데이터 이해하기
wine_data = wine.data
wine_label = wine.target
print(f'Target Names: {wine.target_names}')
print(wine.DESCR)
print('------------------------------------------------------------------\n') # DESCR과 결과값 구분

# train, test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(wine_data, wine_label, test_size=0.2, random_state = 10)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
print('\t\t    [[ Model : Decision Tree ]]')
print(classification_report(y_test, y_pred))

# Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print('\t\t    [[ Model : Random Forest ]]')
print(classification_report(y_test, y_pred))

# SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print('\t\t\t[[ Model : SVM ]]')
print(classification_report(y_test, y_pred))

# SGD Classifier
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)
print('\t\t\t[[ Model : SGD ]]')
print(classification_report(y_test, y_pred))

# Logistic Regression
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
print('\t\t  [[ Model : Logistic Regression ]]')
print(classification_report(y_test, y_pred))

# 와인 분류 종료
print('------------------------------------------------------------------\n')


# 3. 유방암 여부 진단 시작
print('-------------------- Diagnosing Breast Cancer --------------------\n')

# 데이터 준비
breast_cancer = load_breast_cancer()

# 데이터 이해하기
breast_cancer_data = breast_cancer.data
breast_cancer_label = breast_cancer.target
print(f'Target Names: {breast_cancer.target_names}')
print(breast_cancer.DESCR)
print('------------------------------------------------------------------\n') # DESCR과 결과값 구분

# train, test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data, breast_cancer_label, test_size=0.2, random_state = 10)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
print('\t\t    [[ Model : Decision Tree ]]')
print(classification_report(y_test, y_pred))

# Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print('\t\t    [[ Model : Random Forest ]]')
print(classification_report(y_test, y_pred))

# SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print('\t\t\t[[ Model : SVM ]]')
print(classification_report(y_test, y_pred))

# SGD Classifier
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)
print('\t\t\t[[ Model : SGD ]]')
print(classification_report(y_test, y_pred))

# Logistic Regression
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
print('\t\t  [[ Model : Logistic Regression ]]')
print(classification_report(y_test, y_pred))

# 유방암 여부 진단 종료
print('------------------------------------------------------------------\n')

''' 4. 총평
< 모델 선별 및 모델의 성능을 평가하는 방법 >
 - classification_report()로 각 모델의 결과값을 확인한다.
 - datasets에서 제공하는 target(ex. 숫자 0, 1, 2, 3 ..., 클래스 1, 2, 3 등) 마다
 정상적으로 결과값이 제공 되는지 확인한다. 이번 예제의 경우 Breast_cacer dataset 에서
 SGD 모델을 사용한 경우에 한하여 class_2의 값을 0.00으로 인식하는 문제가 있었다. 만약 결과를
 단순히 accuracy_score() 만으로 확인 한다면 이런 문제점을 찾아 내기가 힘들것이다. 이말은 즉,
 사용하는 데이터의 종류나 양, 전처리가 진행된 방식 등에 따라 그에 맞는 모델과 맞지 않는 모델이
 존재할 수 있다는 것이며 따라서 결과를 classification_report()로 상세히 확인 한 뒤, 사용 된
 데이터의 성향과 결과값의 precision, recall 수치 등을 고려하여 가장 적합하다고 생각되는 모델을
 선별하는 것이 올바른 평가 및 모델선별 방법이라고 생각한다.
'''
