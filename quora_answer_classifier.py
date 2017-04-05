from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

training_samples, parameters = map(int, input().split())
#print(training_samples, parameters)
training_data = []
training_label = []
for i in range(training_samples):
    temp = input().strip().split(" ")
    training_label.append(str(temp[1]))
    x=[]
    for j in temp[2:]:
        position,value=j.split(":")
        #print(position,value)
        x.append(float(value))
    training_data.append(x)
    
#print(training_data)
#print(training_label)

clf=RandomForestClassifier(n_estimators = 65, bootstrap = False, max_features = "log2", class_weight = 'auto')


clf.fit(training_data, training_label)

test_samples = (int)(input().strip())
#print(test_samples)
test_data = []
test_data_first = []
for i in range(test_samples):
    temp = input().strip().split(" ")
    test_data_first.append(temp[0])
    x=[]
    for j in temp[1:]:
        position,value=j.split(":")
        #print(position,value)
        x.append(float(value))
    test_data.append(x)
#print(test_data)
#test_data = vectorizer.transform(test_data)
prediction = clf.predict(test_data)
for i in range(len(prediction)):
    print(test_data_first[i],prediction[i],sep=' ')