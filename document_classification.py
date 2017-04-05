#from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier

# opening the txt file
training_file = open("trainingdata.txt")
training_samples = 0
training_data = []
training_label = []
# differentiating the training_input and training_label
for line in training_file:
    line = line.strip()
    item = line.strip().split(" ")
    if(len(item) == 1):
        training_samples = item[0]
    else:
        training_data.append(line[2:])
        training_label.append(item[0])
training_file.close()

#print(training_data)
#print(training_label)
#Convert a collection of raw documents to a matrix of TF-IDF features.
#Equivalent to CountVectorizer followed by TfidfTransformer.
vectorizer = TfidfVectorizer()
#   Fit to data, then transform it.
#   Learn the vocabulary dictionary and return term-document matrix.
x_train = vectorizer.fit_transform(training_data)

#clf = Perceptron() #94.7
# PA always classifies correctly last seen data(most recent update in data) but not with perceptron and SVM 
# because update size constant
# PA is more efficien but weaker to noise than SVM and perceptron 
clf = PassiveAggressiveClassifier() # 95.52
#clf = KNeighborsClassifier(n_neighbors=10) # 75.58

clf.fit(x_train, training_label)
    
total_samples = (int)(input())
test_data = []
for i in range(total_samples):
    test_data.append(input())
#print(test_data)

# Transform documents to document-term matrix.
x_test = vectorizer.transform(test_data)
# predict the result
prediction = clf.predict(x_test)
for i in range(len(prediction)):
    print(prediction[i])