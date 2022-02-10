# Simple ML Model
## _Let's train model based on datset of COVID-19_

The dataset is taken from kaggle and the training is done to predict if the given attributes can lead to Covid 19.

The dataset contains:
- test_date
- cough
- fever
- sore_throat
- shortness_of_breath
- head_ache
- corona_result
- age_60_and_above
- gender
- test_indication

There are total 278594 entries are present in the dataset, with mixed data types. It contain alpha-numeric data types.

**Note** : All the data types in the dataset are converted to numerical values before going through any training to form a model.

![](https://github.com/Sara-cos/Intern-Work/blob/main/int-ml-5/Simple_ML_Model%20(Covid%2019%20model)/Images/Data%20Types%20ML.png)

Here we can see there are a number of datatypes when they are collected originally which changes to **numerical eventually** with various techniques.

#### Let's check the dataset first...

Going through the dataset, its 9 attribute and more than 250000 rows

```
data.head()
```
![](https://github.com/Sara-cos/Intern-Work/blob/main/int-ml-5/Simple_ML_Model%20(Covid%2019%20model)/Images/Dataset.png)



Decribing the dataset provides these insights...

![](https://github.com/Sara-cos/Intern-Work/blob/main/int-ml-5/Simple_ML_Model%20(Covid%2019%20model)/Images/Describe.png)



The dataset has numerical and text values, let's see what are the categories in the textual values.

![](https://github.com/Sara-cos/Intern-Work/blob/main/int-ml-5/Simple_ML_Model%20(Covid%2019%20model)/Images/Desc_1.png)


And it's very clear how the data is distribution, with focus on normalised numerical data type.

```
sns.countplot(final_data['corona_result'])
```
![](https://github.com/Sara-cos/Intern-Work/blob/main/int-ml-5/Simple_ML_Model%20(Covid%2019%20model)/Images/Count_cases.png)
The results show that its not that prominent that the result is positive all the time, rather insterestingly its negetive maximum time.

```
sns.barplot(final_data['fever'], final_data['corona_result'])
```
![](https://github.com/Sara-cos/Intern-Work/blob/main/int-ml-5/Simple_ML_Model%20(Covid%2019%20model)/Images/Fever_count.png)

Here we can see that fever contributes to fair amount of possibility of postive result of COVID 19 giving the insight that it is one of the major factor.



```
sns.barplot(final_data['shortness_of_breath'], final_data['corona_result'])
```
![](https://github.com/Sara-cos/Intern-Work/blob/main/int-ml-5/Simple_ML_Model%20(Covid%2019%20model)/Images/Shortness_breath.png)

Shortness of breathing provides a strong response here. We can see that the positiveness of the test is majorly affected by the attribute of breathlessness

## Model training

So, we went through the dataset and found that some attributes were weigning a lot and some combination were promising in finding or predicting the result or the target attribute.

Going through the whole we found that the "corona_result" attribute which was the column indicating if the results were positive or negetive is our target attribute. So this is what we will be predicting using other columns.

```
# now target is y and features in X
y = final_data['corona_result']
X = final_data.drop(['corona_result'], axis = 1)
```
This will set our x and y values, which is target values and rest values.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
```
This splits the dataset with 8:2 ratio that is 20 percent is testing data or test_size of 0.20, out of the whole dataset.

> Spiltting is a very crucial step if you want to test your model rightaway. Using what we have as in the datset inself, we can compare if the resulted predicted value is close to the value already present in the dataset. 

#### K-Nearest Neighbors
The first to algorithm used for training.The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.

- Initialize K to your chosen number of neighbors
- Calculate the distance between the query example and the current example from the data.
- Add the distance and the index of the example to an ordered collection
- Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
- Pick the first K entries from the sorted collection
- Get the labels of the selected K entries
- If regression, return the mean of the K labels
- If classification, return the mode of the K labels

```
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(X_train, y_train)
```

```
pred = classifier.predict(X_test) 
accuracy_knn = accuracy_score(y_test, pred)
print("KNN accuracy_score: ", accuracy_knn)
scores_dict['K-NearestNeighbors'] = accuracy_knn * 100
```
**KNN accuracy_score:  0.9545397440729374**

#### Random Forest Classifier

Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction.

> A large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models.

```
RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train, y_train)
```

```
predRandomForest = RandomForest.predict(X_test)
accuracy_rf = accuracy_score(y_test, predRandomForest)
print('RandomForest accuracy_score: ', accuracy_rf)
scores_dict['RandomForestClassifier'] = accuracy_rf * 100
```
**RandomForest accuracy_score:  0.9567472495917012**

#### Decision Tree Classifier

Decision Tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning algorithms, the decision tree algorithm can be used for solving regression and classification problems too.

Creating a training model that can use to predict the class or value of the target variable by learning simple decision rules inferred from prior data(training data) is teh requirement in this algorithm.

- For predicting a class label for a record we start from the root of the tree.
- We compare the values of the root attribute with the record’s attribute. On the basis of comparison
- We follow the branch corresponding to that value and jump to the next node.

```
DecisionTree = DecisionTreeClassifier()
DecisionTree = DecisionTree.fit(X_train, y_train)
```
```
pred = DecisionTree.predict(X_test)
accuracy_dt = accuracy_score(y_test, pred)
print('DecisionTree accuracy_score: ', accuracy_dt)
scores_dict['DecisionTreeClassifier'] = accuracy_dt * 100
```
**DecisionTree accuracy_score:  0.9565857247976454**


### The final results and accuracy

We got the accuracy of from all the 3 algorithms used and highest with very small cut difference was random tree forest with about 95.674%

![](https://github.com/Sara-cos/Intern-Work/blob/main/int-ml-5/Simple_ML_Model%20(Covid%2019%20model)/Images/Score.png)
