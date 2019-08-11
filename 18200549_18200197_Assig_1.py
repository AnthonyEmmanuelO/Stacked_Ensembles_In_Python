
# coding: utf-8

# ## Sumit_Chawan 
# 

# In[1]:


from IPython.display import display, HTML, Image

from TAS_Python_Utilities import data_viz
from TAS_Python_Utilities import data_viz_target
from TAS_Python_Utilities import visualize_tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from random import randint
import math

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial import distance
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import linear_model
from sklearn import neighbors
from sklearn.utils import resample
from sklearn.metrics import cohen_kappa_score
import warnings
from sklearn.model_selection import KFold 
import itertools
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
#%qtconsole


# In[2]:


def create_classifier(classifier_type, tree_min_samples_split = 20):

    if classifier_type == "svm":
        c = svm.SVC(probability=True)

    elif classifier_type == "logreg":
        c = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=1000)

    elif classifier_type == "knn":
        c = neighbors.KNeighborsClassifier()

    elif classifier_type == "tree":
        c = tree.DecisionTreeClassifier(min_samples_split = tree_min_samples_split)

    elif classifier_type == "randomforest":
        c = ensemble.RandomForestClassifier()
        
    else:
        c = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=1000)
    
    return c


# In[3]:


# Create a new classifier which is based on the sckit-learn BaseEstimator and ClassifierMixin classes
class StackedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    
    """An ensemble classifier that uses heterogeneous models at the base layer and a aggregatnio model at the aggregation layer. A k-fold cross validation is used to gnerate training data for the stack layer model.

    Parameters
    ----------
    base_estimators: list 
        A list of the classifiers in the ase layer of the ensemble. Supported types are
        - "svm" Support Vector Machine implemented by sklearn.svm.SVC
        - "logreg" Logistic Regression implemented by sklearn.linear_models.LogisticRegression
        - "knn" k Nearest Neighbour implemented by sklearn.neighbors.KNeighborsClassifier
        - "tree" Decision Tree implemented by sklearn.tree.DecisionTreeClassifier
        - "randomforest" RandomForest implemented by sklearn.tree.RandomForestClassifier    
    classifier_duplicates: int, optional (default = 1)
        How many instances of each classifier type listed in base_estimators is included in the ensemble
    stack_layer_classifier: string, optional (default = "logreg')
        The classifier type used at the stack layer. The same classifier types as are supported at the base layer are supported        
    training_folds: int, optional (default = 4)
        How many folds will be used to generate the training set for the stacked layer
        
    Attributes
    ----------
    classes_ : array of shape = [n_classes] 
        The classes labels (single output problem).


    Notes
    -----
    The default values for most base learners are used.

    See also
    --------
    
    ----------
    .. [1]  van der Laan, M., Polley, E. & Hubbard, A. (2007). 
            Super Learner. Statistical Applications in Genetics 
            and Molecular Biology, 6(1) 
            doi:10.2202/1544-6115.1309
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> clf = StackedEnsembleClassifier()
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)

    """
    # Constructor for the classifier object
    def __init__(self, base_estimator_types = ["svm", "logreg", "tree"], base_estimator_duplicates = 8, stack_layer_classifier_type = "logreg"):
        """Setup a SuperLearner classifier .
        Parameters
        ----------
        base_estimator_types: The types of classifiers to include at the base layer
        base_estimator_duplicates: The number of duplicates of each type of classiifer to include
        stack_layer_classifier_type: The type of classifier to include at the stack layer 
        
        Returns
        -------
        Nothing
        """     

        # Initialise class variabels
        self.base_estimator_types = base_estimator_types
        self.base_estimator_type_list = list()
        self.base_estimator_duplicates = base_estimator_duplicates
        self.stack_layer_classifier_type = stack_layer_classifier_type

    # The fit function to train a classifier
    def fit(self, X, y):
        """Build a SuperLearner classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. 
        y : array-like, shape = [n_samples] 
            The target values (class labels) as integers or strings.
        Returns
        -------
        self : object
        """    
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        ########################
        # LEVEL 0
        ########################
        
        # Set up the base classifeirs in the ensemble
        self.classifiers_ = list()
        
        for i in range(0, self.base_estimator_duplicates):
            for t in self.base_estimator_types:

                self.base_estimator_type_list.append(t)      
                c = create_classifier(t, tree_min_samples_split=math.ceil(len(X)*0.05))
                self.classifiers_.append(c)
        
        # Store the number of classifers in the ensemble
        self.n_estimators_ = len(self.classifiers_)

        # Use all training data to train base classifiers
        X_train = X
        y_train = y
        
        # Set up empty arrays to hold stack layer training data
        self.X_stack_train = None #(dtype = float)
        self.y_stack_train = y_train
          
        # Train each base calssifier and generate the stack layer training dataset
        for classifier in self.classifiers_:

            # Extract a bootstrap sample
            X_train_samp, y_train_samp = resample(X_train, y_train, replace=True)    
            
            # Train a base classifier
            classifier.fit(X_train_samp, y_train_samp)
            
            # Make predictions for all instances in the training set
            y_pred = classifier.predict_proba(X_train)

            # Append the predictions ot the stack layer traing set (a bit of hacking here!)
            try:
                self.X_stack_train = np.c_[self.X_stack_train, y_pred]
            except ValueError:
                self.X_stack_train = y_pred
      
        ########################
        # LEVEL 1
        ########################
        
        # Create the stack layer classifier
        self.stack_layer_classifier_ = create_classifier(self.stack_layer_classifier_type, tree_min_samples_split=math.ceil(len(X)*0.05))

        # Train the stack layer using the newly created dataset
        self.stack_layer_classifier_.fit(self.X_stack_train, self.y_stack_train)
            
        # Return the classifier
        return self

    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        """Predict class labels of the input samples X.
        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The input samples. 
        Returns
        -------
        p : array of shape = [n_samples, ].
            The predicted class labels of the input samples. 
        """
        
        # Check is fit had been called by confirming that the teamplates_ dictiponary has been set up
        check_is_fitted(self, ['stack_layer_classifier_'])

        # Check that the input features match the type and shape of the training features
        X = check_array(X)
   
        X_stack_queries = None
              
        # Make a prediction with each base classifier and assemble the stack layer query
        for classifier in self.classifiers_:
            
            y_pred = classifier.predict_proba(X)
            
            try:
                X_stack_queries = np.c_[X_stack_queries, y_pred]
            except ValueError:
                X_stack_queries = y_pred
        
        # Return the prediction made by the stack layer classifier
        return self.stack_layer_classifier_.predict(X_stack_queries)
    
    # The predict function to make a set of predictions for a set of query instances
    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.
        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The input samples. 
        Returns
        -------
        p : array of shape = [n_samples, n_labels].
            The predicted class label probabilities of the input samples. 
        """
        # Check is fit had been called by confirming that the teamplates_ dictiponary has been set up
        check_is_fitted(self, ['stack_layer_classifier_'])

        # Check that the input features match the type and shape of the training features
        X = check_array(X)
        
        X_stack_queries = None
        
        # Make a prediction with each base classifier
        for classifier in self.classifiers_:
            
            y_pred = classifier.predict_proba(X)
                
            try:
                X_stack_queries = np.c_[X_stack_queries, y_pred]
            except ValueError:
                X_stack_queries = y_pred

        # Return the prediction made by the stack layer classifier        
        return self.stack_layer_classifier_.predict_proba(X_stack_queries)


# ## Task-1 

# In[4]:


# Write your code here
# Create a new classifier which is based on the sckit-learn BaseEstimator and ClassifierMixin classes
class StackedEnsembleHoldOutClassifier(BaseEstimator, ClassifierMixin):
   
    # Constructor for the classifier object
    def __init__(self, base_estimator_types = ["svm", "logreg", "tree"], base_estimator_duplicates = 8, stack_layer_classifier_type = "logreg"):
        self.base_estimator_types = base_estimator_types
        self.base_estimator_type_list = list()
        self.base_estimator_duplicates = base_estimator_duplicates
        self.stack_layer_classifier_type = stack_layer_classifier_type

    def fit(self, X, y):
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        ########################
        # LEVEL 0
        ########################
        
        # Set up the base classifeirs in the ensemble
        self.classifiers_ = list()
        for i in range(0, self.base_estimator_duplicates):
            for t in self.base_estimator_types:
                self.base_estimator_type_list.append(t)      
                c = create_classifier(t, tree_min_samples_split=math.ceil(len(X)*0.05))
                self.classifiers_.append(c)
        
        # Store the number of classifers in the ensemble
        self.n_estimators_ = len(self.classifiers_)

        # Use all training data to train base classifiers
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=1)
        
        # Set up empty arrays to hold stack layer training data
        self.X_stack_train = None #(dtype = float)
        self.y_stack_train = y_val
        # Train each base calssifier and generate the stack layer training dataset
        for classifier in self.classifiers_:

            # Extract a bootstrap sample
            X_train_samp, y_train_samp = resample(X_train, y_train, replace=True)    
            
            # Train a base classifier
            classifier.fit(X_train_samp, y_train_samp)
            
            # Make predictions for all instances in the training set
            y_pred = classifier.predict_proba(X_val)
            # Append the predictions ot the stack layer traing set (a bit of hacking here!)
            try:
                self.X_stack_train = np.c_[self.X_stack_train, y_pred]
            except ValueError:
                self.X_stack_train = y_pred
       
        ########################
        # LEVEL 1
        ########################
                
        # Create the stack layer classifier
        self.stack_layer_classifier_ = create_classifier(self.stack_layer_classifier_type, tree_min_samples_split=math.ceil(len(X)*0.05))

        # Train the stack layer using the newly created dataset
        self.stack_layer_classifier_.fit(self.X_stack_train, self.y_stack_train)
            
        # Return the classifier
        return self

    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        
        # Check is fit had been called by confirming that the teamplates_ dictiponary has been set up
        check_is_fitted(self, ['stack_layer_classifier_'])

        # Check that the input features match the type and shape of the training features
        X = check_array(X)
   
        X_stack_queries = None
              
        # Make a prediction with each base classifier and assemble the stack layer query
        for classifier in self.classifiers_:
            
            y_pred = classifier.predict_proba(X)
            
            try:
                X_stack_queries = np.c_[X_stack_queries, y_pred]
            except ValueError:
                X_stack_queries = y_pred
        
        # Return the prediction made by the stack layer classifier
        return self.stack_layer_classifier_.predict(X_stack_queries)
    
    # The predict function to make a set of predictions for a set of query instances
    def predict_proba(self, X):
        # Check is fit had been called by confirming that the teamplates_ dictiponary has been set up
        check_is_fitted(self, ['stack_layer_classifier_'])

        # Check that the input features match the type and shape of the training features
        X = check_array(X)
        
        X_stack_queries = None
        
        # Make a prediction with each base classifier
        for classifier in self.classifiers_:
            
            y_pred = classifier.predict_proba(X)
                
            try:
                X_stack_queries = np.c_[X_stack_queries, y_pred]
            except ValueError:
                X_stack_queries = y_pred

        # Return the prediction made by the stack layer classifier        
        return self.stack_layer_classifier_.predict_proba(X_stack_queries)


# ## Task-2

# In[5]:


# Write your code here
# Create a new classifier which is based on the sckit-learn BaseEstimator and ClassifierMixin classes
class StackedEnsembleKFoldClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, base_estimator_types = ["svm", "logreg", "tree"], base_estimator_duplicates = 8, stack_layer_classifier_type = "logreg",kFold = 5):
        # Initialise class variabels
        self.base_estimator_types = base_estimator_types
        self.base_estimator_type_list = list()
        self.base_estimator_duplicates = base_estimator_duplicates
        self.stack_layer_classifier_type = stack_layer_classifier_type
        self.kFold = kFold

    # The fit function to train a classifier
    def fit(self, X, y):
   
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
          
        # Set up the base classifeirs in the ensemble
        self.classifiers_ = list()
        for i in range(0, self.base_estimator_duplicates):
            for t in self.base_estimator_types:
                self.base_estimator_type_list.append(t)      
                c = create_classifier(t, tree_min_samples_split=math.ceil(len(X)*0.05))
                self.classifiers_.append(c)
        
     
        
        # Set up empty arrays to hold stack layer training data
        #kfold = KFold(self.kFold, True, 1)
        skf = StratifiedKFold(n_splits=self.kFold)
        self.X_stack_combine = None
        self.y_stack_train = None
        #for train, val in kfold.split(X):
        for train, val in skf.split(X, y):
          X_train = X[train]
          X_val = X[val]
          y_train = y[train]
          y_val = y[val]
          
          try:
              self.y_stack_train = np.r_[self.y_stack_train, y_val]
          except ValueError:
              self.y_stack_train = y_val
              
          self.X_stack_train = None
          for classifier in self.classifiers_:
            X_train_samp, y_train_samp = resample(X_train, y_train, replace=True)
            classifier.fit(X_train_samp, y_train_samp)
            y_pred = classifier.predict_proba(X_val)
            try:
              self.X_stack_train = np.c_[self.X_stack_train, y_pred]
            except ValueError:
              self.X_stack_train = y_pred
          
          try:
            self.X_stack_combine = np.r_[self.X_stack_combine, self.X_stack_train]
          except ValueError:
            self.X_stack_combine = self.X_stack_train
          
        self.stack_layer_classifier_ = create_classifier(self.stack_layer_classifier_type, tree_min_samples_split=math.ceil(len(X)*0.05))
        self.stack_layer_classifier_.fit(self.X_stack_combine, self.y_stack_train)
            
        # Return the classifier
        return self

    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        check_is_fitted(self, ['stack_layer_classifier_'])

        X = check_array(X)
   
        X_stack_queries = None
              
        for classifier in self.classifiers_:
            
            y_pred = classifier.predict_proba(X)
            
            try:
                X_stack_queries = np.c_[X_stack_queries, y_pred]
            except ValueError:
                X_stack_queries = y_pred
        
        # Return the prediction made by the stack layer classifier
        return self.stack_layer_classifier_.predict(X_stack_queries)
    
    # The predict function to make a set of predictions for a set of query instances
    def predict_proba(self, X):
        # Check is fit had been called by confirming that the teamplates_ dictiponary has been set up
        check_is_fitted(self, ['stack_layer_classifier_'])

        # Check that the input features match the type and shape of the training features
        X = check_array(X)
        
        X_stack_queries = None
        
        # Make a prediction with each base classifier
        for classifier in self.classifiers_:
            
            y_pred = classifier.predict_proba(X)
                
            try:
                X_stack_queries = np.c_[X_stack_queries, y_pred]
            except ValueError:
                X_stack_queries = y_pred

        # Return the prediction made by the stack layer classifier        
        return self.stack_layer_classifier_.predict_proba(X_stack_queries)


# In[6]:


# Write your code here
# Create a new classifier which is based on the sckit-learn BaseEstimator and ClassifierMixin classes
class StackedEnsembleOneVsOneClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, stack_layer_classifier_type = "logreg",base_classifier_type = "svm"):
        self.base_estimator_type_list = list()
        self.stack_layer_classifier_type = stack_layer_classifier_type
        self.base_classifier_type = base_classifier_type

    # The fit function to train a classifier
    def fit(self, X, y):        
        # Check that X and y have correct shape
        orignalX = X
        orignalY = y
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        
        self.classifiers_ = list()
        
        self.classes_ = unique_labels(y)
        
        strUniqueLabels =''.join(str(x) for x in self.classes_)
        
        data = pd.concat([orignalX, orignalY], axis=1)
        
        label_combination = list(itertools.combinations(strUniqueLabels, 2))
        self.X_stack_train = None
        self.y_stack_train = y
                
        for i in range(0,len(label_combination)):
          combineData = data[(data['label']==int(label_combination[i][0])) | (data['label']==int(label_combination[i][1]))]
          
          X_new = combineData.drop('label',axis=1)
          Y_new = combineData['label']
          
         
          
          classifier = create_classifier(self.base_classifier_type, tree_min_samples_split=math.ceil(len(X)*0.05))
          
          self.classifiers_.append(classifier)
          
          X_train_samp, y_train_samp = resample(X_new, Y_new, replace=True)    
            
          classifier.fit(X_train_samp, y_train_samp)
            
          y_pred = classifier.predict_proba(X)
          
          try:
            self.X_stack_train = np.c_[self.X_stack_train, y_pred]
          except ValueError:
            self.X_stack_train = y_pred
                
        self.stack_layer_classifier_ = create_classifier(self.stack_layer_classifier_type, tree_min_samples_split=math.ceil(len(X)*0.05))
        self.stack_layer_classifier_.fit(self.X_stack_train, self.y_stack_train)
           
        # Return the classifier
        return self

    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        
        # Check is fit had been called by confirming that the teamplates_ dictiponary has been set up
        check_is_fitted(self, ['stack_layer_classifier_'])

        # Check that the input features match the type and shape of the training features
        X = check_array(X)
   
        X_stack_queries = None
              
        # Make a prediction with each base classifier and assemble the stack layer query
        for classifier in self.classifiers_:
            
            y_pred = classifier.predict_proba(X)
            
            try:
                X_stack_queries = np.c_[X_stack_queries, y_pred]
            except ValueError:
                X_stack_queries = y_pred
        
        # Return the prediction made by the stack layer classifier
        return self.stack_layer_classifier_.predict(X_stack_queries)
    
    # The predict function to make a set of predictions for a set of query instances
    def predict_proba(self, X):
       
        # Check is fit had been called by confirming that the teamplates_ dictiponary has been set up
        check_is_fitted(self, ['stack_layer_classifier_'])

        # Check that the input features match the type and shape of the training features
        X = check_array(X)
        
        X_stack_queries = None
        
        # Make a prediction with each base classifier
        for classifier in self.classifiers_:
            
            y_pred = classifier.predict_proba(X)
                
            try:
                X_stack_queries = np.c_[X_stack_queries, y_pred]
            except ValueError:
                X_stack_queries = y_pred

        # Return the prediction made by the stack layer classifier        
        return self.stack_layer_classifier_.predict_proba(X_stack_queries)


# ## Prepared Data

# In[7]:


accuracy_valid = {} # Accuracy on Validation Set
accuracy_train_test = {} # Accuracy on Train Set
accuracy_data_test ={} # Accuracy on Test Data


# In[8]:


data_train =pd.read_csv('fashion-mnist_train.csv')
data_train = data_train.iloc[0:int(0.01*len(data_train)),:]
for i in range(1,len(data_train.iloc[0])):
  data_train['pixel'+str(i)] = data_train['pixel'+str(i)].fillna(0)
 


# In[9]:


X = data_train.drop('label',axis=1)
y = data_train['label']


# In[10]:


X_train_plus_valid, X_test, y_train_plus_valid, y_test     = train_test_split(X, y, random_state=0,                                     test_size = 0.3)

X_train, X_valid, y_train, y_valid     = train_test_split(X_train_plus_valid,                                         y_train_plus_valid,                                         random_state=0,                                         train_size = 0.5/0.7)


# In[11]:


data_test = pd.read_csv('fashion-mnist_test.csv')


# In[12]:


data_test_X = data_test[data_test.columns[1:]]
data_test_Y = np.array(data_test["label"])


# ## Task-3

# In[ ]:


stack_ensemble_clf = StackedEnsembleClassifier()


# In[18]:


stack_ensemble_clf.fit(X_train,y_train)


# In[ ]:


y_pred = stack_ensemble_clf.predict(X_valid)


# In[20]:


accuracy = metrics.accuracy_score(y_valid, y_pred) # , normalize=True, sample_weight=None
accuracy_valid["StackedEnsembleClassifier"] = accuracy
print("Accuracy on Training Set: " +  str(accuracy))
print(metrics.classification_report(y_valid, y_pred))
print("Confusion Matrix for Training Set")
display(pd.crosstab(np.array(y_valid), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[ ]:


y_pred = stack_ensemble_clf.predict(X_test)


# In[22]:


accuracy = metrics.accuracy_score(y_test, y_pred) # , normalize=True, sample_weight=None
accuracy_train_test["StackedEnsembleClassifier"] = accuracy
print("Accuracy on Testing set: " +  str(accuracy))
print(metrics.classification_report(y_test, y_pred))
print("Confusion Matrix for Testing set:")
display(pd.crosstab(np.array(y_test), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[ ]:


### Train the StackedEnsembleClassifier on the training data and test it with the test data


# In[23]:


stack_ensemble_clf.fit(X,y)


# In[ ]:


# Predictions for Test_dataset
y_pred = stack_ensemble_clf.predict(data_test_X)


# In[25]:


accuracy = metrics.accuracy_score(data_test_Y, y_pred) # , normalize=True, sample_weight=None
accuracy_data_test["StackedEnsembleClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(data_test_Y, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(data_test_Y), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[ ]:


stack_hold_out_ensemble_clf = StackedEnsembleHoldOutClassifier()


# In[27]:


stack_hold_out_ensemble_clf.fit(X_train, y_train)


# In[ ]:


y_pred = stack_hold_out_ensemble_clf.predict(X_valid)


# In[29]:


accuracy = metrics.accuracy_score(y_valid, y_pred) # , normalize=True, sample_weight=None
accuracy_valid["StackedEnsembleHoldOutClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_valid, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_valid), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[ ]:


y_pred = stack_hold_out_ensemble_clf.predict(X_test)


# In[31]:


accuracy = metrics.accuracy_score(y_test, y_pred) # , normalize=True, sample_weight=None
accuracy_train_test["StackedEnsembleHoldOutClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_test, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_test), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[32]:


stack_hold_out_ensemble_clf.fit(X, y)


# In[ ]:


y_pred = stack_hold_out_ensemble_clf.predict(data_test_X)


# In[34]:


accuracy = metrics.accuracy_score(data_test_Y, y_pred) # , normalize=True, sample_weight=None
accuracy_data_test["StackedEnsembleHoldOutClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(data_test_Y, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(data_test_Y), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# ## Performance of StackedEnsembleClassifierKFold

# In[ ]:


stack_Kfold_ensemble_clf = StackedEnsembleKFoldClassifier()


# In[39]:


stack_Kfold_ensemble_clf.fit(X_train, y_train)


# In[ ]:


y_pred = stack_Kfold_ensemble_clf.predict(X_valid)


# In[41]:


accuracy = metrics.accuracy_score(y_valid, y_pred) # , normalize=True, sample_weight=None
accuracy_valid["StackedEnsembleKFoldClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_valid, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_valid), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[ ]:


y_pred = stack_Kfold_ensemble_clf.predict(X_test)


# In[43]:


accuracy = metrics.accuracy_score(y_test, y_pred) # , normalize=True, sample_weight=None
accuracy_train_test["StackedEnsembleKFoldClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_test, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_test), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[44]:


stack_Kfold_ensemble_clf.fit(X, y)


# In[ ]:


y_pred = stack_Kfold_ensemble_clf.predict(data_test_X)


# In[46]:


accuracy = metrics.accuracy_score(data_test_Y, y_pred) # , normalize=True, sample_weight=None
accuracy_data_test["StackedEnsembleKFoldClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(data_test_Y, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(data_test_Y), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# ## Graphs for Task 3

# In[ ]:


objects = ('StackedEnsemble', 'HoldOut', 'KFold')
y_pos = np.arange(len(objects))
performance = [accuracy_valid['StackedEnsembleClassifier'],accuracy_valid['StackedEnsembleHoldOutClassifier'],accuracy_valid['StackedEnsembleKFoldClassifier']]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Perfomance on valid Set')
 
plt.show()


# In[ ]:


objects = ('StackedEnsemble', 'HoldOut', 'KFold')
y_pos = np.arange(len(objects))
performance = [accuracy_train_test['StackedEnsembleClassifier'],accuracy_train_test['StackedEnsembleHoldOutClassifier'],accuracy_train_test['StackedEnsembleKFoldClassifier']]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Perfomance on train test Set')
 
plt.show()


# In[ ]:


objects = ('StackedEnsemble', 'HoldOut', 'KFold')
y_pos = np.arange(len(objects))
performance = [accuracy_data_test['StackedEnsembleClassifier'],accuracy_data_test['StackedEnsembleHoldOutClassifier'],accuracy_data_test['StackedEnsembleKFoldClassifier']]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Perfomance on test Data')
 
plt.show()


# ## Task 4: Comparing the Performance of various Stack-Layer Approaches with some standard Approaches

# In[47]:


my_decision_tree = tree.DecisionTreeClassifier(criterion="entropy")
my_decision_tree.fit(X_train,y_train)


# In[ ]:


y_pred = my_decision_tree.predict(X_valid)


# In[54]:


accuracy = metrics.accuracy_score(y_valid, y_pred) # , normalize=True, sample_weight=None
accuracy_valid["DecisionTreeClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_valid, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_valid), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[ ]:


y_pred = my_decision_tree.predict(X_test)


# In[56]:


accuracy = metrics.accuracy_score(y_test, y_pred) # , normalize=True, sample_weight=None
accuracy_train_test["DecisionTreeClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_test, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_test), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[57]:


my_decision_tree.fit(X,y)


# In[ ]:


y_pred = my_decision_tree.predict(data_test_X)


# In[59]:


accuracy = metrics.accuracy_score(data_test_Y, y_pred) # , normalize=True, sample_weight=None
accuracy_data_test["DecisionTreeClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(data_test_Y, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(data_test_Y), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# ## Gridsearch Approach

# In[ ]:


cv_folds =2
model_tuned_params_list = dict()


# In[61]:


param_grid ={'criterion': ['gini', "entropy"],              'max_depth': list(range(3, 50, 3)),              'min_samples_split': [200]}

# Perform the search
tuned_tree = GridSearchCV(tree.DecisionTreeClassifier(),                                 param_grid, cv=cv_folds, verbose = 2,                             return_train_score=True)
tuned_tree.fit(X_train_plus_valid, y_train_plus_valid)

# Print details
print("Best parameters set found on development set:")
display(tuned_tree.best_params_)
model_tuned_params_list["Tuned Tree"] = tuned_tree.best_params_
display(tuned_tree.best_score_)
display(tuned_tree.cv_results_)


# In[ ]:


tuned_tree.fit(X_train,y_train)


# In[ ]:


y_pred = tuned_tree.predict(X_valid)


# In[ ]:


accuracy = metrics.accuracy_score(y_valid, y_pred) # , normalize=True, sample_weight=None
accuracy_valid["PrunedDecisionTreeClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_valid, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_valid), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[ ]:


y_pred = tuned_tree.predict(X_test)


# In[ ]:


accuracy = metrics.accuracy_score(y_test, y_pred) # , normalize=True, sample_weight=None
accuracy_train_test["PrunedDecisionTreeClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_test, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_test), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[74]:


tuned_tree.fit(X,y)


# In[ ]:


y_pred = tuned_tree.predict(data_test_X)


# In[76]:


accuracy = metrics.accuracy_score(data_test_Y, y_pred) # , normalize=True, sample_weight=None
accuracy_data_test["PrunedDecisionTreeClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(data_test_Y, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(data_test_Y), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# ## Bagging

# In[78]:


bagging_model = ensemble.BaggingClassifier(base_estimator = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf = 50),                                       n_estimators=10)
bagging_model.fit(X_train,y_train)


# In[ ]:


y_pred = bagging_model.predict(X_valid)


# In[82]:


accuracy = metrics.accuracy_score(y_valid, y_pred) # , normalize=True, sample_weight=None
accuracy_valid["BaggingClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_valid, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_valid), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[ ]:


y_pred = bagging_model.predict(X_test)


# In[ ]:


accuracy = metrics.accuracy_score(y_test, y_pred) # , normalize=True, sample_weight=None
accuracy_train_test["BaggingClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_test, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_test), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[ ]:


bagging_model.fit(X,y)


# In[ ]:


y_pred = bagging_model.predict(data_test_X)


# In[ ]:


accuracy = metrics.accuracy_score(data_test_Y, y_pred) # , normalize=True, sample_weight=None
accuracy_data_test["BaggingClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(data_test_Y, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(data_test_Y), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# ## Best Params for Grid Search Approach

# In[ ]:


param_grid = [
 {'n_estimators': list(range(50, 501, 50)),
  'base_estimator': [tree.DecisionTreeClassifier(criterion="entropy", max_depth = 6, min_samples_leaf = 200)]}
]


# In[ ]:


tuned_grid_model = GridSearchCV(ensemble.BaggingClassifier(), param_grid, cv=cv_folds, verbose = 2)


# In[91]:


tuned_grid_model.fit(X_train_plus_valid, y_train_plus_valid)


# In[92]:


print("Best params found:")
print(tuned_grid_model.best_params_)
model_tuned_params_list["Tuned Bagging"] = tuned_grid_model.best_params_
print(tuned_grid_model.best_score_)


# In[ ]:


tuned_grid_model.fit(X_train,y_train)


# In[ ]:


y_pred = tuned_grid_model.predict(X_valid)


# In[ ]:


accuracy = metrics.accuracy_score(y_valid, y_pred) # , normalize=True, sample_weight=None
accuracy_valid["TunedBaggingClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_valid, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_valid), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[ ]:


y_pred = tuned_grid_model.predict(X_test)


# In[98]:


accuracy = metrics.accuracy_score(y_test, y_pred) # , normalize=True, sample_weight=None
accuracy_train_test["TunedBaggingClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_test, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_test), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[ ]:


tuned_grid_model.fit(X,y)


# In[ ]:


y_pred = tuned_grid_model.predict(data_test_X)


# In[ ]:


accuracy = metrics.accuracy_score(data_test_Y, y_pred) # , normalize=True, sample_weight=None
accuracy_data_test["TunedBaggingClassifier"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(data_test_Y, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(data_test_Y), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[ ]:


objects = ('Stacked', 'HoldOut', 'KFold','Tree','PrunedTree','Bagg','TunedBagg')
y_pos = np.arange(len(objects))
performance = [accuracy_valid['StackedEnsembleClassifier'],accuracy_valid['StackedEnsembleHoldOutClassifier'],accuracy_valid['StackedEnsembleKFoldClassifier'],accuracy_valid['DecisionTreeClassifier'],accuracy_valid['PrunedDecisionTreeClassifier'],accuracy_valid['BaggingClassifier'],accuracy_valid['TunedBaggingClassifier']]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Perfomance on valid set')
 
plt.show()


# In[ ]:


objects = ('Stacked', 'HoldOut', 'KFold','Tree','PrunedTree','Bagg','TunedBagg')
y_pos = np.arange(len(objects))
performance = [accuracy_train_test['StackedEnsembleClassifier'],accuracy_train_test['StackedEnsembleHoldOutClassifier'],accuracy_train_test['StackedEnsembleKFoldClassifier'],accuracy_train_test['DecisionTreeClassifier'],accuracy_train_test['PrunedDecisionTreeClassifier'],accuracy_train_test['BaggingClassifier'],accuracy_train_test['TunedBaggingClassifier']]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Perfomance on Train Test set')
 
plt.show()


# In[ ]:


objects = ('Stacked', 'HoldOut', 'KFold','Tree','PrunedTree','Bagg','TunedBagg')
y_pos = np.arange(len(objects))
performance = [accuracy_data_test['StackedEnsembleClassifier'],accuracy_data_test['StackedEnsembleHoldOutClassifier'],accuracy_data_test['StackedEnsembleKFoldClassifier'],accuracy_data_test['DecisionTreeClassifier'],accuracy_data_test['PrunedDecisionTreeClassifier'],accuracy_data_test['BaggingClassifier'],accuracy_data_test['TunedBaggingClassifier']]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Perfomance on Test set')
 
plt.show()


# ## Task 5 One Vs One

# In[10]:


# Write your code here
# Create a new classifier which is based on the sckit-learn BaseEstimator and ClassifierMixin classes
class StackedEnsembleOneVsOneClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, stack_layer_classifier_type = "logreg",base_classifier_type = "svm"):
        self.base_estimator_type_list = list()
        self.stack_layer_classifier_type = stack_layer_classifier_type
        self.base_classifier_type = base_classifier_type

    # The fit function to train a classifier
    def fit(self, X, y):        
        # Check that X and y have correct shape
        orignalX = X
        orignalY = y
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        
        self.classifiers_ = list()
        
        self.classes_ = unique_labels(y)
        
        strUniqueLabels =''.join(str(x) for x in self.classes_)
        
        data = pd.concat([orignalX, orignalY], axis=1)
        
        label_combination = list(itertools.combinations(strUniqueLabels, 2))
        self.X_stack_train = None
        self.y_stack_train = y
                
        for i in range(0,len(label_combination)):
          combineData = data[(data['label']==int(label_combination[i][0])) | (data['label']==int(label_combination[i][1]))]
          
          X_new = combineData.drop('label',axis=1)
          Y_new = combineData['label']
          
         
          
          classifier = create_classifier(self.base_classifier_type, tree_min_samples_split=math.ceil(len(X)*0.05))
          
          self.classifiers_.append(classifier)
          
          X_train_samp, y_train_samp = resample(X_new, Y_new, replace=True)    
            
          classifier.fit(X_train_samp, y_train_samp)
            
          y_pred = classifier.predict_proba(X)
          
          try:
            self.X_stack_train = np.c_[self.X_stack_train, y_pred]
          except ValueError:
            self.X_stack_train = y_pred
                
        self.stack_layer_classifier_ = create_classifier(self.stack_layer_classifier_type, tree_min_samples_split=math.ceil(len(X)*0.05))
        self.stack_layer_classifier_.fit(self.X_stack_train, self.y_stack_train)
           
        # Return the classifier
        return self

    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        
        # Check is fit had been called by confirming that the teamplates_ dictiponary has been set up
        check_is_fitted(self, ['stack_layer_classifier_'])

        # Check that the input features match the type and shape of the training features
        X = check_array(X)
   
        X_stack_queries = None
              
        # Make a prediction with each base classifier and assemble the stack layer query
        for classifier in self.classifiers_:
            
            y_pred = classifier.predict_proba(X)
            
            try:
                X_stack_queries = np.c_[X_stack_queries, y_pred]
            except ValueError:
                X_stack_queries = y_pred
        
        # Return the prediction made by the stack layer classifier
        return self.stack_layer_classifier_.predict(X_stack_queries)
    
    # The predict function to make a set of predictions for a set of query instances
    def predict_proba(self, X):
       
        # Check is fit had been called by confirming that the teamplates_ dictiponary has been set up
        check_is_fitted(self, ['stack_layer_classifier_'])

        # Check that the input features match the type and shape of the training features
        X = check_array(X)
        
        X_stack_queries = None
        
        # Make a prediction with each base classifier
        for classifier in self.classifiers_:
            
            y_pred = classifier.predict_proba(X)
                
            try:
                X_stack_queries = np.c_[X_stack_queries, y_pred]
            except ValueError:
                X_stack_queries = y_pred

        # Return the prediction made by the stack layer classifier        
        return self.stack_layer_classifier_.predict_proba(X_stack_queries)


# ## Task 6 Evaluate the Performance of the StackedEnsembleClassifierOneVsOne Algorithm

# In[13]:


stacked_ensembleOneVsOne_clf= StackedEnsembleOneVsOneClassifier(stack_layer_classifier_type = "logreg",base_classifier_type = "logreg")


# In[14]:


stacked_ensembleOneVsOne_clf.fit(X_train, y_train)


# In[15]:


y_pred = stacked_ensembleOneVsOne_clf.predict(X_valid)


# In[16]:


accuracy = metrics.accuracy_score(y_valid, y_pred) # , normalize=True, sample_weight=None
accuracy_valid["StackedEnsembleOneVsOne_Log_n_Log"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_valid, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_valid), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[17]:


y_pred = stacked_ensembleOneVsOne_clf.predict(X_test)


# In[18]:


accuracy = metrics.accuracy_score(y_test, y_pred) # , normalize=True, sample_weight=None
accuracy_train_test["StackedEnsembleOneVsOne_Log_n_Log"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_test, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_test), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[19]:


### Training stacked_ensembleOneVsOne_clf with training data and test data


# In[20]:


stacked_ensembleOneVsOne_clf.fit(X, y)


# In[21]:


y_pred = stacked_ensembleOneVsOne_clf.predict(data_test_X)


# In[22]:


accuracy = metrics.accuracy_score(data_test_Y, y_pred) # , normalize=True, sample_weight=None
accuracy_data_test["StackedEnsembleOneVsOne_Log_n_Log"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(data_test_Y, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(data_test_Y), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[23]:


stacked_ensembleOneVsOne_clf= StackedEnsembleOneVsOneClassifier(stack_layer_classifier_type = "logreg",base_classifier_type = "tree")


# In[24]:


stacked_ensembleOneVsOne_clf.fit(X_train, y_train)


# In[25]:


y_pred = stacked_ensembleOneVsOne_clf.predict(X_valid)


# In[26]:


accuracy = metrics.accuracy_score(y_valid, y_pred) # , normalize=True, sample_weight=None
accuracy_valid["StackedEnsembleOneVsOne_tree_n_Log"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_valid, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_valid), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[27]:


y_pred = stacked_ensembleOneVsOne_clf.predict(X_test)


# In[28]:


accuracy = metrics.accuracy_score(y_test, y_pred) # , normalize=True, sample_weight=None
accuracy_train_test["StackedEnsembleOneVsOne_tree_n_Log"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(y_test, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(y_test), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[29]:


stacked_ensembleOneVsOne_clf.fit(X, y)


# In[30]:


y_pred = stacked_ensembleOneVsOne_clf.predict(data_test_X)


# In[31]:


accuracy = metrics.accuracy_score(data_test_Y, y_pred) # , normalize=True, sample_weight=None
accuracy_data_test["StackedEnsembleOneVsOne_tree_n_Log"] = accuracy
print("Accuracy on Test Data: " +  str(accuracy))
print(metrics.classification_report(data_test_Y, y_pred))
print("Confusion Matrix for Test Data:")
display(pd.crosstab(np.array(data_test_Y), y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# In[32]:


print(accuracy_valid)


# In[33]:


print(accuracy_train_test)


# In[34]:


print(accuracy_data_test)


# In[ ]:


objects = ('Stk', 'Hold', 'KFold','Tree','PruTree','Bagg','TunBagg','11log','11tree')
y_pos = np.arange(len(objects))
performance = [accuracy_valid['StackedEnsembleClassifier'],accuracy_valid['StackedEnsembleHoldOutClassifier'],accuracy_valid['StackedEnsembleKFoldClassifier'],accuracy_valid['DecisionTreeClassifier'],accuracy_valid['PrunedDecisionTreeClassifier'],accuracy_valid['BaggingClassifier'],accuracy_valid['TunedBaggingClassifier'],accuracy_valid['StackedEnsembleOneVsOne_Log_n_Log'],accuracy_valid['StackedEnsembleOneVsOne_tree_n_Log']]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Perfomance on Test set')
 
plt.show()


# In[ ]:


objects = ('Stk', 'Hold', 'KFold','Tree','PruTree','Bagg','TunBagg','11log','11tree')
y_pos = np.arange(len(objects))
performance = [accuracy_train_test['StackedEnsembleClassifier'],accuracy_train_test['StackedEnsembleHoldOutClassifier'],accuracy_train_test['StackedEnsembleKFoldClassifier'],accuracy_train_test['DecisionTreeClassifier'],accuracy_train_test['PrunedDecisionTreeClassifier'],accuracy_train_test['BaggingClassifier'],accuracy_train_test['TunedBaggingClassifier'],accuracy_train_test['StackedEnsembleOneVsOne_Log_n_Log'],accuracy_train_test['StackedEnsembleOneVsOne_tree_n_Log']]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Perfomance on Test set')
 
plt.show()


# In[ ]:


objects = ('Stk', 'Hold', 'KFold','Tree','PruTree','Bagg','TunBagg','11log','11tree')
y_pos = np.arange(len(objects))
performance = [accuracy_data_test['StackedEnsembleClassifier'],accuracy_data_test['StackedEnsembleHoldOutClassifier'],accuracy_data_test['StackedEnsembleKFoldClassifier'],accuracy_data_test['DecisionTreeClassifier'],accuracy_data_test['PrunedDecisionTreeClassifier'],accuracy_data_test['BaggingClassifier'],accuracy_data_test['TunedBaggingClassifier'],accuracy_data_test['StackedEnsembleOneVsOne_Log_n_Log'],accuracy_data_test['StackedEnsembleOneVsOne_tree_n_Log']]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Perfomance on Test set')
 
plt.show()

