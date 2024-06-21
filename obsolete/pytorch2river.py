import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from river import metrics
from river import stream
# from river import tree,neighbors,naive_bayes,ensemble,linear_model
# from river.drift import DDM, ADWIN
# import lightgbm as lgb
import time
# import torch
import torch.nn as nn
# import river
from river import compat
from river import optim
from river import preprocessing
from river import compat
# from river import datasets
# from river import evaluate
from river import metrics
from river import preprocessing
from torch import nn
from torch import optim
from torch import manual_seed
# print(river.__version__)

# Define a simple PyTorch neural network model for binary classification
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.dense1 = nn.Linear(input_size, 10)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(10, 2)  # Assuming binary classification

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        return x
df = pd.read_csv("./data/IoT_2020_b_0.01_fs.csv")

# Train-test split
# 10% training set, and 90% test set
X = df.drop(['Label'],axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.1, test_size = 0.9, shuffle=False,random_state = 0)

# Define a generic adaptive learning function
# The argument "model" means an online adaptive learning algorithm
def adaptive_learning(model, X_train, y_train, X_test, y_test):
    metric = metrics.Accuracy() # Use accuracy as the metric
    i = 0 # count the number of evaluated data points
    t = [] # record the number of evaluated data points
    m = [] # record the real-time accuracy
    yt = [] # record all the true labels of the test set
    yp = [] # record all the predicted labels of the test set

    # Learn the training set
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        model.learn_one(xi1,yi1)

    # Predict the test set
    for xi, yi in stream.iter_pandas(X_test, y_test):
        y_pred= model.predict_one(xi)  # Predict the test sample
        model.learn_one(xi,yi) # Learn the test sample
        metric = metric.update(yi, y_pred) # Update the real-time accuracy
        t.append(i)
        m.append(metric.get()*100)
        yt.append(yi)
        yp.append(y_pred)
        i = i+1
    print("Accuracy: "+str(round(accuracy_score(yt,yp),4)*100)+"%")
    print("Precision: "+str(round(precision_score(yt,yp),4)*100)+"%")
    print("Recall: "+str(round(recall_score(yt,yp),4)*100)+"%")
    print("F1-score: "+str(round(f1_score(yt,yp),4)*100)+"%")
    return t, m

# Define a figure function that shows the real-time accuracy changes
def acc_fig(t, m, name):
    plt.rcParams.update({'font.size': 15})
    plt.figure(1,figsize=(10,6))
    sns.set_style("darkgrid")
    plt.clf()
    plt.plot(t,m,'-b',label='Avg Accuracy: %.2f%%'%(m[-1]))

    plt.legend(loc='best')
    plt.title(name+' on IoTID20 dataset', fontsize=15)
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy (%)')

    plt.draw()

_ = manual_seed(00)
def build_pytorch_model(n_features):
    net = nn.Sequential(
        nn.Linear(n_features, 5),
        nn.Linear(5,5),
        nn.Linear(5,5),
        nn.Linear(5,5),
        nn.Linear(5,1),
        nn.Sigmoid()
    )
    return net

if __name__ == "__main__":
    # Assuming X_train has your feature data to determine input size
    input_size = X_train.shape[1]

    pytorch_model = SimpleNN(input_size=input_size)
   # Wrap the PyTorch model with River's PyTorch2River
    model = compat.PyTorch2RiverClassifier(
        build_fn=build_pytorch_model,
        loss_fn=nn.BCELoss,
        optimizer=optim.Adam,  # Pass optimizer class, parameters are specified later
        learning_rate=1e-3,  # Pass learning rate
    )

    # Optional: Wrap the model with preprocessing steps, such as normalization
    model = preprocessing.StandardScaler() | model
    start = time.time()
    name = "Neural Network model"
    t, m5 = adaptive_learning(model, X_train, y_train, X_test, y_test) # Learn the model on the dataset
    # acc_fig(t, m5, name) # Draw the figure of how the real-time accuracy changes with the number of samples
    end = time.time()
    print("Time: "+str(end - start))