import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import time

# Import the online learning metrics and algorithms from the River library
from river import metrics
from river import stream
from river import tree,neighbors,naive_bayes,ensemble,linear_model
from river.drift import DDM, ADWIN

import multiprocessing

df = pd.read_csv("./data/IoT_2020_b_0.01_fs.csv")

# split the data into train and test
X = df.drop(['Label'],axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.1, test_size = 0.9, shuffle=False, random_state = 0)

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



# Define four unique algorithms for demonstration
# Each takes a value and a weight
def algorithm_1(value,model,xi):
    y_pred1= model.predict_one(xi) 
    return (value + 1)

def algorithm_2(value,model,xi):
    return (value * 2) 

def algorithm_3(value,model,xi):
    return (value - 1)

def algorithm_4(value,model,xi):
    return (value ** 2) 

# Worker function that processes data with a given algorithm and considers the weight
def worker(worker_id, data, algorithm, conn, model):
    response = "okay"
    

    
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        model.learn_one(xi1,yi1)

    for xi, yi in stream.iter_pandas(X_test, y_test):
            y_pred1= model.predict_one(xi) 
            conn.send(y_pred1)
            print(f"Worker {worker_id} sent result: {y_pred1}")
            response = conn.recv()
            print("Response: ",response)

    # for value in data:
    #     # Apply the algorithm 
    #     result = algorithm(value,model)
    #     # Send result to parent
    #     conn.send(result)
    #     # Receive new response from parent
    #     response = conn.recv()
    conn.close()

if __name__ == "__main__":
    data_array = [1, 2, 3, 4, 5]  # The shared data array
    algorithms = [algorithm_1, algorithm_2, algorithm_3, algorithm_4]  # The list of algorithms
    models = [ensemble.AdaptiveRandomForestClassifier(n_models=3),ensemble.SRPClassifier(n_models=3),ensemble.AdaptiveRandomForestClassifier(n_models=3,drift_detector=DDM(),warning_detector=DDM()),ensemble.SRPClassifier(n_models=3,drift_detector=DDM(),warning_detector=DDM())]


    for xi, yi in stream.iter_pandas(X_train, y_train):
        processes = []
        parent_connections = []
        child_connections = []
        for i,algorithm in enumerate(algorithms):
            parent_conn, child_conn = multiprocessing.Pipe()
            parent_connections.append(parent_conn)
            child_connections.append(child_conn)
            # model.learn_one(xi,yi)
            process = multiprocessing.Process(target=worker, args=(i, data_array, algorithm, child_conn, models[i]))
            processes.append(process)
            process.start()

    # for i, algorithm in enumerate(algorithms):
    #     parent_conn, child_conn = multiprocessing.Pipe()
    #     parent_connections.append(parent_conn)
    #     child_connections.append(child_conn)
        
    #     process = multiprocessing.Process(target=worker, args=(i, data_array, algorithm, child_conn, models[i]))
    #     processes.append(process)
    #     process.start()

    for _ in stream.iter_pandas(X_test, y_test):
        results = []
        # Collect results from children
        for conn in parent_connections:
            results.append(conn.recv())

        for i, conn in enumerate(parent_connections):
            conn.send("okay")

    # Example loop for 5 iterations assuming 5 elements in the data array
    # for _ in range(len(data_array)):
    #     results = []
    #     # Collect results from children
    #     for conn in parent_connections:
    #         results.append(conn.recv())
        
    #     # Evaluate results and send back new weights (simplified example)
    #     # Here, you would insert logic to evaluate results and decide on weights
    #     # weights = [1, 1, 1, 1]  # Placeholder for new weights based on evaluation
        
    #     for i, conn in enumerate(parent_connections):
    #         # conn.send(weights[i])
    #         conn.send("okay")
    
    # Close parent connections and join processes
    for conn in parent_connections:
        conn.close()
    
    for process in processes:
        process.join()

    print("Processing complete.")
