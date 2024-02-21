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
import signal
import sys
import os
from functools import partial
from time import sleep

# class for sending message to subprocess
class ParentMessage:
    def __init__(self, xi, yi):
        self.xi = xi
        self.yi = yi
    def __str__(self):
        return f"xi: {self.xi}, yi: {self.yi}"
    def __repr__(self):
        return f"xi: {self.xi}, yi: {self.yi}"

# class for sending message to main process
class ChildMessage:
    def __init__(self, worker_id, y_pred, y_prob, ypro0, ypro1, e):
        self.worker_id = worker_id
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.ypro0 = ypro0
        self.ypro1 = ypro1
        self.e = e
    def __str__(self):
        return f"worker_id: {self.worker_id}, y_pred: {self.y_pred}, y_prob: {self.y_prob}"
    def __repr__(self):
        return f"worker_id: {self.worker_id}, y_pred: {self.y_pred}, y_prob: {self.y_prob}"


df = pd.read_csv("./data/IoT_2020_b_0.01_fs.csv")

# split the data into train and test
X = df.drop(['Label'],axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.1, test_size = 0.9, shuffle=False, random_state = 0)

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


# Worker function that processes data with a given algorithm and considers the weight
def worker(lock, worker_id, conn, model):
    metric = metrics.Accuracy()

    signal.signal(signal.SIGTERM, partial(cleanup,lock, conn))

    while True:
        # receive parent message object
        p_msg = conn.recv()
        
        y_pred = model.predict_one(p_msg.xi)
        y_prob = model.predict_proba_one(p_msg.xi)
        
        model.learn_one(p_msg.xi,p_msg.yi)
        metric = metric.update(p_msg.yi, y_pred)
        e = 1 - metric.get()

        if y_pred == 1:
            ypro0 = 1-y_prob[1]
            ypro1 = y_prob[1]
        else:
            ypro0 = y_prob[0]
            ypro1 = 1-y_prob[0]


        #create child object 
        msg =   ChildMessage(worker_id, y_pred, y_prob, ypro0, ypro1, e)

        conn.send(msg)

# signal handler to close the connection
def cleanup(lock,conn,signum, frame):
    with lock:
        conn.close()
        sys.exit(0)



if __name__ == "__main__":
    
    models = [ensemble.AdaptiveRandomForestClassifier(n_models=3),ensemble.SRPClassifier(n_models=3),ensemble.AdaptiveRandomForestClassifier(n_models=3,drift_detector=DDM(),warning_detector=DDM()),ensemble.SRPClassifier(n_models=3,drift_detector=DDM(),warning_detector=DDM())]
    
    # learn the models
    i = 0
    for xi, yi in stream.iter_pandas(X_train, y_train):
        i = i + 1
        for model in models:
            model.learn_one(xi,yi)
   
    j = 0
    processes = []
    parent_connections = []
    child_connections = []
    lock = multiprocessing.Lock()

    # generate processes for each model
    for model in models:
            j = j + 1
            parent_conn, child_conn = multiprocessing.Pipe()
            parent_connections.append(parent_conn)
            child_connections.append(child_conn)
 
            process = multiprocessing.Process(target=worker, args=(lock,j, child_conn, model))
            processes.append(process)
            process.start()
    k = 0
    t = []
    m = []
    yt = []
    yp = []

    # send data to each process and receive the results
    for xi, yi in stream.iter_pandas(X_train, y_train):
        results = []
        metric = metrics.Accuracy()

        msg = ParentMessage(xi, yi)
        for _, conn in enumerate(parent_connections):
           conn.send(msg)
        
        for conn in parent_connections:
            # receive child message object
            results.append(conn.recv())
        
        sorted_results = sorted(results, key=lambda x: x.worker_id)

        ep = 0.001
        # linear version
        # ea = 1/(e1+ep) + 1/(e2+ep) + 1/(e3+ep) + 1/(e4+ep)
        # parallel version
        ea = 0
        for result in sorted_results:
            ea += 1/(result.e+ep)

        # linear version
        # w1 = (1/(e1+ep))/ea
        #parallel version
        w = []
        for result in sorted_results:
            w.append((1/(result.e+ep))/ea)  

        # linear version
        #y_prob_0 = w1*ypro10 + w2*ypro20 + w3*ypro30 + w4*ypro40
        #y_prob_1 = w1*ypro11 + w2*ypro21 + w3*ypro31 + w4*ypro41
        #parallel version
        y_prob_0 = 0
        y_prob_1 = 0
        for i, result in enumerate(sorted_results):
            y_prob_0 += w[i]*result.ypro0
            y_prob_1 += w[i]*result.ypro1

        if y_prob_0 > y_prob_1:
            y_pred = 0
        else:
            y_pred = 1

        metric = metric.update(yi, y_pred)
        
        t.append(k)
        m.append(metric.get()*100)
        yt.append(yi)
        yp.append(y_pred)
        k = k+1
        
        # print(results)

    print("Accuracy: "+str(round(accuracy_score(yt,yp),4)*100)+"%")
    print("Precision: "+str(round(precision_score(yt,yp),4)*100)+"%")
    print("Recall: "+str(round(recall_score(yt,yp),4)*100)+"%")
    print("F1-score: "+str(round(f1_score(yt,yp),4)*100)+"%")


    # send sigterm to all processes
    for proc in processes:
        os.kill(proc.pid, signal.SIGTERM)


    # Close parent connections and join processes
    for conn in parent_connections:
        conn.close()
    
    # Wait for all processes to finish
    for process in processes:
        process.join()

    name = "Parallel"
    acc_fig(t, m, name)
    plt.show()

    # print("Processing complete.")
