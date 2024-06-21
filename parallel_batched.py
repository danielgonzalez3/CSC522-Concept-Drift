import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from river import metrics
from river import stream
from river import tree,ensemble,forest
from river.drift import ADWIN
from river.drift.binary import DDM
from PIL import Image
import os
import time

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

df = pd.read_csv("./data/6LoWPANHeader.csv")
# split the data into train and test
X = df.drop(['Label'],axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.1, test_size = 0.9, shuffle=False, random_state = 0)


# df = pd.read_csv("./data/IoT_2020_b_0.01_fs.csv")
# # split the data into train and test
# X = df.drop(['Label'],axis=1)
# y = df['Label']
# X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.1, test_size = 0.9, shuffle=False, random_state = 0)

# df = pd.read_csv("./data/cic_0.01km.csv")
# X = df.drop(['Labelb'],axis=1)
# y = df['Labelb']
# X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.1, test_size = 0.9, shuffle=False,random_state = 0)


# Define a figure function that shows the real-time accuracy changes
def acc_fig(t, m, name):
    plt.rcParams.update({'font.size': 15})
    plt.figure(1,figsize=(10,6)) 
    plt.clf() 
    plt.plot(t,m,'-b',label='Avg Accuracy: %.2f%%'%(m[-1]))

    plt.legend(loc='best')
    plt.title(name+' on IoTID20 dataset', fontsize=15)
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy (%)')

    plt.draw()


# Worker function that processes data with a given algorithm and considers the weight
def worker(lock, worker_id, conn, model):
    metricx = metrics.Accuracy()

    signal.signal(signal.SIGTERM, partial(cleanup,lock, conn))

    batch_size = 600
    batch_data = []
    batch_labels = []
    batch_predictions = []

    # e = 1 - metricx.get()
    e = 0


    while True:
        # receive parent message object
        p_msg = conn.recv()
        
        batch_data.append(p_msg.xi)
        batch_labels.append(p_msg.yi)
        

        if (len(batch_data) == batch_size):
            # print(f"Worker {worker_id} processing batch data")
            for xi, yi, pred in zip(batch_data, batch_labels, batch_predictions):
                model.learn_one(xi,yi)
                metricx.update(p_msg.yi, pred)
                e = 1 - metricx.get()
            batch_data = []
            batch_labels = []
            # error rate should only be evaluated after the batch is processed



        y_pred = model.predict_one(p_msg.xi)
        y_prob = model.predict_proba_one(p_msg.xi)

        batch_predictions.append(y_pred)
        
        # model.learn_one(p_msg.xi,p_msg.yi)
        # metricx.update(p_msg.yi, y_pred)
        # e = 1 - metricx.get()

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
    start = time.time()
    # models = [ensemble.AdaptiveRandomForestClassifier(n_models=3),ensemble.SRPClassifier(n_models=3),ensemble.AdaptiveRandomForestClassifier(n_models=3,drift_detector=DDM(),warning_detector=DDM()),ensemble.SRPClassifier(n_models=3,drift_detector=DDM(),warning_detector=DDM())]
    
    #   PWPAE Models 
    models = [
        forest.adaptive_random_forest.ARFClassifier(n_models=3),
        ensemble.SRPClassifier(n_models=3),
        forest.adaptive_random_forest.ARFClassifier(n_models=3,drift_detector=DDM(),warning_detector=DDM()),
        ensemble.SRPClassifier(n_models=3,drift_detector=DDM(),warning_detector=DDM())
    ]

    # Proposed Models
    # models = [
    #     forest.adaptive_random_forest.ARFClassifier(n_models=3),
    #     ensemble.AdaBoostClassifier(model=tree.HoeffdingTreeClassifier(), n_models=3),
    #     forest.adaptive_random_forest.ARFClassifier(n_models=3,drift_detector=DDM(),warning_detector=DDM()),
    #     ensemble.BOLEClassifier(model=tree.HoeffdingTreeClassifier(), n_models=3)
    # ]

    # learn the models
    for xi, yi in stream.iter_pandas(X_train, y_train):
        for model in models:
            model.learn_one(xi,yi)
   
    j = 0
    processes = []
    parent_connections = []
    child_connections = []
    lock = multiprocessing.Lock()
    metric = metrics.Accuracy()

    i = 0
    t = []
    m = []
    yt = []
    yp = []
    
    # generate processes for each model
    for model in models:
        j = j + 1
        parent_conn, child_conn = multiprocessing.Pipe()
        parent_connections.append(parent_conn)
        child_connections.append(child_conn)

        process = multiprocessing.Process(target=worker, args=(lock,j, child_conn, model))
        processes.append(process)
        process.start()

    last_w = []

    # send data to each process and receive the results
    for xi, yi in stream.iter_pandas(X_test, y_test):
        # print(f"Type of yi: {type(yi)}, Value of yi: {yi}")

        results = []

        msg = ParentMessage(xi, yi)
        for _, conn in enumerate(parent_connections):
           conn.send(msg)
        
        for conn in parent_connections:
            # receive child message object
            results.append(conn.recv())
        
        sorted_results = sorted(results, key=lambda x: x.worker_id)

        ep = 0.001
        ea = 0
        for result in sorted_results:
            ea += 1/(result.e+ep)
        
        w = []
        for result in sorted_results:
            w.append((1/(result.e+ep))/ea)

        # if length of w and last_w are the same
        # iterate across each element and check if they are the same

        # print(len(w))
        # print(len(last_w))

        # if len(w) == len(last_w):
        #     for x,y in zip(w,last_w):
        #         if x != y:
        #             for z in w:
        #                 print(z)
        #             break


        last_w = w.copy()


        y_prob_0 = 0
        y_prob_1 = 0
        for k, result in enumerate(sorted_results):
            y_prob_0 += w[k]*result.ypro0
            y_prob_1 += w[k]*result.ypro1
        if y_prob_0 > y_prob_1:
            y_pred = 0
        else:
            y_pred = 1

        metric.update(yi, y_pred)
        
        t.append(i)
        m.append(metric.get()*100)
        yt.append(yi)
        yp.append(y_pred)
        i = i+1
        
    print("parallel batched:")         
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
    end = time.time()
    print("Time: "+str(end - start))
    # plt.show()


    # print("Processing complete.")
