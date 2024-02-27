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

class batch_data:
    def __init__(self, datasource, interval):
        self.ds = datasource
        self.interval = interval
        self.done = False
    def gen(self):
        try:
            for i in range(self.interval):
                yield next(self.ds)
        except StopIteration:
            self.done = True

class AdaptiveModel:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.metric = metrics.Accuracy()
        self.history = {'t': [], 'accuracy': []}

    def learn(self, X, y):
        start_time = time.time()
        for xi, yi in stream.iter_pandas(X, y):
            self.model.learn_one(xi, yi)
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nLearning for {self.name} took {duration:.2f} seconds\n") 

    def evaluate(self, X, y):
        start_time = time.time()
        true_labels = []
        predicted_labels = []
        for xi, yi in stream.iter_pandas(X, y):
            y_pred = self.model.predict_one(xi)
            self.model.learn_one(xi, yi)  # Online learning update
            self.metric.update(yi, y_pred)
            self.history['t'].append(len(self.history['t']))
            self.history['accuracy'].append(self.metric.get() * 100)
            true_labels.append(yi)
            predicted_labels.append(y_pred)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"{self.name}")
        print(f"Evaluation for {self.name} took {duration:.2f} seconds")
        print("Accuracy:", round(accuracy_score(true_labels, predicted_labels), 4) * 100, "%")
        print("Precision:", round(precision_score(true_labels, predicted_labels, average='macro'), 4) * 100, "%")
        print("Recall:", round(recall_score(true_labels, predicted_labels, average='macro'), 4) * 100, "%")
        print("F1-score:", round(f1_score(true_labels, predicted_labels, average='macro'), 4) * 100, "%")

    def evaluate_batch(self, X, y, batch_size):
        start_time = time.time()
        true_labels = []
        predicted_labels = []
        i = 0


        data = batch_data(stream.iter_pandas(X, y), batch_size)
        while not data.done:
            i1 = i
            batch_pairs = []
            for xi, yi in data.gen():
                y_pred = self.model.predict_one(xi)
                self.metric.update(yi, y_pred)
                self.history['t'].append(i)
                self.history['accuracy'].append(self.metric.get() * 100)
                true_labels.append(yi)
                predicted_labels.append(y_pred)
                batch_pairs.append((xi, yi))
                i = i+1
            for j in range(i - i1):
                self.model.learn_one(*batch_pairs[j])
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"{self.name}")
        print(f"Evaluation for {self.name} took {duration:.2f} seconds")
        print("Accuracy:", round(accuracy_score(true_labels, predicted_labels), 4) * 100, "%")
        print("Precision:", round(precision_score(true_labels, predicted_labels, average='macro'), 4) * 100, "%")
        print("Recall:", round(recall_score(true_labels, predicted_labels, average='macro'), 4) * 100, "%")
        print("F1-score:", round(f1_score(true_labels, predicted_labels, average='macro'), 4) * 100, "%")

    def plot_accuracy(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['t'], self.history['accuracy'], label=f'Accuracy of {self.name}')
        plt.legend(loc='best')
        plt.title(f'Real-Time Accuracy of {self.name}')
        plt.xlabel('Number of Samples')
        plt.ylabel('Accuracy (%)')

        os.makedirs('result', exist_ok=True)

        plt.savefig(f'result/{self.name.replace(" ", "_")}_accuracy_plot.png', bbox_inches='tight')
        plt.close()

def plot_ensemble(t, m, name):
    plt.figure(1,figsize=(10,6))
    plt.plot(t,m,'-b',label='Avg Accuracy: %.2f%%'%(m[-1]))
    plt.legend(loc='best')
    plt.title(f'Real-Time Accuracy of {name}')
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy (%)')

    os.makedirs('result', exist_ok=True)

    plt.savefig(f'result/{name}_accuracy_plot.png', bbox_inches='tight')
    plt.close()

def PWPAE(X_train, y_train, X_test, y_test):
    metric = metrics.Accuracy()
    metric1 = metrics.Accuracy()
    metric2 = metrics.Accuracy()
    metric3 = metrics.Accuracy()
    metric4 = metrics.Accuracy()

    i=0
    t = []
    m = []
    yt = []
    yp = []

    hat1 = forest.adaptive_random_forest.ARFClassifier(n_models=3) # ARF-ADWIN
    hat2 = ensemble.SRPClassifier(n_models=3) # SRP-ADWIN
    hat3 = forest.adaptive_random_forest.ARFClassifier(n_models=3,drift_detector=DDM(),warning_detector=DDM()) # ARF-DDM
    hat4 = ensemble.SRPClassifier(n_models=3,drift_detector=DDM(),warning_detector=DDM()) # SRP-DDM

    # The four base learners learn the training set
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        hat1.learn_one(xi1,yi1)
        hat2.learn_one(xi1,yi1)
        hat3.learn_one(xi1,yi1)
        hat4.learn_one(xi1,yi1)

    # Predict the test set
    for xi, yi in stream.iter_pandas(X_test, y_test):
        # The four base learner predict the labels
        y_pred1= hat1.predict_one(xi)
        y_prob1= hat1.predict_proba_one(xi)
        hat1.learn_one(xi,yi)

        y_pred2= hat2.predict_one(xi)
        y_prob2= hat2.predict_proba_one(xi)
        hat2.learn_one(xi,yi)

        y_pred3= hat3.predict_one(xi)
        y_prob3= hat3.predict_proba_one(xi)
        hat3.learn_one(xi,yi)

        y_pred4= hat4.predict_one(xi)
        y_prob4= hat4.predict_proba_one(xi)
        hat4.learn_one(xi,yi)

        # Record their real-time accuracy
        metric1.update(yi, y_pred1)
        metric2.update(yi, y_pred2)
        metric3.update(yi, y_pred3)
        metric4.update(yi, y_pred4)

        # Calculate the real-time error rates of four base learners
        e1 = 1-metric1.get()
        e2 = 1-metric2.get()
        e3 = 1-metric3.get()
        e4 = 1-metric4.get()


        ep = 0.001 # The epsilon used to avoid dividing by 0
        # Calculate the weight of each base learner by the reciprocal of its real-time error rate
        ea = 1/(e1+ep)+1/(e2+ep)+1/(e3+ep)+1/(e4+ep)
        w1 = 1/(e1+ep)/ea
        w2 = 1/(e2+ep)/ea
        w3 = 1/(e3+ep)/ea
        w4 = 1/(e4+ep)/ea

        # Make ensemble predictions by the classification probabilities
        if  y_pred1 == 1:
            ypro10=1-y_prob1[1]
            ypro11=y_prob1[1]
        else:
            ypro10=y_prob1[0]
            ypro11=1-y_prob1[0]
        if  y_pred2 == 1:
            ypro20=1-y_prob2[1]
            ypro21=y_prob2[1]
        else:
            ypro20=y_prob2[0]
            ypro21=1-y_prob2[0]
        if  y_pred3 == 1:
            ypro30=1-y_prob3[1]
            ypro31=y_prob3[1]
        else:
            ypro30=y_prob3[0]
            ypro31=1-y_prob3[0]
        if  y_pred4 == 1:
            ypro40=1-y_prob4[1]
            ypro41=y_prob4[1]
        else:
            ypro40=y_prob4[0]
            ypro41=1-y_prob4[0]

        # Calculate the final probabilities of classes 0 & 1 to make predictions
        y_prob_0 = w1*ypro10+w2*ypro20+w3*ypro30+w4*ypro40
        y_prob_1 = w1*ypro11+w2*ypro21+w3*ypro31+w4*ypro41

        if (y_prob_0>y_prob_1):
            y_pred = 0
            y_prob = y_prob_0
        else:
            y_pred = 1
            y_prob = y_prob_1

        # Update the real-time accuracy of the ensemble model
        metric.update(yi, y_pred)

        t.append(i)
        m.append(metric.get()*100)
        yt.append(yi)
        yp.append(y_pred)

        i=i+1
    print("Accuracy: "+str(round(accuracy_score(yt,yp),4)*100)+"%")
    print("Precision: "+str(round(precision_score(yt,yp),4)*100)+"%")
    print("Recall: "+str(round(recall_score(yt,yp),4)*100)+"%")
    print("F1-score: "+str(round(f1_score(yt,yp),4)*100)+"%")
    return t, m

def merge_results(directory='result', output_filename='merged_result.png'):
    images = [img for img in os.listdir(directory) if img.endswith('.png')]
    num_images = len(images)
    
    # For simplicity, create a grid that is roughly square
    num_columns = int(num_images**0.5)
    num_rows = (num_images + num_columns - 1) // num_columns
    
    if num_images == 0:
        print("No PNG images found in the directory.")
        return
    
    sample_image = Image.open(os.path.join(directory, images[0]))
    img_width, img_height = sample_image.size
    sample_image.close()
    
    merged_image = Image.new('RGB', (img_width * num_columns, img_height * num_rows), 'white')
    
    for index, img_name in enumerate(images):
        img = Image.open(os.path.join(directory, img_name))
        x_offset = (index % num_columns) * img_width
        y_offset = (index // num_columns) * img_height
        merged_image.paste(img, (x_offset, y_offset))
        img.close()
    
    merged_image.save(os.path.join(directory, output_filename))
    print(f"\n Merged image saved as {os.path.join(directory, output_filename)}.")

def main():
    df = pd.read_csv("./data/IoT_2020_b_0.01_fs.csv")
    X = df.drop(['Label'], axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, test_size=0.9, shuffle=False, random_state=0)

    print(dir(tree))
    print(dir(ensemble))
    models = [
        AdaptiveModel(forest.adaptive_random_forest.ARFClassifier(n_models=3, drift_detector=ADWIN()), "IoT_2020-ARF-ADWIN"),
        AdaptiveModel(forest.adaptive_random_forest.ARFClassifier(n_models=3, drift_detector=DDM()), "IoT_2020-ARF-DDM"),
        AdaptiveModel(forest.aggregated_mondrian_forest.AMFClassifier(n_estimators=10), "IoT_2020-AMF"),
        AdaptiveModel(tree.ExtremelyFastDecisionTreeClassifier(), "IoT_2020-EFDT"),
        AdaptiveModel(tree.HoeffdingAdaptiveTreeClassifier(), "IoT_2020-HAT"),
        AdaptiveModel(tree.HoeffdingTreeClassifier(), "IoT_2020-HTC"),
        AdaptiveModel(tree.SGTClassifier(), "IoT_2020-SGT"),
        AdaptiveModel(ensemble.ADWINBaggingClassifier(model=tree.HoeffdingTreeClassifier(), n_models=3), "IoT_2020-ADWIN-BA-HT"),
        AdaptiveModel(ensemble.ADWINBoostingClassifier(model=tree.HoeffdingTreeClassifier(), n_models=3), "IoT_2020-ADWIN-BO-HT"),
        AdaptiveModel(ensemble.AdaBoostClassifier(model=tree.HoeffdingTreeClassifier(), n_models=3), "IoT_2020-ADA-HT"),
        AdaptiveModel(ensemble.BOLEClassifier(model=tree.HoeffdingTreeClassifier(), n_models=3), "IoT_2020-BOLE-HT"),
        AdaptiveModel(ensemble.BaggingClassifier(model=tree.HoeffdingTreeClassifier(), n_models=3), "IoT_2020-BAG-HT"),
        AdaptiveModel(ensemble.LeveragingBaggingClassifier(model=tree.HoeffdingTreeClassifier(), n_models=3), "IoT_2020-LEVBAG-HT"),
        AdaptiveModel(ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(), n_models=3), "IoT_2020-SRP-HT"),
        AdaptiveModel(ensemble.SRPClassifier(n_models=3, drift_detector=ADWIN()), "IoT_2020-SRP-ADWIN"),
        AdaptiveModel(ensemble.SRPClassifier(n_models=3, drift_detector=DDM()), "IoT_2020-SRP-DDM"),
    ]

    for model in models:
        model.learn(X_train, y_train)
        # model.evaluate(X_test, y_test)
        model.evaluate_batch(X_test, y_test, 600)
        model.plot_accuracy()

    t, m = PWPAE(X_train, y_train, X_test, y_test)
    plot_ensemble(t, m, "IoT_2020-PWPAE")

    # TODO: ensure merged_result.png has results stacked
    merge_results(directory='result', output_filename='IoT_2020-merged-result.png')

if __name__ == "__main__":
    main()
