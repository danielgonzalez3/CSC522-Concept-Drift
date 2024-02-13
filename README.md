
## Overview

This project is designed to demonstrate an ensemble of various online adaptive learning algorithms for real-time data stream processing. It utilizes the `river` library extensively, a powerful tool for machine learning in Python that excels at handling streaming data. The project showcases how to implement adaptive models, such as Adaptive Random Forests and others, to predict outcomes based on streaming IoT data. The goal is to achieve high accuracy in predictions while adapting to potential changes in data patterns (concept drift).

### Features

-   Implementation of various adaptive learning models using the `river` library.
-   Real-time accuracy tracking of model predictions.
-   Ensemble methods for improved prediction accuracy.
-   Demonstration of handling streaming data for machine learning.

## Installation

To set up this project locally, follow these steps:

1.  **Create a Virtual Environment**
    
    `python3 -m venv venv` 
    
2.  **Activate the Virtual Environment**
    
    For Windows:
    
    `.\venv\Scripts\activate` 
    
    For macOS/Linux:
    
    `source venv/bin/activate` 
    
3.  **Install Required Packages and Execute Model**

Ensure pip is up to date:

`python -m ensurepip --default-pip` 

Install the required packages:

`pip3 install -r requirements.txt`

Execute Model:

`python3 PWPAE.py`