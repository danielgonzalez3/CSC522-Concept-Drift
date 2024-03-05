from river import compat
from river import datasets
from river import evaluate
from river import metrics
from river import preprocessing
from torch import nn
from torch import optim
from torch import manual_seed

_ = manual_seed(0)

def build_torch_mlp_classifier(n_features):
    net = nn.Sequential(
        nn.Linear(n_features, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 1),
        nn.Sigmoid()
    )
    return net
model = compat.PyTorch2RiverClassifier(
    build_fn= build_torch_mlp_classifier,
    loss_fn=nn.BCELoss,
    optimizer_fn=optim.Adam,
    learning_rate=1e-3
)
dataset = datasets.Phishing()
metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset=dataset, model=model, metric=metric)
# Accuracy: 74.38%
