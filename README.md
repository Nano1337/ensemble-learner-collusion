# Ensemble Learner Collusion

Reproducing issues in the paper "[Joint Training of Deep Ensembles Fails due to Learner Collusion](https://arxiv.org/abs/2301.11323)" 

Briefly, there seems to be some sort of diversity term that controls the exact difference between the performance of a jointly learned ensemble vs individual learners with individual loss functions of which predictions are ensembled afterwards. Consistently, we see that individual learners perform better than joint ensembles and that different types of models that are trained through a joint optimization scheme achieve different levels of performance on various benchmark datasets (e.g. ImageNet, CIFAR, etc.). 

This preliminary reproduction of the CIFAR results presented in the paper above will serve as an initial investigation into this phenomenon and possible fixes. My hypothesis is that spectral norms for weight init and updates from "[A Spectral Condition for Feature Learning](https://arxiv.org/abs/2310.17813)" by Greg at xAI may alleviate some of the issues behind training here.

Recall, to find the spectral norm of matrix Q, we first find B = Q^T * Q, find the eigenvalues of B, and then take the large eigenvalue (spectral radius) and sqrt it. 

## Setting up

First, create a virtual environment and install dependencies. 
```bash 
python -m venv venv
```

And then activate it: 
```bash 
source venv/bin/activate
```

Then, install all dependencies with pip: 
```bash 
pip install -r requirements.txt
```

# Results so far: 
Note: results reported are for 1 run with uncontrolled seed. 

For validation after 80 epochs
Training 1 model: 64% acc, train loss 1.83
Training 3 models with fusion: acc 59%, train loss ???
Training 3 models with bagging: acc 62% train loss 1.88

next step: train until convergence (current lr is 1e-3, paper lr is 0.1, just increase it lol)

Note: network was change to width*0.5 instead of 0.7, will need to retrain baselines

# Same settings but with muP: 
For validation after 80 epochs: 
training 1 model: acc 69%, train loss 0.082
training 3 models with fusion: acc 61%, train loss 1.88

Other observations: seems like it converged much faster? Like within half the time. Doesn't seem to fix the ensembling problem - I think it mostly has to do with the way the probabilities are directly averaged, definitely has negative effects. Seems like train loss settles at about the same place but I think this is a good starting research problem: Investigate how to scale model ensembles correctly which feeds into scaling modalities

