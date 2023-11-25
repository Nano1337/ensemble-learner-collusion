# Ensemble Learner Collusion

Reproducing issues in the (paper)[https://arxiv.org]. 

Briefly, there seems to be some sort of diversity term that controls the exact difference between the performance of a jointly learned ensemble vs individual learners with individual loss functions of which predictions are ensembled afterwards. Consistently, we see that individual learners perform better than joint ensembles and that different types of models that are trained through a joint optimization scheme achieve different levels of performance on various benchmark datasets (e.g. ImageNet, CIFAR, etc.). 

This preliminary reproduction of the CIFAR results presented in the paper above will serve as an initial investigation into this phenomenon and possible fixes. My hypothesis is that spectral norms for weight init and updates from Greg at xAI may alleviate some of the issues behind training here.

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



