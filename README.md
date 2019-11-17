# Few-shot meta-learning
This repository contains the implementations of many meta-learning algorithms to solve the few-shot learning problem in Pytorch, including:
- [Model-Agnostic Meta-Learning (MAML)](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)
- [Probabilistic Model-Agnostic Meta-Learning (PLATIPUS)](https://papers.nips.cc/paper/8161-probabilistic-model-agnostic-meta-learning.pdf)
- [Bayesian Model-Agnostic Meta-Learning (BMAML)](https://papers.nips.cc/paper/7963-bayesian-model-agnostic-meta-learning.pdf)
- [Amortized Bayesian Meta-Learning](https://openreview.net/pdf?id=rkgpy3C5tX)
- [Uncertainty in Model-Agnostic Meta-Learning using Variational Inference (VAMPIRE)](https://arxiv.org/abs/1907.11864)

These have been tested to work with Pytorch 1.2.

## Data source
### Regression
The data source for regression is generated from `DataGeneratorT.py`

### Classification
Omniglot and mini-ImageNet are the two datasets considered. Their organization follows the `torchvision.datasets.ImageFolder`.
```
dataset
│__alphabet1_character1 (or class1)
|__alphabet2_character2 (or class2)
...
|__alphabetn_characterm (or classz)
```

## Run
Before running, please go to each script and modify the path to save the files by looking for `dst_folder_root` and `dst_folder`. The epoch here is locally defined through `expected_total_tasks_per_epoch` tasks (e.g. 10k tasks = 1 epoch), and therefore, different from the definition of epoch in conventional machine learning.
To run, copy and paste the command at the beginning of each algorithm script and change the configurable parameters (if needed).

## Test
The command for testing is slightly different from running. This can be done by provide the file (or epoch) we want to test through `resume_epoch=xx`, where `xx` is the id of the file. It is followed by the parameter `--test`, and:
- `--no_uncertainty`: quanlitative result in regression, or average accuracy in classification
- `--uncertainty`: outputs a csv-file with accuracy and prediction class probability of each image. This will be then used to calculate the calibration error.