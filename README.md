# Few-shot meta-learning
This repository contains the implementations of many meta-learning algorithms to solve the few-shot learning problem in Pytorch, including:
- [Model-Agnostic Meta-Learning (MAML)](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)
- ~~[Probabilistic Model-Agnostic Meta-Learning (PLATIPUS)](https://papers.nips.cc/paper/8161-probabilistic-model-agnostic-meta-learning.pdf)~~
- [Prototypical Networks (protonet)](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf)
- [Bayesian Model-Agnostic Meta-Learning (BMAML)](https://papers.nips.cc/paper/7963-bayesian-model-agnostic-meta-learning.pdf)
- [Amortized Bayesian Meta-Learning](https://openreview.net/pdf?id=rkgpy3C5tX)
- [Uncertainty in Model-Agnostic Meta-Learning using Variational Inference (VAMPIRE)](http://openaccess.thecvf.com/content_WACV_2020/papers/Nguyen_Uncertainty_in_Model-Agnostic_Meta-Learning_using_Variational_Inference_WACV_2020_paper.pdf)

These have been tested to work with Pytorch 1.7.

## New updates
The old implementation requires to code a "functional" form of `torch.nn.Module` of the model of interest. This is, however, inconvenient when changing the network architecture.

Fortunately, Facebook Research has developed [__higher__][higher repo] - a library that can easily convert any "conventional" `torch.nn.Module` into its "functional" form. For example:
```
# define a network
resnet18 = torchvision.models.resnet18(pretrain=False)

# get its parameters
params = list(resnet18.paramters())

# convert the network to its functional form
f_resnet18 = higher.patch.make_functional(restnet18)

# forward
y1 = f_resnet18.forward(x=x1, params=params)

# update params
new_params = update_parameter(params)

# forward on different data with new paramter
y2 = f_resnet18.forward(x=x2, params=new_params)
```

Hence, we only need to load or specify the "conventional" model written in PyTorch without re-implementing its "functional" form. I have made a few common models in `CommonModels.py`, so that we can load them to use immediately.

Although [__higher__][higher repo] provides convenient APIs to track gradients, it does not allow us to use the "first-order" approximate, resulting in more memory and longer training time. I have created a work-around solution to enable the "first-order" approximation, and controlled this by setting `--first-order=True` when running the code.

In addition, I rewrite a more readable code to load data.

Current update is available for most popular algorithms, including MAML, ABML, VAMPIRE and ProtoNet. Please refer to `classification_meta_learning.py`. I will try to update for regression soon.

## Operation mechanism explanation
The implementation is mainly in `classification_meta_learning.py` with some auxilliary classes and functions in `_utils.py`. The operation principle of the implementation can be divided into 3 steps:

### Step 1: initialize hyper-net and base-net
Recall the nature of the meta-learning as:

&theta; &rarr; __w__ &rarr; y &larr; __x__,

where &theta; denotes the parameter of the hyper-net, __w__ is the base-model parameter, and (__x__, y) is the data.

In the implementation, I follow this generative process, where the hyper-net will generate the base-net. It can be summarized in the following _pseudo-code_:

```
# initialization
base_net = ResNet18()
hyper_net = hyper_net_cls(base_net=base_net)

# learning and inference
base_net = hyper_net.forward()
```
- MAML: the hyper-net is the initialization of the base-net. Hence, the generative process follows identity operator, and hence, `hyper_net_cls` is defined as the class `IdentityNet` in `_utils.py`.
- ABML and VAMPIRE: the base-net parameter is a sample drawn from a diagonal Gaussian distribution parameterized by the meta-parameter. Hence, the hyper-net is designed to simulate this sampling process. In this case, `hyper_net_cls` is the class `NormalVariationalNet` in `_utils.py`.
- Prototypical network is different from the above algorithms due to its metric-learning nature. In the implementation, the base-net is a result of identity operation from the hyper-net (similar to MAML).

Why is it such a complicated implementation? It is to allow us to share the common procedures of many meta-learning algorithms. If it is not cleared to you, please open an issue or send me an email. I am happy to discuss to improve the readability of the code further.

### Step 2: task adaptation (often known as inner-loop)
There are 2 sub-functions corresponding to MAML-like algorithms and protonet.

#### `adapt_to_episode_innerloop`
The idea is simple:
1. Generate base-net(s) from the hyper-net
2. Use the generated base-net(s) to calculate loss on _training_ (also known as _support_) data
3. Minimize the loss w.r.t. the parameter of the hyper-net
4. Return the learnt hyper-net for that particular task

#### `adapt_to_episode_protonet`
Calculate and return the prototypes

### Step 3: evaluate on validation subset
The learnt hyper-net, in the case of MAML-like algorithms, or the prototypes, in the case of prototypical networks, are used to predict the labels of the data in the validation subset. 
- In training, the predicted labels are used to calculate the loss, and the parameter of the hyper-net is updated to minimize that loss.
- In testing, the predicted labels are used to compute the prediction accuracy.

## Data source
### Regression
The data source for regression is generated from `DataGeneratorT.py`

### Classification
Initially, Omniglot and mini-ImageNet are the two datasets considered. They are organized following the `torchvision.datasets.ImageFolder`.
```
dataset
│__alphabet1_character1 (or class1)
|__alphabet2_character2 (or class2)
...
|__alphabetn_characterm (or classz)
```
If this your setup, please modify the code to use `ImageFolderGenerator` as the `EpisodeGeneratorClass`.

You can also keep the original structure of Omniglot (train -> alphabets -> characters), and directly run the code without modification. I had this setup in mind when coding the implementation.

For the extracted feature, which I call `miniImageNet_640`, the train-test split is in pickle format. Each pickle file consists of a tuple `all_class, all_data`, where:
- `all_class` is a dictionary where keys are the names of classes, and the values are their corresponding names of images belong to those classes,
- `all_data` is also a dictionary where keys are the names of all images, and values are the vector values of the corresponding images.

You can also download [the resized Omniglot](https://www.dropbox.com/s/w1do3wi0wzzo4jw/omniglot.zip?dl=0) and [the miniImageNet with extracted features](https://www.dropbox.com/s/z48ioy2s2bjbu93/miniImageNet_640.zip?dl=0) from my shared Dropbox. Please note that the extracted features of miniImageNet dataset is done by the authors of LEO nets from DeepMind. The one from my Dropbox is a modified version with scaling and putting into a proper data structure for the ease of use.

## Run
Before running, please go to each script and modify the path to save the files by looking for `dst_folder_root` and `dst_folder`.

For the legacy implementation such as BMAML, please go to the `utils.py` and modify the default dataset folders (or you can do it by passing the corresponding argument in the `load_dataset` function). The epoch here is locally defined through `expected_total_tasks_per_epoch` tasks (e.g. 10k tasks = 1 epoch), and therefore, different from the definition of epoch in conventional machine learning.

To run, copy and paste the command at the beginning of each algorithm script and change the configurable parameters (if needed).

[Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) is also integrated into the implementation. Hence, you can open it and monitor the training on your favourite browser.

## Final note
If you feel this repository useful, please give a star. It might not mean much, but that lets me know that what I am doing is helpful, and encourages me to do more.

In addition, please consider to give a star to the [__higher__][higher repo] repository developed by Facebook. Without it, we still suffer from the arduous re-implementation of model "functional" form.

---
(legacy)

## Test
The command for testing is slightly different from running. This can be done by provide the file (or epoch) we want to test through `resume_epoch=xx`, where `xx` is the id of the file. It is followed by the parameter `--test`, and:
- `--no_uncertainty`: quanlitative result in regression, or output the accuracy of each task in classification,
- `--uncertainty`: outputs a csv-file with accuracy and prediction class probability of each image. This will be then used to calculate the calibration error.

[higher repo]: https://github.com/facebookresearch/higher
