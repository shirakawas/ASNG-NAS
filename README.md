## Adaptive Stochastic Natural Gradient Method for One-Shot Neural Architecture Search
This repository contains the code for the following paper:

Youhei Akimoto, Shinichi Shirakawa, Nozomu Yoshinari, Kento Uchida, Shota Saito, and Kouhei Nishida: Adaptive Stochastic Natural Gradient Method for One-Shot Neural Architecture Search, 36th International Conference on Machine Learning (ICML) (2019) (Accepted)


If you use this code for your research, please cite our paper:

```
@inproceedings{AkimotoICML2019,
    author = {Youhei Akimoto and Shinichi Shirakawa and Nozomu Yoshinari and Kento Uchida and Shota Saito and Kouhei Nishida},
    title = {Adaptive Stochastic Natural Gradient Method for One-Shot Neural Architecture Search},
    booktitle = {36th International Conference on Machine Learning (ICML)},
    year = {2019}
}
```

###  Directory structure

```
.
├── README.md
├── classification (source codes of 3.2. CIFAR-10 classification task)
├── inpainting (source codes of 3.3. Celeb-A inpainting task)
└── toy (source codes of 3.1. Toy Problem)
```

### Requirements
We used the [PyTorch](https://pytorch.org/) version 0.4.1 for neural network implementation. We tested the codes on the following environment:

- Ubuntu 16.04 LTS
- GPU: NVIDIA GeForce GTX 1080Ti (for image classification and inpainting)
- CUDA version 9.2
- Python 3.6.5
- Python package version
    - numpy 1.14.3
    - scipy 1.1.0
    - torch 0.4.1
    - torchvision 0.2.1
    - scikit-image 0.14


### Usage
#### Classification (experiment of section 3.2)

```
$ cd classification
$ python main_classification.py
```

1. move to the directory of `classification`
1. run the program as `python main_classification.py`
    * If you want to run the code with a different setting, please directly modify the function call of `experiment` in `python main_classification.py`


#### Inpainting (experiment of section 3.3)
```
$ cd inpainting
$ python main_inpainting_cat.py -d celebA -p ~/data/ -c Center -g 0 -o ./result/
```

The main script `main_inpainting_cat.py` corresponds to the architecture encoding method using categorical variables, and `main_inpainting_int.py` corresponds to the architecture encoding method using the mix of categorical and integer variables. Please see our paper in details.

1. move to the directory of `inpainting`
1. prepare the celebA image dataset (train / test splitting)
    * we refer the training / test data splitting in <https://github.com/sg-nm/Evolutionary-Autoencoders>.
    * e.g., training dataset is located at `~/data/celebA/train/img/`, and test dataset is located at `~/data/celebA/test/img/`
1. run the program as `python main_inpainting_cat.py -d celebA -p ~/data/ -c Center -g 0 -o ./result/`
    * `-p` : path to the celebA dataset
    * `-c` : mask type ('Center' or 'RandomPixel' or 'RandomHalf')
    * `-g` : gpu id
    * `-o` : output directory


#### Toy Problem (experiment of section 3.1)
```
$ cd toy
$ python main_toy.py
```
1. move to the directory of `toy`
1. run the program as `python main_toy.py` (this program is run on CPU)
    * default: ASNG with $eps_x=0.05$, $\alpha=1.5$, and $\delta^{0}_{\theta}=1.0$
    * If you want to run the code with different setting, please directly modify the following function call in `python main_benchmark.py`
    `experiment(alg='ASNG', eta_x=0.05, eta_theta_factor=0., alpha=1.5, K=5, D=30, maxite=100000)`
