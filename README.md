# InfoGAN -- Caution: forked and modified by nhdaly

# Fork Notes:
I have made the following changes:

1. Added code to generate output images with the trained model after training is completed. This is used in mnist and celebA.
2. Added code to run the celebA experiment, which requires downloading the celebA database separately. 

Except **note** that right now the celebA experiment doesn't work correctly. :D The output definitely does not look right.

All the new files I've created for the celebA experiment, I created as `.ipynb` files, but they also autogenerate `.py` scripts in the same directory. If you want to play with the Notebook files (which you should totally do! It's the best!), you can click the below button to launch a Jupyter Notebook!:

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/nhdaly/infogan)


You'll need to download the "celebA" dataset, which you can do from here: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. I used the [`img_align_celeba.zip`](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing) set, and extracted it to `/celebA/img_align_celeba/`. I'm actually not sure yet if it's possible to download that into the Binder instance, though... :/

---------------------------

# Original README:

Code for reproducing key results in the paper [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) by Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel.

## Dependencies

This project currently requires the dev version of TensorFlow available on Github: https://github.com/tensorflow/tensorflow. As of the release, the latest commit is [79174a](https://github.com/tensorflow/tensorflow/commit/79174afa30046ecdc437b531812f2cb41a32695e).

In addition, please `pip install` the following packages:
- `prettytensor`
- `progressbar`
- `python-dateutil`

## Running in Docker

```bash
$ git clone git@github.com:openai/InfoGAN.git
$ docker run -v $(pwd)/InfoGAN:/InfoGAN -w /InfoGAN -it -p 8888:8888 gcr.io/tensorflow/tensorflow:r0.9rc0-devel
root@X:/InfoGAN# pip install -r requirements.txt
root@X:/InfoGAN# python launchers/run_mnist_exp.py
```

## Running Experiment

We provide the source code to run the MNIST example:

```bash
PYTHONPATH='.' python launchers/run_mnist_exp.py
```

You can launch TensorBoard to view the generated images:

```bash
tensorboard --logdir logs/mnist
```
