# NVIDIA Jetson training on MNIST with TensorFlow

Trains MNIST on an NVIDIA Jetson using TensorFlow.

The NVIDIA Jetson is an embedded device with onboard GPU, GPIO pins, camera
slot, and more. While embedded GPU devices are traditionally used for edge
inference, not training, you may decide that you would like to train a model
on-board. Some reasons you might want to do this include:

* You want to experiment with GPU and other hardware configurations for model
training.
* You have no other available GPUs and want to run a long-lived training
job--something that free platforms like Google Colab do not support.
* You want to explore ways to perform online learning based on data at the
edge.

I have not been able to find a "Hello, world" example for on-board training, so
I implemented one myself and documented the results here. There are more
advanced examples of training available in the [jetson-inference](https://github.com/dusty-nv/jetson-inference/tree/master/python/training)
repository, but these examples are (1) written in PyTorch, and (2) not very
minimal. Specifically, it is not immediately clear to me from those projects
how one configures the GPU to maximum effect. In this project, I demonstrate
GPU training on a very simple problem: MNIST.

## Training

All commands are executed on-board.

NVIDIA recommends using one of their [preconfigured Docker containers](https://github.com/dusty-nv/jetson-containers)
for running machine learning code on-board. We will use one of the TensorFlow
containers available for our L4T release. You can check the L4T version with
the following.

```bash
cat /etc/nv_tegra_release
# R32 (release), REVISION: 6.1, GCID: 27863751, BOARD: t210ref, EABI: aarch64, DATE: Mon Jul 26 19:20:30 UTC 2021
```

We are using r32.6.1. For that release, there are two TensorFlow containers: a
TFv1.15 and TFv2.5. We will use the latter. As explained in the Jetson
containers repository linked above, you probably want to run the container with
the helper shell script to save you from having to type out the volume mounts,
X11 forwarding (if applicable), etc.

```bash
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers/
scripts/docker_run.sh -c nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3
```

We are now in our TensorFlow container, and all we need to do is run our
training script. You could copy the Python file over manually, or you could
clone the repository. Below we do the latter.

```bash
apt update
apt install git
git clone https://github.com/kostaleonard/nvidia-jetson-tensorflow-mnist.git
cd nvidia-jetson-tensorflow-mnist/
python3 train_mnist.py
```

## GPU notes

In [train_mnist.py](train_mnist.py), we set GPU parameters in
`configure_gpu()`. If you were to not call this function before training, you
would likely have out of memory errors. I do not know the reason for this
defect, but [moderator posts](https://forums.developer.nvidia.com/t/jetson-nano-running-out-of-memory-resourceexhaustederror-oom-when-allocating-tensor-with-shape-3-3-512-1024/154513/5)
on the NVIDIA developer forums state that you need to limit the memory
available to TensorFlow. For a 2GB Jetson, they recommend limiting to 1GB(!) of
memory. They also advise setting memory growth.

During training, you can monitor GPU usage with `tegrastats`. With the training
script's default settings, the GPU memory appears to be nearly maximally
utilized. The 8GB of swap space is not used much, but it may not be needed for
such a small dataset.

```bash
tegrastats
...
# RAM 1889/1980MB (lfb 16x256kB) SWAP 683/8192MB (cached 58MB) CPU [6%@102,5%@102,15%@102,12%@102] EMC_FREQ 0% GR3D_FREQ 0% PLL@27C CPU@28C PMIC@50C GPU@29C AO@36.5C thermal@28.5C
...
```

## Other notes

More Jetson examples can be found at the [jetson-inference repository](https://github.com/dusty-nv/jetson-inference).
