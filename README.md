# Reed Mauzy's Master's Thesis

This is the code used for the thesis I wrote for my M.S. in computer science.

## Description

A physical unclonable function, or PUF, is a chip designed to take advantage of the unavoidable variability in silicon fabrication so that if given the same binary challenge, it and only it will be able to reproduce its binary response. This has obvious applications in cybersecurity; unfortunately, no truly secure PUF has been proposed yet, as most designs are vulnerable to machine-learning attacks. The design studied here is the FF-arbiter PUF, in which two signals race to an arbiter that outputs a 0 or 1 depending on which signal reaches the arbiter first and the challenge bits determine which of two paths the signals travel down. There can also be feed-forward loops in the PUF, which help to obfuscate some of the bits in the challenge. This code is a proof of concept for a method to speed up dealing with feed-forward loops by running several neural networks in parallel on the same GPU.

## Requirements

This code can be cloned from GitHub and run without being installed, but it does have some dependencies: [MPI for Python](https://mpi4py.readthedocs.io/en/stable/), a C or FORTRAN implementation of MPI, such as [Open MPI](https://www.open-mpi.org/) for Linux or [MS-MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi) for Windows, and [Keras](https://keras.io/), with [TensorFlow](https://www.tensorflow.org/) being recommended as the Keras backend. It's also pointless to run it on a computer without a discrete GPU, such as any graphics card made by AMD or NVIDIA, since the thesis is about running neural networks in parallel on a GPU.

## Usage

In the Anaconda Powershell command prompt, activate an environment with the requisite packages installed, navigate to the directory with main.py in it, and run (as an example):

```bash
mpiexec -n 1 python main.py --max_proc 8
```

```mpiexec -n 1 python main.py``` should be exactly as above, but ```--max_proc 8``` is an optional argument. The full argument list is:

- ```--stages```: How many stages should be in the simulated PUF, which is also the length of the challenges. Defaults to 64.
- ```--challenges```: How many challenges should be simulated. Defaults to 625,000. ```(Number of challenges)/1,000``` should be divisible by ```--max_proc```; you'll get an error otherwise.
- ```--num_loops```: How many feed-forward loops should be in the PUF. Defaults to 1.
- ```--overlap```: Whether the feed-forward loops are allowed to overlap. Defaults to True.
- ```--runs```: How many times to run the experiment. Defaults to 1.
- ```--solver```: Which neural-network optimization algorithm to use. Defaults to the Adam optimizer.
- ```--minibatch```: The size of the batches the neural networks use. Defaults to 200.
- ```--num_layers```: How many layers the neural networks should have. Defaults to 1.
- ```--num_neurons```: The base number of neurons per layer (increases exponentially with the number of loops). Defaults to 2.
- ```--max_proc```: Number of child processes. Defaults to 5.

The results will be saved in the directory one level up from ```main.py``` in a plain text file with a name like ```xor#_(#)bit.txt```, with the first ```#``` replaced with the number of loops and the second ```#``` replaced with the number of stages.