# Abdomen-CT-multi-organ-segmentation-using-token-based-MLP-Mixer
**This is the official repository for the paper "[Abdomen CT multi‐organ segmentation using token‐based MLP‐Mixer]"
(https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.16135)".**

Notice in this github, the hyper-parameter might be a little bit different with my paper since my school does not grant me access to it. Sad. But the overall architecture and the final performance are very close to my paper.

# Required packages

The requires packages are in environment.yaml.

Create an environment using Anaconda:
```
conda env create -f \your directory\environment.yaml
```


# Usage

The usage is in the jupyter notebook TDM main.ipynb. Including how to build a diffusion process, how to build a network, and how to call the diffusion process to train, and sample new synthetic images. However, we give simple example below:
