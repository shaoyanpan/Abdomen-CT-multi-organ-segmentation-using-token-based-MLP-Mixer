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

# Required data saving format
In default, your data should be saved as:

![dataformat](https://github.com/shaoyanpan/Abdomen-CT-multi-organ-segmentation-using-token-based-MLP-Mixer/assets/89927506/3dd16045-1d08-4aed-bc31-50f2f2a557d0)

But feel free to change your directory as long as you change the data address in the "Set the data folder for data reading" section accordingly.

# Usage

The usage is in the jupyter notebook TOKEN_MLP_MAIN.ipynb. Including data preprocessing (for CT as example), how to call the network (MLP_mixer.py), and how to train and evaluate it. Moreover, we give a simple example below if anyone just want to call the network:



**Call the network**
```
# in_channels: color channel for the input, usually 1 for medical images
# out_channels: number of the segmentation classes, # of organs + 1(background)
# depth: depth of the network
# feature_size: Token size, controling how dense of the information extracted by the token. Larger -> more information, but easier to overfit.
# hidden_size: Layer size, similar to the convolutional channel in CNNs. Larger -> more information, but easier to overfit.
# But notice, hidden_size = 512 means 64,128,256,512 since depth = 4.
# mlp_dim: MLP layer size in the MLP_Mixer, controling how much you want to learn from the token. Larger -> more information, but easier to overfit.

from MLP_mixer import *
model =  MLP_MIXER(
    in_channels=1,
    out_channels=class_num,
    depth = 4,
    feature_size=512,
    hidden_size=512,
    mlp_dim=512,
).to(device)

```

# Visual examples for CT organ segmentation
![Figure  3](https://github.com/shaoyanpan/Abdomen-CT-multi-organ-segmentation-using-token-based-MLP-Mixer/assets/89927506/f1f7c188-6034-42f7-aadb-c620f41ee004)
