# DeepCFD
# Dataset and Code

A toy dataset and the code for this project can be downloaded using the following https://zenodo.org/record/3666056/files/DeepCFD.zip?download=1

The folder includes the files dataX and dataY, in which the first file provides the input information on the geometry of 981 channel flow samples, whereas the dataY file provides their ground-truth CFD solution for the velocity (Ux and Uy) and thepressure (p) fields using the simpleFOAM solver. Figure 1 describes the structure of each of these files in detail:

![CFDAI](./SuppMaterial.eps)

Both dataX and dataY have the same dimensions (Ns, Nc, Nx, Ny), in which the first axis is the number of samples (Ns), the second axis is the number of channels (Nc), and third and fourth axes are the number of elements in x and y (Nx and Ny). Regarding the input dataX, the first channel is the SDF calculated from the obstacle's surface, the second channel is the multi-label flow region channel, and the third channel is the SDF from the top/bottom surfaces. For the output dataY file, the first channel is the Ux horizontal velocity component, the second channel is the Uy vertical velocity component, and the third channel is the pressure field.

An example of how to train the DeepCFD model using the settings described in the paper is provided in the "DeepCFD.py" script. A few useful functions are provided in the "functions.py" file, such as a plotting function to visualize the outcome of the model. Moreover, templates with all networks investigated in this study can be found in the folder "Models", including both "AutoEncoder" and "UNet" architecture types with one or multiple decoders.
