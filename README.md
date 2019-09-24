# cnnCFD
Convolutional Neural Network for non-uniform steady-state 2-D CFD solutions

This is the script for the initial-attempt-quick-and-dirty implementation of using a CNN network to learn to make predictions of non-uniform steady state.

The dataset is contains CFD results of 961 simulations of a flow around a randomly shaped obstacle based on five original shapes (circular, square, forward/backward triangle, and diamond). The dataset is composed of files (Xs and Ys). Xs contains the input features (SDF, X, Y information) and Ys contains the output information (Velocity in X, Velocity in Y, and pressure).

The Xs file contains 4 channels and but I'm ignoring the channel in dimension 1 because it didn't help much. I suggest you to do the same, at least for now. The first channel from the rest is the SDF (signed distance function) as in the Autodesk paper. The other 2 are spatial information (x and Y).

The initial network first creates an spatial representation of the geometry with convolutional operations from the input Xs. This latent spatial representation is then used to make predictions for both velocity components and pressure in a set of three parallel deconvolution operations.

The network I have there right now is the one that gave me the following results:







It would be nice to do hyperparameter search before we can go further. There are many paremeters that can be changed. Once we find the best parameters, we will have our baseline Vanilla-CNN model. Then later we can see if we can improve the results even more by incorporating a neural ODE in the encoding part of the network.

I can push my initial implementation with the neural ODE later to the repository as well, but I think for now it would make sense to get a good baseline first with this hyperparameter search before we proceed.
