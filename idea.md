Train a large n-layered neural net.
Freeze the network.

Create a smaller k-layered neural net where the net is significantly smaller than the original net.
Take the same dataset
Input to each layer of the larger net becomes the input to the smaller net with 1 extra dimension which represents the layer number.
Expected output of the smaller net to the input is the output of the larger net at that layer number.

Train the smaller net.
