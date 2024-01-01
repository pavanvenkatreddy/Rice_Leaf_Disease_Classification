<h1> Rice Leaf Disease classification using Convolutional Neural Networks </h1>

* A CNN network with several convolutional layers, batch normalization, ReLU activation, and max-pooling.
* After the convolutional layers, there's a global average pooling layer and an inversion layer that reverses the color channels.
Following this, there are additional dense layers for further feature extraction.
* Finally, there are two output branches: one for classification (classification_output) and one for a binary task (negative_output).
* The model iis very rubust and occupies only 5.6 MB with a ~98% accuracy.
