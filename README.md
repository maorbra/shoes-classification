# shoes-classification

**Data-Set** - The dataset is images organized in such a way that there are two categories: one for men's or women's shoes and the other for left or right shoes, After that I split the dataset to **train** and **test_m** for male and **test_w** for woman

**ML_DL_Assignment** - We get a dataset of men and women shoes and we need to clesify if the shoes is from the same pair or not while the data set is arreanged with laft and righr shoe.

ML_DL_Functions - 2 deep learning models, the first is **CNN** which is **Convolutional Neural Network** particularly well-suited for processing grid-like data, such as images, and the second is **CNN_Channel** which is in the context of CNNs, channels refer to the depth of the input data. Channels represent different aspects or features of the input data. 

**CNN Modes** - A Convolutional Neural Network (CNN) is composed of several layers, each serving a specific purpose in processing the input data:

**Input Layer:**

The input layer represents the input data, such as an image. The image is represented as a multidimensional array (or tensor) of pixel values. For example, a colored image of size 64x64 pixels will have dimensions (64, 64, 3)—the width, height, and number of color channels (Red, Green, Blue).

**Convolutional Layers:**

These layers apply convolutional filters (also called kernels) to the input data to create feature maps. A convolutional filter slides (or "convolves") over the input data, multiplying the filter's weights by the input values to produce a transformed output.
The convolution operation captures local patterns and features such as edges, textures, or more complex structures in images.
Pooling Layers:

**Pooling layers** 

(often called subsampling or downsampling layers) reduce the dimensionality of feature maps while retaining important information. The most common pooling operation is max pooling, which takes the maximum value in each patch of the feature map.
Pooling helps reduce the number of parameters and computations in the network and helps make the model invariant to small translations of the input image.

**Fully Connected (Dense) Layers:**

After several convolutional and pooling layers, the high-level reasoning is performed via fully connected layers. These layers have neurons connected to every neuron in the previous layer, similar to a traditional feedforward neural network.
The final fully connected layer usually has as many neurons as the number of classes in the classification problem and uses a softmax activation function to produce a probability distribution over the classes.

**Activation Functions:**

Non-linear activation functions like ReLU (Rectified Linear Unit) are commonly used after convolutional layers to introduce non-linearity, allowing the network to learn more complex patterns.

**Output Layer:**

Output layer provides the final prediction. For example, in a classification task, it will output a probability for each class.
