Area for storing trained models

For now, models of the format size of input, number of training images, rotated or not, number of epochs

For example, 64_33000_1_5.h5 == 33,000 images of 64x64 size, each of which were rotated to create 3 other subsets were used to train a model over 5 epochs each

Models can be loaded into keras using the load_model method in keras (https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)

Hope this makes some sense! Let me know if there's a better naming convention - obviously this tells nothing about architecture
