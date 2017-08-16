from data_helper import DataHelper
import pandas as pd

helper = DataHelper()

train_y = helper.get_label_matrix()

haze_indices = train_y[train_y.haze==1].index.tolist()


num_epochs = 2
batch_size = 64

for epoch in range(num_epochs):

    print('EPOCH: ', epoch)

    haze_batch = helper.batch_generator(haze_indices, batch_size, image_size=(256, 256))
    batch_idx =0
    for images,aug,labels in haze_batch:

        print(batch_idx, images.shape, aug.shape)
        batch_idx += 1



D = helper.get_label_matrix()

gen = helper.train_batch_generator(32, image_size=(32,32))

for batch in gen:
    print(len(batch))

