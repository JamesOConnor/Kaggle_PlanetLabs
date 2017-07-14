from batch_generator import *


train_gen = train_batch_generator(10, (256, 256))
test_gen = test_batch_generator(10, (256, 256))

for (x,y) in train_gen:

    print((x))

