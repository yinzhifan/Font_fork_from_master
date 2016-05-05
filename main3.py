# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:26:06 2016

@author: shengx

Main function
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:54:40 2016

@author: shengx
"""

#%% Load Data

import numpy as np
import theano
from theano import tensor as T

from Font import *
from utility import *
from NeuralNets import *


basis_size = 36
font_dir = 'Fonts'
input_letter = ['B','A','S','Q']
output_letter = ['R']

n_train_batches = 10
n_epochs = 50000       #original:1500

output_num = 4
batch_size = 1

lamb1 = 0.01
lamb2 = 0.01

f0size = 1
f1size = 1

learning_rate = 1

#%% compare output
n = 0

Fonts = Font(basis_size, font_dir, input_letter, output_letter )
#%%
trainInput, trainOutput, testInput, testOutput = Fonts.getLetterSets(n_train_batches * batch_size, output_num * batch_size)
trainInput = 1 - trainInput
trainOutput = 1 - trainOutput
testInput = 1 - testInput
testOutput = 1 - testOutput


n_train = trainInput.shape[0]
n_test = testInput.shape[0]
input_size = len(input_letter) * basis_size * basis_size
output_size = len(output_letter) * basis_size * basis_size
image_size = basis_size * basis_size

trainInput = trainInput.reshape((n_train,image_size*len(input_letter)))
trainOutput = trainOutput.reshape((n_train,image_size*len(output_letter)))
testInput = testInput.reshape((n_test,image_size*len(input_letter)))
testOutput = testOutput.reshape((n_test,image_size*len(output_letter)))
trainInput, trainOutput = shared_dataset(trainInput, trainOutput)


#%% building neural networks


rng1 = np.random.RandomState(1234)
rng2 = np.random.RandomState(2345)
rng3 = np.random.RandomState(1567)
rng4 = np.random.RandomState(1124)
nkerns = [2, 2]


# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')
y = T.imatrix('y')

print('...building the model')

layer00_input = x[:,0:image_size].reshape((batch_size, 1, basis_size, basis_size))
layer01_input = x[:,image_size:2 * image_size].reshape((batch_size, 1, basis_size, basis_size))
layer02_input = x[:,2 * image_size:3 * image_size].reshape((batch_size, 1, basis_size, basis_size))
layer03_input = x[:,3 * image_size:4 * image_size].reshape((batch_size, 1, basis_size, basis_size))
# first convolutional layer
# image original size 50X50, filter size 5X5, filter number nkerns[0]
# after filtering, image size reduced to (36 - 3 + 1) = 34
# after max pooling, image size reduced to 34 / 2 = 17
layer00 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer00_input,
        image_shape=(batch_size, 1, basis_size, basis_size),   # input image shape
        filter_shape=(nkerns[0], 1, 3, 3),
        poolsize=(2, 2)
    )
layer01 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer01_input,
        image_shape=(batch_size, 1, basis_size, basis_size),   # input image shape
        filter_shape=(nkerns[0], 1, 3, 3),
        poolsize=(2, 2)
    )
layer02 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer02_input,
        image_shape=(batch_size, 1, basis_size, basis_size),   # input image shape
        filter_shape=(nkerns[0], 1, 3, 3),
        poolsize=(2, 2)
    )
layer03 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer03_input,
        image_shape=(batch_size, 1, basis_size, basis_size),   # input image shape
        filter_shape=(nkerns[0], 1, 3, 3),
        poolsize=(2, 2)
    )


# second convolutional layer
# input image size 23X23, filter size 4X4, filter number nkerns[1]
# after filtering, image size (17 - 4 + 1) = 14
# after max pooling, image size reduced to 14 / 2 = 7
layer10 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer00.output,
        image_shape=(batch_size, nkerns[0], 17, 17),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2)
    )
layer11 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer01.output,
        image_shape=(batch_size, nkerns[0], 17, 17),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2)
    )
layer12 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer02.output,
        image_shape=(batch_size, nkerns[0], 17, 17),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2)
    )
layer13 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer03.output,
        image_shape=(batch_size, nkerns[0], 17, 17),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2)
    )

# layer 2 input size = 2 * 4 * 7 * 7 =  392
layer2_input = T.concatenate([layer10.output.flatten(2), layer11.output.flatten(2), layer12.output.flatten(2), layer13.output.flatten(2)],
                              axis = 1)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer2_input,
        n_in=nkerns[1] * len(input_letter) * 7 * 7,
        n_out=50,
        activation=T.nnet.sigmoid
    )

layer5 = HiddenLayer(
    np.random.RandomState(np.random.randint(10000)),
    input=layer2.output,
    n_in=50,
    n_out=50,
    activation=T.nnet.sigmoid
)

layer6 = HiddenLayer(
    np.random.RandomState(np.random.randint(10000)),
    input=layer5.output,
    n_in=50,
    n_out=50,
    activation=T.nnet.sigmoid
)

layer7 = HiddenLayer(
    np.random.RandomState(np.random.randint(10000)),
    input=layer6.output,
    n_in=50,
    n_out=50,
    activation=T.nnet.sigmoid
)

layer3 = HiddenLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer7.output,
        n_in=50,
        n_out=50,
        activation=T.nnet.sigmoid
    )

layer4 = BinaryLogisticRegression(
        np.random.RandomState(np.random.randint(10000)),
        input=layer3.output,
        n_in=50,
        n_out=basis_size * basis_size,
    )

error = ((y - layer4.y_pred)**2).sum()

params = (layer4.params
        + layer3.params
        + layer2.params
        + layer5.params
        + layer6.params
        + layer7.params
        + layer10.params + layer11.params + layer12.params + layer13.params
        + layer00.params + layer01.params + layer02.params + layer03.params)

cost = layer4.negative_log_likelihood(y) + lamb1 * ((params[0])**2).sum() + lamb2 * ((params[1])**2).sum()

grads = T.grad(cost, params)

updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

#
#test_model = theano.function(
#        inputs = [index],
#        outputs = cost,
#        givens={
#            x: testInput[index * batch_size: (index + 1) * batch_size],
#            y: testOutput[index * batch_size: (index + 1) * batch_size]
#        }
#    )

train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates=updates,
        givens={
            x: trainInput[index * batch_size: (index + 1) * batch_size],
            y: trainOutput[index * batch_size: (index + 1) * batch_size]
        }
    )

#%% training the model

epoch = 0
costlist = []


while (epoch < n_epochs):
    epoch = epoch + 1
    total = 0
    for minibatch_index in range(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)
        total += minibatch_avg_cost
        iter = (epoch - 1) * n_train_batches + minibatch_index
        #print(('   epoch %i, minibatch %i/%i.') % (epoch, minibatch_index +1, n_train_batches))
    if (epoch % 100 == 0):
        print(('   epoch %i') % (epoch))
        total = total/n_train_batches
        print(total)
        costlist += [total]
#test_losses = [test_model(i) for i in range(n_test_batches)]
#test_score = np.mean(test_losses)

theano.function
#%% predict output
if not os.path.exists('test'):
    os.makedirs('test')

# print(error)
predict_model = theano.function(
    inputs=[x, y],
    outputs=[layer4.p_y_given_x, cost],
    on_unused_input='ignore',
    allow_input_downcast=True
)


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

test_cost_list = []
for testindex in range(output_num):
    predicted_values = predict_model(testInput[testindex * batch_size:(testindex + 1) * batch_size],
                                     testOutput[testindex * batch_size:(testindex + 1) * batch_size])
    output_img = predicted_values[0]
    test_cost = predicted_values[1]
    output_img = output_img.reshape(batch_size,basis_size,basis_size)
    output_img = np.asarray(output_img, dtype = 'float64') /256
    plt.figure(1)

    le = len(input_letter)
    siz = 10*le + 200

    for x in range(len(input_letter)):

        plt.subplot(siz + x + 1)
        plt.imshow(testInput[testindex,x * image_size:  (x+1)*image_size].reshape((basis_size,basis_size)),interpolation="nearest",cmap='Greys')
    plt.subplot(siz + le + 2)
    """
    plt.imshow(output_img[n,:,:],interpolation="nearest",cmap='Greys')
    """
    """
    plt.imshow(testInput[n,:,:],interpolation="nearest",cmap='Greys')
    """
    plt.imshow(output_img[0,:,:],interpolation="nearest",cmap='Greys')
    plt.subplot(siz + le + 1)
    plt.imshow(testOutput[testindex,:].reshape((basis_size,basis_size)),interpolation="nearest",cmap='Greys')
    x = 0
    st = 'test/c-4l-2cov-as-f-i-l-'+ str(learning_rate) + '-' + str(lamb1) + '-' + str(lamb2) + '-'+ str(n_train_batches) + '-' + str(n_epochs) +'-'+ str(batch_size)
    while os.path.exists(st + '-' + str(x) + '.png'):
        x += 1
    plt.savefig(st + '-' + str(x) +'.png')
    plt.show()
    print(test_cost)
    test_cost_list += [test_cost]

# c: Chinese test training
# 6l: 6 layers in total; or nl
# 2cov: 2 convolutional layers
# as: autosave, f: font.py changes, i: image display change (L -> 1)
# l: lambda
# name style: surffi -d -a -b -c -num
# -d learning rate
# -a n_train_batches
# -b n_epochs      #original: 1500
# -c batch_size    #original: 50
# -num in the end: the name for same parameter images.


fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.plot(costlist)
fig.savefig(st + '-' + str(x) + 'cost_graph.png')
plt.close(fig)

"""st2 = '4lasfi-'+ str(learning_rate) + '-' + str(n_train_batches) + '-' + str(n_epochs) +'-'+ str(batch_size)
text_index = 0
textfile = open('testparams/'+st2 + '-' + str(x) + '.txt', 'w')
text = str(params)
textfile.write(st)
textfile.close()"""
textfile = open('paramrecord', 'a')
textfile.write(st + '-' + str(x) + '\n'
               + "learning rate :" + str(learning_rate) + '\n'
               + 'test number: ' + str(output_num) +'\n'
               + 'lambda: ' + str(lamb1) + '/' + str(lamb2) + '\n'
               + 'filer size: ' + str(f0size) + '/' + str(f1size) + '\n'
               + 'training cost: ' + str(total) +'\n'
               + 'testing cost: ' + str(sum(test_cost_list) / len(test_cost_list))
               +'\n \n')
textfile.close()

