# -*- coding: utf-8 -*-

import cPickle, gzip, numpy
import theano
import theano.tensor as T

def SharedDataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')
#end SharedDataset()



class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input

        if W is None :
            W_values = numpy.asarray(
                rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)),
                            high=numpy.sqrt(6. / (n_in + n_out)),
                            size=(n_in, n_out)),
                dtype=theano.config.floatX)

            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
        #end if W is None

        if b is None :
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        #end if b is None


        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        if activation is None:
            self.output = lin_output
        else :
            activation(lin_output)


        # parameters of the model
        self.params = [self.W, self.b]
    #end __init()__


#end HiddenLayer()



if __name__ == '__main__':
    # Load the dataset
    f = gzip.open('../mnist_datasets/small_mnist.pkl.gz', 'rb')
    train_set, dev_set, test_set = cPickle.load(f)
    f.close()

    """ 
    以training set為例, 有train_set[0], train_set[1]兩個組成

    ：train_set[0] ＝ 5000x784的特徵矩陣；也就是說：它是有5000筆資料的特徵向量(每筆資料是784維的特徵向量)
    ：train_set[1] ＝ 1x5000的向量；也就是說：它是每筆資料對應的label    
    
    """

    # loads the dataset into shared variables
    train_set_x, train_set_y = SharedDataset(train_set)
    dev_set_x, dev_set_y = SharedDataset(dev_set)
    test_set_x, test_set_y = SharedDataset(test_set)
    

    batch_size = 500    # size of the minibatch

    # accessing the third minibatch of the training set
    data  = train_set_x[2 * batch_size: 3 * batch_size]
    label = train_set_y[2 * batch_size: 3 * batch_size]


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size  #50000 / batch_size
    n_dev_batches = dev_set_x.get_value(borrow=True).shape[0] / batch_size      #10000 / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size    #10000 / batch_size

    index = T.lscalar() 
    ls_x = test_set_x[index * batch_size: (index + 1) * batch_size]
    ls_y = test_set_y[index * batch_size: (index + 1) * batch_size]

    print index
    print ls_x[0][0]
    #print ls_y
    print '.'*10
    print train_set[1][0]
    print '.'*10
    print '＝'*8
    print train_set_x.get_value(borrow=True).shape[0]
    print dev_set_x.get_value(borrow=True).shape[0]
    print test_set_x.get_value(borrow=True).shape[0]
    print '＝'*8
    print len(train_set)
    print train_set[1][0:10]
    print '-'*8
    print len(dev_set)
    #print dev_set
    print '-'*8
    print len(test_set)
    #print test_set
    print '＝'*8
    print n_train_batches, ' ', n_dev_batches, ' ', n_test_batches
    
#if __name__ == '__main__':