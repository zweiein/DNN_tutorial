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


def ChooseSamplseByLabel(corpus_set, label_list, len_of_set, len_of_sub_sample):
#產生
#傳入corpus_set (例如：train_set, dev_set, test_set)，
    max = len_of_sub_sample
    set_x_label = []
    set_y_label = []
    set_x = corpus_set[0]
    set_y = corpus_set[1]

    for l in range(0, len(label_list)):
        for index in range(0, len_of_set): # for all labels
            if len_of_sub_sample > 0 and set_y[index] == int(label_list[l]):
                set_x_label.append(set_x[index])
                set_y_label.append(label_list[l])
                len_of_sub_sample = len_of_sub_sample - 1
            #end if
        #for y
        len_of_sub_sample = max
    #end for all label

    # combine x and y to a list,
    ## x stored in list[0]
    ## y stored in list[1]
    result = []
    result.append(set_x_label)
    result.append(set_y_label)

    return result
#ChooseSamplseByLabel()




if __name__ == '__main__':
    # Load the dataset
    f = gzip.open('../mnist_datasets/mnist.pkl.gz', 'rb')
    train_set, dev_set, test_set = cPickle.load(f)
    f.close()

    """
    以training set為例, 有train_set[0], train_set[1]兩個組成

    ：train_set[0] ＝ 5000x784的特徵矩陣；也就是說：它是有5000筆資料的特徵向量(每筆資料是784維的特徵向量)
    ：train_set[1] ＝ 1x5000的向量；也就是說：它是每筆資料對應的label

    """

    # loads the dataset into shared variables
    #train_set_x, train_set_y = SharedDataset(train_set)
    #dev_set_x, dev_set_y = SharedDataset(dev_set)
    #test_set_x, test_set_y = SharedDataset(test_set)

    batch_size = 500    # size of the minibatch

    target_label_list = []
    for i in range(0, 5): target_label_list.append(i)

    """
    訓練集有50000筆資料(所有label)，每個label抓100筆出來，我們只用0-4；共五個label
    """
    print 'Create training set'
    len_of_set = 50000 # training set has 50000 samples
    new_train_set = ChooseSamplseByLabel(train_set, target_label_list, len_of_set, 100)
    train_set_x, train_set_y = new_train_set[0], new_train_set[1]

    print len(train_set_x)
    print len(train_set_y)
    print train_set_y[-1]
    print '...'*10

    """
    dev集與測試集有10000筆資料(所有label)，每個label抓20筆出來，我們只用0-4；共五個label
    """
    print 'Create dev set and test set'
    len_of_set = 10000 # training set has 50000 samples
    new_dev_set = ChooseSamplseByLabel(dev_set, target_label_list, len_of_set, 20)
    dev_set_x, dev_set_y = new_dev_set[0] , new_dev_set[1]

    new_test_set = ChooseSamplseByLabel(test_set, target_label_list, len_of_set, 20)
    test_set_x, test_set_y = new_test_set[0], new_test_set[1]

    print len(dev_set_x)
    print len(dev_set_y)
    print dev_set_y[-1]
    #print train_set_y[0]
    print '...'*10

    print len(test_set_x)
    print len(test_set_y)
    print test_set_y[-1]
    #print train_set_y[0]
    print '...'*10


    """
    寫入cPickle檔
    """
    dataset = []
    dataset.append(new_train_set)
    dataset.append(new_dev_set)
    dataset.append(new_test_set)

    f = gzip.open('../mnist_datasets/small_mnist.pkl.gz','wb')
    cPickle.dump(dataset, f, protocol=2)
    f.close()

    print 'Dump OK! --> smaller_mnist.pkl.gz'
#if __name__ == '__main__':