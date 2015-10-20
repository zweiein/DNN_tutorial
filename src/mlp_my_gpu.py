# -*- coding: utf-8 -*-

"""Before use:   Ensure to add your cuda directory in the environment variable: $PATH
       example: export PATH=/usr/local/cuda/bin:${PATH}
"""

#The current division (/) operator has an ambiguous meaning for
#    numerical arguments: it returns the floor of the mathematical
#    result of division if the arguments are ints or longs, but it
#    returns a reasonable approximation of the division result if the
#    arguments are floats or complex.  This makes expressions expecting
#    float or complex results error-prone when integers are not
#    expected but possible as inputs.
from __future__ import division
#1.0 / 2.0 --> 0.5                            1 / 2 --> 0.5
#1.0 / 2   --> 0.5        after import        4 / 2 --> 2.0
#1 / 2.0   --> 0.5          division          1 // 2 --> 0
#1 / 2     --> 0              ===>            4 // 2 --> 2

import cPickle, gzip
import theano
import theano.tensor as T
import numpy
from pycuda import driver, compiler, gpuarray, tools

from Cheetah.Template import Template
from os import path
import pycuda.autoinit

#import prep_read_LLPLLR_features as pr
import math

# -- default parameters
DEFAULT_BLOCK_SIZE = 16
DEFAULT_WORK_SIZE = 1
DEFAULT_UNROLL = 0
DEFAULT_SPILL = False
DEFAULT_PREFETCH = False


CURR_PATH = path.dirname(path.abspath(__file__))
TEMPLATE_FILENAME = path.join(CURR_PATH, "gpu_matrixmul.cu")


# ------------------------------------------------------------------------------
def Matrixmul_opt(matrix_a, matrix_b,
                  block_size = DEFAULT_BLOCK_SIZE,
                  work_size = DEFAULT_WORK_SIZE,
                  unroll = DEFAULT_UNROLL,
                  spill = DEFAULT_SPILL,
                  prefetch = DEFAULT_PREFETCH):
    #matrix_a is the feature matrix, matrix_b is the weight matrix
    a_height, a_width = matrix_a.shape
    b_height, b_width = matrix_b.shape
    
    assert a_width == b_height #if not equal, raise an error

    # -- pad input matrices appropriately
    a_height_padded = int(numpy.ceil(a_height/block_size)) * block_size
    a_width_padded = int(numpy.ceil(a_width/block_size)) * (block_size*work_size)
    matrix_a_padded = numpy.zeros((a_height_padded, a_width_padded), numpy.float32)
    matrix_a_padded[:a_height,:a_width] = matrix_a

    b_height_padded = a_width_padded
    b_width_padded = int(numpy.ceil(b_width/(block_size*work_size))) * (block_size*work_size)
    matrix_b_padded = numpy.zeros((b_height_padded, b_width_padded), numpy.float32)
    matrix_b_padded[:b_height, :b_width] = matrix_b

    c_height_padded = a_height_padded
    c_width_padded = b_width_padded

    # -- upload padded input matrices to the GPU
    matrix_a_gpu = gpuarray.to_gpu(matrix_a_padded)
    matrix_b_gpu = gpuarray.to_gpu(matrix_b_padded)

    # -- create empty container matrix for the result (C = A * B)
    matrix_c_gpu = gpuarray.zeros((c_height_padded, c_width_padded), numpy.float32)

    # -- generate and compile the code
    # prepare the template parameters
    template_params = { 
        'BLOCK_SIZE': block_size, 
        'WORK_SIZE': work_size, 
        'UNROLL': unroll, 
        'SPILL': spill, 
        'PREFETCH': prefetch, 
        'A_WIDTH': a_width_padded,
        'A_HEIGHT': a_height_padded,
        'B_WIDTH': b_width_padded,
        }
    
    # run the template engine to get the code
    kernel_code = Template(
        file = TEMPLATE_FILENAME,
        searchList = [template_params],
        )
    
    # compile the code
    module = compiler.SourceModule(kernel_code)
    
    # get the kernel from the module
    matrixmul_func = module.get_function("matrixMul")

    # some info about the module
    print "number of registers used:", matrixmul_func.num_regs

    # block of threads
    # ATTENTION: block is (threadDim.x, threadDim.y, threadDim.z) 
    #            and not (threadDim.z, threadDim.y, threadDim.x)
    block =  block_size, block_size, 1
    
    # grid of blocks 
    # ATTENTION: it's (blockDim.x, blockDim.y) 
    #            and not (blockDim.y, blockDim.x)
    grid = int(c_width_padded / block_size /work_size), int(c_height_padded / block_size)

    # -- call the kernel on the GPU
    # Note that when we use time_kernel=True pycuda will automatically synchronize the kernel 
    # to make sure that the timing is correct. If you time the code yourself, you'll have to
    # synchronize the current Context.
    gpu_time = matrixmul_func(
        # -- output
        matrix_c_gpu,
        # -- inputs
        matrix_a_gpu, matrix_b_gpu,
        # -- grid of blocks
        grid = grid, 
        # -- block of threads
        block = block, 
        # -- time the kernel (approx.)
        time_kernel = True,
        )

    # get the GPU matrix back to CPU memory
    matrix_c_padded = matrix_c_gpu.get()
    matrix_c = matrix_c_padded[:a_height, :b_width]

    return matrix_c, gpu_time
#end Matrixmul_opt()


def MySigmoid(x):
    temp = -x
    return 1 / (1 + numpy.exp(temp))
#end MySigmoid()

def MySoftmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    out = e_x / e_x.sum()
    return out
#end MySoftmax()


def Relu(x):
    return T.switch(x < 0, 0, x)
#end relu()


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




if __name__ == "__main__":
    # Load the dataset
    f = gzip.open('../mnist_datasets/small_mnist.pkl.gz', 'rb')
    train_set, dev_set, test_set = cPickle.load(f)
    f.close()

    """
    以training set為例, 有train_set[0], train_set[1]兩個組成

    ：train_set[0] ＝ 5000x784的特徵矩陣；也就是說：它是有5000筆資料的特徵向量(每筆資料是784維的特徵向量)
    ：train_set[1] ＝ 1x5000的向量；也就是說：它是每筆資料對應的label

    """
    learn_rate = 0.1
    # loads the dataset into shared variables
    #train_set_x, train_set_y = SharedDataset(train_set)
    #dev_set_x, dev_set_y = SharedDataset(dev_set)
    #test_set_x, test_set_y = SharedDataset(test_set)

    train_set_x, train_set_y = train_set[0], train_set[1]
    dev_set_x, dev_set_y = dev_set[0], dev_set[1]
    test_set_x, test_set_y = test_set[0], test_set[1]

    minibatch_size = 20

    """#######################################
    #        對desire_output_label做前處理     #
    ##########################################
    """
    # train_set_y是個vector，每個維度代表每筆訓練資料的label
    # 先將vector擴展到matrix，每個row是one hot表示法
    matrix_desire_output = []
    for i in range(0, len(train_set_y)):
        temp_vec = numpy.zeros((5,), dtype=numpy.float32) #create an empty vector with 10 elements
        current_label = train_set_y[i]
        temp_vec[current_label] = 1
        matrix_desire_output.append(temp_vec)
    #end for
    matrix_desire_output = numpy.asarray(matrix_desire_output, dtype=numpy.float32)
    vec_desire_output = numpy.asarray(train_set_y, dtype=numpy.float32)



    """#######################################
    #Feed-forward矩陣乘法：input * weight_ih   #
    ##########################################
    """

    # setting matrix sizes
    feature_height = 500
    feature_width = 784 #784 dims
    w_ih_height = feature_width
    w_ih_width = 600  #600 neurons


    matrix_feature = numpy.asarray(train_set_x, dtype=numpy.float32) #OK

    # 回傳乘完的矩陣以及GPU時間
    # create random weight matrices
    numpy.random.seed(30)

    #matrix_weight = numpy.random.random_sample((w_ih_height, w_ih_width)).astype(numpy.float32)
    w_ih = numpy.random.random_sample((w_ih_height, w_ih_width)).astype(numpy.float32) #ok
    w_ih = w_ih / w_ih_height  # input 784 dims

    #bias高度跟I*W後結果的高度相同
    b_i_vector = numpy.random.random_sample(w_ih_width).astype(numpy.float32) #先產生一個row的vector
    b_ih = numpy.tile(b_i_vector, (feature_height, 1)) #然後複製同一個vector疊高成matrix
    b_ih = b_ih / feature_height #500 samples

    # compute reference on the cpu to verify GPU computation
    #matrix_reference = numpy.dot(matrix_feature, w_ih)

    # -- this is a good place to auto-tune the code (using the optimization kwargs)
    # (note that you may need more that one iteration to get accurate timing estimates)
    matrix_input_to_hidden_result, gpu_time = Matrixmul_opt(matrix_feature, w_ih)
    matrix_input_to_hidden_result = MySigmoid(matrix_input_to_hidden_result + b_ih)

    # check for correctness
    #diff = matrix_result - matrix_reference
    #error = np.absolute(diff).max()
    #assert error <= 1e-2
    #l2norm = np.linalg.norm(diff)
    #print "l2norm: ", l2norm

    """########################################
    #Feed-forward矩陣乘法：hidden * weight_ho   #
    ###########################################
    """

    # setting matrix sizes
    w_ih_height = feature_width
    w_ih_width = 600  #600 neurons

    w_ho_height = w_ih_width
    w_ho_width = 5

    # 回傳乘完的矩陣以及GPU時間
    # create random weight matrices
    numpy.random.seed(31)

    #matrix_weight = numpy.random.random_sample((w_ih_height, w_ih_width)).astype(numpy.float32)
    w_ho = numpy.random.random_sample((w_ho_height, w_ho_width)).astype(numpy.float32) #ok
    w_ho = w_ho / w_ho_height  # 600 neurons

    #bias的高度跟H*W後結果的高度相同
    b_h_vector = numpy.random.random_sample(w_ho_width).astype(numpy.float32) #先產生一個row的vector
    b_ho = numpy.tile(b_h_vector, (matrix_input_to_hidden_result.shape[0], 1)) #然後複製同一個vector疊高成matrix

    b_ho = b_ho / matrix_input_to_hidden_result.shape[0] #500 samples
    print 'b_ho ', b_ho.shape


    # compute reference on the cpu to verify GPU computation
    #matrix_reference = numpy.dot(matrix_input_to_hidden_result, w_ho)

    # -- this is a good place to auto-tune the code (using the optimization kwargs)
    # (note that you may need more that one iteration to get accurate timing estimates)
    matrix_hidden_to_output_result, gpu_time = Matrixmul_opt(matrix_input_to_hidden_result, w_ho)
    matrix_hidden_to_output_result = MySigmoid(matrix_hidden_to_output_result + b_ho)

    """########################################
    #Feed-forward矩陣乘法：最後一層做softmax      #
    ###########################################
    """

    net_output = []
    for i in range(0, len(matrix_hidden_to_output_result)):
        net_output.append(MySoftmax(matrix_hidden_to_output_result[i]))

    net_output = numpy.asarray(net_output, dtype=numpy.float32)


    """########################################
    #              Back-Propagation           #
    ###########################################
    """
    #network輸出與desire output做誤差計算，採用negative log-likelihood （最小化）
    # choose loss index
    loss_vec = []
    e_L = []
    for i in range(0, net_output.shape[0]):
        current_label = vec_desire_output[i]
        current_loss = -math.log(net_output[i][current_label])

        this_row_e_L = net_output[i] - matrix_desire_output[i]
        #grad_w = learn_rate * j_w / 500

        e_L.append(this_row_e_L)
        loss_vec.append(current_loss)
    #end for choose loss

    e_L = numpy.asarray(e_L)
    e_L_trans = numpy.transpose(e_L)



    print 'e_L_trans', e_L_trans.shape
    print 'i2h ', matrix_input_to_hidden_result.shape
    delta_w, gpu_time = Matrixmul_opt(e_L_trans, matrix_input_to_hidden_result)
    delta_w = delta_w / 600
    delta_b = e_L_trans / 500


    new_w =  w_ho - learn_rate * numpy.transpose(delta_w)


    """########################################
    #                印出目前結果               #
    ###########################################
    """

    # print some stats
    print "gpu time:", gpu_time
    gflop = matrix_input_to_hidden_result.size * (feature_width * 2.) / (1000**3.)
    gflops = gflop / gpu_time
    print "gflops:", gflops

    # print multiplication result
    print '='*80, 'input to hidden', '='*80
    print "////////input to hidden///////"
    print 'input-to-hidden H=I*Wih',matrix_input_to_hidden_result.shape
    print matrix_input_to_hidden_result


    # print some stats
    print '='*80, 'hidden to output', '='*80
    print "////////hidden to output///////"
    print "gpu time:", gpu_time
    gflop = matrix_hidden_to_output_result.size * (w_ho_width * 2.) / (1000**3.)
    gflops = gflop / gpu_time
    print "gflops:", gflops

    # print multiplication result
    print '-'*80
    print 'hidden-to-output Z=H*Woh', matrix_hidden_to_output_result.shape
    print matrix_hidden_to_output_result

    print '='*80, 'output softmax', '='*80
    print "//////// output softmax ///////"
    print 'net_output  ', net_output.shape
    print 'matrix_desire_output ', matrix_desire_output.shape
    print 'vec_desire_output', len(vec_desire_output)
    print net_output
    print '-'*80

    print '='*80, 'delta_w', '='*80
    print "//////// delta_w ///////"
    print 'delta_w ', delta_w.shape
    print 'e_L ', e_L.shape

    print delta_w
    print '-'*80

    print '='*80, 'delta_b', '='*80
    print "//////// delta_b ///////"
    print 'delta_b ', delta_b.shape

    print delta_b
    print '-'*80

    print '='*80, 'new_w', '='*80
    print "//////// new_w ///////"
    print 'new_w ', new_w.shape

    print new_w
    print '-'*80



    #print matrix_label
#end if __name__ == "__main__"