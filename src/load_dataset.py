import cPickle, gzip, numpy

def LoadDataSet(path_and_file_name):
    # Load the dataset
    f = gzip.open(path_and_file_name, 'rb')
    return_set = cPickle.load(f)
    f.close()
    return return_set
#end LoadDataSet()