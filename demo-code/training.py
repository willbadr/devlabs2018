import os, urllib
import mxnet as mx
import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
import json
import numpy as np
# define a simple data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def download_resnet(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
            urllib.urlretrieve(url, filename)
        
def get_model(prefix, epoch):
    download_resnet(prefix+'-symbol.json')
    download_resnet(prefix+'-%04d.params' % (epoch,))

get_model('http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50', 0)
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-50', 0)

def download(url):
    dir = "dataset"
    train_filename = "102flowers-train.rec"
    validation_filename = "102flowers-valid.rec"
    train_path = os.getcwd() + '/' + dir + '/' + train_filename
    valid_path = os.getcwd() + '/' + dir + '/' + validation_filename
    if not os.path.isdir('./dataset'):
        os.makedirs('dataset')
        print(os.getcwd() + '/' + dir + '/' + train_filename)
        if not os.path.exists(train_path):
            urllib.urlretrieve(url, train_filename)
        if not os.path.exists(valid_path):
            urllib.urlretrieve(url, validation_filename)
    else:
        if not os.path.exists(train_path):
            urllib.urlretrieve(url, train_path)
        if not os.path.exists(valid_path):
            urllib.urlretrieve(url, valid_path)

def get_iterators(batch_size, data_shape=(3, 224, 224)):
    download('https://sagemaker-us-west-2-625616379791.s3.amazonaws.com/102flowers_dataset/train/102flowers-train.rec?AWSAccessKeyId=AKIAI6XQ45DLTC3PZ2XQ&Expires=1524404348&Signature=QF0zrJqi6LGezHfHPJqMiXOn6h4%3D')

    download('https://sagemaker-us-west-2-625616379791.s3.amazonaws.com/102flowers_dataset/validation/102flowers-valid.rec?AWSAccessKeyId=AKIAI6XQ45DLTC3PZ2XQ&Expires=1524404396&Signature=FGDEFu60GfOi2OFRmgquVInAK%2B8%3D')
    
    #sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-50', 0)
    train = mx.io.ImageRecordIter(
        path_imgrec         = './102flowers-train.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True)
    val = mx.io.ImageRecordIter(
        path_imgrec         = './102flowers-valid.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False)
    return (train, val)
    
    
def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)
    
def train(channel_input_dirs, hyperparameters, model_dir, hosts, num_gpus, **kwargs):    
    ctx = mx.gpu()
    num_classes = 102
    batch_per_gpu = 10
    num_gpus = 1
    
    # get the hyperparameters
    batch_size = hyperparameters.get('batch_size', 100)
    epochs = hyperparameters.get('epochs', 1)
    learning_rate = hyperparameters.get('learning_rate', 0.1)
    momentum = hyperparameters.get('momentum', 0.9)
    log_interval = hyperparameters.get('log_interval', 100)
    
    #Download the resnet model
    get_model('http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50', 0)
    sym, args, aux_params = mx.model.load_checkpoint('resnet-50', 0)
    
    #replace the last model.
    (new_sym, arg_params) = get_fine_tune_model(sym, args, num_classes)
    #devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=new_sym, context=ctx)
    batch_size = batch_per_gpu * num_gpus
    (train, val) = get_iterators(batch_size)
    mod.fit(train, val, 
        num_epoch= epochs,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),
        kvstore='device',
        optimizer='sgd',
        epoch_end_callback  = mx.callback.do_checkpoint('%s/102flowers' % model_dir, 1),
        optimizer_params={'learning_rate': learning_rate},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc')
    metric = mx.metric.Accuracy()
    #return mod.score(val, metric)
    return mod

def save(net, model_dir):
    print("Saving the model..")
    # save the model
    #y = net(mx.sym.var('data'))
    net.save_params('%s/102flowers-0000.params' % model_dir)
    #y.save('%s/model.json' % model_dir)
    #net.collect_params().save('%s/model.params' % model_dir)
    
def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    
    sym, arg_params, aux_params = mx.model.load_checkpoint('%s/102flowers' % model_dir, 0)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    return mod
    
    

def transform_fn(net, data, input_content_type, output_content_type):
    from collections import namedtuple
    Batch = namedtuple('Batch', ['data'])
    net.forward(Batch([mx.nd.array(data)]))
    prob = net.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    response = np.argsort(prob)[::-1]
    
    return response
