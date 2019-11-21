import os

from easydict import EasyDict as edict

config = edict()

config.TRAIN = edict()
config.TRAIN.num_gpus = 8
config.TRAIN.batch_size = 1024
config.TRAIN.save_interval = 5000
config.TRAIN.log_interval = 1
# config.TRAIN.n_epoch = 50 
config.TRAIN.num_steps = 200
config.TRAIN.lr_init = 1e-3
config.TRAIN.lr_decay_factor = 0.333
config.TRAIN.lr_decay_every_step = 500
config.TRAIN.weight_decay_factor = 5e-4
config.TRAIN.display_step = 10
config.TRAIN.train_mode = "parallel"

config.MODEL = edict()
config.MODEL.model_path = 'models'
config.MODEL.name = 'simple_conv_net'
config.MODEL.num_input = 784 # MNIST_data_input = 28*28
config.MODEL.num_classes = 10
config.MODEL.dropout = 0.75
# config.MODEL.hin = 
# config.MODEL.win = 
# config.MODEL.hout = 
# config.MODEL.wout = 

config.DATA = edict()
config.DATA.data_path = 'data'
config.DATA.train_images_path = 'data/MNIST/'
config.DATA.annotations_path = 'data/MNIST/'

config.LOG = edict()
config.LOG.vis_path = 'vis'