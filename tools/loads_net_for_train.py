
from math import ceil
import numpy as np

import tensorflow as tf

from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add
from keras.layers import Concatenate, Dropout, Dense
from keras.models import Model

from keras import backend as K
from keras.layers import Layer

# from tools.train_starter import train_starter

# tf padding is 'SAME' not 'same' as Keras
DEFAULT_PADDING = 'SAME'


def make_deconv_filter(name, f_shape, trainable=True):
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    # init = tf.constant_initializer(value=weights, dtype=tf.float32)
    # var = tf.get_variable(name, shape=weights.shape, initializer=init, trainable=trainable)
    init = K.constant_initializer(value=weights, dtype=K.float32)
    var = K.get_variable(name, shape=weights.shape, initializer=init, trainable=trainable)

    return var


def deconv(input, k_h, k_w, c_o, s_h, s_w, name, reuse=None, padding=DEFAULT_PADDING, trainable=True):
    #self.validate_padding(padding)
    c_i = input.get_shape()[-1]
    # with tf.variable_scope(name, reuse=reuse) as scope:
    with K.variable_scope(name, reuse=reuse) as scope:
        # Compute shape out of input
        # in_shape = tf.shape(input)
        in_shape = K.shape(input)
        h = in_shape[1] * s_h
        w = in_shape[2] * s_w
        new_shape = [in_shape[0], h, w, c_o]
        # output_shape = tf.stack(new_shape)
        output_shape = K.stack(new_shape)

        # filter
        f_shape = [k_h, k_w, c_o, c_i]
        weights = make_deconv_filter('weights', f_shape, trainable)
    # return tf.nn.conv2d_transpose(input, weights, output_shape, [1, s_h, s_w, 1], padding=padding, name=scope.name)
    return K.nn.conv2d_transpose(input, weights, output_shape, [1, s_h, s_w, 1], padding=padding, name=scope.name)


class Deconv(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # return K.dot(x, self.kernel)
        return deconv(x, 4, 4, num_units, 2, 2, name='upscore_conv5', trainable=False)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

  
def smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights, sigma=1.0):
    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = tf.multiply(vertex_weights, vertex_diff)
    abs_diff = tf.abs(diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_diff, 1. / sigma_2)))
    in_loss = tf.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
            + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss = tf.div( tf.reduce_sum(in_loss), tf.reduce_sum(vertex_weights) + 1e-10 )
    return loss

def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
      print 'Appending horizontally-flipped training examples...'
      imdb.append_flipped_images()
      print 'done'

  return imdb.roidb


class vgg16_convs():

    def __init__(self):

        self.num_units = 64
        self.keep_prob_queue = 0.5 # YX uses keep_prob
        self.rate = 1 - self.keep_prob_queue # keras uses rate
        self.scale = 1.
        self.height=480
        self.width=640
        self.batch_sz=1
        self.num_classes=3
        self.vertex_reg=1

        input = Input(shape=(self.height, self.width, 3, ))

        self.model = self.build(input, self.height, self.width, self.batch_sz, self.num_classes, self.vertex_reg)


    def build(self, input=None, height=480, width=640, batch_sz=1, num_classes=3, vertex_reg=1):

        # 1st block
        conv1_1 = Conv2D(64, (3,3), name='conv1_1', padding='same', activation='relu')(input)
        conv1_2 = Conv2D(64, (3,3), name='conv1_2', padding='same', activation='relu')(conv1_1)
        pool1 = MaxPooling2D((2,2), strides=(2,2), name='pool1')(conv1_2)

        # 2nd block
        conv2_1 = Conv2D(128, (3,3), name='conv2_1', padding='same', activation='relu')(pool1)
        conv2_2 = Conv2D(128, (3,3), name='conv2_2', padding='same', activation='relu')(conv2_1)
        pool2 = MaxPooling2D((2,2), strides=(2,2), name='pool2')(conv2_2)

        # 3rd block
        conv3_1 = Conv2D(256, (3,3), name='conv3_1', padding='same', activation='relu')(pool2)
        conv3_2 = Conv2D(256, (3,3), name='conv3_2', padding='same', activation='relu')(conv3_1)
        conv3_3 = Conv2D(256, (3,3), name='conv3_3', padding='same', activation='relu')(conv3_2)
        pool3 = MaxPooling2D((2,2), strides=(2,2), name='pool3')(conv3_3)

        # 4th block
        conv4_1 = Conv2D(512, (3,3), name='conv4_1', padding='same', activation='relu')(pool3)
        conv4_2 = Conv2D(512, (3,3), name='conv4_2', padding='same', activation='relu')(conv4_1)
        conv4_3 = Conv2D(512, (3,3), name='conv4_3', padding='same', activation='relu')(conv4_2)
        pool4 = MaxPooling2D((2,2), strides=(2,2), name='pool4')(conv4_3)

        # 5th block
        conv5_1 = Conv2D(512, (3,3), name='conv5_1', padding='same', activation='relu')(pool4)
        conv5_2 = Conv2D(512, (3,3), name='conv5_2', padding='same', activation='relu')(conv5_1)
        conv5_3 = Conv2D(512, (3,3), name='conv5_3', padding='same', activation='relu')(conv5_2)

        # vertex pred head
        num_units = self.num_units
        num_classes = self.num_classes
        scale= self.scale

        score_conv5 = Conv2D(num_units, (1,1), name='score_conv5', padding='same', activation='relu')(conv5_3)
        upscore_conv5 = deconv(score_conv5, 4, 4, num_units, 2, 2, name='upscore_conv5', trainable=False)

        score_conv4 = Conv2D(num_units, (1,1), name='score_conv4', padding='same', activation='relu')(conv4_3)

        add_score = Add()([score_conv4, upscore_conv5])
        dropout = Dropout(self.rate, name='dropout')(add_score)
        upscore = deconv(dropout, int(16*scale), int(16*scale), num_units, int(8*scale), int(8*scale), name='upscore', trainable=False)

        # 'prob' and 'label_2d' will be added later. 'gt_label_weight' cannot be added because of hard_label C++ module

        # vertex_reg = 1
        # num_classes = 3
        if vertex_reg:
            score_conv5_vertex = Conv2D(128, (1,1), name='score_conv5_vertex', padding='same', activation='relu')(conv5_3)
            upscore_conv5_vertex = deconv(score_conv5_vertex, 4, 4, 128, 2, 2, name='upscore_conv5_vertex')
            
            score_conv4_vertex = Conv2D(128, (1,1), name='score_conv4_vertex', padding='same', activation='relu')(conv4_3)
            
            add_score_vertex = Add()([score_conv4_vertex, upscore_conv5_vertex])
            dropout_vertex = Dropout(self.rate, name='dropout_vertex')(add_score_vertex)
            upscore_vertex = deconv(dropout_vertex, int(16*scale), int(16*scale), 128, int(8*scale), int(8*scale), name='upscore_vertex', trainable=False)
            vertex_pred = Conv2D(3*num_classes, (1,1), name='vertex_pred', padding='same', activation='relu')(upscore_vertex)

        
        model = Model(inputs=input, outputs=vertex_pred, name='vgg16')
        return model



if __name__ == '__main__':


    # trainner = train_starter()
    tf.reset_default_graph()

    # batch_sz = 50

    # input = Input(shape=(480, 640, 3, ))

    vgg16 = vgg16_convs()


    # vertex_targets = tf.placeholder(tf.float32, shape=[None, None, None, 3*num_classes])
    # vertex_weights = tf.placeholder(tf.float32, shape=[None, None, None, 3*num_classes])
    # VERTEX_W = 5.0

    # loss_vertex = VERTEX_W * smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights)
