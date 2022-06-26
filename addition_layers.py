
import tensorflow as tf
from keras import backend as K
#from keras.engine.topology import Layer
from tensorflow.python.keras.layers import Layer
from keras.layers import ZeroPadding2D, Cropping2D
from keras.layers import Activation, BatchNormalization, Conv2D, Lambda, UpSampling2D, Input, Conv2DTranspose, MaxPooling2D, Dropout
from keras.layers import  concatenate, add, average, multiply

def mvn(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1,2), keepdims=True)
    std = K.std(tensor, axis=(1,2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)
    
    return mvn

def crop(tensors):
    '''
    List of 2 tensors, the second tensor having larger spatial dimensions.
    '''
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = t._keras_shape
        h_dims.append(h)
        w_dims.append(w)
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = int(crop_h % 2)
    rem_w = int(crop_w % 2)
    tt_h = int(crop_h / 2)
    tt_w = int(crop_w / 2)
    crop_h_dims = (tt_h, tt_h + rem_h)
    crop_w_dims = (tt_w, tt_w + rem_w)
    cropped = Cropping2D(cropping=(crop_h_dims, crop_w_dims))(tensors[1])
    
    return cropped

class MaxPoolingWithIndices(Layer):
    def __init__(self, pool_size,strides,padding='SAME',**kwargs):
        super(MaxPoolingWithIndices, self).__init__(**kwargs)
        self.pool_size=pool_size
        self.strides=strides
        self.padding=padding
        return
    def call(self,x):
        pool_size=self.pool_size
        strides=self.strides
        if isinstance(pool_size,int):
            ps=[1,pool_size,pool_size,1]
        else:
            ps=[1,pool_size[0],pool_size[1],1]
        if isinstance(strides,int):
            st=[1,strides,strides,1]
        else:
            st=[1,strides[0],strides[1],1]
        output1,output2=tf.nn.max_pool_with_argmax(x,ps,st,self.padding)
        return [output1,output2]
    def compute_output_shape(self, input_shape):
        if isinstance(self.pool_size,int):
            output_shape=(input_shape[0],input_shape[1]//self.pool_size,input_shape[2]//self.pool_size,input_shape[3])
        else:
            output_shape=(input_shape[0],input_shape[1]//self.pool_size[0],input_shape[2]//self.pool_size[1],input_shape[3])
        return [output_shape,output_shape]


class UpSamplingWithIndices(Layer):
    def __init__(self, **kwargs):
        super(UpSamplingWithIndices, self).__init__(**kwargs)
        return
    def call(self,x):
        argmax=K.cast(K.flatten(x[1]),'int32')
        max_value=K.flatten(x[0])
        with tf.compat.v1.variable_scope(self.name):
            input_shape=K.shape(x[0])
            batch_size=input_shape[0]
            image_size=input_shape[1]*input_shape[2]*input_shape[3]
            output_shape=[input_shape[0],input_shape[1]*2,input_shape[2]*2,input_shape[3]]
            indices_0=K.flatten(tf.multiply(K.reshape(tf.range(batch_size),(batch_size,1)),K.ones_like((1,image_size),dtype='int32')))
            indices_1=argmax%(image_size*4)//(output_shape[2]*output_shape[3])
            indices_2=argmax%(output_shape[2]*output_shape[3])//output_shape[3]
            indices_3=argmax%output_shape[3]
            indices=tf.stack([indices_0,indices_1,indices_2,indices_3])
            output=tf.scatter_nd(K.transpose(indices),max_value,output_shape)
            return output
    def compute_output_shape(self, input_shape):
        shape_x, shape_argmax = input_shape
        return shape_x[0],shape_x[1]*2,shape_x[2]*2,shape_x[3]

def standard_block(input_tensor, stage, nb_filter, kernel_size=3, act = 'relu', thickness = 2, kernel_reg = None, normalize_layer = None):
    x = input_tensor

    if thickness == 0:
      return x
    
    for i in range(thickness):
      x = Conv2D(nb_filter, (3,3), name='conv'+stage+'_'+str(i+1), kernel_initializer = 'he_normal', padding='same', kernel_regularizer = kernel_reg)(x)
      #x = Dropout(dropout_rate, name='dp'+stage+'_'+str(i+1))(x)
      if normalize_layer == 'bn':
        x = BatchNormalization(name = 'bn'+stage+'_'+str(i+1))(x)
      elif normalize_layer == 'mvn':
        x = Lambda(mvn, name = 'lambda'+stage+'_'+str(i+1))(x)
      x = Activation(act, name='act'+stage+'_'+str(i+1))(x)

    return x

def standard_block_t2(input_tensor, stage, nb_filter, kernel_size=3, act = 'relu', thickness = 2, kernel_reg = None, normalize_layer = None):
    x = input_tensor

    if thickness == 0:
      return x
    
    for i in range(thickness):
      x = Conv2D(nb_filter, (kernel_size, kernel_size), name='conv'+stage+'_'+str(i+1), activation = act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer = kernel_reg)(x)
      #x = Dropout(dropout_rate, name='dp'+stage+'_'+str(i+1))(x)
      if normalize_layer == 'bn':
        x = BatchNormalization(name = 'bn'+stage+'_'+str(i+1))(x)
      elif normalize_layer == 'mvn':
        x = Lambda(mvn, name = 'lambda'+stage+'_'+str(i+1))(x)

    return x

def expend_as(tensor, rep, name):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep},
                       name='psi_up' + name)(tensor)
    return my_repeat

def AttnGatingBlock(x, g, inter_shape, name, normalize_layer = 'bn'):
    ''' take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same geature channels (theta_x)
    then, upsample g to be same size as x
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''

    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', name='xl' + name)(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same', name='g_up' + name)(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same', name='psi' + name)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3], name)
    y = multiply([upsample_psi, x], name='q_attn' + name)

    result = Conv2D(shape_x[3], (1, 1), padding='same', name='q_attn_conv' + name)(y)
    if normalize_layer == 'bn':
        result_bn = BatchNormalization(name='q_attn_bn' + name)(result)
    elif normalize_layer == 'mvn':
        result_bn = Lambda(mvn, name='q_attn_mvn' + name)(result)
    return result_bn


def UnetConv2D(input, outdim, is_batchnorm, name, kinit = 'glorot_normal'):
    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name + '_1')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_act')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name + '_2')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_act')(x)
    return x


def UnetGatingSignal(input, name, normalize_layer = 'bn', act = 'relu'):
    ''' this is simply 1x1 convolution, bn, activation '''
    shape = K.int_shape(input)
    x = Conv2D(shape[3], (1, 1), strides=(1, 1), padding="same", name=name + '_conv')(input)
    if normalize_layer == 'bn':
        x = BatchNormalization(name=name + '_bn')(x)
    elif normalize_layer == 'mvn':
        x = Lambda(mvn, name=name + 'mvn')(x)
    x = Activation(act, name=name + '_act')(x)
    return x

def UnetGatingSignal_t2(input, name, nb_filter, normalize_layer = 'bn', act = 'relu'):
    ''' this is simply 1x1 convolution, bn, activation '''
    x = Conv2D(nb_filter, (1, 1), strides=(1, 1), padding="same", name=name + '_conv')(input)
    if normalize_layer == 'bn':
        x = BatchNormalization(name=name + '_bn')(x)
    elif normalize_layer == 'mvn':
        x = Lambda(mvn, name=name + 'mvn')(x)
    x = Activation(act, name=name + '_act')(x)
    return x

##############################################################################################
#Att_unet_t3 additional functions
def attention_block_2d(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = multiply([x, rate])
    return att_x


def attention_up_and_concate(down_layer, layer,inter_ratio = 0.25, opt_skip=concatenate):
    # in_channel = down_layer.get_shape().as_list()[3]
    in_channel = down_layer._keras_shape[-1]
    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2))(down_layer)
    layer = attention_block_2d(x=layer, g=up, inter_channel=int(in_channel*inter_ratio))
    # my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    # concate = my_concat([up, layer])
    concate = opt_skip([up, layer])
    return concate

def attention_up_and_concate_t2(down_layer, layer,inter_ratio = 0.25, opt_skip=concatenate):
    # in_channel = down_layer.get_shape().as_list()[3]
    in_channel = down_layer._keras_shape[-1]
    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2))(down_layer)
    layer = attention_block_2d(x=layer, g=up, inter_channel=int(in_channel*inter_ratio))
    # my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    # concate = my_concat([up, layer])
    concate = opt_skip([Conv2DTranspose(in_channel, [2,2], strides=[2,2], padding='same')(down_layer), layer])
    return concate