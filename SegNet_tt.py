from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.layers import Activation,BatchNormalization,Conv2D,Lambda,UpSampling2D, add, average, Input, multiply, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, GlobalAveragePooling2D, Dense
from addition_layers import MaxPoolingWithIndices, UpSamplingWithIndices, mvn, standard_block
from losses_and_metrics import dice_coef_loss

def Seg_Net(img_size, num_class=1, act = 'relu',kernel_reg = None, nb_filter = [64,128,256,512,512], normalize_layer = None, opt = 'Adam', lossfnc = dice_coef_loss,
                skip_con = False, opt_skip = add, use_dropout = False, droprate=[0,0,0,0,0.5], thickness = [2,2,3,3,3], normalize_input = True):

    if isinstance(thickness, list):
      assert len(nb_filter) == len(thickness), '\'nb_filter\' and \'thickness\' must have same length'
    elif isinstance(thickness, int):
      val_thick = thickness
      thickness=[val_thick for i in nb_filter]
    else:
      raise Exception('\'thickness\' should be a \'list\' or a \'int\'')
    if use_dropout:
      if isinstance(droprate, list):
        assert len(nb_filter) == len(droprate), '\'nb_filter\' and \'droprate\' must have same length'
      else:
        raise Exception('\'droprate\' should be a \'list\' or a \'int\'')
    
    img_input = Input(shape=img_size, name='main_input')
    if normalize_input:
      pool = Lambda(mvn)(img_input)
    else:
      pool = img_input
    down_fm = []
    down_id = []
    len_x = len(nb_filter)
    for i, sfilter in enumerate(nb_filter):
      conv = standard_block(pool, stage=str(i+1)+'1', nb_filter=sfilter, thickness = thickness[i], act = act, normalize_layer = normalize_layer, kernel_reg = kernel_reg)
      if skip_con:
        down_fm.append(conv)
      if use_dropout and droprate[i]:
        conv = Dropout(rate = 0.5)(conv)
      pool, indice = MaxPoolingWithIndices(pool_size = (2,2), strides = (2,2), padding = "VALID", name='pool'+str(i+1))(conv)
      down_id.append(indice)
    
    conv = pool

    for i in range(len_x-1):
      up = UpSamplingWithIndices()([conv, down_id[len_x-i-1]])
      if skip_con:
        up = opt_skip([up, down_fm[len_x-i-1]], name='merge'+str(len_x-i)+'2')
      conv = standard_block(up, stage=str(len_x-i)+'2', nb_filter=nb_filter[len_x-i-1], thickness = thickness[len_x-i-1]-1, act = act, normalize_layer = normalize_layer, kernel_reg = kernel_reg)
      conv = standard_block(conv, stage=str(len_x-i)+'2'+'_pp', nb_filter=nb_filter[len_x-i-2], thickness = 1, act = act, normalize_layer = normalize_layer, kernel_reg = kernel_reg)
    
    up = UpSamplingWithIndices()([conv, down_id[0]])
    if skip_con:
      up = opt_skip([up, down_fm[0]], name='merge'+str(len_x-i-1)+'2')
    conv = standard_block(up, stage='12', nb_filter=nb_filter[0], thickness = thickness[0], act = act, normalize_layer = normalize_layer, kernel_reg = kernel_reg)

    net_output = Conv2D(num_class, (1, 1), activation='sigmoid', name='output', kernel_initializer = 'he_normal', padding='same')(conv)

    model = Model(input=img_input, output=net_output)
    model.compile(optimizer = opt, loss = lossfnc, metrics = [dice_coef, jaccard_coef])

    return model