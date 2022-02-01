import tensorflow as tf

class WSSNet():
    def build_network(self, input_layer):
        print('UNet BN')
        padding = 'SAME'
        channel_nr = 64
        
        [xyz0, xyz1, xyz2, v1, v2] = input_layer

        input_layer = tf.keras.layers.concatenate([xyz0, xyz1, xyz2, v1, v2])
        
        # === Starting U-Net ===
        # =========== Downward blocks =========== 
        conv1 = conv2d(input_layer, kernel_size=3, filters=channel_nr, padding='PERIODIC', activation='relu')
        conv1 = conv2d(conv1, kernel_size=3, filters=channel_nr, padding='PERIODIC', activation='relu')
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        pool1 = tf.keras.layers.MaxPooling2D()(conv1)

        conv2 = conv2d(pool1, kernel_size=3, filters=channel_nr * 2, padding=padding, activation='relu')
        conv2 = conv2d(conv2, kernel_size=3, filters=channel_nr * 2, padding=padding, activation='relu')
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        pool2 = tf.keras.layers.MaxPooling2D()(conv2)

        conv3 = conv2d(pool2, kernel_size=3, filters=channel_nr * 4, padding=padding, activation='relu')
        conv3 = conv2d(conv3, kernel_size=3, filters=channel_nr * 4, padding=padding, activation='relu')
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        pool3 = tf.keras.layers.MaxPooling2D()(conv3)

        # =========== bottom layer =========== 
        conv4 = conv2d(pool3, kernel_size=3, filters=channel_nr * 8, padding=padding, activation='relu')
        conv4 = conv2d(conv4, kernel_size=3, filters=channel_nr * 8, padding=padding, activation='relu')
        conv4 = tf.keras.layers.BatchNormalization()(conv4)

        # =========== Upward blocks =========== 
        up5 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(conv4)
        up5 = conv2d(up5, kernel_size=3, filters=channel_nr * 4, padding=padding, activation='relu')
        merge5 = tf.keras.layers.concatenate([conv3,up5])

        conv5 = conv2d(merge5, kernel_size=3, filters=channel_nr * 4, padding=padding, activation='relu')
        conv5 = conv2d(conv5, kernel_size=3, filters=channel_nr * 4, padding=padding, activation='relu')
        conv5 = tf.keras.layers.BatchNormalization()(conv5)


        up6 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(conv5)
        up6 = conv2d(up6, kernel_size=3, filters=channel_nr * 2, padding=padding, activation='relu')
        merge6 = tf.keras.layers.concatenate([conv2,up6])

        conv6 = conv2d(merge6, kernel_size=3, filters=channel_nr * 2, padding=padding, activation='relu')
        conv6 = conv2d(conv6, kernel_size=3, filters=channel_nr * 2, padding=padding, activation='relu')
        conv6 = tf.keras.layers.BatchNormalization()(conv6)

        up7 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(conv6)
        up7 = conv2d(up7, kernel_size=3, filters=channel_nr, padding=padding, activation='relu')
        merge7 = tf.keras.layers.concatenate([conv1,up7])

        conv7 = conv2d(merge7, kernel_size=3, filters=channel_nr, padding=padding, activation='relu')
        conv7 = conv2d(conv7, kernel_size=3, filters=channel_nr, padding=padding, activation='relu')
        
        # output layer, 3 channels (vector)
        wss = conv2d(conv7, kernel_size=3, filters=3, padding=padding, activation=None)
    
        
        return wss
    
def conv2d(x, kernel_size, filters=64, padding='SYMMETRIC', activation=None, initialization=None, use_bias=True, constraint=None, strides=1):
    if padding == 'PERIODIC':
        p = (kernel_size - 1) // 2
        x = periodic_padding_flexible(x, axis=1,padding=p)
        x = tf.pad(x, [[0,0],[0,0],[p,p],[0,0]], 'SYMMETRIC')
        x = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, kernel_initializer=initialization, use_bias=use_bias, kernel_constraint=constraint, strides=strides)(x)
    elif padding == 'SYMMETRIC' or padding == 'REFLECT':
        p = (kernel_size - 1) // 2
        x = tf.pad(x, [[0,0],[p,p],[p,p],[0,0]], padding)
        x = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, kernel_initializer=initialization, use_bias=use_bias, kernel_constraint=constraint, strides=strides)(x)
    else:
        assert padding in ['SAME', 'VALID']
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, activation=activation, kernel_initializer=initialization, use_bias=use_bias, kernel_constraint=constraint, strides=strides)(x)
    return x

def periodic_padding_flexible(tensor, axis,padding=1):
    """
        https://stackoverflow.com/questions/39088489/tensorflow-periodic-padding
        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: number of cells to pad, int or tuple

        return: padded tensor
    """
    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(padding,int):
        padding = (padding,)

    ndim = len(tensor.shape)
    for ax,p in zip(axis,padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left

        ind_right = [slice(-p,None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right]
        left = tensor[ind_left]
        middle = tensor
        tensor = tf.concat([right,middle,left], axis=ax)

    return tensor