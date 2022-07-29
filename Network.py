import tensorflow as tf
from keras import datasets, layers, models, callbacks, optimizers, losses, Sequential, Model, Input

# Net1: resblock-maxpool
# Spatial-attention module: attention map multiplied with the input feature maps
# Net2: resblock-avgpool-dense
# Triplet cross-entropy
# class/hash


def TripletLoss(bq, bp, bn):
    margin = 0.5
    mse_loss = losses.mean_squared_error(reduction='none')
    margin_val = margin * bq.shape[1]
    squared_loss_pos = tf.reduce_mean(mse_loss(bq, bp), dim=1) #reduce_mean on positive and anchor
    squared_loss_neg = tf.reduce_mean(mse_loss(bq, bn), dim=1) #reduce_mean on negative and anchor
    zeros = tf.zeros_like(squared_loss_neg)
    loss = tf.math.reduce_max(zeros, margin_val - squared_loss_neg + squared_loss_pos)
    return tf.math.reduce_max(loss)

class Network:
    def __init__(self):
        self.model = self.tripletnetwork(shape=(67,1,64,128,128))
        self.compile = self.model.compile(optimizer = optimizers.adam_v2.Adam(lr=1e-4),
                           metrics=['accuracy'])
        # Need to properly incorporate loss into above
        self.fit = self.model.fit
        self.predict = self.model.predict
        self.evaluate = self.model.evaluate
        self.summary = self.model.summary

    def resblock(self, x, filters):
        def net(inp):
            l1 = layers.Conv3D(filters, strides=2, kernel_size=3, padding='same', activation='relu',
                               data_format='channels_first')(inp)
            l2 = layers.BatchNormalization(axis=1)(l1)
            l3 = layers.Conv3D(filters, strides=2, kernel_size=3, padding='same', activation='relu')(l2)
            l4 = layers.BatchNormalization(axis=1)(l3)
            return l4

        f = net(x)
        h = f + x
        res = layers.ReLU()(h)
        return res

    def spatial(self, x, filters):
        def net(inp):
            l1a = layers.MaxPool3D(pool_size=3, strides=2, padding='same')
            l1b = tf.math.reduce_mean(inp, keepdims=True)
            l1c = tf.math.reduce_max(inp, keepdims=True)
            l2 = tf.concat([l1a, l1b, l1c], axis=1)
            l3 = layers.Conv3D(filters, strides=2, kernel_size=3, padding='same', activation='sigmoid')(l2)
            return l3
        return net(x)

    def net1(self, inp):
        l1 = self.resblock(inp, 16)
        l2 = layers.MaxPool3D(3, strides=2, padding='same')(l1)
        return l2

    def net2(self, inp):
        l1 = self.resblock(inp, 8)
        l2 = layers.AvgPool3D(pool_size=3, strides=2, padding='same')(l1)
        return l2

    def model(self, inp, hash_size=36, type_size=5):
        x = self.net1(inp)
        x = self.spatial(x, 16)
        x = self.net2(x)
        x = self.resblock(x, 1)
        x_hash = layers.Dense(hash_size)(x)
        x_type = layers.Dense(type_size)(x)
        return Model(inputs=inp, outputs=[x_hash, x_type], name='ATH')

    def tripletnetwork(self, shape):
        ce_loss = losses.BinaryCrossentropy
        xq = Input(shape=shape, name='xq')
        xp = Input(shape=shape, name='xp')
        xn = Input(shape=shape, name='xn')
        yq = Input(shape=shape, name='yq')
        yp = Input(shape=shape, name='yp')
        yn = Input(shape=shape, name='yn')
        Q_hash, Q_type = self.model(xq)
        P_hash, P_type = self.model(xp)
        N_hash, N_type = self.model(xn)
        hash_loss = TripletLoss(Q_hash, P_hash, N_hash)
        type_loss = ce_loss(Q_type, yq) + ce_loss(P_type, yp), ce_loss(N_type, yn)
        loss = hash_loss + type_loss
        return Model(inputs=[xq, xp, xn, yq, yp, yn], outputs=loss)
