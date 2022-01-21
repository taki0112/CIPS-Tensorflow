from ops import *

##################################################################################
# Generator Layers
##################################################################################
class ToRGB(tf.keras.layers.Layer):
    def __init__(self, fmaps, **kwargs):
        super(ToRGB, self).__init__(**kwargs)
        self.fmaps = fmaps

        self.conv = ModulatedConv2D(fmaps=3, style_fmaps=fmaps, kernel=1, up=False, down=False, demodulate=False,
                                    resample_kernel=None, gain=1.0, lrmul=1.0, fused_modconv=True, name='conv')
        self.apply_bias = BiasAct(lrmul=1.0, act='linear', name='bias')

    def call(self, inputs, skip=None, training=None, mask=None):
        x, w = inputs

        x = self.conv([x, w])
        x = self.apply_bias(x)

        if skip is not None:
            x = x + skip

        return x

class StyledConv(tf.keras.layers.Layer):
    def __init__(self, fmaps, style_fmaps, kernel, resample_kernel=(1, 3, 3, 1), up=False, down=False, demodulate=True, fused_modconv=True, gain=1.0, lrmul=1.0, **kwargs):
        super(StyledConv, self).__init__(**kwargs)
        resample_kernel = list(resample_kernel)
        self.conv = ModulatedConv2D(fmaps, style_fmaps, kernel, resample_kernel, up, down, demodulate, fused_modconv, gain, lrmul)
        self.apply_noise = Noise(name='noise')
        self.apply_bias_act = BiasAct(lrmul=lrmul, act='lrelu', name='bias')

    def call(self, inputs, training=None, mask=None):
        x, y = inputs
        x = self.conv([x, y])
        x = self.apply_noise(x)
        x = self.apply_bias_act(x)

        return x

class ConLinear(tf.keras.layers.Layer):
    def __init__(self, out_ch, is_fist=False, **kwargs):
        super(ConLinear, self).__init__(**kwargs)
        in_ch = 2

        if is_fist:
            weight_initializer = tf.initializers.RandomUniform(minval=-np.sqrt(9 / in_ch), maxval=np.sqrt(9 / in_ch))
            self.conv = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=1, strides=1, use_bias=True,
                                               data_format='channels_first', kernel_initializer=weight_initializer)
        else:
            weight_initializer = tf.initializers.random_uniform(minval=-np.sqrt(3 / in_ch), maxval=np.sqrt(3 / in_ch))
            self.conv = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=1, strides=1, use_bias=True,
                                               data_format='channels_first', kernel_initializer=weight_initializer)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        return x

class LFF(tf.keras.layers.Layer):
    def __init__(self, hidden_size, **kwargs):
        super(LFF, self).__init__(**kwargs)

        self.ffm = ConLinear(hidden_size, is_fist=True)

    def call(self, inputs, training=None, mask=None):
        x = self.ffm(inputs)
        x = tf.sin(x)

        return x

class ConstantInput(tf.keras.layers.Layer):
    def __init__(self, channel, size=4, **kwargs):
        super(ConstantInput, self).__init__(**kwargs)

        const_init = tf.random.normal(shape=(1, channel, size, size), mean=0.0, stddev=1.0)
        self.const = tf.Variable(const_init, name='const', trainable=True)

    def call(self, inputs, training=None, mask=None):
        batch = inputs.shape[0]
        x = tf.tile(self.const, multiples=[batch, 1, 1, 1])

        return x

##################################################################################
# Discriminator Layers
##################################################################################
class FromRGB(tf.keras.layers.Layer):
    def __init__(self, fmaps, **kwargs):
        super(FromRGB, self).__init__(**kwargs)
        self.fmaps = fmaps

        self.conv = Conv2D(fmaps=self.fmaps, kernel=1, up=False, down=False,
                           resample_kernel=None, gain=1.0, lrmul=1.0, name='conv')
        self.apply_bias_act = BiasAct(lrmul=1.0, act='lrelu', name='bias')

    def call(self, inputs, training=None, mask=None):
        y = self.conv(inputs)
        y = self.apply_bias_act(y)
        return y


class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, n_f0, n_f1, **kwargs):
        super(DiscriminatorBlock, self).__init__(**kwargs)
        self.gain = 1.0
        self.lrmul = 1.0
        self.n_f0 = n_f0
        self.n_f1 = n_f1
        self.resnet_scale = 1. / tf.sqrt(2.)

        # conv_0
        self.conv_0 = Conv2D(fmaps=self.n_f0, kernel=3, up=False, down=False,
                             resample_kernel=None, gain=self.gain, lrmul=self.lrmul, name='conv_0')
        self.apply_bias_act_0 = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_0')

        # conv_1 down
        self.conv_1 = Conv2D(fmaps=self.n_f1, kernel=3, up=False, down=True,
                             resample_kernel=[1, 3, 3, 1], gain=self.gain, lrmul=self.lrmul, name='conv_1')
        self.apply_bias_act_1 = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_1')

        # resnet skip
        self.conv_skip = Conv2D(fmaps=self.n_f1, kernel=1, up=False, down=True,
                                resample_kernel=[1, 3, 3, 1], gain=self.gain, lrmul=self.lrmul, name='skip')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        residual = x

        # conv0
        x = self.conv_0(x)
        x = self.apply_bias_act_0(x)

        # conv1 down
        x = self.conv_1(x)
        x = self.apply_bias_act_1(x)

        # resnet skip
        residual = self.conv_skip(residual)
        x = (x + residual) * self.resnet_scale
        return x

class DiscriminatorLastBlock(tf.keras.layers.Layer):
    def __init__(self, n_f0, n_f1, **kwargs):
        super(DiscriminatorLastBlock, self).__init__(**kwargs)
        self.gain = 1.0
        self.lrmul = 1.0
        self.n_f0 = n_f0
        self.n_f1 = n_f1

        self.minibatch_std = MinibatchStd(group_size=4, num_new_features=1, name='minibatchstd')

        # conv_0
        self.conv_0 = Conv2D(fmaps=self.n_f0, kernel=3, up=False, down=False,
                             resample_kernel=None, gain=self.gain, lrmul=self.lrmul, name='conv_0')
        self.apply_bias_act_0 = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_0')

        # dense_1
        self.dense_1 = Dense(self.n_f1, gain=self.gain, lrmul=self.lrmul, name='dense_1')
        self.apply_bias_act_1 = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_1')

    def call(self, x, training=None, mask=None):
        x = self.minibatch_std(x)

        # conv_0
        x = self.conv_0(x)
        x = self.apply_bias_act_0(x)

        # dense_1
        x = self.dense_1(x)
        x = self.apply_bias_act_1(x)
        return x