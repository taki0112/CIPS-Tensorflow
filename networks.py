from layers import *
from tensorflow.keras import Sequential

##################################################################################
# CIPS Generator Networks
##################################################################################

class CIPSskip(tf.keras.Model):
    def __init__(self, g_params, **kwargs):
        super(CIPSskip, self).__init__(**kwargs)

        self.size = g_params['img_size']
        self.hidden_size = g_params['hidden_size']
        self.w_dim = g_params['w_dim']
        self.labels_dim = g_params['labels_dim']
        self.featuremaps = g_params['featuremaps']
        self.n_mapping = g_params['n_mapping']

        self.lff = LFF(self.hidden_size)
        self.emb = ConstantInput(self.hidden_size, self.size)

        self.convs = []
        self.to_rgbs = []
        self.log_size = int(np.log2(self.size))

        self.n_intermediate = self.log_size - 1
        self.to_rgb_stride = 2

        if self.labels_dim > 0:
            self.labels_embedding = LabelEmbedding(embed_dim=self.w_dim, name='labels_embedding')

        self.conv1 = StyledConv(fmaps=self.featuremaps[0], style_fmaps=self.featuremaps[0]*2, kernel=1, resample_kernel=[1, 3, 3, 1])

        for i in range(self.log_size - 1):
            fmaps = self.featuremaps[i]
            style_fmaps2 = fmaps
            if i == 0:
                style_fmaps = fmaps
            else:
                style_fmaps = self.featuremaps[i-1]
            self.convs.append(StyledConv(fmaps=fmaps, style_fmaps=style_fmaps, kernel=1, resample_kernel=[1, 3, 3, 1]))
            self.convs.append(StyledConv(fmaps=fmaps, style_fmaps=style_fmaps2, kernel=1, resample_kernel=[1, 3, 3, 1]))
            self.to_rgbs.append(ToRGB(fmaps=style_fmaps2))

        self.mapping_layers = [PixelNorm()]

        for i in range(self.n_mapping):
            self.mapping_layers.append(Dense(fmaps=self.w_dim))

        self.mapping_layers = Sequential(self.mapping_layers)

    def build(self, input_shape):
        self.coords = get_coords(batch_size=input_shape[0][0], height=self.size, width=self.size)

    @tf.function
    def set_as_moving_average_of(self, src_net, beta=0.99, beta_nontrainable=0.0):
        for cw, sw in zip(self.weights, src_net.weights):
            assert sw.shape == cw.shape

            if 'w_avg' in cw.name:
                cw.assign(lerp(sw, cw, beta_nontrainable))
            else:
                cw.assign(lerp(sw, cw, beta))
        return

    def call(self, inputs, truncation_psi=1.0, ret_w=True, input_is_latent=False, training=None, mask=None):
        latent, labels = inputs
        if truncation_psi < 1:
            truncation_latent = tf.reduce_mean(tf.random.normal(shape=[4096, self.w_dim]), axis=0, keepdims=True)
            latent = truncation_latent + truncation_psi * (latent - truncation_latent)

        # embed label if any
        if self.labels_dim > 0:
            y = self.labels_embedding(labels)
            latent = tf.concat([latent, y], axis=1)

        if not input_is_latent:
            latent = self.mapping_layers(latent)
        fourier = self.lff(self.coords)

        batch_size, _, height, width = self.coords.shape

        if training and width == height == self.size:
            coord_emb = self.emb(fourier)
        else:
            coord_emb = self.emb(fourier)
            coord_emb = grid_sample_tf(coord_emb, self.coords)

        x = tf.concat([fourier, coord_emb], axis=1) # channel concat
        rgb = None
        x = self.conv1([x, latent])
        for i in range(self.n_intermediate):
            for j in range(self.to_rgb_stride):
                x = self.convs[i*self.to_rgb_stride + j]([x, latent])

            rgb = self.to_rgbs[i]([x, latent], skip=rgb)

        if ret_w:
            return rgb, latent
        else:
            return rgb

##################################################################################
# Discriminator Networks
##################################################################################
class Discriminator(tf.keras.Model):
    def __init__(self, d_params, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        # discriminator's (resolutions and featuremaps) are reversed against generator's
        self.labels_dim = d_params['labels_dim']
        self.r_resolutions = d_params['resolutions'][::-1]
        self.r_featuremaps = d_params['featuremaps'][::-1]

        # stack discriminator blocks
        res0, n_f0 = self.r_resolutions[0], self.r_featuremaps[0]
        self.initial_fromrgb = FromRGB(fmaps=n_f0, name='{:d}x{:d}/FromRGB'.format(res0, res0))
        self.blocks = []

        for index, (res0, n_f0) in enumerate(zip(self.r_resolutions[:-1], self.r_featuremaps[:-1])):
            n_f1 = self.r_featuremaps[index + 1]
            self.blocks.append(DiscriminatorBlock(n_f0=n_f0, n_f1=n_f1, name='{:d}x{:d}'.format(res0, res0)))

        # set last discriminator block
        res = self.r_resolutions[-1]
        n_f0, n_f1 = self.r_featuremaps[-2], self.r_featuremaps[-1]
        self.last_block = DiscriminatorLastBlock(n_f0, n_f1, name='{:d}x{:d}'.format(res, res))

        # set last dense layer
        self.last_dense = Dense(max(self.labels_dim, 1), gain=1.0, lrmul=1.0, name='last_dense')
        self.last_bias = BiasAct(lrmul=1.0, act='linear', name='last_bias')



    # @ tf.function
    def call(self, inputs, training=None, mask=None):
        images, labels = inputs

        x = self.initial_fromrgb(images)
        for block in self.blocks:
            x = block(x)

        x = self.last_block(x)

        logit = self.last_dense(x)
        logit = self.last_bias(logit)

        if self.labels_dim > 0:
            logit = tf.reduce_sum(logit * labels, axis=1, keepdims=True)

        scores_out = logit

        return scores_out