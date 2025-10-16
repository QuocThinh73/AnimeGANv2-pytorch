import tensorflow as tf

weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
weight_regularizer = None

#############################
# Activation Function
#############################


def flatten(x):
    return tf.compat.v1.layers.flatten(x)


def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.nn.tanh(x)


def sigmoid(x):
    return tf.nn.sigmoid(x)

#############################
# Normalization Function
#############################


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def instance_norm(x, scope='instance_norm'):
    return tf.keras.layers.GroupNormalization(
        groups=-1, axis=-1, epsilon=1e-5, center=True, scale=True, name=scope
    )(x)


def layer_norm(x, scope='layer_norm'):
    return tf.keras.layers.LayerNormalization(
        axis=-1,
        center=True,
        scale=True,
        epsilon=1e-5,
        name=scope
    )(x)


def batch_norm(x, is_training=True, scope='batch_norm'):
    bn = tf.keras.layers.BatchNormalization(
        momentum=0.9,
        epsilon=1e-5,
        center=True,
        scale=True,
        name=scope
    )
    return bn(x, training=is_training)

#############################
# Layer
#############################


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.compat.v1.get_variable(
        "u", [1, w_shape[-1]], initializer=tf.compat.v1.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def convolution(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, spectral_normalization=False, scope='conv0'):
    with tf.compat.v1.variable_scope(scope):
        if (kernel - stride) % 2 == 0:
            pad_top = pad_bottom = pad_left = pad_right = pad
        else:
            pad_top = pad_left = pad
            pad_bottom = pad_right = kernel - stride - pad

        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom],
                       [pad_left, pad_right], [0, 0]])
        elif pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [
                       pad_left, pad_right], [0, 0]], mode="REFLECT")

        if spectral_normalization:
            w = tf.compat.v1.get_variable(
                "kernel",
                shape=[kernel, kernel, x.shape[-1], channels],
                initializer=weight_init,
                regularizer=weight_regularizer
            )
            x = tf.nn.conv2d(
                input=x,
                filters=spectral_norm(w),
                strides=[1, stride, stride, 1],
                padding='VALID'
            )
            if use_bias:
                bias = tf.compat.v1.get_variable(
                    "bias", [channels], initializer=tf.zeros_initializer())
                x = tf.nn.bias_add(x, bias)
        else:
            x = tf.compat.v1.layers.conv2d(
                inputs=x,
                filters=channels,
                kernel_size=kernel,
                kernel_initializer=weight_init,
                kernel_regularizer=weight_regularizer,
                strides=stride,
                use_bias=use_bias
            )
        return x


def deconvolution(x, channels, kernel=4, stride=2, use_bias=True, spectral_normalization=False, scope='deconv0'):
    with tf.compat.v1.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        output_shape = [x_shape[0], tf.shape(
            x)[1] * stride, tf.shape(x)[2] * stride, channels]
        if spectral_normalization:
            w = tf.compat.v1.get_variable(
                "kernel",
                shape=[kernel, kernel, channels, x.get_shape()[-1]],
                initializer=weight_init,
                regularizer=weight_regularizer
            )
            x = tf.nn.conv2d_transpose(
                inputs=x,
                filters=spectral_norm(w),
                output_shape=output_shape,
                strides=[1, stride, stride, 1],
                padding='SAME'
            )
            if use_bias:
                bias = tf.compat.v1.get_variable(
                    "bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
        else:
            x = tf.compat.v1.layers.conv2d_transpose(
                inputs=x,
                filters=channels,
                kernel_size=kernel,
                kernel_initializer=weight_init,
                kernel_regularizer=weight_regularizer,
                strides=stride,
                padding='SAME',
                use_bias=use_bias
            )
        return x

#############################
# Residual Block
#############################


def resblock(x_init, channels, use_bias=True, scope='resblock0'):
    with tf.compat.v1.variable_scope(scope):
        with tf.compat.v1.variable_scope('res1'):
            x = convolution(x_init, channels, kernel=3, stride=1,
                            pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.compat.v1.variable_scope('res2'):
            x = convolution(x, channels, kernel=3, stride=1,
                            pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

        return x + x_init

#############################
# Loss Function
#############################


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def L2_loss(x, y):
    size = tf.size(x, out_type=tf.float32)
    return tf.nn.l2_loss(x - y) * 2.0 / size


def Huber_loss(x, y):
    return tf.keras.losses.Huber(delta=1.0, reduction='sum_over_batch_size')(x, y)


def discriminator_loss(loss_func, real, gray, fake, real_blur):
    real_loss = gray_loss = fake_loss = real_blur_loss = 0

    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        real_loss = -tf.reduce_mean(real)
        gray_loss = -tf.reduce_mean(gray)
        fake_loss = -tf.reduce_mean(fake)
        real_blur_loss = -tf.reduce_mean(real_blur)

    if loss_func == 'lsgan':
        real_loss = tf.reduce_mean(tf.square(real - 1.0))
        gray_loss = tf.reduce_mean(tf.square(gray))
        fake_loss = tf.reduce_mean(tf.square(fake))
        real_blur_loss = tf.reduce_mean(tf.square(real_blur))

    if loss_func == 'gan' or loss_func == 'dragan':
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(real), logits=real))
        gray_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(gray), logits=gray))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(fake), logits=fake))
        real_blur_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(real_blur), logits=real_blur))

    if loss_func == 'hinge':
        real_loss = tf.reduce_mean(relu(1.0 - real))
        gray_loss = tf.reduce_mean(relu(1.0 + gray))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))
        real_blur_loss = tf.reduce_mean(relu(1.0 + real_blur))

    loss = real_loss + gray_loss + fake_loss + 0.1 * real_blur_loss
    return loss


def generator_loss(loss_func, fake):
    fake_loss = 0

    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp' or loss_func == 'hinge':
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan':
        fake_loss = tf.reduce_mean(tf.square(fake - 1.0))

    if loss_func == 'gan' or loss_func == 'dragan':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(fake), logits=fake))

    return fake_loss


def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    GramMatrix = tf.matmul(tf.transpose(x, [0, 2, 1]), x)
    GramMatrixNorm = GramMatrix / tf.cast((tf.size(x) // b), tf.float32)
    return GramMatrixNorm


def content_loss(vgg, real, fake):
    vgg.build(real)
    real_feature_map = vgg.conv4_3_no_activation

    vgg.build(fake)
    fake_feature_map = vgg.conv4_3_no_activation

    return L1_loss(real_feature_map, fake_feature_map)


def style_loss(style, fake):
    return L1_loss(gram(style), gram(fake))


def content_style_loss(vgg, real, anime, fake):
    vgg.build(real)
    real_feature_map = vgg.conv4_3_no_activation

    vgg.build(fake)
    fake_feature_map = vgg.conv4_3_no_activation

    vgg.build(anime[:fake_feature_map.shape[0]])
    anime_feature_map = vgg.conv4_3_no_activation

    contentLoss = L1_loss(real_feature_map, fake_feature_map)
    styleLoss = style_loss(anime_feature_map, fake_feature_map)

    return contentLoss, styleLoss


def color_loss(content, fake):
    content = rgb2yuv(content)
    fake = rgb2yuv(fake)

    YLoss = L1_loss(content[:, :, :, 0], fake[:, :, :, 0])
    ULoss = Huber_loss(content[:, :, :, 1], fake[:, :, :, 1])
    VLoss = Huber_loss(content[:, :, :, 2], fake[:, :, :, 2])
    return YLoss + ULoss + VLoss


def rgb2yuv(rgb):
    rgb = (rgb + 1.0) / 2.0
    return tf.image.rgb_to_yuv(rgb)
