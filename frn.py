from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects


class FilterResponseNormNd(Layer):
    def __init__(self, ndim, num_features, eps=1e-6,
                 learnable_eps=False, **kwargs):
        super(FilterResponseNormNd, self).__init__(**kwargs)
        assert ndim in [3, 4, 5], \
            'FilterResponseNorm for '+str(ndim)+'d not implemented.'

        shape = (1, ) + (1, ) * (ndim - 2) + (num_features,)
        self.eps = tf.Variable(initial_value=tf.ones(shape=shape)*eps, trainable=learnable_eps)
        self.build(shape)

    def build(self, shape):

        self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer='ones')

        self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer='zeros')

        self.tau = self.add_weight(shape=shape,
                                    name='tau',
                                    initializer='zeros')

    def call(self, x, **kwargs):

        tensor_input_shape = K.int_shape(x)
        avg_dims = tuple(range(1, len(tensor_input_shape)-1))
        nu2 = Lambda(lambda x : K.mean(tf.pow(x, 2),axis=avg_dims, keepdims=True))(x)
        xs = Lambda(lambda x : tf.math.rsqrt(tf.add(nu2,tf.math.abs(self.eps))))(nu2)
        x = Multiply()([xs,x])
        
        return tf.math.maximum(self.gamma * x + self.beta, self.tau)

    def get_config(self):
        config = {
            'learnable_eps': self.learnable_eps,
            'eps': self.eps
        }
        base_config = super(FilterResponseNormNd, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class FilterResponseNorm1d(FilterResponseNormNd):

    def __init__(self, num_features, eps=1e-6, learnable_eps=False, **kwargs):
        super(FilterResponseNorm1d, self).__init__(
            3, num_features, eps=eps, learnable_eps=learnable_eps, **kwargs)


class FilterResponseNorm2d(FilterResponseNormNd):

    def __init__(self, num_features, eps=1e-6, learnable_eps=False, **kwargs):
        super(FilterResponseNorm2d, self).__init__(
            4, num_features, eps=eps, learnable_eps=learnable_eps, **kwargs)


class FilterResponseNorm3d(FilterResponseNormNd):

    def __init__(self, num_features, eps=1e-6, learnable_eps=False, **kwargs):
        super(FilterResponseNorm3d, self).__init__(
            5, num_features, eps=eps, learnable_eps=learnable_eps, **kwargs)

def install_frn():
    get_custom_objects().update({'FilterResponseNorm1d': FilterResponseNorm1d})
    get_custom_objects().update({'FilterResponseNorm2d': FilterResponseNorm2d})
    get_custom_objects().update({'FilterResponseNorm3d': FilterResponseNorm3d})
