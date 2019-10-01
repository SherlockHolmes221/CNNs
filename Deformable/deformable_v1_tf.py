import tensorflow as tf
from keras.initializers import RandomNormal
from keras.layers import Conv2D


class ConvOffset2D(Conv2D):
    """
    ConvOffset2D卷积层学习2D的偏移量，使用双线性插值输出变形后采样值
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init

        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2D layer in Keras
        """

        self.filters = filters
        # 注意通道数翻倍，输出的特征图表示偏移量x,y
        super(ConvOffset2D, self).__init__(
            self.filters * 2, (3, 3), padding='same', use_bias=False,
            kernel_initializer=RandomNormal(0, init_normal_stddev),
            **kwargs
        )

    def call(self, x):
        """Return the deformed featured map"""
        x_shape = x.get_shape()

        # 卷积输出得到2倍通道的feature map，获取到偏移量 大小为(batch,h,w,2c)
        offsets = super(ConvOffset2D, self).call(x)

        # offsets reshape成: (b*c, h, w, 2) 共有b*c个map.大小为h,w
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # 将输入x也切换成这样: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # 双线性采样得到采样后的X_offset: (b*c, h, w)
        x_offset = self._tf_batch_map_offsets(x, offsets)

        # 再变原本的shape，即x_offset: (b, h, w, c)
        x_offset = self._to_b_h_w_c(x_offset, x_shape)

        return x_offset

    def compute_output_shape(self, input_shape):
        """Output shape is the same as input shape

        Because this layer does only the deformation part
        """
        return input_shape

    def _to_bc_h_w_2(self, x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2]), 2))
        return x

    def _to_bc_h_w(self, x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2])))
        return x

    def _to_b_h_w_c(self, x, x_shape):
        """(b*c, h, w) -> (b, h, w, c)"""
        x = tf.reshape(
            x, (-1, int(x_shape[3]), int(x_shape[1]), int(x_shape[2]))
        )
        x = tf.transpose(x, [0, 2, 3, 1])
        return x

    def _tf_batch_map_offsets(self, input, offsets, order=1):
        """Batch map offsets into input

        Parameters
        ---------
        input : tf.Tensor. shape = (b, s, s)
        offsets: tf.Tensor. shape = (b, s, s, 2)

        Returns
        -------
        tf.Tensor. shape = (b, s, s)
        """

        input_shape = tf.shape(input)
        batch_size = input_shape[0]
        input_size = input_shape[1]

        offsets = tf.reshape(offsets, (batch_size, -1, 2))
        grid = tf.meshgrid(
            tf.range(input_size), tf.range(input_size), indexing='ij'
        )
        grid = tf.stack(grid, axis=-1)
        grid = tf.cast(grid, 'float32')
        grid = tf.reshape(grid, (-1, 2))
        grid = self._tf_repeat_2d(grid, batch_size)
        coords = offsets + grid # 实际的采样坐标

        mapped_vals = self._tf_batch_map_coordinates(input, coords) # 双线性插值
        return mapped_vals

    def _tf_repeat_2d(self, a, repeats):
        """Tensorflow version of np.repeat for 2D"""

        assert len(a.get_shape()) == 2
        a = tf.expand_dims(a, 0)
        a = tf.tile(a, [repeats, 1, 1])
        return a

    def _tf_batch_map_coordinates(self,input, coords, order=1):
        """Batch version of tf_map_coordinates

        Only supports 2D feature maps

        Parameters
        ----------
        input : tf.Tensor. shape = (b, s, s)
        coords : tf.Tensor. shape = (b, n_points, 2)

        Returns
        -------
        tf.Tensor. shape = (b, s, s)
        """
        input_shape = tf.shape(input)
        batch_size = input_shape[0]
        input_size = input_shape[1]
        n_coords = tf.shape(coords)[1]

        # 包装加上偏移后的Position没有超过边界
        coords = tf.clip_by_value(coords, 0, tf.cast(input_size, 'float32') - 1)

        # 获取采样的四个角坐标，用于双线性插值
        coords_lt = tf.cast(tf.floor(coords), 'int32')
        coords_rb = tf.cast(tf.ceil(coords), 'int32')
        coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
        coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)

        idx = self._tf_repeat(tf.range(batch_size), n_coords)

        # 得到像素值
        def _get_vals_by_coords(input, coords):
            indices = tf.stack([
                idx, self._tf_flatten(coords[..., 0]), self._tf_flatten(coords[..., 1])
            ], axis=-1)
            vals = tf.gather_nd(input, indices)
            vals = tf.reshape(vals, (batch_size, n_coords))
            return vals

        # 获取对应坐标像素值
        vals_lt = _get_vals_by_coords(input, coords_lt)
        vals_rb = _get_vals_by_coords(input, coords_rb)
        vals_lb = _get_vals_by_coords(input, coords_lb)
        vals_rt = _get_vals_by_coords(input, coords_rt)

        # 双线性插值
        coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
        vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
        vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
        mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]

        # 返回双线性插值采样值
        return mapped_vals

    def _tf_flatten(self, a):
        """Flatten tensor"""
        return tf.reshape(a, [-1])


    def _tf_repeat(self, a, repeats, axis=0):
        """TensorFlow version of np.repeat for 1D"""
        # https://github.com/tensorflow/tensorflow/issues/8521
        assert len(a.get_shape()) == 1

        a = tf.expand_dims(a, -1)
        a = tf.tile(a, [1, repeats])
        a = self._tf_flatten(a)
        return a

