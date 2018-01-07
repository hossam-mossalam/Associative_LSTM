import numpy as np
import tensorflow as tf
from numpy.random import permutation
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.math_ops import sigmoid, tanh

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class _LayerRNNCell(rnn_cell.RNNCell):
  """Subclass of RNNCells that act like proper `tf.Layer` objects.

  For backwards compatibility purposes, most `RNNCell` instances allow their
  `call` methods to instantiate variables via `tf.get_variable`.  The underlying
  variable scope thus keeps track of any variables, and returning cached
  versions.  This is atypical of `tf.layer` objects, which separate this
  part of layer building into a `build` method that is only called once.

  Here we provide a subclass for `RNNCell` objects that act exactly as
  `Layer` objects do.  They must provide a `build` method and their
  `call` methods do not access Variables `tf.get_variable`.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size, s] for s in self.state_size`.
      scope: `VariableScope` for the created subgraph; if not provided,
        defaults to standard `tf.layers.Layer` behavior.

    Returns:
      A pair containing:

      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    # Bypass RNNCell's variable capturing semantics for LayerRNNCell.
    # Instead, it is up to subclasses to provide a proper build
    # method.  See the class docstring for more details.
    return base_layer.Layer.__call__(self, inputs, state, scope=scope)

class AssociativeLSTMCell(_LayerRNNCell):

  """Associative LSTM Recurrent Unit cell

  (cf. https://arxiv.org/abs/1602.03032).

  """

  def __init__(self, cell_size, num_copies, input_keys = 1, output_keys = 1,
               initializer=None, num_proj=None, forget_bias=1.0,
               state_is_tuple=True, activation = None, reuse = None,
               name = None):
    """Initialize the parameters for an Associative LSTM cell.

    Args:
      cell_size: int, The number of units per copy in the ALSTM cell
      num_copies: int, The number of memory copies in the ALSTM cell
      input_keys: int, The number of inputs to be used.
      output_keys: int, The number of outputs to be used.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      num_unit_shards: How to split the weight matrix.  If >1, the weight
        matrix is stored across num_unit_shards.
      activation: Activation function of the inner states.
    """
    super(AssociativeLSTMCell, self).__init__(_reuse=reuse, name=name)

    if cell_size % 2 != 0:
      raise ValueError("cell_size must be an even number")

    self._cell_size = cell_size
    self._num_copies = num_copies
    self._input_keys = input_keys
    self._output_keys = output_keys
    self._initializer = initializer
    self._num_proj = num_proj

    # Generating key permutations for each copy.
    self._permutations = np.array(
            [permutation(self._cell_size // 2) for _
                 in range(self._num_copies)])
    self._permutations = tf.concat([self._permutations,
                                    self._permutations + self._cell_size // 2],
                                    axis = 1, name = 'concat6')

    if num_proj:
      if num_proj % output_keys != 0:
        raise ValueError("num_proj must be divisible by output_keys")
      self._state_size = (rnn_cell.LSTMStateTuple(cell_size, num_proj))
      self._output_size = num_proj
    else:
      num_proj = cell_size * output_keys
      self._state_size = (rnn_cell.LSTMStateTuple(cell_size, cell_size))
      self._output_size = cell_size

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  @property
  def memory_copies(self):
    return self._num_copies

  def _permute(self, input_, scope = None):
    '''
    Permutes input_ based on self._permutations.

    Args:
    input_ : 3-d tensor of dimensions : #keys x batch_size x cell_size

    Returns:
    output : 4-d tensor of dimensions :
        #keys x num_copies x batch_size x cell_size
    '''
    with tf.variable_scope(scope or type(self).__name__,
                           initializer=self._initializer):
      output = [[tf.gather(e, self._permutations) for e in tf.unstack(key)]
                    for key in tf.unstack(input_)]
      output = tf.transpose(output, [0,2,1,3])
      return output

  def _complex_multiplication(self, u, v, scope = None):
    '''
    Returns the mutliplication of the two complex numbers u and v.

    Args:
    u: a 4-d tensor
    v: a 4-d tensor

    Returns:
    output: a 4-d tensor produced by the complex multiplication of u and v in
        the 4th dimension.
    '''
    with tf.variable_scope(scope or type(self).__name__,
                           initializer=self._initializer):
      u_r = u[:,:,:,:self._cell_size//2]
      u_i = u[:,:,:,self._cell_size//2:]
      v_r = v[:,:,:,:self._cell_size//2]
      v_i = v[:,:,:,self._cell_size//2:]
      out_r = u_r * v_r - u_i * v_i
      out_i = u_r * v_i + u_i * v_r
      output = tf.concat([out_r, out_i], 3, name = 'concat1')
      return output

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value

    num_proj = self._cell_size if self._num_proj is None else self._num_proj

    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + num_proj,
          int((2.5 + self._input_keys + self._output_keys) * self._cell_size)],
          initializer=self._initializer)

    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[int((2.5 + self._input_keys + self._output_keys) * self._cell_size)],
        initializer=tf.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state, scope=None):
    """Run one step of Associative LSTM.

    Args:
      inputs: input Tensor, 2D, batch x cell_size.
      state: a tuple of state Tensors, both `2-D`, with column sizes `c_state`
          and `m_state`.
      scope: VariableScope for the created subgraph; defaults to
          "AssociativeLSTMCell".

    Returns:
      A tuple containing:

      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           cell_size otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    num_proj = self._cell_size if self._num_proj is None else self._num_proj

    (c_prev, m_prev) = state

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]

    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    # bs x (input_size + num_proj)
    cell_inputs = tf.concat([inputs, m_prev], 1, name = 'concat2')

    # bs x ((2.5 + _input_keys + _output_keys) * cell_size)
    lstm_matrix = tf.matmul(cell_inputs, self._kernel)
    lstm_matrix = tf.nn.bias_add(lstm_matrix, self._bias)

    # i = input_gate, f = forget_gate, o = output_gate
    # bs x (cell_size // 2)
    i, f, o = tf.split(value = lstm_matrix[:, :int(1.5 * self._cell_size)],
                       axis = 1, num_or_size_splits = 3)

    # u
    # bs x cell_size
    u = tf.split(lstm_matrix[:, int(1.5 * self._cell_size):int(2.5 * self._cell_size)],
                 axis = 1, num_or_size_splits = 1)[0]

    # ri
    # _input_keys x bs x cell_size
    input_keys = tf.split(lstm_matrix[:,
                          int(2.5 * self._cell_size):
                          int((2.5 + self._input_keys) * self._cell_size)],
                          axis = 1, num_or_size_splits = 1)[0]
    input_keys = tf.reshape(input_keys,
        [self._input_keys, -1, self._cell_size])

    # ro
    # _output_keys x bs x cell_size
    output_keys = tf.split(lstm_matrix[:,
                              int((2.5 + self._input_keys) * self._cell_size):],
                              axis = 1, num_or_size_splits = 1)[0]
    output_keys = tf.reshape(output_keys,
        [self._output_keys, -1, self._cell_size])

    # applying the sigmoid activation function
    # bs x (cell_size // 2)
    i = sigmoid(i)
    f = sigmoid(f)
    o = sigmoid(o)

    # appending gates
    # bs x cell_size
    i = tf.concat([i, i], 1, name = 'concat3')
    f = tf.concat([f, f], 1, name = 'concat4')
    o = tf.concat([o, o], 1, name = 'concat5')

    # applying tanh activation function
    # bs x cell_size
    u = tanh(u)
    # _input_keys x bs x cell_size
    input_keys = tanh(input_keys)
    # _output_keys x bs x cell_size
    output_keys = tanh(output_keys)

    # applying permutations
    #_input_keys x num_copies x batch_size x cell_size
    input_keys = self._permute(input_keys, scope = 'input_keys')
    #_output_keys x num_copies x batch_size x cell_size
    output_keys = self._permute(output_keys, scope = 'output_keys')

    # memory copies update
    # num_copies x bs x cell_size
    memory_update = self._complex_multiplication(
        input_keys, tf.expand_dims(tf.expand_dims(u * i, 0), 0))
    memory_update = tf.reduce_mean(memory_update, 0)

    # memory copies forget
    # num_copies x bs x cell_size
    memory_forget = tf.expand_dims(f, 0) * c_prev

    # updating memory
    # num_copies x bs x cell_size
    c = memory_forget + memory_update

    # reading refers to the reading gate
    # _output_keys x bs x cell_size
    reading_gate = tanh(tf.reduce_mean(
        self._complex_multiplication(output_keys, tf.expand_dims(c, 0)), 1))

    # bs x num_proj
    m = tf.expand_dims(o, 0) * reading_gate
    m = tf.transpose(m, [1,0,2])
    m = tf.reshape(m, [-1, self._num_proj])

    new_state = rnn_cell.LSTMStateTuple(c, m)

    return m, new_state

  def zero_state(self, batch_size, dtype):
    c = tf.zeros([self._num_copies, batch_size, self._cell_size], dtype = dtype)
    h = tf.zeros([batch_size, self._cell_size * self._output_keys],
                 dtype = dtype)
    return rnn_cell.LSTMStateTuple(c, h)
