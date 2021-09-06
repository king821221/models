import tensorflow as tf

from official.nlp.transformer.model_utils import map_updates_to_vocab 
from official.nlp.transformer.model_utils import tf_print


##########
# Copy Stack specifically for copy mechanism in tasks as text summarization
# CopyGenerator reference: https://github.com/OpenNMT/OpenNMT-py/blob/fb14f48cb4e185c5169183a65ae26e532e522764/onmt/modules/copy_generator.py
##########
class CopyStack(tf.keras.layers.Layer):
  """Transformer copy mechanism stack.

  The copy mechanism is to generate logits per word in V+X based on a combination
  of P(g) + P(c), P(g) is the generation probability, P(c) is the copy probability

  """

  def __init__(self, params):
    super(CopyStack, self).__init__()
    self.params = params

  def build(self, input_shape):
    """Builds the copy stack."""
    params = self.params
    activation = params.get('copy_activation') or tf.nn.tanh
    self.copy_projection = tf.keras.layers.Dense(params["hidden_size"],
                                                 activation=activation,
                                                 name='copy_projection')
    super(CopyStack, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self,
           inputs,
           encoder_outputs,
           decoder_outputs,
           input_bias=None,
           **kwargs):
    """Return the output of the copy mechanism.

    Args:
      decoder_outputs: A tensor with shape [batch_size, target_length,
        hidden_size].
      encoder_outputs: A tensor with shape [batch_size, input_length,
        hidden_size]
      generate_logits: A tensor with shape [batch_size, target_length, vocab_size]
      inputs: A tensor with shape [batch_size, input_length]
      input_bias: A tensor with shape [batch_size, .., input_length], the
        bias for encoder-decoder attention layer.
      training: A bool, whether in training mode or not.

    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """
    training = kwargs['training']
    print("copy stack encoder outputs {}".format(encoder_outputs.shape))
    encoder_outputs = self.copy_projection(encoder_outputs, training=training)
    # encoder_outputs: [batch_size, hidden_size, input_length]
    encoder_outputs = tf.transpose(encoder_outputs, (0, 2, 1))
    # enc_dec_c_f: [batch_size, target_length, input_length]
    enc_dec_c_f = tf.matmul(decoder_outputs, encoder_outputs)
    print("copy stack enc_dec_c_f {}".format(enc_dec_c_f.shape))
    enc_dec_c_f = tf_print(enc_dec_c_f, 'copy_stack_enc_dec_c_f')
    if input_bias is not None:
      input_bias = tf.reshape(input_bias, [tf.shape(input_bias)[0], 1, tf.shape(input_bias)[-1]])
      enc_dec_c_f += input_bias 
    enc_dec_c_f = tf_print(enc_dec_c_f, 'copy_stack_enc_dec_c_f_biased')
    tf.summary.histogram('copy_gen_enc_dec_c_f', enc_dec_c_f)
    prob_fn = self.params.get('prob_fn') or tf.nn.softmax
    enc_dec_probs = prob_fn(enc_dec_c_f)
    enc_dec_probs = tf_print(enc_dec_probs, 'copy_stack_enc_dec_probs')
    # copy_probs: [batch_size, target_length, vocab_size]
    print("copy stack enc_dec_probs {} input sequence {}".format(enc_dec_probs, inputs))
    tf.summary.histogram('copy_gen_enc_dec_probs', enc_dec_probs)
    copy_probs = map_updates_to_vocab(enc_dec_probs, inputs, self.params['vocab_size'])
    copy_probs = tf_print(copy_probs, 'copy_stack_copy_probs')
    return copy_probs

class GenCopyMixer(tf.keras.layers.Layer):

  def __init__(self, params):
    super(GenCopyMixer, self).__init__()
    self.params = params

  def build(self, input_shape):
    """Builds the copy stack."""
    params = self.params
    activation = params.get('gen_prob_activation') or tf.nn.sigmoid
    self.gen_prob_projection = tf.keras.layers.Dense(1,
                                                     activation=activation,
                                                     name='gen_prob_projection')
    super(GenCopyMixer, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def compute_mix_prob(self, decoder_outputs, **kwargs):
    mix_prob = self.gen_prob_projection(decoder_outputs)
    return mix_prob

  def call(self,
           generative_probs,
           copy_probs,
           decoder_outputs,
           **kwargs):
    mix_prob_r = self.compute_mix_prob(decoder_outputs, **kwargs)
    print("Mix gen with copy with prob {}".format(mix_prob_r.shape))
    mix_prob_r = tf_print(mix_prob_r, 'mix_prob_r')
    tf.summary.histogram('copy_gen_mix_prob', mix_prob_r)
    tf.summary.histogram('copy_gen_copy_probs', copy_probs)
    tf.summary.histogram('copy_gen_gen_probs', generative_probs)
    return mix_prob_r * generative_probs + (1 - mix_prob_r) * copy_probs 
