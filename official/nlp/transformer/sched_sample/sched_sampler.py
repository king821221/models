import tensorflow as tf

class FlipThresholdDecayFunc(object):

    def __init__(self):
        pass

    def __call__(self, steps, **kwargs):
        return self.call(steps, **kwargs)

    def call(self, steps, **kwargs):
        raise NotImplementedError

class InverseSigmoidDecay(FlipThresholdDecayFunc):

    def __init__(self, k):
        self.k = k
        assert self.k > 1

    def call(self, steps, **kwargs):
        return self.k/(self.k + tf.exp(steps/self.k))

class ExponentialDecay(FlipThresholdDecayFunc):

    def __init__(self, k):
        self.k = k
        assert self.k < 1

    def call(self, steps, **kwargs):
        return tf.pow(self.k, steps)

class LinearDecay(FlipThresholdDecayFunc):

    def __init__(self, gamma, k, c):
        self.gamma = gamma
        self.k = k
        self.c = c

    def call(self, steps, **kwargs):
        return tf.maximum(self.gamma, self.k - self.c * steps)

def create_sched_sample_prob_thr_fn(params):
    prob_thr_fn_k = params['prob_thr_fn']
    if prob_thr_fn_k == 'inverse_sigmoid':
      k = params['inverse_sigmoid_k']
      return InverseSigmoidDecay(k)
    elif prob_thr_fn_k == 'linear':
      k = params['linear_k']
      c = params['linear_c']
      gamma = params['linear_gamma']
      return LinearDecay(gamma, k, c)
    elif prob_thr_fn_k == 'exp':
      k = params['exp_k']
      return ExponentialDecay(k)
    raise KeyError('Invalid prob threshold func {}'.format(prob_thr_fn_k))

class ScheduledSampler(tf.keras.layers.Layer):

    def __init__(self, params):
      self.prob_thr_func = create_sched_sample_prob_thr_fn(params)
      super(ScheduledSampler, self).__init__()

    def call(self,
             target, # [batch_size, target_sequence_length]
             logits,  # [batch_size, target_sequence_length, vocab_size]
             **kwargs):
      # convert logits to log probs
      log_probs = tf.nn.log_softmax(logits)
      # samples: [batch_size, target_sequence_length, 1]
      log_probs_r = tf.reshape(log_probs, [-1, tf.shape(log_probs)[-1]])
      samples = tf.random.categorical(log_probs_r, 1)
      # samples: [batch_size, target_sequence_length]
      samples = tf.reshape(samples, [tf.shape(log_probs)[0], tf.shape(log_probs)[1]])
      # flip_threshold: [1, target_sequence_length]
      flip_threshold = self.prob_thr_func(tf.range(tf.shape(target)[1])+1, **kwargs)
      flip_threshold = tf.expand_dims(flip_threshold, 0)
      # choose_prob: [batch_size, target_sequence_length]
      choose_prob = tf.random.uniform((tf.shape(samples)[0],
                                       tf.shape(samples)[1]))
      samples = tf.cast(samples, target.dtype)
      choose_prob = tf.cast(choose_prob, logits.dtype)
      flip_threshold = tf.cast(flip_threshold, logits.dtype)
      return tf.where(tf.less(choose_prob, flip_threshold),
                      target,
                      samples)

