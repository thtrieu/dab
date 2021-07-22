"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.bin import t2t_trainer
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
import problems
import tensorflow as tf


@registry.register_model
class TransformerExtraTokenToDecoder(transformer.Transformer):

  def __init__(self, *args, **kwargs):
    super(TransformerExtraTokenToDecoder, self).__init__(*args, **kwargs)

    def _prepare_extra_token_decoder_fn(targets, hparams, features):
      decoder_input, decoder_self_attention_bias = transformer.transformer_prepare_decoder(
          targets, hparams, features)
      # [batch, length, hidden_dim], [None, 1, length, length]

      batch_size = decoder_input.shape[0]
      hidden_dim = hparams.hidden_size
      num_extras = hparams.extra_tokens
      extra_tokens = tf.get_variable(
          'extra_tokens', [1, num_extras, hidden_dim],
          initializer=tf.random_normal_initializer(0.0, hidden_dim**-0.5))
      extra_tokens = tf.repeat(
          extra_tokens, batch_size, axis=0)  # [batch, num_extras, hidden_dim]
      decoder_input = tf.concat(
          [extra_tokens, decoder_input], axis=1)  # [batch, num_extras+len, hidden_dim]

      batch_size = decoder_self_attention_bias.shape[0]

      decoder_self_attention_bias = tf.pad(
          decoder_self_attention_bias,
          [[0, 0], [0, 0], [num_extras, 0], [num_extras, 0]],
          constant_values=0.0,
      )
      return decoder_input, decoder_self_attention_bias

    self._prepare_decoder_fn = _prepare_extra_token_decoder_fn
  
  def decode(self, *args, **kwargs):
    hparams = self._hparams
    decoder_output = super(
        TransformerExtraTokenToDecoder, self).decode(*args, **kwargs)
    return decoder_output[:, hparams.extra_tokens:, :]


@registry.register_hparams
def transformer_tall9():
  hparams = transformer.transformer_big()
  hparams.hidden_size = 768
  hparams.filter_size = 3072
  hparams.num_hidden_layers = 9
  hparams.num_heads = 12
  hparams.add_hparam("extra_tokens", 8)
  return hparams
  

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(t2t_trainer.main)
