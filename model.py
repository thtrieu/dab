from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
import tensorflow as tf


@registry.register_model
class Transformerextratokentodecoderv2(transformer.Transformer):

  def encode(self, *args, **kwargs):
    encoder_output, encoder_decoder_attention_bias = super(
        Transformerextratokentodecoderv2, self).encode(*args, **kwargs)
    hparams = self._hparams
    batch_size = encoder_output.shape[0]
    hidden_dim = int(encoder_output.shape[-1])
    num_extras = hparams.extra_tokens
    extra_tokens = tf.get_variable(
        'extra_tokens', [1, num_extras, hidden_dim],
        initializer=tf.random_normal_initializer(0.0, hidden_dim**-0.5))
    extra_tokens = tf.repeat(
        extra_tokens, batch_size, axis=0)  # [batch, num_extras, hidden_dim]
    encoder_output = tf.concat(
        [extra_tokens, encoder_output], axis=1)  # [batch, num_extras+len, hidden_dim]
    
    encoder_decoder_attention_bias = tf.pad(
        encoder_decoder_attention_bias,
        [[0, 0], [0, 0], [0, 0], [num_extras, 0]],
        constant_values=0.0,
    )
    return encoder_output, encoder_decoder_attention_bias



@registry.register_hparams
def transformer_tall9():
  hparams = transformer.transformer_big()
  hparams.hidden_size = 768
  hparams.filter_size = 3072
  hparams.num_hidden_layers = 9
  hparams.num_heads = 12
  hparams.add_hparam("extra_tokens", 8)
  return hparams
  