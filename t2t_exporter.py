#!/usr/bin/env python
"""t2t-exporter."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.utils import registry


from tensor2tensor.serving import export
import tensorflow as tf
import problems

@registry.register_hparams
def transformer_tall9():
  hparams = transformer.transformer_big()
  hparams.hidden_size = 768
  hparams.filter_size = 3072
  hparams.num_hidden_layers = 9
  hparams.num_heads = 12
  return hparams

def main(argv):
  export.main(argv)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
