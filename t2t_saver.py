"""Decode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import layers
from tensor2tensor.models import transformer
import problems
import tensorflow as tf
import decoding
import os
from tensor2tensor.bin import t2t_decoder
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import registry

flags = tf.flags
FLAGS = flags.FLAGS

if __name__ == '__main__':
  # tf.logging.set_verbosity(tf.logging.INFO)
  # tf.app.run(t2t_decoder.main)
  trainer_lib.set_random_seed(FLAGS.random_seed)

  hp, decode_hp, _ = decoding.create_hp_and_estimator(
      FLAGS.problem, FLAGS.data_dir, FLAGS.checkpoint_path
  )

  hp.model_dir = FLAGS.checkpoint_path or FLAGS.output_dir

  model_cls = t2t_decoder.registry.model(FLAGS.model)

  model = model_cls(
      hp, 
      tf.estimator.ModeKeys.EVAL,
      data_parallelism=None,
      decode_hparams=decode_hp
  )

  ckpt_path = FLAGS.checkpoint_path or FLAGS.output_dir
  import pdb; pdb.set_trace()
  model.initialize_from_ckpt(ckpt_path)
  
  # features = {
  #         "inputs": layers.Input(shape=[1, 128]),
  #         "targets": layers.Input(shape=[1, 128]),
  #         "target_space_id": pass
  # }

  # predictions, _ = this_model(features)

  # model = keras.Model(inputs, predictions)
  # model.save('Translation_Vien_Base1m')
