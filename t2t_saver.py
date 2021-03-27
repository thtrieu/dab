"""Decode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.models import transformer
import problems
import tensorflow as tf
import decoding
import os
import t2t_decoder
from tensor2tensor.utils import registry

flags = tf.flags
FLAGS = flags.FLAGS

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  # tf.app.run(t2t_decoder.main)

  # trainer_lib.set_random_seed(FLAGS.random_seed)

  hp = trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problem)

  decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
  decode_hp.shards = FLAGS.decode_shards
  decode_hp.shard_id = FLAGS.worker_id
  decode_in_memory = FLAGS.decode_in_memory or decode_hp.decode_in_memory
  decode_hp.decode_in_memory = decode_in_memory

  model_cls = t2t_decoder.registry.model(FLAGS.model)

  run_config = t2t_decoder.t2t_trainer.create_run_config(hp)

  model = model_cls(
      hp, 
      tf.estimator.ModeKeys.EVAL,
      data_parallelism=None,
      decode_hparams=decode_hp
  )

  model.get_scaffold_fn(FLAGS.checkpoint_path or FLAGS.output_dir)()

  model.save('Translation_Vien_Base1m')
