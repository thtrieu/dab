"""Back Translation to augment a dataset."""

from __future__ import print_function
from __future__ import division

from tensor2tensor.data_generators import translate_envi
from tensor2tensor.utils import registry


# End-of-sentence marker.
EOS = translate_envi.EOS

# For English-Vietnamese the IWSLT'15 corpus
# from https://nlp.stanford.edu/projects/nmt/ is used.



# For development 1,553 parallel sentences are used.
_VIEN_TEST_DATASETS = [[
    "https://github.com/stefan-it/nmt-en-vi/raw/master/data/dev-2012-en-vi.tgz",  # pylint: disable=line-too-long
    ("tst2012.vi", "tst2012.en")
]]

_ENVI_TEST_DATASETS = [[
    "https://github.com/stefan-it/nmt-en-vi/raw/master/data/dev-2012-en-vi.tgz",  # pylint: disable=line-too-long
    ("tst2012.en", "tst2012.vi")
]]


# The original dataset has 133K parallel sentences.
_K133VIEN_TRAIN_DATASETS = [[
    "https://github.com/stefan-it/nmt-en-vi/raw/master/data/train-en-vi.tgz",  # pylint: disable=line-too-long
    ("train.vi", "train.en")
]]

@registry.register_problem
class TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _K133VIEN_TRAIN_DATASETS if train else _VIEN_TEST_DATASETS


_K200ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("k200train.en", "k200train.vi")
]]

@registry.register_problem
class K200TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _K200ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_K500ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("k500train.en", "k500train.vi")
]]


@registry.register_problem
class K500TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _K500ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_K500ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("Fb_Wiki_k500train.en", "Fb_Wiki_k500train.vi")
]]


@registry.register_problem
class Fb_Wiki_K500TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_K500ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_K200VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("k200train.vi", "k200train.en")
]]


@registry.register_problem
class K200TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _K200VIEN_TRAIN_DATASETS if train else _VIEN_TEST_DATASETS


_K500VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("k500train.vi", "k500train.en")
]]

@registry.register_problem
class K500TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _K500VIEN_TRAIN_DATASETS if train else _VIEN_TEST_DATASETS



_FB_WIKI_K500VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("Fb_Wiki_k500train.vi", "Fb_Wiki_k500train.en")
]]


@registry.register_problem
class Fb_Wiki_K500TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_K500VIEN_TRAIN_DATASETS if train else _VIEN_TEST_DATASETS