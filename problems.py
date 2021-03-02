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





_11CLASSES_ENVI_DATASETS = [
    ['', ('train.en.fixed.append128.tag', 'train.vi.fixed.append128.tag')],  # original.
    ['', ('medical.en.fixed.train.append128.tag', 'medical.vi.fixed.train.append128.tag')],
    ['', ('Gnome.en.fixed.filter.train.tag', 'Gnome.vi.fixed.filter.train.tag')],
    ['', ('Kde4.en.fixed.filter.train.tag', 'Kde4.vi.fixed.filter.train.tag')],
    ['', ('open_sub_en.txt.fixed.tag', 'open_sub_vi.txt.fixed.tag')],
    ['', ('law.en.fixed.filter.fixed.tag', 'law.vi.fixed.filter.fixed.tag')],
    ['', ('fbwiki.en.fixed.tag', 'fbwiki.vi.fixed.tag')],
    ['', ('qed.en.fixed.tag', 'qed.vi.fixed.tag')],
    ['', ('Ubuntu_tp_en.txt.fixed.filter.train.tag', 'Ubuntu_tp_vi.txt.fixed.filter.train.tag')],
    ['', ('ELRC_2922.en-vi.en.fixed.append128.tag', 'ELRC_2922.en-vi.vi.fixed.append128.tag')],
    ['', ('bible_uedin.en.fixed.append128.tag', 'bible_uedin.vi.fixed.append128.tag')],
    ['', ('lyric_en.txt.fixed.append128.tag', 'lyric_vi.txt.fixed.append128.tag')],
    ['', ('m21book_add2train.en.fixed.append128.tag', 'm21book_add2train.vi.fixed.append128.tag')],
    ['', ('tatoeba.en.fixed.append128.tag', 'tatoeba.vi.fixed.append128.tag')],
    ['', ('ted2020.en.fixed.filter.fixed.append128.tag', 'ted2020.vi.fixed.filter.fixed.append128.tag')],
    ['', ('vnsn.en.filter.fixed.append128.tag', 'vnsn.vi.filter.fixed.append128.tag')],
    ['', ('youtube.fixed.en.fixed.append128.tag', 'youtube.fixed.vi.fixed.append128.tag')],
    ['', ('youtube.teded.en.fixed.filter.fixed.append128.tag', 'youtube.teded.vi.fixed.filter.fixed.append128.tag')],
]



_11CLASSES_VIEN_DATASETS = [
    [url, (vi, en)] for [url, (en, vi)] in _11CLASSES_ENVI_DATASETS
]



@registry.register_problem
class TranslateClass11VienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _11CLASSES_VIEN_DATASETS if train else _VIEN_TEST_DATASETS




@registry.register_problem
class TranslateClass11EnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _11CLASSES_ENVI_DATASETS if train else _ENVI_TEST_DATASETS



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


_FB_WIKI_K760ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_k760train.en", "fb_wiki_k760train.vi")
]]


@registry.register_problem
class FbWikiK760TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_K760ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_M6_AVG61_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.avg61.en", "train.6m.avg61.vi")
]]


@registry.register_problem
class M6Avg61TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_AVG61_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_M6_AVG25_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.avg25.en", "train.6m.avg25.vi")
]]


@registry.register_problem
class M6Avg25TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_AVG25_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_M6_LONGEST_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.longest.en", "train.6m.longest.vi")
]]


@registry.register_problem
class M6LongestTranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_LONGEST_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_M6_SHORTEST_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.shortest.en", "train.6m.shortest.vi")
]]


@registry.register_problem
class M6ShortestTranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_SHORTEST_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_M6_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m6train.en", "fb_wiki_m6train.vi")
]]


@registry.register_problem
class FbWikiM6TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M6_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_M12_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m12train.en", "fb_wiki_m12train.vi")
]]


@registry.register_problem
class FbWikiM12TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M12_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_M18_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m18train.en", "fb_wiki_m18train.vi")
]]


@registry.register_problem
class FbWikiM18TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M18_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_M30_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m30train.en", "fb_wiki_m30train.vi")
]]


@registry.register_problem
class FbWikiM30TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M30_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_FB_WIKI_BESTM30_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_bestm30train.en", "fb_wiki_bestm30train.vi")
]]


@registry.register_problem
class FbWikiBestm30TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_BESTM30_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_BOOK_M30_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("book_m30train.en", "book_m30train.vi")
]]


@registry.register_problem
class Bookm30TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _BOOK_M30_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_AND_BOOK_M45_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_and_book_m45train.en", "fb_wiki_and_book_m45train.vi")
]]


@registry.register_problem
class FbWikiAndBookM45TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_AND_BOOK_M45_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

###################################### VI-EN #######################################

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



_FB_WIKI_K760VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_k760train.vi", "fb_wiki_k760train.en")
]]


@registry.register_problem
class FbWikiK760TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_K760VIEN_TRAIN_DATASETS if train else _VIEN_TEST_DATASETS


_M6_AVG61_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.avg61.vi", "train.6m.avg61.en")
]]


@registry.register_problem
class M6Avg61TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_AVG61_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_M6_AVG25_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.avg25.vi", "train.6m.avg25.en")
]]


@registry.register_problem
class M6Avg25TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_AVG25_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS



_M6_LONGEST_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.longest.vi", "train.6m.longest.en")
]]


@registry.register_problem
class M6LongestTranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_LONGEST_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_M6_SHORTEST_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.shortest.vi", "train.6m.shortest.en")
]]


@registry.register_problem
class M6ShortestTranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_SHORTEST_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_M6_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m6train.vi", "fb_wiki_m6train.en")
]]


@registry.register_problem
class FbWikiM6TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M6_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_M12_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m12train.vi", "fb_wiki_m12train.en")
]]


@registry.register_problem
class FbWikiM12TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M12_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_M18_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m18train.vi", "fb_wiki_m18train.en")
]]


@registry.register_problem
class FbWikiM18TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M18_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_FB_WIKI_M30_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m30train.vi", "fb_wiki_m30train.en")
]]


@registry.register_problem
class FbWikiM30TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M30_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_FB_WIKI_BESTM30_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_bestm30train.vi", "fb_wiki_bestm30train.en")
]]


@registry.register_problem
class FbWikiBestm30TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_BESTM30_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_BOOK_M30_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("book_m30train.vi", "book_m30train.en")
]]


@registry.register_problem
class Bookm30TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _BOOK_M30_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_FB_WIKI_AND_BOOK_M45_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_and_book_m45train.vi", "fb_wiki_and_book_m45train.en")
]]


@registry.register_problem
class FbWikiAndBookM45TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_AND_BOOK_M45_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

