"""
The :mod:`in2ai.play.spam` module includes utilities to load 
datasets, including methods to load and fetch popular reference datasets. 
It also features some utilities for training and testing spam detection.
"""
from ._lingspam import fetch_lingspam
from ._lingspam import create_pipelines_lingspam
from ._spambase import fetch_spambase
from ._spambase import create_pipeline_spambase


from ._core import StopWordRemovalTransformer
from ._core import LemmatizeTransformer
from ._core import DocEmbeddingVectorizer

