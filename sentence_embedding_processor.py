from abc import abstractmethod

from stanza import Document
from stanza.models.common.doc import Sentence
from stanza.pipeline.processor import Processor, register_processor
import numpy as np
from sentence_transformers import SentenceTransformer


def set_embedding(sentence: Sentence, embedding: np.ndarray[float]):
    """For the custom embedding property of Sentence"""
    sentence._embedding = embedding


# Add a custom embedding property to Sentence.
Sentence.add_property('embedding', default=None,
                      getter=lambda self: self._embedding,
                      setter=set_embedding)


class SentenceEmbeddingProcessor(Processor):
    """
    An abstract base stanza processor for creating sentence embeddings
    for each sentence in the document.
    """

    def __init__(self, config, pipeline, device):
        super().__init__(config, pipeline, device)

    def process(self, doc: Document) -> Document:
        for sentence in doc.sentences:
            embedding = self.vectorize(sentence)
            sentence.embedding = embedding
        return doc

    @abstractmethod
    def vectorize(self, sentence: Sentence) -> np.ndarray[float]:
        """
        Create a sentence embedding from the embedding.

        :param sentence: the sentence.
        :return: an embedding as a numpy array.
        """
        raise NotImplementedError()


@register_processor('sbert-embedder')
class SBERTProcessor(SentenceEmbeddingProcessor):
    """
    A sentence embedding processor for the stanza pipeline, which creates
    sentence embeddings using a Swedish SBERT model.
    """

    REQUIRES_DEFAULT = {'tokenize'}
    PROVIDES_DEFAULT = {'sbert-embedder'}

    def __init__(self, config, pipeline, device):
        super().__init__(config, pipeline, device)

        self.model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')

    def vectorize(self, sentence: Sentence) -> np.ndarray[float]:
        return self.model.encode(sentence.text, convert_to_numpy=True)
