"""
Tokenize the text using the
"""

from stanza.models.common.doc import Sentence
from stanza import Document, Pipeline, DownloadMethod

import lsa
import sentence_embedding_processor  # import in order to register processor.
from paragraph import Paragraph
import numpy as np
import pprint as pp


class SBERTLSA:
    """Parses and analyzes a text with Latent Semantic Analysis. A Swedish SBERT model
    is used for creating the sentence embeddings."""

    def __init__(self):
        self.parser = Pipeline(lang='sv', processors='tokenize, sbert-embedder',
                               download_method=DownloadMethod.REUSE_RESOURCES)

    def compute_metrics(self, text:str) -> dict[str, float]:
        """
        Compute each LSA metric on the text.

        :param text:
        :return:
        """
        doc = self.parser(text)  # type: Document
        results = {}  # type: dict[str, float]

        results['LSASS1'], results['LSASS1d'] = adjacent_sentences_overlap(doc.sentences)
        results['LSASSa'], results['LSASSad'] = all_sentences_overlap(doc.sentences)
        # results['LSASSp'], results['LSASSpd'] = all_sentences_overlap_paragraph(doc.paragraphs)
        # results['LSAPP1'], results['LSAPP1d'] = adjacent_paragraphs_overlap(doc.paragraphs)
        results['LSAGN'], results['LSAGNd'] = given_new(doc.sentences)

        return results


def adjacent_sentences_overlap(sentences:list[Sentence]) -> tuple[float, float]:
    """
    LSASS1: LSA overlap, adjacent sentences, mean
    LSASS1d: LSA overlap, adjacent sentences, standard deviation
    :param sentences:
    :return:
    """
    embeddings = get_sentences_embeddings(sentences)
    return lsa.lsa_adjacent_overlap(embeddings)


def all_sentences_overlap(sentences: list[Sentence]) -> tuple[float, float]:
    """
    LSASSa: LSA overlap, all sentences, mean.
    LSASSad: LSA overlap, all sentences, standard deviation.

    Note: this is not an official Coh-Metrix index.

    :param sentences:
    :return:
    """
    embeddings = get_sentences_embeddings(sentences)
    return lsa.lsa_all_overlap(embeddings)


def all_sentences_overlap_paragraph(paragraphs: list[Paragraph]) -> tuple[float, float]:
    """
    LSASSp: LSA overlap, all sentences in paragraph, mean.
    LSASSpd: LSA overlap, all sentences in paragraph, standard deviation.

    :param paragraphs:
    :return: a tuple (mean, std)
    """
    overlaps = [all_sentences_overlap(paragraph.sentences) for paragraph in paragraphs]
    return np.mean(overlaps), np.std(overlaps)


def adjacent_paragraphs_overlap(paragraphs: list[Paragraph]) -> tuple[float, float]:
    """
    LSAPP1: LSA overlap, adjacent paragraphs, mean.
    LSAPP1d: LSA overlap, adjacent paragraphs, standard deviation.

    :param paragraphs:
    :return:
    """
    raise NotImplementedError()


def given_new(sentences:list[Sentence]) -> tuple[float, float]:
    """
    LSAGN: LSA given/new, sentences, mean
    LSAGNd: LSA given/new, sentences, standard deviation

    :param sentences:
    :return:
    """
    embeddings = get_sentences_embeddings(sentences)
    return lsa.lsa_given_new(embeddings)


# UTILITY FUNCTIONS ----------------------------------------------

def get_sentences_embeddings(sentences: list[Sentence]) -> list[lsa.Embedding]:
    """Return a list of sentence embeddings for the sentences."""
    return [sentence.embedding for sentence in sentences]


# TESTING --------------------------------------------------------


def test_sbert_lsa():
    sbert_lsa = SBERTLSA()
    text = "Bakterier (Bacteria) eller eubakterier (Eubacteria) är encelliga mikroorganismer utan cellkärna " \
           "och andra membranomslutna organeller; de har dock ribosomer. Bakterier räknas till prokaryoterna " \
           "som även inkluderar domänen arkéer. Bakterier är vanligtvis ett antal mikrometer långa och väger " \
           "ett antal hundra femtogram. Bakterier kan ha ett varierande utseende, bland annat formade som sfärer, " \
           "spiraler (helix) eller stavar. Studier av bakterier kallas för bakteriologi och är en gren inom " \
           "mikrobiologin. Bakterier kan hittas i alla ekosystem på jorden, i varma källor, bland radioaktivt " \
           "avfall, i havsvatten och djupt ned i jordskorpan. Vissa bakterier kan till och med överleva i extrem " \
           "kyla och i vakuum. I genomsnitt finns 40 miljoner bakterier i ett gram jord och en miljon bakterier i " \
           "en milliliter färskvatten."

    metrix = sbert_lsa.compute_metrics(text)
    pp.pprint(metrix)


if __name__ == '__main__':
    test_sbert_lsa()
