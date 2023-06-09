"""
LSA Coh-Metrix measures using Stanza for managing the language data.
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

        :param text: a Swedish text as a string.
        :return: a dictionary of the computed metrics, where the key is metric name as a
        string, and the value is the metric as a float.
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
    The semantic overlap between each adjacent sentence pair in the text. This
    consists of LSASS1 (the mean overlap) and LSASS1d (the standard deviation overlap).

    :param sentences: a list of sentences making up the text.
    :return: the LSASS1 and LSASS1d as a tuple of floats: (LSASS1, LSASS1d)
    """
    embeddings = get_sentences_embeddings(sentences)
    return lsa.lsa_adjacent_overlap(embeddings)


def all_sentences_overlap(sentences: list[Sentence]) -> tuple[float, float]:
    """
    The semantic overlap between every possible sentence pair in the text. This
    consists of LSASSa (the mean overlap) and LSASSad (the standard deviation overlap).

    Note: LSASSa and LSASSad are not official Coh-Metrix indexes. These are
    generalizations of the LSASSp and LSASSpd indexes.

    :param sentences: a list of sentences making up the text.
    :return: the LSASSa and LSASSad as a tuple of floats: (LSASSa, LSASSad)
    """
    embeddings = get_sentences_embeddings(sentences)
    return lsa.lsa_all_overlap(embeddings)


def all_sentences_overlap_paragraph(paragraphs: list[Paragraph]) -> tuple[float, float]:
    """
    The semantic overlap between every possible sentence pair within each paragraph. This
    consists of LSASSp (the mean overlap) and LSASSpd (the standard deviation overlap).

    :param paragraphs: a list of paragraphs making up the text.
    :return: the LSASSp and LSASSpd as a tuple of floats: (LSASSp, LSASSpd)
    """
    overlaps = [all_sentences_overlap(paragraph.sentences) for paragraph in paragraphs]
    return np.mean(overlaps), np.std(overlaps)


def adjacent_paragraphs_overlap(paragraphs: list[Paragraph]) -> tuple[float, float]:
    """
    The semantic overlap between each adjacent paragraph pair in the text. This
    consists of LSAPP1 (the mean overlap) and LSAPP1d (the standard deviation overlap).

    :param paragraphs: a list of paragraphs making up the text.
    :return: the LSAPP1 and LSAPP1d as a tuple of floats: (LSAPP1, LSAPP1d)
    """
    raise NotImplementedError()


def given_new(sentences:list[Sentence]) -> tuple[float, float]:
    """
    The given/new ratio for each sentence in relation to its preceding sentences.
    This consists of LSAGN (the mean given/new ratio) and LSAGNd (the standard
    deviation given/new ratio).

    :param sentences: a list of sentences making up the text.
    :return: the LSAGN and LSAGNd as a tuple of floats: (LSAGN, LSAGNd)
    """
    embeddings = get_sentences_embeddings(sentences)
    return lsa.lsa_given_new(embeddings)


# UTILITY FUNCTIONS ----------------------------------------------

def get_sentences_embeddings(sentences: list[Sentence]) -> list[lsa.Embedding]:
    """Return a list of sentence embeddings for the sentences."""
    return [sentence.embedding for sentence in sentences]


# TESTING --------------------------------------------------------


def test_sbert_lsa():
    # Test by parsing a text and computing its metrics.
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
