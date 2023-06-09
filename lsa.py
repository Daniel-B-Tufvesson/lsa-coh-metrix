"""
Math functions for Latent Semantic Analysis.

These operate purely on the numerical level, that is, everything are
represented as embeddings. There are no references to textual features,
such as words, sentences or paragraphs. This means the script is standalone
to any NLP libraries or NLP data objects (except for the embeddings).
"""

import numpy as np

Embedding = np.ndarray[float]


def lsa_adjacent_overlap(embeddings: list[Embedding]) -> tuple[float, float]:
    """
    Compute the mean and standard deviation overlap between all adjacent
    embedding pairs.

    This is suitable for LSASS1, LSASS1d, LSAPP1, LSAPP1d.

    :param embeddings: a list of embeddings.
    :return: the mean and standard deviation as a tuple: (mean, std).
    """
    overlaps = [cosine_similarity(embeddings[i], embeddings[i + 1])
                for i in range(len(embeddings) - 1)]
    return np.mean(overlaps), np.std(overlaps)


def lsa_all_overlap(embeddings: list[Embedding]) -> tuple[float, float]:
    """
    Compute the mean and standard deviation overlap between all possible
    embedding pairs.

    This is suitable for LSASSp, LSASSpd.

    :param embeddings: a list of embeddings.
    :return: the mean and standard deviation as a tuple: (mean, std).
    """
    overlaps = [cosine_similarity(embeddings[i], embeddings[j])
                for i in range(len(embeddings))
                for j in range(len(embeddings))
                if i != j]
    return np.mean(overlaps), np.std(overlaps)


def lsa_given_new(embeddings: list[Embedding]) -> tuple[float, float]:
    """
    Compute the given/new ratio of semantic information in a text.

    Suitable for LSAGN and LSAGNd.

    :param embeddings: a list of embeddings.
    :return: the mean and standard deviation as a tuple: (mean, std).
    """
    if len(embeddings) < 2:
        raise ValueError("There must be at least two embeddings: ", len(embeddings))

    def compute_givenness(embedding, previous_embeddings):
        # Project onto previous embeddings
        projection = project_onto_subspace(embedding, previous_embeddings)

        # Compute the amount of new information.
        new_information = embedding - projection
        len_old = np.linalg.norm(projection)  # norm computes the length of the vector.
        len_new = np.linalg.norm(new_information)

        # Return the ratio between old and new information.
        return len_old / (len_new + len_old)

    givenness = [compute_givenness(embeddings[i], embeddings[:i])
                 for i in range(1, len(embeddings))]
    return np.mean(givenness), np.std(givenness)


# VECTOR ALGEBRA ---------------------------------


def cosine_similarity(embedding_1: Embedding, embedding_2: Embedding) -> float:
    """
    Compute the normalized cosine between the two embeddings.

    Here normalized means that the resulting cosine value is in the range 0 to 1,
    in contrast to -1 and 1. 1 indicates complete similarity, while 0 indicated
    complete opposites.

    :param embedding_1: the first embedding.
    :param embedding_2: the second embedding.
    :return: the normalized cosine between the embeddings.
    """
    # Use dot product to compute cosine: cos(x) = u·v / (|u||v|)
    dot = np.dot(embedding_1, embedding_2)
    len1 = np.linalg.norm(embedding_1)  # norm computes the length of the vector.
    len2 = np.linalg.norm(embedding_2)
    cosine = dot / (len1 * len2)

    return (cosine + 1) / 2  # Normalize.


def project_onto_subspace(vector: np.ndarray[float], other_vectors: list[np.ndarray[float]]) -> np.ndarray:
    """
    Project a vector onto the subspace defined by a list of vectors.
    The subspace vectors do not have to be orthogonal.

    :param vector: the vector to project.
    :param other_vectors: a list of vectors specifying the subspace.
    :return: the projected vector onto the subspace.
    """

    orthogonal_subspace = orthogonalize(other_vectors)
    sub_space_projection = np.zeros_like(vector)

    # Project onto subspace.
    for ortho_vec in orthogonal_subspace:
        sub_space_projection += project(vector, ortho_vec)

    return sub_space_projection


def project(vector_1: np.ndarray[float], vector_2: np.ndarray[float]) -> np.ndarray:
    """
    Project the first vector onto the second.

    :param vector_1: the vector to be projected.
    :param vector_2: the vector to be projected onto.
    :return: the resulting projected vector.
    """

    # proj = (v1·v2 / |v2|^2) * v2
    dot = np.dot(vector_1, vector_2)
    len2 = np.dot(vector_2, vector_2)
    return vector_2.copy() * (dot / len2)


def orthogonalize(vectors: list[np.ndarray[float]]) -> list[np.ndarray[float]]:
    """
    Orthogonalize the subspace specified by the vectors. This uses the Gram-Schmidt
    orthogonalization process. Note, however, that the resulting orthogonalized vectors
    are not normalized.

    :param vectors: a list of vectors specifying the subspace.
    :return: a new list of orthogonal vectors specifying the subspace.
    """

    remaining = vectors.copy()
    ortho_space = [remaining.pop().copy()]

    while len(remaining) > 0:
        next_vec = remaining.pop().copy()  # Copy so we don't modify the original embedding.
        ortho_projections = [project(next_vec, ortho_vec) for ortho_vec in ortho_space]

        # Subtract projections from the vector.
        for projection in ortho_projections:
            next_vec -= projection

        ortho_space.append(next_vec)

    return ortho_space
