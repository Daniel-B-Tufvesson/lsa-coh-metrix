from stanza.models.common.doc import Sentence


class Paragraph:

    def __init__(self):
        self.sentences = []  # type: list[Sentence]


class ParagraphProcessor:
    pass