'''Common NLP tools.'''
import spacy


# tokenizers
def split_tokenizer(text):
    """
    Simple & fast tokenizer.
    """
    return text.split(' ')


def make_spacy_tokenizer(model_name):
    """
    Make a spacy tokenizer
    """
    model = spacy.load(model_name)

    def spacy_tokenizer(string):
        """spacy tokenbizer"""
        return [t.text for t in model.tokenizer(string)]

    return spacy_tokenizer
