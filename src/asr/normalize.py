from transformers import WhisperTokenizer

# TODO: Test that all models normalise the same way to validate why it doesnt really matter which is chosen
DEFAULT_NORMALIZER = "openai/whisper-small"


# Speakers for Afrispeech were told to speak punctuation in full which are removed during normalisation
# semi colon or semicolon matches before colon
AFRISPEECH_PUNCTUATION_TERMS = (
    # new line, next line,
    "full stop",
    "question mark",
    "exclamation mark",
    "open bracket",
    "close bracket",
    "open parenthesis",
    "close parenthesis"
    "comma",
    "semi colon",
    "semicolon",
    "colon",
    "slash",
    "hyphen",
    "dash",
    "period",
    "query"
)



def remove_duplicate_whitespace(text: str) -> str:
    """
    Removes repeated and leading/trailing whitespace
    """

    return " ".join(text.split())


def normalize_text(input: str, dataset: str) -> str:
    if input is None:
            raise ValueError(
                "Cannot normalize None text."
            )

    # Stage 1: normalise text using Whisper's tokenizer
    tokenizer = WhisperTokenizer.from_pretrained(DEFAULT_NORMALIZER)
    normalized_text = tokenizer.normalize(input)

    # Stage 2: dataset specific normalization
    dataset = dataset.lower().strip() 
    if (dataset == "afrispeech"):
         normalized_text = remove_exact_phrases(normalized_text, AFRISPEECH_PUNCTUATION_TERMS)

    # Stage 3: remove repeated whitespace
    normalized_text = remove_duplicate_whitespace(normalized_text)  


def remove_exact_phrases(text, phrases):
    """
    Remove phrases from an already normalised string (lowercase, no punctuation, 6 -> six, etc)
 
    Works on lists of whole words rather than by replacing substrings, so a
    phrase can never match inside a longer word. Without this, removing
    "colon" would also remove it from "colonoscopy".
    """

    # split the text into words
    tokens = text.split()
 
    # split each phrase into a list of words
    phrase_tokens = []
    for phrase in phrases:
        phrase_tokens.append(phrase.split())
 
    # ensuring longer phrases are removed first ("semi colon" before "colon")
    phrase_tokens.sort(key=len, reverse=True)
 
    kept_tokens = []
    removed = 0
    i = 0
 
    while i < len(tokens):
        matched = False
 
        for phrase in phrase_tokens:
            length = len(phrase)  # the length of a phrase is the number of words

            # if a match is found, the index moves past the phrase so that it is not kept (removed)
            if tokens[i:i + length] == phrase:
                i = i + length
                removed = removed + 1
                matched = True
                break
 
        if not matched:
            kept_tokens.append(tokens[i])
            i = i + 1
 
    return " ".join(kept_tokens), removed


# def remove_exact_phrases(text: str, phrases: list) -> tuple:

#     # padding so only full words are removed (eg only the word "colon" not within another word like in "colonoscopy")
#     text = " " + text + " "
#     removed_count = 0

#     # ensuring longer phrases are removed first ("semi colon" before "colon")
#     phrases = sorted(phrases, key=len, reverse=True)

#     for phrase in phrases:
#         target = " " + phrase + " "

#         # A while loop because two punctuation words in a row share a
#         # space, so a single pass would only remove the first one.
#         while target in text:
#             text = text.replace(target, " ", 1)
#             removed_count = removed_count + 1

#     # Collapse any extra spaces and strip the padding.
#     return " ".join(text.split()), removed_count