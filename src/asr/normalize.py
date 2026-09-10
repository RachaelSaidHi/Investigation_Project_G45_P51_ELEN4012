from transformers import WhisperTokenizer

# TODO: Test that all models normalise the same way to validate why it doesnt really matter which is chosen
DEFAULT_NORMALIZER = "openai/whisper-small"

# runs once when the module is first imported instead of being loaded every time the normalize_text gets called
tokenizer = WhisperTokenizer.from_pretrained(DEFAULT_NORMALIZER)


# Speakers for Afrispeech were told to speak punctuation in full which are removed during normalisation
# semi colon or semicolon matches before colon
AFRISPEECH_PUNCTUATION_TERMS = (
    "full stop",
    "question mark",
    "exclamation mark",
    "open bracket",
    "close bracket",
    "open parenthesis",
    "close parenthesis",
    "comma",
    "semi colon",
    "semicolon",
    "colon",
    "slash",
    "hyphen",
    "dash",
    "period",
    "query"
    # "new line", "next line" may be needed
)


def remove_duplicate_whitespace(text: str) -> str:
    """
    Removes repeated and leading/trailing whitespace
    """

    return " ".join(text.split())


def remove_exact_phrases(text, phrases):
    """
    Remove phrases from an already normalised string (lowercase, no punctuation, six -> 6, etc)
 
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


def normalize_text(input_text: str, dataset: str) -> tuple:
    """
    Normalise a reference or hypothesis transcript before computing WER.
    The same function is applied to both the reference and the hypothesis
    so that they are always compared under identical rules.

    Three stages:

        1. Whisper's English normaliser: lowercases the text, removes
           punctuation symbols, converts number words to digits
           ("six" -> "6"), expands contractions and applies the
           British/American spelling map. 

        2. Dataset-specific correction. For AfriSpeech only, spoken
           punctuation words are removed. 

        3. Collapse repeated whitespace left behind by the earlier stages or from the reference transcript itself.
    """

    if input_text is None:
        raise ValueError(
            "Cannot normalize None text."
        )

    # Stage 1: normalise text using Whisper's tokenizer
    normalized_text = tokenizer.normalize(input_text)

    # Stage 2: dataset specific normalization
    dataset = dataset.lower().strip() 

    if dataset not in ("afrispeech", "speech_accent_archive", "edacc"):
        raise ValueError("Unknown dataset: " + dataset)
    
    removed = 0
    if dataset == "afrispeech":
         normalized_text, removed = remove_exact_phrases(normalized_text, AFRISPEECH_PUNCTUATION_TERMS)

    # Stage 3: remove repeated whitespace
    normalized_text = remove_duplicate_whitespace(normalized_text)  

    # returns the normalised text and the number of words removed 
    return normalized_text, removed


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