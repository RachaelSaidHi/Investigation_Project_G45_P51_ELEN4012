from jiwer import process_words

reference_raw = """
Please call Stella. Ask her to bring these things with her from the store:
Six spoons of fresh snow peas, five thick slabs of blue cheese...
"""

hypothesis_raw = transcription

reference_norm = processor.tokenizer.normalize(reference_raw)
hypothesis_norm = processor.tokenizer.normalize(hypothesis_raw)

result = process_words(reference_norm, hypothesis_norm)

print("Reference:", reference_norm)
print("Hypothesis:", hypothesis_norm)

print("WER:", result.wer)
print("Substitutions:", result.substitutions)
print("Deletions:", result.deletions)
print("Insertions:", result.insertions)