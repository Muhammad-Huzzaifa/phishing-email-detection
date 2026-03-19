import re
import tensorflow as tf

def clean_text(text):
    if text is None:
        return ""

    return re.sub(r"\s+", " ", str(text).lower().strip())

def validate_input(text):
    cleaned = clean_text(text or "")
    if not cleaned:
        raise ValueError("Input text is empty. Please provide an email message.")
    return cleaned

def tfidf_vectorize(text, tfidf_vectorizer):
    cleaned = clean_text(text)
    return tfidf_vectorizer.transform([cleaned])

def tokenize_and_pad(text, tokenizer, maxlen=500):
    cleaned = clean_text(text)
    sequences = tokenizer.texts_to_sequences([cleaned])
    return tf.keras.utils.pad_sequences(
        sequences,
        maxlen=maxlen,
        padding="post",
        truncating="post",
        dtype="int32",
    )

def decode_prediction(pred_class, positive_label="Phishing"):
    return positive_label if int(pred_class) == 1 else "Safe"
