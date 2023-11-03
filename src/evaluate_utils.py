import evaluate
from pythainlp.tokenize import word_tokenize


def evaluate_wer_thai(prediction_text: str, reference_text: str):
    # Create word list using pythainlp word_tokenize
    predictions = word_tokenize(
        prediction_text, engine="newmm", keep_whitespace=False, join_broken_num=True
    )
    references = word_tokenize(
        reference_text, engine="newmm", keep_whitespace=False, join_broken_num=True
    )

    # Evaluate word list
    wer = evaluate.load("wer")
    wer_score = wer.compute(predictions=predictions, references=references)
    return wer_score


def evaluate_cer_thai(prediction_text: str, reference_text: str):
    # Create word list using pythainlp word_tokenize
    predictions = word_tokenize(
        prediction_text, engine="newmm", keep_whitespace=False, join_broken_num=True
    )
    references = word_tokenize(
        reference_text, engine="newmm", keep_whitespace=False, join_broken_num=True
    )

    # Evaluate word list
    cer = evaluate.load("cer")
    cer_score = cer.compute(predictions=predictions, references=references)
    return cer_score


def evaluate_wer_eng(prediction_text: str, reference_text: str):
    # Create word list for english text
    predictions = prediction_text.split(" ")
    references = reference_text.split(" ")

    # Evaluate word list
    wer = evaluate.load("wer")
    wer_score = wer.compute(predictions=predictions, references=references)
    return wer_score


def evaluate_cer_eng(prediction_text: str, reference_text: str):
    # Create word list for english text
    predictions = prediction_text.split(" ")
    references = reference_text.split(" ")

    # Evaluate word list
    cer = evaluate.load("cer")
    cer_score = cer.compute(predictions=predictions, references=references)
    return cer_score


if __name__ == "__main__":
    # Input
    prediction_text = "กินข้าว"
    reference_text = "กินน้ำ"
    wer_score = evaluate_wer_thai(prediction_text, reference_text)
    cer_score = evaluate_cer_thai(prediction_text, reference_text)

    print(f"WER: {wer_score}")
    print(f"CER: {cer_score}")
