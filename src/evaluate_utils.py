import deepcut
import jiwer

from src.normalizers import EnglishTextNormalizer


def isd(preds: list[str], actuals: list[str], debug=False) -> int:
    dp = [[0 for _ in range(len(preds) + 1)] for _ in range(len(actuals) + 1)]

    for row in range(len(dp)):
        for col in range(len(dp[0])):
            if row == 0 or col == 0:
                dp[row][col] = max(row, col)
                continue

            if preds[col - 1] != actuals[row - 1]:
                dp[row][col] = (
                    min(dp[row - 1][col], dp[row][col - 1], dp[row - 1][col - 1]) + 1
                )
            else:
                dp[row][col] = min(
                    dp[row - 1][col], dp[row][col - 1], dp[row - 1][col - 1]
                )

    if debug:
        print(*dp, sep="\n")

    return dp[-1][-1]


def wer_th(pred: str, actual: str, **kwargs) -> float:
    preds = deepcut.tokenize(pred)
    actuals = deepcut.tokenize(actual)
    err = isd(preds, actuals, **kwargs)
    return err / len(actuals)


def wer_eng(pred: str, actual: str, **kwargs) -> float:
    # Normalize text before calculate
    normalizer = EnglishTextNormalizer()
    pred = normalizer(pred)
    actual = normalizer(actual)

    # Split sentence to word
    preds = pred.split(" ")
    actuals = actual.split(" ")

    # Calculte wer
    err = isd(preds, actuals, **kwargs)
    return err / len(actuals)


def cer_th(pred: str, actual: str):
    return jiwer.cer(hypothesis=pred, reference=actual)


def cer_eng(pred: str, actual: str):
    # Normalize text before calculate
    normalizer = EnglishTextNormalizer()
    pred = normalizer(pred)
    actual = normalizer(actual)
    return jiwer.cer(hypothesis=pred, reference=actual)


if __name__ == "__main__":
    # Input eng
    with open(
        "data/result/whisper/large-v1/en_Digital_Roadshow_Q1_2023_NVD/[EN] Digital Roadshow Q1_2023 NVD _ NIRVANA DEVELOPMENT PUBLIC COMPANY LIMITED.txt",
        "r",
    ) as f:
        prediction_text = f.read()
    with open(
        "data/sample_output/Transcription Digital Roadshow Nirvana.txt",
        "r",
    ) as f:
        reference_text = f.read()
    wer_score = wer_eng(prediction_text, reference_text)
    cer_score = cer_eng(prediction_text, reference_text)

    print(f"WER: {wer_score}")
    print(f"CER: {cer_score}")

    # Input th
    with open(
        "data/result/whisper/large-v1/th_Oppday_Q2_2023_IP_interfama/[TH] Oppday Q2_2023 IP บมจ. อินเตอร์ ฟาร์มา.txt",
        "r",
    ) as f:
        prediction_text = f.read()
    with open(
        "data/sample_output/Transcript Oppday Inter Pharma.txt",
        "r",
    ) as f:
        reference_text = f.read()
    wer_score = wer_th(prediction_text, reference_text)
    cer_score = cer_th(prediction_text, reference_text)

    print(f"WER: {wer_score}")
    print(f"CER: {cer_score}")

    # Input eng
    with open(
        "data/result/whisper/large-v1/en-th_Oppday_Q2_2023_CCET/[EN_TH] Oppday Q2_2023 CCET บมจ. แคล-คอมพ์ อีเล็คโทรนิคส์ (ประเทศไทย).txt",
        "r",
    ) as f:
        prediction_text = f.read()
    with open(
        "data/sample_output/Transcript  Oppday CCET.txt",
        "r",
    ) as f:
        reference_text = f.read()
    wer_score = wer_eng(prediction_text, reference_text)
    cer_score = cer_eng(prediction_text, reference_text)

    print(f"WER: {wer_score}")
    print(f"CER: {cer_score}")
