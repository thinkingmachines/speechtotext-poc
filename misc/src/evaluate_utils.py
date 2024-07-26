# flake8: noqa: C901
import deepcut
import jiwer
import pandas as pd

from src.normalizers import EnglishTextNormalizer


def alignedPrint(steps: list, preds: list[str], actuals: list[str], score: float):
    """
    This funcition is to print the result of comparing reference and hypothesis sentences in an aligned way.

    Attributes:
        steps   -> the list of steps.
        actuals      -> the list of words produced by splitting reference sentence.
        preds      -> the list of words produced by splitting hypothesis sentence.
        score -> the rate calculated based on edit distance.
    """
    print("REF:", end=" ")
    for i in range(len(steps)):
        if steps[i] == "i":
            count = 0
            for j in range(i):
                if steps[j] == "d":
                    count += 1
            index = i - count
            print(" " * (len(preds[index])), end=" ")
        elif steps[i] == "s":
            count1 = 0
            for j in range(i):
                if steps[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if steps[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(actuals[index1]) < len(preds[index2]):
                print(
                    actuals[index1] + " " * (len(preds[index2]) - len(actuals[index1])),
                    end=" ",
                )
            else:
                print(actuals[index1], end=" "),
        else:
            count = 0
            for j in range(i):
                if steps[j] == "i":
                    count += 1
            index = i - count
            print(actuals[index], end=" "),
    print("\nHYP:", end=" ")
    for i in range(len(steps)):
        if steps[i] == "d":
            count = 0
            for j in range(i):
                if steps[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(actuals[index])), end=" ")
        elif steps[i] == "s":
            count1 = 0
            for j in range(i):
                if steps[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if steps[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(actuals[index1]) > len(preds[index2]):
                print(
                    preds[index2] + " " * (len(actuals[index1]) - len(preds[index2])),
                    end=" ",
                )
            else:
                print(preds[index2], end=" ")
        else:
            count = 0
            for j in range(i):
                if steps[j] == "d":
                    count += 1
            index = i - count
            print(preds[index], end=" ")
    print("\nEVA:", end=" ")
    for i in range(len(steps)):
        if steps[i] == "d":
            count = 0
            for j in range(i):
                if steps[j] == "i":
                    count += 1
            index = i - count
            print("D" + " " * (len(actuals[index]) - 1), end=" ")
        elif steps[i] == "i":
            count = 0
            for j in range(i):
                if steps[j] == "d":
                    count += 1
            index = i - count
            print("I" + " " * (len(preds[index]) - 1), end=" ")
        elif steps[i] == "s":
            count1 = 0
            for j in range(i):
                if steps[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if steps[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(actuals[index1]) > len(preds[index2]):
                print("S" + " " * (len(actuals[index1]) - 1), end=" ")
            else:
                print("S" + " " * (len(preds[index2]) - 1), end=" ")
        else:
            count = 0
            for j in range(i):
                if steps[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(actuals[index])), end=" ")
    print(f"\nWER: {score}")


def get_step(preds: list[str], actuals: list[str], dp: list[list]) -> dict:
    """
    This function is to get the list of steps in the process of dynamic programming.

    Attributes:
        preds -> the list of words produced by splitting hypothesis sentence.
        actuals -> the list of words produced by splitting reference sentence.
        dp -> the matrix built when calulating the editting distance of preds and actuals.
    """
    x = len(preds)
    y = len(actuals)
    steps = []
    insertions, substitutions, deletions = 0, 0, 0
    inserted_words, deleted_words, substituted_words = [], [], []
    while True:
        if x == 0 and y == 0:
            break
        elif (
            x >= 1
            and y >= 1
            and dp[y][x] == dp[y - 1][x - 1]
            and actuals[y - 1] == preds[x - 1]
        ):
            steps.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and dp[y][x] == dp[y - 1][x] + 1:
            steps.append("i")
            inserted_words.append(actuals[y - 1])
            insertions += 1
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and dp[y][x] == dp[y - 1][x - 1] + 1:
            steps.append("s")
            substituted_words.append((preds[x - 1], actuals[y - 1]))
            substitutions += 1
            x = x - 1
            y = y - 1
        else:
            steps.append("d")
            deleted_words.append(preds[x - 1])
            deletions += 1
            x = x - 1
            y = y
    return {
        "steps": steps[::-1],
        "insertions": insertions,
        "substitutions": substitutions,
        "deletions": deletions,
        "inserted_words": inserted_words,
        "substituted_words": substituted_words,
        "deleted_words": deleted_words,
    }


def isd(preds: list[str], actuals: list[str], debug=False) -> tuple:
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

    # if debug:
    # print(*dp, sep="\n")

    # Backtrack to get edit operations counts
    step = get_step(preds, actuals, dp)
    columns = [
        "wer",
        "insertions",
        "deletions",
        "substitutions",
        "inserted_words",
        "deleted_words",
        "substituted_words",
    ]
    summary = pd.DataFrame(
        [
            [
                dp[-1][-1] / len(actuals),
                step["insertions"],
                step["deletions"],
                step["substitutions"],
                step["inserted_words"],
                step["deleted_words"],
                step["substituted_words"],
            ]
        ],
        columns=columns,
    )
    if debug:
        alignedPrint(step["steps"], actuals, preds, dp[-1][-1] / len(actuals))
        print(summary)

    return (dp[-1][-1], step, summary)


def wer_th(pred: str, actual: str, **kwargs) -> float:
    preds = deepcut.tokenize(pred)
    actuals = deepcut.tokenize(actual)
    err, step, _ = isd(preds, actuals, **kwargs)
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
    err, steps, _ = isd(preds, actuals, **kwargs)
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
    wer_score = wer_eng(
        "I went to go to market place place", "I want to go to the market", debug=True
    )

    # Input eng
    # with open(
    #     "data/result/whisper/large-v1/en_Digital_Roadshow_Q1_2023_NVD/[EN] Digital Roadshow Q1_2023 NVD _ NIRVANA DEVELOPMENT PUBLIC COMPANY LIMITED.txt",
    #     "r",
    # ) as f:
    #     prediction_text = f.read()
    # with open(
    #     "data/sample_output/Transcription Digital Roadshow Nirvana.txt",
    #     "r",
    # ) as f:
    #     reference_text = f.read()
    # wer_score = wer_eng(prediction_text, reference_text)
    # cer_score = cer_eng(prediction_text, reference_text)

    # print(f"WER: {wer_score}")
    # print(f"CER: {cer_score}")

    # Input th
    # with open(
    #     "data/result/whisper/large-v3/th_Oppday_Q2_2023_IP_interfama/[TH] Oppday Q2_2023 IP บมจ. อินเตอร์ ฟาร์มา.txt",
    #     "r",
    # ) as f:
    #     prediction_text = f.read()
    # with open(
    #     "data/sample_output/Transcript Oppday Inter Pharma.txt",
    #     "r",
    # ) as f:
    #     reference_text = f.read()
    # wer_score = wer_th(prediction_text, reference_text)
    # cer_score = cer_th(prediction_text, reference_text)

    # print(f"WER: {wer_score}")
    # print(f"CER: {cer_score}")

    # Input eng
    # with open(
    #     "data/result/whisper/large-v1/en-th_Oppday_Q2_2023_CCET/[EN_TH] Oppday Q2_2023 CCET บมจ. แคล-คอมพ์ อีเล็คโทรนิคส์ (ประเทศไทย).txt",
    #     "r",
    # ) as f:
    #     prediction_text = f.read()
    # with open(
    #     "data/sample_output/Transcript  Oppday CCET.txt",
    #     "r",
    # ) as f:
    #     reference_text = f.read()
    # wer_score = wer_eng(prediction_text, reference_text)
    # cer_score = cer_eng(prediction_text, reference_text)

    # print(f"WER: {wer_score}")
    # print(f"CER: {cer_score}")
