import json
import os
import numpy as np
import pandas as pd
import urllib.request
from urllib.parse import unquote
from joblib import Parallel, delayed
from tqdm import tqdm


train = [
    "Xeon1OCR_round1_train1_20210526.csv",
    "Xeon1OCR_round1_train_20210524.csv",
    "Xeon1OCR_round1_train2_20210526.csv"
]
test = [
    "Xeon1OCR_round1_test1_20210528.csv",
    "Xeon1OCR_round1_test2_20210528.csv",
    "Xeon1OCR_round1_test3_20210528.csv"
]


def toPaddleStyle(jso):
    out = []
    if "option" in jso[1]:
        if jso[1]["option"] == "底部朝下":
            ord = [0, 1, 2, 3]
        elif jso[1]["option"] == "底部朝右":
            ord = [3, 0, 1, 2]
        elif jso[1]["option"] == "底部朝上":
            ord = [2, 3, 0, 1]
        elif jso[1]["option"] == "底部朝左":
            ord = [1, 2, 3, 0]
    for l in jso[0]:
        out.append({
            "transcription": l["text"][10:-2],
            "points": np.asarray(l["coord"])
            .astype(float)
            .reshape((4, 2))[ord]
            .tolist()
        })
    return json.dumps(out, ensure_ascii=False)


def down_image(url, dst_dir):
    filename = url.split('/')[-1]
    dst_path = os.path.join(dst_dir, filename)
    if os.path.exists(dst_path):
        return
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    urllib.request.urlretrieve(url, dst_path)


def download():
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_DYNAMIC'] = 'FALSE'

    df = []
    for csv in train:
        df.append(pd.read_csv(csv))
    df = pd.concat(df)
    df["链接"] = df["原始数据"].apply(lambda x: json.loads(x)["tfspath"])
    df["链接"].to_csv("train.txt", header=False, index=False)

    urls = [row["链接"] for (_, row) in df.iterrows()]
    Parallel(n_jobs=-1)(delayed(down_image)(url, "train") for url in tqdm(urls))

    print("???")

    test_df = []
    for i, csv in enumerate(test):
        df = pd.read_csv(csv)
        test_df.append(df)
        df["链接"] = df["原始数据"].apply(lambda x: json.loads(x)["tfspath"])
        df["链接"].to_csv(f"test{i+1}.txt", header=False, index=False)

        urls = [row["链接"] for (_, row) in df.iterrows()]
        Parallel(n_jobs=-1)(
            delayed(down_image)(url, f"test{i + 1}") for url in tqdm(urls))


def prebuild():
    valid_ratio = 0.1

    train["图片"] = train["原始数据"].apply(lambda x: json.loads(x)["tfspath"].split("/")[-1])
    train["答案"] = train["融合答案"].apply(lambda x: toPaddleStyle(json.loads(x)))
    valid = np.zeros((len(train),), dtype=bool)
    valid[: int(len(train) * valid_ratio)] = True
    np.random.shuffle(valid)

    train.loc[~valid, ["图片", "答案"]].to_csv(
        "train_label.txt", header=False, index=False, sep="\t", quoting=3
    )
    train.loc[valid, ["图片", "答案"]].to_csv(
        "valid_label.txt", header=False, index=False, sep="\t", quoting=3
    )


def main():
    download()


if __name__ == "__main__":
    main()
