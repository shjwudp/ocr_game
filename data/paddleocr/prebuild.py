import json
import os
import numpy as np
import pandas as pd
import urllib.request
from urllib.parse import unquote
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import typing
import pathlib

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
    pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)
    for i in range(6):
        try:
            urllib.request.urlretrieve(url, dst_path)
            break
        except TimeoutError as err:
            print("url={}, err={}".format(url, err))
            time.sleep(5 ** i)


def download(train, test) -> typing.Tuple[pd.Series, pd.Series]:
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_DYNAMIC'] = 'FALSE'

    train_df = []
    for csv in train:
        train_df.append(pd.read_csv(csv))
    train_df = pd.concat(train_df)
    train_df["链接"] = train_df["原始数据"].apply(lambda x: json.loads(x)["tfspath"])
    train_df["链接"].to_csv("train.txt", header=False, index=False)
    train_df.to_csv("all_train.txt")

    urls = [row["链接"] for (_, row) in train_df.iterrows()]
    Parallel(n_jobs=-1)(delayed(down_image)(url, "train") for url in tqdm(urls))

    test_df = []
    for i, csv in enumerate(test):
        df = pd.read_csv(csv)
        test_df.append(df)
        df["链接"] = df["原始数据"].apply(lambda x: json.loads(x)["tfspath"])
        df["链接"].to_csv(f"test{i+1}.txt", header=False, index=False)

        urls = [row["链接"] for (_, row) in df.iterrows()]
        Parallel(n_jobs=4)(
            delayed(down_image)(url, f"test{i + 1}") for url in tqdm(urls))

    return train_df, test_df


def prebuild(train):
    valid_ratio = 0.1

    # train["图片"] = train["原始数据"].apply(lambda x: unquote(unquote(json.loads(x)["tfspath"].split("/")[-1])))
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


def infer(test, test_df):
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        # det_model_dir="ch_ppocr_server_v2.0_det_infer",
        det_model_dir="pretrained_model_infer",
        cls_model_dir="ch_ppocr_mobile_v2.0_cls_infer",
        rec_model_dir="ch_ppocr_server_v2.0_rec_infer",
        use_angle_cls=True,
        cls=True
    )
    for i in range(3):
        resdict = {}
        for img in test_df[i]['链接']:
            # name = img.split('/')[-1][:-4]
            name = img.split('/')[-1][:-4]
            # unquote_name = unquote(unquote(name))[:-4]
            points = []
            transcriptions = []
            result = ocr.ocr(f"test{i + 1}/{name}.jpg")
            for line in result:
                points.append(sum(line[0], []))
                transcriptions.append(line[1][0])

            resdict[name] = {
                "pointsList": points,
                "transcriptionsList": transcriptions,
                "ignoreList": [False] * len(points),
                "classesList": [1] * len(points),
            }
        with open(f"{test[i][:-4]}.json", "w") as f:
            json.dump(resdict, f, ensure_ascii=False)


def main():
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

    train_df, test_df = download(train, test)
    print(train_df)
    prebuild(train_df)
    infer(test, test_df)


if __name__ == "__main__":
    main()
