import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=Path, required=True,
                        help="Directory path to search files containing ground truth bounding boxes.")
    parser.add_argument("--gt_format", default="xyrb",
                        help="Format of the coordinates of ground truth bounding boxes.")
    parser.add_argument("--det_dir", type=Path, required=True,
                        help="Directory path to search files containing detected bounding boxes.")
    parser.add_argument("--det_format", default="xyrb",
                        help="Format of the coordinates of the detected bounding boxes.")
    parser.add_argument("--output", type=Path, default="output",
                        help="Directory path to output results.")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IOU threshold.")
    args = parser.parse_args()
    # fmt: on

    return args


def load_gt_bboxes(dirpath, bbox_format):
    """Load ground truth bounding boxes from files.
    
    Args:
        dirpath (Path): Directory path to search files containing ground truth bounding boxes.
        bbox_format (str): "xyrb" or "xywh". Format of the coordinates of ground truth bounding boxes.
    
    Raises:
        FileNotFoundError: Raise when `dirpath` does not exists.
        ValueError: Raise when `bbox_format` is not either "xyrb" or "xywh".
    
    Returns:
        DataFrame: Ground truth bounding boxes.
    """
    if not dirpath.exists():
        raise FileNotFoundError(f"Directory '{dirpath}' does not exists.")

    if bbox_format not in ["xyrb", "xywh"]:
        raise ValueError(
            f"bbox_format should be 'xyrb' or 'xywh'. '{bbox_format}' passed."
        )

    bboxes = []
    for path in sorted(dirpath.glob("*.txt")):
        lines = open(path).read().splitlines()
        for line in lines:
            label, *coords = line.split()
            coords = list(map(float, coords))

            if bbox_format == "xyrb":
                xmin, ymin, xmax, ymax = coords
            elif bbox_format == "xywh":
                x, y, w, h = coords
                xmin, ymin, xmax, ymax = x, y, x + w, y + h

            bboxes.append([path.name, label, xmin, ymin, xmax, ymax])

    bboxes = pd.DataFrame(
        bboxes, columns=["Filename", "Label", "Xmin", "Ymin", "Xmax", "Ymax"]
    )

    return bboxes


def load_det_bboxes(dirpath, bbox_format):
    """Load detected bounding boxes from files.
    
    Args:
        dirpath (Path): Directory path to search files containing detected bounding boxes.
        bbox_format (str): "xyrb" or "xywh". Format of the coordinates of detected bounding boxes.
    
    Raises:
        FileNotFoundError: Raise when `dirpath` does not exists.
        ValueError: Raise when `bbox_format` is not either "xyrb" or "xywh".
    
    Returns:
        DataFrame: Detected bounding boxes.
    """
    if not dirpath.exists():
        raise FileNotFoundError(f"Directory '{dirpath}' does not exists.")

    if bbox_format not in ["xyrb", "xywh"]:
        raise ValueError(
            f"bbox_format should be 'xyrb' or 'xywh'. '{bbox_format}' passed."
        )

    bboxes = []
    for path in sorted(dirpath.glob("*.txt")):
        lines = open(path).read().splitlines()
        for line in lines:
            label, score, *coords = line.split()
            score = float(score)
            coords = list(map(float, coords))

            if bbox_format == "xyrb":
                xmin, ymin, xmax, ymax = coords
            elif bbox_format == "xywh":
                x, y, w, h = coords
                xmin, ymin, xmax, ymax = x, y, x + w, y + h

            bboxes.append([path.name, label, score, xmin, ymin, xmax, ymax])

    bboxes = pd.DataFrame(
        bboxes, columns=["Filename", "Label", "Score", "Xmin", "Ymin", "Xmax", "Ymax"]
    )

    return bboxes


def check_det_bboxes(gt_bboxes, det_bboxes, iou_threshold):
    """Check if the detected bounding boxes are correct.

    Args:
        gt_bboxes (DataFrame): ground truth bounding boxes.
        det_bboxes (DataFrame): detected bounding boxes.
        iou_threshold (float): iou threshold
    
    Returns:
        tuple: (ground truth bounding boxes, detected bounding boxes)
    """
    gt_bboxes = gt_bboxes.copy()
    det_bboxes = det_bboxes.copy()

    # ground truth の矩形が検出した矩形と紐付いているかどうかを記録する列を追加する。
    gt_bboxes["Match"] = False  
    # 検出した矩形が正解したかどうかを記録する列を追加する。
    det_bboxes["Correct"] = False
    # スコアが高い順にソートする。
    det_bboxes.sort_values("Score", ascending=False, inplace=True)

    for det_bbox in det_bboxes.itertuples():
        # 検出した矩形と同じ画像の正解の矩形を取得する。
        corr_gt_bboxes = gt_bboxes[gt_bboxes["Filename"] == det_bbox.Filename]

        if corr_gt_bboxes.empty:
            continue  # ground truth が存在しない場合

        # IOU を計算し、IOU が最大の ground truth の矩形を選択する。
        a = np.array([det_bbox.Xmin, det_bbox.Ymin, det_bbox.Xmax, det_bbox.Ymax])
        b = corr_gt_bboxes[["Xmin", "Ymin", "Xmax", "Ymax"]].values
        iou = calc_iou(a, b)

        # 検出した矩形 det_idx と正解の矩形 gt_idx が対応づけられた
        gt_idx = corr_gt_bboxes.index[iou.argmax()]

        if iou.max() >= iou_threshold and not gt_bboxes.loc[gt_idx, "Match"]:
            # IOU が閾値以上、かつ選択した矩形がまだ他の検出した矩形と紐付いていない場合、
            # 正解と判定する。
            gt_bboxes.loc[gt_idx, "Match"] = True
            det_bboxes.loc[det_bbox.Index, "Correct"] = True

    return gt_bboxes, det_bboxes


def calc_pr_curve(gt_bboxes, det_bboxes):
    """Calculate precision and recall for each threshold.

    Args:
        gt_bboxes (DataFrame): ground truth bounding boxes.
        det_bboxes (DataFrame): detected bounding boxes.
    
    Returns:
        tuple: (precision, recall)
    """
    TP = det_bboxes["Correct"]
    FP = ~det_bboxes["Correct"]
    n_positives = len(gt_bboxes.index)

    acc_TP = TP.cumsum()
    acc_FP = FP.cumsum()
    precision = acc_TP / (acc_TP + acc_FP)
    recall = acc_TP / n_positives

    return precision, recall


def calc_average_precision(recall, precision):
    """Calculate average precision (AP).
    
    Args:
        recall (array-like): Precision for each threshold.
        precision (array-like): Recall for each threshold.

    Returns:
        tuple: (AP, modified precision, modified recall)
    """
    modified_recall = np.concatenate([[0], recall, [1]])
    modified_precision = np.concatenate([[0], precision, [0]])

    # 末尾から累積最大値を計算する。
    modified_precision = np.maximum.accumulate(modified_precision[::-1])[::-1]

    # AP を計算する。
    average_precision = (np.diff(modified_recall) * modified_precision[1:]).sum()

    return modified_precision, modified_recall, average_precision


def calc_iou(a_bbox, b_bboxes):
    """Calculate intersection over union (IOU).
    
    Args:
        a (array-like): 1-D Array with shape (4,) representing bounding box.
        b (array-like): 2-D Array with shape (NumBoxes, 4) representing bounding boxes.
    
    Returns:
        [type]: [description]
    """
    # 短形 a_bbox と短形 b_bboxes の共通部分を計算する。
    xmin = np.maximum(a_bbox[0], b_bboxes[:, 0])
    ymin = np.maximum(a_bbox[1], b_bboxes[:, 1])
    xmax = np.minimum(a_bbox[2], b_bboxes[:, 2])
    ymax = np.minimum(a_bbox[3], b_bboxes[:, 3])
    i_bboxes = np.column_stack([xmin, ymin, xmax, ymax])

    # 矩形の面積を計算する。
    a_area = calc_area(a_bbox)
    b_area = np.apply_along_axis(calc_area, 1, b_bboxes)
    i_area = np.apply_along_axis(calc_area, 1, i_bboxes)

    # IOU を計算する。
    iou = i_area / (a_area + b_area - i_area)

    return iou


def calc_area(bbox):
    """Calculate area of boudning box.
    
    Args:
        bboxes (array-like): 1-D Array with shape (4,) representing bounding box.
    
    Returns:
        float: Areea
    """
    # 矩形の面積を計算する。
    # 共通部分がない場合は、幅や高さは負の値になるので、その場合、幅や高さは 0 とする。
    width = max(0, bbox[2] - bbox[0] + 1)
    height = max(0, bbox[3] - bbox[1] + 1)

    return width * height


def calc_metrics(gt_bboxes, det_bboxes, class_, iou_threshold):
    """Calculate result for specific class.
    
    Args:
        gt_bboxes (DataFrame): DataFrame representing ground truth bounding boxes.
        det_bboxes (DataFrame): DataFrame representing detected bounding boxes.
        class_ (str): Class to calculate result.
        iou_threshold (float): IOU threshold.
    
    Returns:
        dict: Result for specific class.
    """
    # 対象クラスの正解及び検出した矩形を抽出する。
    taget_gt_bboxes = gt_bboxes[gt_bboxes["Label"] == class_]
    taget_det_bboxes = det_bboxes[det_bboxes["Label"] == class_]

    # TP, FP を計算する。
    taget_gt_bboxes, taget_det_bboxes = check_det_bboxes(
        taget_gt_bboxes, taget_det_bboxes, iou_threshold
    )

    # PR 曲線を計算する。
    precision, recall = calc_pr_curve(taget_gt_bboxes, taget_det_bboxes)

    modified_precision, modified_recall, average_precision = calc_average_precision(
        recall.values, precision.values
    )

    result = {
        "class": class_,
        "precision": precision,
        "recall": recall,
        "modified_precision": modified_precision,
        "modified_recall": modified_recall,
        "average_precision": average_precision,
    }

    return result


def plot_pr_curve(result, save_path=None):
    """Plot precision recall (PR) cureve.
    
    Args:
        result (dict): Result for specific class.
        save_path (Path, optional): Path to save created figure. Defaults to None.
    """
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_title(
        "Precision Recall Curve\n"
        f"Class: {result['class']}, AP: {result['average_precision']:.2%}"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    # PR 曲線を描画する。
    ax.plot(result["recall"], result["precision"], label="Precision")

    # 修正した PR 曲線を描画する。
    ax.step(
        result["modified_recall"],
        result["modified_precision"],
        "--k",
        label="Modified Precision",
    )

    # 図を保存する。
    fig.savefig(save_path)


def aggregate(results):
    """Aggregate each class result and create DataFrame.
    
    Args:
        results (list of dicts): list of dicts representing all class results.
    
    Returns:
        DataFrame: DataFrame representing all class results.
    """
    metrics = []
    for result in results:
        metrics.append({"Class": result["class"], "AP": result["average_precision"]})

    metrics = pd.DataFrame(metrics)
    metrics.set_index("Class", inplace=True)
    metrics.sort_index(inplace=True)
    
    # mAP を計算する。
    metrics.loc["mAP"] = metrics["AP"].mean()

    return metrics


def main():
    args = parse_args()

    # ground truth 及び検出した矩形一覧を読み込む。
    gt_bboxes = load_gt_bboxes(args.gt_dir, args.gt_format)
    det_bboxes = load_det_bboxes(args.det_dir, args.det_format)

    # クラス一覧を取得する。
    classes = gt_bboxes["Label"].unique()

    # クラスごとに計算する。
    results = []
    for class_ in classes:
        result = calc_metrics(gt_bboxes, det_bboxes, class_, iou_threshold=args.iou)
        results.append(result)

    # 各クラスの AP をデータフレームでまとめる。
    metrics = aggregate(results)
    print(metrics)

    # 結果を出力する。
    args.output.mkdir(exist_ok=True)
    for ret in results:
        save_path = args.output / f"{ret['class']}.png"
        plot_pr_curve(ret, save_path)
    save_path = args.output / f"metrics.csv"
    metrics.to_csv(save_path)


if __name__ == "__main__":
    main()
