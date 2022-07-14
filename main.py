import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=Path, required=True,
                        help="txt path containing ground truth bounding boxes.")
    parser.add_argument("--det_path", type=Path, required=True,
                        help="txt path containing detected bounding boxes.")

    parser.add_argument("--output", type=Path, default="output",
                        help="Directory path to output results.")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IOU threshold.")
    args = parser.parse_args()
    
    return args


def load_gt_bboxes(filepath):
    if not filepath.exists():
        raise FileNotFoundError(f"Directory '{filepath}' does not exists.")

    bboxes = []
    with open(filepath) as f:
        txt = f.readlines()
        for line in txt:
            image_path = line.strip().split()[0]
            targets = line.strip().split()[1:]

            for target in targets:
                xmin, ymin, zmin, xmax, ymax, zmax, label = target.split(",")
                bboxes.append([image_path, label, float(xmin), float(ymin), float(zmin), float(xmax), float(ymax), float(zmax)])

    bboxes = pd.DataFrame(
        bboxes, columns=["Filename", "Label", "Xmin", "Ymin", "Zmin", "Xmax", "Ymax", "Zmax"]
    )

    return bboxes


def load_det_bboxes(filepath):
    if not filepath.exists():
        raise FileNotFoundError(f"Directory '{filepath}' does not exists.")

    bboxes = []
    with open(filepath) as f:
        txt = f.readlines()
        for line in txt:
            image_path = line.strip().split()[0]
            targets = line.strip().split()[1:]

            for target in targets:
                score, xmin, ymin, zmin, xmax, ymax, zmax, label = target.split(",")
                bboxes.append([image_path, label, float(score), float(xmin), float(ymin), float(zmin), float(xmax), float(ymax), float(zmax)])

    bboxes = pd.DataFrame(
        bboxes, columns=["Filename", "Label", "Score", "Xmin", "Ymin", "Zmin", "Xmax", "Ymax", "Zmax"]
    )

    return bboxes


def check_det_bboxes(gt_bboxes, det_bboxes, iou_threshold):
    gt_bboxes = gt_bboxes.copy()
    det_bboxes = det_bboxes.copy()

    gt_bboxes["Match"] = False  
    det_bboxes["Correct"] = False
    det_bboxes.sort_values("Score", ascending=False, inplace=True)

    for det_bbox in det_bboxes.itertuples():
        corr_gt_bboxes = gt_bboxes[gt_bboxes["Filename"] == det_bbox.Filename]

        if corr_gt_bboxes.empty:
            continue

        a = np.array([det_bbox.Xmin, det_bbox.Ymin, det_bbox.Zmin, det_bbox.Xmax, det_bbox.Ymax, det_bbox.Zmax])
        b = corr_gt_bboxes[["Xmin", "Ymin", "Zmin", "Xmax", "Ymax", "Zmax"]].values
        iou = calc_iou(a, b)

        gt_idx = corr_gt_bboxes.index[iou.argmax()]

        if iou.max() >= iou_threshold and not gt_bboxes.loc[gt_idx, "Match"]:
            gt_bboxes.loc[gt_idx, "Match"] = True
            det_bboxes.loc[det_bbox.Index, "Correct"] = True

    return gt_bboxes, det_bboxes


def calc_pr_curve(gt_bboxes, det_bboxes):
    TP = det_bboxes["Correct"]
    FP = ~det_bboxes["Correct"]
    n_positives = len(gt_bboxes.index)

    acc_TP = TP.cumsum()
    acc_FP = FP.cumsum()
    precision = acc_TP / (acc_TP + acc_FP)
    recall = acc_TP / n_positives

    return precision, recall


def calc_average_precision(recall, precision):
    modified_recall = np.concatenate([[0], recall, [1]])
    modified_precision = np.concatenate([[0], precision, [0]])

    modified_precision = np.maximum.accumulate(modified_precision[::-1])[::-1]

    average_precision = (np.diff(modified_recall) * modified_precision[1:]).sum()

    return modified_precision, modified_recall, average_precision


def calc_iou(a_bbox, b_bboxes):
    
    xmin = np.maximum(a_bbox[0], b_bboxes[:, 0])
    ymin = np.maximum(a_bbox[1], b_bboxes[:, 1])
    zmin = np.maximum(a_bbox[2], b_bboxes[:, 2])
    xmax = np.minimum(a_bbox[3], b_bboxes[:, 3])
    ymax = np.minimum(a_bbox[4], b_bboxes[:, 4])
    zmax = np.minimum(a_bbox[5], b_bboxes[:, 5])
    i_bboxes = np.column_stack([xmin, ymin, zmin, xmax, ymax, zmax])

    a_area = calc_area(a_bbox)
    b_area = np.apply_along_axis(calc_area, 1, b_bboxes)
    i_area = np.apply_along_axis(calc_area, 1, i_bboxes)

    iou = i_area / (a_area + b_area - i_area)

    return iou


def calc_area(bbox):
    
    width = max(0, bbox[3] - bbox[0] + 1)
    height = max(0, bbox[4] - bbox[1] + 1)
    depth = max(0, bbox[5] - bbox[2] + 1)

    return width * height * depth


def calc_metrics(gt_bboxes, det_bboxes, class_, iou_threshold):
    taget_gt_bboxes = gt_bboxes[gt_bboxes["Label"] == class_]
    taget_det_bboxes = det_bboxes[det_bboxes["Label"] == class_]

    taget_gt_bboxes, taget_det_bboxes = check_det_bboxes(
        taget_gt_bboxes, taget_det_bboxes, iou_threshold
    )

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
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_title(
        "Precision Recall Curve\n"
        f"Class: {result['class']}, AP: {result['average_precision']:.2%}"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.plot(result["recall"], result["precision"], label="Precision")

    ax.step(
        result["modified_recall"],
        result["modified_precision"],
        "--k",
        label="Modified Precision",
    )

    fig.savefig(save_path)


def aggregate(results):
    metrics = []
    for result in results:
        metrics.append({"Class": result["class"], "AP": result["average_precision"]})

    metrics = pd.DataFrame(metrics)
    metrics.set_index("Class", inplace=True)
    metrics.sort_index(inplace=True)
    
    metrics.loc["mAP"] = metrics["AP"].mean()

    return metrics


def main():
    args = parse_args()

    gt_bboxes = load_gt_bboxes(args.gt_path)
    det_bboxes = load_det_bboxes(args.det_path)

    classes = gt_bboxes["Label"].unique()

    results = []
    for class_ in classes:
        result = calc_metrics(gt_bboxes, det_bboxes, class_, iou_threshold=args.iou)
        results.append(result)

    metrics = aggregate(results)
    print(metrics)

    args.output.mkdir(exist_ok=True)
    for ret in results:
        save_path = args.output / f"{ret['class']}.png"
        plot_pr_curve(ret, save_path)
    save_path = args.output / f"metrics.csv"
    metrics.to_csv(save_path)


if __name__ == "__main__":
    main()
