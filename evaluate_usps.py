from pathlib import Path
import numpy as np
from PIL import Image
from ann_model import NeuralNetwork

BASE_DIR = Path(__file__).resolve().parent


def shift_to_center_of_mass(arr_28: np.ndarray) -> np.ndarray:
    """Căn ảnh 28x28 [0, 1] theo trọng tâm gần (13.5, 13.5)."""
    total_mass = float(np.sum(arr_28))
    if total_mass <= 1e-6:
        return arr_28

    rows, cols = np.indices(arr_28.shape, dtype=np.float32)
    cy = float(np.sum(rows * arr_28)) / total_mass
    cx = float(np.sum(cols * arr_28)) / total_mass

    shift_y = int(round(13.5 - cy))
    shift_x = int(round(13.5 - cx))

    img_u8 = np.clip(arr_28 * 255.0, 0, 255).astype(np.uint8)
    shifted = Image.fromarray(img_u8, mode="L").transform(
        (28, 28),
        Image.Transform.AFFINE,
        (1, 0, -shift_x, 0, 1, -shift_y),
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )
    return np.asarray(shifted, dtype=np.float32) / 255.0


def preprocess_digit_image(pil_img: Image.Image) -> np.ndarray:
    """Chuẩn hóa ảnh chữ số về vector (784,) [0.0, 1.0], nền đen chữ sáng, căn trọng tâm."""
    gray = pil_img.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0

    # Tự nhận diện nếu nền sáng hơn chữ (viền sáng hơn) thì đảo màu về nền đen chữ sáng
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if float(np.mean(border)) > 0.5:
        arr = 1.0 - arr

    arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    coords = np.argwhere(arr_u8 > 25)
    if coords.size == 0:
        return arr.reshape(784)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    cropped = Image.fromarray(arr_u8, mode="L").crop((x_min, y_min, x_max, y_max))

    w, h = cropped.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * 20 / w)))
    else:
        new_h = 20
        new_w = max(1, int(round(w * 20 / h)))

    resized = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas_28 = Image.new("L", (28, 28), color=0)
    canvas_28.paste(resized, ((28 - new_w) // 2, (28 - new_h) // 2))

    centered = shift_to_center_of_mass(np.asarray(canvas_28, dtype=np.float32) / 255.0)
    return centered.reshape(784)


def load_dataset_by_digit_folders(root_dir: Path):
    """
    Quét toàn bộ thư mục con có tên là chữ số '0'..'9' bên trong root_dir
    (hỗ trợ cả trường hợp lồng sâu hơn như root_dir/Test/0/*.png).
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    X_list, y_list = [], []

    for digit in range(10):
        matching_dirs = [p for p in root_dir.rglob(str(digit)) if p.is_dir()]
        for d_dir in sorted(matching_dirs):
            for img_path in sorted(d_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in exts:
                    with Image.open(img_path) as im:
                        X_list.append(preprocess_digit_image(im))
                        y_list.append(digit)

    if not X_list:
        raise ValueError(f"Không tìm thấy ảnh nào trong các thư mục '0'..'9' dưới: {root_dir}")

    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10):
    """Tính Confusion Matrix (Hàng: Dự đoán | Cột: Thực tế), Accuracy, Precision, Recall và F1."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[p, t] += 1

    total = int(np.sum(cm))
    accuracy = float(np.trace(cm)) / total if total > 0 else 0.0

    per_class = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(np.sum(cm[c, :]) - tp)
        fn = int(np.sum(cm[:, c]) - tp)
        support = int(np.sum(cm[:, c]))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class.append({
            "digit": c,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    macro_precision = float(np.mean([m["precision"] for m in per_class]))
    macro_recall = float(np.mean([m["recall"] for m in per_class]))
    macro_f1 = float(np.mean([m["f1"] for m in per_class]))

    return accuracy, cm, per_class, (macro_precision, macro_recall, macro_f1)


def format_cm_aligned(cm: np.ndarray) -> str:
    """Định dạng ma trận nhầm lẫn thẳng cột có tiêu đề lớp 0..9."""
    col_width = max(5, len(str(int(np.max(cm)))) + 1)
    header = f"{'Pred \\ True':<12} | " + " ".join(f"{c:>{col_width}d}" for c in range(cm.shape[1]))
    sep = "-" * len(header)
    lines = [header, sep]
    for r in range(cm.shape[0]):
        row_vals = " ".join(f"{int(cm[r, c]):>{col_width}d}" for c in range(cm.shape[1]))
        lines.append(f"Lớp {r:<7} | {row_vals}")
    return "\n".join(lines)


def format_usps_report(
    model_name: str,
    dataset_name: str,
    total_samples: int,
    accuracy: float,
    cm: np.ndarray,
    per_class: list,
    macro_scores: tuple,
) -> str:
    """Định dạng báo cáo đánh giá chi tiết trên tập USPS, thẳng hàng từng cột."""
    macro_p, macro_r, macro_f1 = macro_scores
    table_header = (
        f"{'Lớp':<6} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | "
        f"{'TP':>5} | {'FP':>5} | {'FN':>5} | {'Tổng mẫu':>8}"
    )
    table_sep = "-" * len(table_header)

    lines = [
        "=" * 78,
        "BÁO CÁO ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP DỮ LIỆU",
        f"File trọng số : {model_name}",
        f"Thư mục dữ liệu: {dataset_name} (Tổng số mẫu: {total_samples})",
        f"Overall Acc    : {accuracy * 100:.2f}%",
        f"Macro Avg      : Precision {macro_p * 100:.2f}% | Recall {macro_r * 100:.2f}% | F1-Score {macro_f1 * 100:.2f}%",
        "=" * 78,
        "",
        "MA TRẬN NHẦM LẪN (Hàng: Dự đoán | Cột: Thực tế)",
        format_cm_aligned(cm),
        "",
        "CHI TIẾT TỪNG CHỮ SỐ",
        table_header,
        table_sep,
    ]

    for item in per_class:
        lines.append(
            f"Số {item['digit']:<3} | "
            f"{item['precision'] * 100:>9.2f}% | "
            f"{item['recall'] * 100:>9.2f}% | "
            f"{item['f1'] * 100:>9.2f}% | "
            f"{item['tp']:>5d} | "
            f"{item['fp']:>5d} | "
            f"{item['fn']:>5d} | "
            f"{item['support']:>8d}"
        )

    lines.append("=" * 78)
    return "\n".join(lines)


def evaluate_usps(
    model_path: str = "ann_weights_init.npz",
    data_folder: str = "Numerals",
    report_txt: str = "usps_evaluation_report.txt",
    cm_csv: str = "usps_confusion_matrix.csv",
):
    """Chạy đánh giá trên tập dữ liệu, in đầy đủ chỉ số thẳng cột và lưu báo cáo ra file."""
    weights_file = BASE_DIR / model_path
    if not weights_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {weights_file}")

    root_dir = BASE_DIR / data_folder
    if not root_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {root_dir}")

    model = NeuralNetwork.load_model(str(weights_file))
    X_usps, y_usps = load_dataset_by_digit_folders(root_dir)

    probs = model.forward(X_usps)
    y_pred = np.argmax(probs, axis=1)

    accuracy, cm, per_class, macro_scores = compute_metrics(y_usps, y_pred)
    report_text = format_usps_report(
        model_name=weights_file.name,
        dataset_name=root_dir.name,
        total_samples=len(y_usps),
        accuracy=accuracy,
        cm=cm,
        per_class=per_class,
        macro_scores=macro_scores,
    )

    print(report_text)

    txt_path = BASE_DIR / report_txt
    txt_path.write_text(report_text, encoding="utf-8")

    csv_path = BASE_DIR / cm_csv
    np.savetxt(csv_path, cm, delimiter=",", fmt="%d")

    print(f"Đã lưu báo cáo chi tiết tại: {txt_path}")
    print(f"Đã lưu ma trận nhầm lẫn CSV tại: {csv_path}")


if __name__ == "__main__":
    evaluate_usps(
        model_path="ann_weights.npz",
        data_folder="Numerals",
        report_txt="usps_evaluation_report.txt",
        cm_csv="usps_confusion_matrix.csv",
    )