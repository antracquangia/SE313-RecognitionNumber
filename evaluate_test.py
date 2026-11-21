import os
import urllib.request
from pathlib import Path
import numpy as np
from ann_model import NeuralNetwork

BASE_DIR = Path(__file__).resolve().parent


def load_mnist_test(filename="mnist.npz"):
    """Tải (nếu chưa có) và trả về tập X_test, y_test đã chuẩn hóa [0.0, 1.0]."""
    mnist_path = BASE_DIR / filename
    if not mnist_path.exists():
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        urllib.request.urlretrieve(url, mnist_path)

    data = np.load(mnist_path)
    X_test = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
    y_test = data["y_test"]
    return X_test, y_test


def compute_test_metrics(y_true, y_pred, num_classes=10):
    """Tính Confusion Matrix (Hàng: Dự đoán | Cột: Thực tế), Accuracy và Precision/Recall/F1."""
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

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class.append({
            "digit": c,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return accuracy, cm, per_class


def format_report(model_path, accuracy, cm, per_class):
    """Định dạng nội dung báo cáo đánh giá trên tập Test."""
    lines = [
        "=" * 64,
        f"BÁO CÁO ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST",
        f"File trọng số: {model_path}",
        f"Độ chính xác tổng thể (Test Accuracy): {accuracy * 100:.2f}%",
        "=" * 64,
        "",
        "MA TRẬN NHẦM LẪN (Hàng: Dự đoán | Cột: Thực tế)",
        str(cm),
        "",
        f"{'Lớp':<6} | {'Precision':<11} | {'Recall':<11} | {'F1-Score':<11} | {'TP/FP/FN'}",
        "-" * 64,
    ]

    for item in per_class:
        lines.append(
            f"Số {item['digit']:<3} | "
            f"{item['precision'] * 100:>9.2f}% | "
            f"{item['recall'] * 100:>9.2f}% | "
            f"{item['f1'] * 100:>9.2f}% | "
            f"{item['tp']}/{item['fp']}/{item['fn']}"
        )

    lines.append("=" * 64)
    return "\n".join(lines)


def save_test_evaluation(
    model_path="ann_weights_finetuned.npz",
    report_txt="test_evaluation_report.txt",
    cm_csv="test_confusion_matrix.csv",
):
    """Chạy đánh giá trên tập Test và lưu kết quả ra file riêng."""
    weights_file = BASE_DIR / model_path
    if not weights_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {weights_file}")

    model = NeuralNetwork.load_model(str(weights_file))
    X_test, y_test = load_mnist_test()

    probs = model.forward(X_test)
    y_pred = np.argmax(probs, axis=1)

    accuracy, cm, per_class = compute_test_metrics(y_test, y_pred)
    report_text = format_report(str(weights_file.name), accuracy, cm, per_class)

    print(report_text)

    txt_path = BASE_DIR / report_txt
    txt_path.write_text(report_text, encoding="utf-8")

    csv_path = BASE_DIR / cm_csv
    np.savetxt(csv_path, cm, delimiter=",", fmt="%d")

    print(f"\nĐã lưu báo cáo chi tiết tại: {txt_path}")
    print(f"Đã lưu ma trận nhầm lẫn CSV tại: {csv_path}")


if __name__ == "__main__":
    save_test_evaluation(model_path="ann_weights.npz")