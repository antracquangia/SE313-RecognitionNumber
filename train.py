import copy
import os
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ann_model import NeuralNetwork

BASE_DIR = Path(__file__).resolve().parent


def load_mnist_npz(filename="mnist.npz"):
    """Tải (nếu chưa có) và chuẩn hóa dữ liệu MNIST về [0.0, 1.0]."""
    mnist_path = BASE_DIR / filename
    if not mnist_path.exists():
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        urllib.request.urlretrieve(url, mnist_path)

    data = np.load(mnist_path)
    X_train = data["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
    X_test = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
    return X_train, data["y_train"], X_test, data["y_test"]


def stratified_train_val_split(X, y, val_ratio=0.1, seed=42):
    """Chia tập train/val để tỷ lệ ảnh giữa các chữ số (0-9) gần như nhau."""
    rng = np.random.default_rng(seed)
    train_idx_list = []
    val_idx_list = []

    for digit in np.unique(y):
        digit_indices = np.flatnonzero(y == digit)
        rng.shuffle(digit_indices)

        n_val = int(round(len(digit_indices) * val_ratio))
        val_idx_list.append(digit_indices[:n_val])
        train_idx_list.append(digit_indices[n_val:])

    train_idx = rng.permutation(np.concatenate(train_idx_list))
    val_idx = rng.permutation(np.concatenate(val_idx_list))

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def get_lr_by_epoch(base_lr=0.05, epoch=0, decay=0.96, min_lr=1e-4):
    """Giảm dần learning rate theo từng epoch để tránh vọt qua điểm tối ưu."""
    return max(min_lr, base_lr * (decay ** epoch))


def augment_batch_light(X_batch_flat, rng, max_shift=2, max_angle=5.0, scale_range=(0.9, 1.1)):
    """Augmentation nhẹ trên ảnh 28x28 cho batch huấn luyện."""
    batch_28 = X_batch_flat.reshape(-1, 28, 28)
    out_28 = np.empty_like(batch_28, dtype=np.float32)

    for i in range(batch_28.shape[0]):
        img_u8 = np.clip(batch_28[i] * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_u8, mode="L")

        angle = float(rng.uniform(-max_angle, max_angle))
        dx = float(rng.integers(-max_shift, max_shift + 1))
        dy = float(rng.integers(-max_shift, max_shift + 1))

        theta = np.deg2rad(angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        cx, cy = 13.5, 13.5

        a = cos_t
        b = sin_t
        c = (cx - dx) - a * cx - b * cy

        d = -sin_t
        e = cos_t
        f = (cy - dy) - d * cx - e * cy

        transformed = pil_img.transform(
            (28, 28),
            Image.Transform.AFFINE,
            (a, b, c, d, e, f),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )

        arr_aug = np.asarray(transformed, dtype=np.float32) / 255.0
        gain = float(rng.uniform(scale_range[0], scale_range[1]))
        out_28[i] = np.clip(arr_aug * gain, 0.0, 1.0)

    return out_28.reshape(-1, 784)


def one_hot(y, num_classes=10):
    """Chuyển nhãn số nguyên (0-9) sang vector one-hot (N, 10)."""
    res = np.zeros((y.size, num_classes), dtype=np.float32)
    res[np.arange(y.size), y] = 1.0
    return res


def compute_mse_loss(y_true, y_pred):
    """Tính sai số bình phương trung bình (MSE)."""
    return float(np.mean(np.square(y_true - y_pred)))


def save_checkpoint_npz(filepath, model, history):
    """Lưu trọng số tốt nhất cùng lịch sử huấn luyện vào file .npz."""
    save_dict = {
        "layer_sizes": np.array(model.layer_sizes),
        "train_loss": np.array(history["train_loss"]),
        "val_loss": np.array(history["val_loss"]),
        "train_acc": np.array(history["train_acc"]),
        "val_acc": np.array(history["val_acc"]),
    }
    for idx, (w, b) in enumerate(zip(model.weights, model.biases)):
        save_dict[f"w_{idx}"] = w
        save_dict[f"b_{idx}"] = b
    np.savez(filepath, **save_dict)


def plot_learning_curves(history, save_path="learning_curves_init.png"):
    """Vẽ và lưu đường cong Loss và Accuracy qua các epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    plt.plot(epochs, history["val_loss"], "r-s", label="Validation Loss", markersize=4)
    plt.title("Loss Curve (Initial Training)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc", markersize=4)
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], "r-s", label="Validation Acc", markersize=4)
    plt.title("Accuracy Curve (Initial Training)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    seed = 42
    rng = np.random.default_rng(seed)

    X_full, y_full, X_test, y_test = load_mnist_npz()
    X_train, y_train_raw, X_val, y_val_raw = stratified_train_val_split(
        X_full, y_full, val_ratio=0.1, seed=seed
    )

    # Kiểm tra nhanh số lượng từng số ở Train và Val
    print("Phân bố số lượng từng chữ số (Train | Val):")
    for d in range(10):
        print(f"  Số {d}: {np.sum(y_train_raw == d):4d} | {np.sum(y_val_raw == d):3d}")

    y_train_oh = one_hot(y_train_raw)
    y_val_oh = one_hot(y_val_raw)

    # Khởi tạo mô hình từ đầu
    nn = NeuralNetwork(layer_sizes=[784, 128, 10])
    epochs = 35
    batch_size = 64
    base_lr = 0.05
    lr_decay = 0.96
    momentum = 0.85

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    best_weights = None
    best_biases = None
    num_samples = X_train.shape[0]

    for epoch in range(epochs):
        current_lr = get_lr_by_epoch(base_lr=base_lr, epoch=epoch, decay=lr_decay)
        perm = rng.permutation(num_samples)
        X_shuf = X_train[perm]
        y_shuf = y_train_oh[perm]

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            X_aug = augment_batch_light(
                X_shuf[start:end], rng, max_shift=2, max_angle=5.0, scale_range=(0.9, 1.1)
            )
            nn.forward(X_aug)
            nn.backward(y_shuf[start:end], current_lr, momentum)

        t_out = nn.forward(X_train)
        v_out = nn.forward(X_val)

        t_loss = compute_mse_loss(y_train_oh, t_out)
        v_loss = compute_mse_loss(y_val_oh, v_out)
        t_acc = float(np.mean(np.argmax(t_out, axis=1) == y_train_raw))
        v_acc = float(np.mean(np.argmax(v_out, axis=1) == y_val_raw))

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} (lr={current_lr:.5f}) | "
            f"Train Acc: {t_acc * 100:.2f}% | Val Acc: {v_acc * 100:.2f}%"
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_weights = copy.deepcopy(nn.weights)
            best_biases = copy.deepcopy(nn.biases)

    nn.weights = best_weights
    nn.biases = best_biases

    out_weights = BASE_DIR / "ann_weights_init.npz"
    save_checkpoint_npz(out_weights, nn, history)

    csv_data = np.column_stack((
        np.arange(1, epochs + 1),
        history["train_loss"],
        history["val_loss"],
        history["train_acc"],
        history["val_acc"],
    ))
    np.savetxt(
        BASE_DIR / "training_history_init.csv",
        csv_data,
        delimiter=",",
        header="epoch,train_loss,val_loss,train_acc,val_acc",
        comments="",
        fmt="%.6f",
    )

    test_acc = float(np.mean(np.argmax(nn.forward(X_test), axis=1) == y_test))
    print(f"\nĐã lưu '{out_weights.name}' | Best Val Acc: {best_val_acc * 100:.2f}% | Test Acc: {test_acc * 100:.2f}%")
    plot_learning_curves(history)