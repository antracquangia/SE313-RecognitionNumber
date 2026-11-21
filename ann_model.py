import numpy as np

def sigmoid(x):
    """Hàm kích hoạt Sigmoid, giới hạn miền giá trị để tránh tràn số."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(output):
    """Đạo hàm hàm Sigmoid theo giá trị đầu ra o: o * (1 - o)."""
    return output * (1.0 - output)

class NeuralNetwork:
    def __init__(self, layer_sizes, seed=42):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        self.prev_weight_updates = []
        self.prev_bias_updates = []
        self.rng = np.random.default_rng(seed)

        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            # Khởi tạo cân đối phù hợp với tầng dùng Sigmoid để hạn chế bão hòa sớm
            scale = np.sqrt(1.0 / in_dim)
            w = self.rng.normal(0.0, scale, size=(in_dim, out_dim)).astype(np.float32)
            b = np.zeros((1, out_dim), dtype=np.float32)
            
            self.weights.append(w)
            self.biases.append(b)
            self.prev_weight_updates.append(np.zeros_like(w))
            self.prev_bias_updates.append(np.zeros_like(b))

    def forward(self, X):
        """Lan truyền xuôi qua các tầng."""
        self.activations = [X]
        current_input = X
        for i in range(self.num_layers - 1):
            net = np.dot(current_input, self.weights[i]) + self.biases[i]
            current_input = sigmoid(net)
            self.activations.append(current_input)
        return current_input

    def backward(self, y_true, learning_rate, momentum):
        """Lan truyền ngược sai số và cập nhật trọng số bằng Gradient Descent + Momentum."""
        m = y_true.shape[0]
        output = self.activations[-1]
        error = y_true - output
        deltas = [error * sigmoid_derivative(output)]

        for i in range(self.num_layers - 2, 0, -1):
            error = np.dot(deltas[0], self.weights[i].T)
            delta_h = error * sigmoid_derivative(self.activations[i])
            deltas.insert(0, delta_h)

        for i in range(len(self.weights)):
            grad_w = np.dot(self.activations[i].T, deltas[i]) / m
            grad_b = np.sum(deltas[i], axis=0, keepdims=True) / m

            weight_update = learning_rate * grad_w + momentum * self.prev_weight_updates[i]
            bias_update = learning_rate * grad_b + momentum * self.prev_bias_updates[i]

            self.weights[i] += weight_update
            self.biases[i] += bias_update

            self.prev_weight_updates[i] = weight_update
            self.prev_bias_updates[i] = bias_update

    def train(self, X_train, y_train, epochs, batch_size, learning_rate, momentum):
        """Vòng lặp huấn luyện theo mini-batch."""
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            self.rng.permutation(num_samples)
            X_train, y_train = X_train[indices], y_train[indices]

            epoch_loss = 0.0
            num_batches = int(np.ceil(num_samples / batch_size))

            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                predictions = self.forward(X_batch)
                epoch_loss += np.mean(np.square(y_batch - predictions))
                self.backward(y_batch, learning_rate, momentum)

            print(f"Epoch {epoch + 1}/{epochs} - Loss (MSE): {epoch_loss / num_batches:.4f}")

    def evaluate(self, X_test, y_test_labels):
        """Đánh giá độ chính xác (%) trên tập kiểm thử."""
        predictions = self.forward(X_test)
        predicted_labels = np.argmax(predictions, axis=1)
        return np.mean(predicted_labels == y_test_labels) * 100

    def predict_single(self, X):
        """Dự đoán nhãn (0-9) và độ tin cậy (%) cho 1 mẫu."""
        probs = self.forward(X)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class]) * 100
        return pred_class, confidence

    def save_model(self, filepath="ann_weights.npz"):
        """Xuất kiến trúc mạng, weights và biases ra file nén NumPy."""
        save_dict = {"layer_sizes": np.array(self.layer_sizes)}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            save_dict[f"w_{i}"] = w
            save_dict[f"b_{i}"] = b
        np.savez(filepath, **save_dict)
        print(f"Đã lưu mô hình thành công tại: {filepath}")

    @classmethod
    def load_model(cls, filepath="ann_weights.npz"):
        """Nạp lại mô hình từ file đã huấn luyện."""
        data = np.load(filepath)
        layer_sizes = data["layer_sizes"].tolist()
        model = cls(layer_sizes)
        for i in range(len(layer_sizes) - 1):
            model.weights[i] = data[f"w_{i}"]
            model.biases[i] = data[f"b_{i}"]
        return model