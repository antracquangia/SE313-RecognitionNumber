import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ann_model import NeuralNetwork


class DigitRecognizerGUI:
    def __init__(self, root, model_path="ann_weights.npz"):
        self.root = root
        self.root.title("Nhận diện chữ số viết tay (ANN - NumPy)")
        self.root.resizable(False, False)

        self.canvas_size = 280
        self.brush_radius = 7

        # Nạp mô hình đã lưu
        self.model = NeuralNetwork.load_model(model_path)

        # Ảnh gốc độ phân giải cao để vẽ mượt, sau đó mới resize về 28x28
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None

        self._build_ui()

    def _build_ui(self):
        main_frame = tk.Frame(self.root, padx=14, pady=14, bg="#18181b")
        main_frame.pack()

        # Khu vực bên trái: Canvas vẽ & nút bấm
        left_frame = tk.Frame(main_frame, bg="#18181b")
        left_frame.pack(side=tk.LEFT, padx=(0, 16))

        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#000000",
            highlightthickness=2,
            highlightbackground="#3f3f46",
        )
        self.canvas.pack()

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_draw)

        btn_frame = tk.Frame(left_frame, bg="#18181b")
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        btn_predict = tk.Button(
            btn_frame,
            text="Dự đoán",
            command=self.predict_digit,
            bg="#2563eb",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=6,
            cursor="hand2",
        )
        btn_predict.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        btn_clear = tk.Button(
            btn_frame,
            text="Xóa",
            command=self.clear_canvas,
            bg="#3f3f46",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=6,
            cursor="hand2",
        )
        btn_clear.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))

        # Khu vực bên phải: Kết quả dự đoán & xác suất 0-9
        right_frame = tk.Frame(main_frame, bg="#18181b", width=210)
        right_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            right_frame,
            text="KẾT QUẢ DỰ ĐOÁN",
            bg="#18181b",
            fg="#a1a1aa",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")

        self.lbl_result = tk.Label(
            right_frame,
            text="?",
            bg="#18181b",
            fg="#4ade80",
            font=("Segoe UI", 42, "bold"),
        )
        self.lbl_result.pack(pady=(2, 0))

        self.lbl_conf = tk.Label(
            right_frame,
            text="Độ tin cậy: --%",
            bg="#18181b",
            fg="#e4e4e7",
            font=("Segoe UI", 10),
        )
        self.lbl_conf.pack(pady=(0, 10))

        self.prob_labels = {}
        for digit in range(10):
            row = tk.Frame(right_frame, bg="#18181b")
            row.pack(fill=tk.X, pady=1)

            tk.Label(
                row,
                text=f"{digit}:",
                bg="#18181b",
                fg="#d4d4d8",
                font=("Consolas", 9),
                width=3,
                anchor="w",
            ).pack(side=tk.LEFT)

            lbl_val = tk.Label(
                row,
                text="0.0%",
                bg="#18181b",
                fg="#a1a1aa",
                font=("Consolas", 9),
                anchor="e",
            )
            lbl_val.pack(side=tk.RIGHT)
            self.prob_labels[digit] = lbl_val

    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self._draw_point(event.x, event.y)

    def paint(self, event):
        if self.last_x is not None and self.last_y is not None:
            r = self.brush_radius
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                fill="#ffffff",
                width=r * 2,
                capstyle=tk.ROUND,
                smooth=True,
            )
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255,
                width=r * 2,
                joint="curve",
            )
        self.last_x = event.x
        self.last_y = event.y

    def _draw_point(self, x, y):
        r = self.brush_radius
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="#ffffff", outline="#ffffff")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255, outline=255)

    def reset_draw(self, _event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.lbl_result.config(text="?")
        self.lbl_conf.config(text="Độ tin cậy: --%")
        for d in range(10):
            self.prob_labels[d].config(text="0.0%", fg="#a1a1aa")

    def shift_to_center_of_mass(self, arr_28):
        """Dịch ảnh 28x28 (giá trị [0, 1]) sao cho trọng tâm gần vị trí (13.5, 13.5)."""
        total_mass = float(np.sum(arr_28))
        if total_mass <= 1e-6:
            return arr_28

        rows, cols = np.indices(arr_28.shape, dtype=np.float32)
        cy = float(np.sum(rows * arr_28)) / total_mass
        cx = float(np.sum(cols * arr_28)) / total_mass

        shift_y = int(round(13.5 - cy))
        shift_x = int(round(13.5 - cx))

        img_u8 = np.clip(arr_28 * 255.0, 0, 255).astype(np.uint8)
        pil_shifted = Image.fromarray(img_u8, mode="L").transform(
            (28, 28),
            Image.Transform.AFFINE,
            (1, 0, -shift_x, 0, 1, -shift_y),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )
        return np.asarray(pil_shifted, dtype=np.float32) / 255.0

    def preprocess_image(self):
        """Cắt vùng chữ số, thu nhỏ về 20x20, đặt vào 28x28 và căn theo trọng tâm."""
        arr = np.array(self.image, dtype=np.uint8)
        coords = np.argwhere(arr > 20)
        if coords.size == 0:
            return None

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        cropped = self.image.crop((x_min, y_min, x_max, y_max))

        # Giữ tỉ lệ khung hình và resize vào vùng tối đa 20x20
        w, h = cropped.size
        if w > h:
            new_w = 20
            new_h = max(1, int(round(h * 20 / w)))
        else:
            new_h = 20
            new_w = max(1, int(round(w * 20 / h)))

        resized = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Đặt vào khung 28x28 ban đầu rồi căn theo trọng tâm
        canvas_28 = Image.new("L", (28, 28), color=0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        canvas_28.paste(resized, (paste_x, paste_y))

        norm_arr = np.array(canvas_28, dtype=np.float32) / 255.0
        centered_arr = self.shift_to_center_of_mass(norm_arr)
        return centered_arr.reshape(1, 784)

    def predict_digit(self):
        x_input = self.preprocess_image()
        if x_input is None:
            self.lbl_result.config(text="?")
            self.lbl_conf.config(text="Hãy vẽ một chữ số!")
            return

        probs = self.model.forward(x_input)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class]) * 100

        self.lbl_result.config(text=str(pred_class))
        self.lbl_conf.config(text=f"Độ tin cậy: {confidence:.1f}%")

        for d in range(10):
            pct = float(probs[d]) * 100
            color = "#4ade80" if d == pred_class else "#a1a1aa"
            self.prob_labels[d].config(text=f"{pct:5.1f}%", fg=color)


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerGUI(root, model_path="ann_weights.npz")
    root.mainloop()