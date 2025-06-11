# Dán toàn bộ nội dung của file heart.py gốc của bạn vào đây
# (Lớp HeartSignal và tất cả các hàm import)
# ... (toàn bộ mã từ heart.py) ...
from math import cos, pi
import numpy as np
import cv2
import os, glob
import tkinter as tk
import time

class HeartSignal:
    # ... (giữ nguyên toàn bộ mã của lớp HeartSignal) ...
    def __init__(self, curve="heart", title="Love U", frame_num=60, seed_points_num=2000, seed_num=None, highlight_rate=0.3,
                 background_img_dir="", set_bg_imgs=False, bg_img_scale=0.2, bg_weight=0.3, curve_weight=0.7, frame_width=1080, frame_height=960, scale=10.1,
                 base_color=None, highlight_points_color_1=None, highlight_points_color_2=None, wait=100):
        super().__init__()
        self.curve = curve
        self.title = title
        self.highlight_points_color_2 = highlight_points_color_2
        self.highlight_points_color_1 = highlight_points_color_1
        self.highlight_rate = highlight_rate
        self.base_color = base_color
        self.m_star, self.n_star = None, None
        star_curve = {"star-5": (5, 2), "star-6": (6, 2), "star-7": (7, 3), "star-7-1": (7, 3), "star-7-2": (7, 2)}
        if "star" in curve:
            self.n_star, self.m_star = star_curve[curve]

        self.curve_weight = curve_weight
        img_paths = glob.glob(background_img_dir + "/*")
        self.bg_imgs = []
        self.set_bg_imgs = set_bg_imgs
        self.bg_weight = bg_weight
        if os.path.exists(background_img_dir) and len(img_paths) > 0 and set_bg_imgs:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                self.bg_imgs.append(img)
            first_bg = self.bg_imgs[0]
            width = int(first_bg.shape[1] * bg_img_scale)
            height = int(first_bg.shape[0] * bg_img_scale)
            first_bg = cv2.resize(first_bg, (width, height), interpolation=cv2.INTER_AREA)

            # 对齐图片，自动裁切中间
            new_bg_imgs = [first_bg, ]
            for img in self.bg_imgs[1:]:
                width_close = abs(first_bg.shape[1] - img.shape[1]) < abs(first_bg.shape[0] - img.shape[0])
                if width_close:
                    # resize
                    height = int(first_bg.shape[1] / img.shape[1] * img.shape[0])
                    width = first_bg.shape[1]
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                    # crop and fill
                    if img.shape[0] > first_bg.shape[0]:
                        crop_num = img.shape[0] - first_bg.shape[0]
                        crop_top = crop_num // 2
                        crop_bottom = crop_num - crop_top
                        img = np.delete(img, range(crop_top), axis=0)
                        img = np.delete(img, range(img.shape[0] - crop_bottom, img.shape[0]), axis=0)
                    elif img.shape[0] < first_bg.shape[0]:
                        fill_num = first_bg.shape[0] - img.shape[0]
                        fill_top = fill_num // 2
                        fill_bottom = fill_num - fill_top
                        img = np.concatenate([np.zeros([fill_top, width, 3]), img, np.zeros([fill_bottom, width, 3])], axis=0)
                else:
                    width = int(first_bg.shape[0] / img.shape[0] * img.shape[1])
                    height = first_bg.shape[0]
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                    # crop and fill
                    if img.shape[1] > first_bg.shape[1]:
                        crop_num = img.shape[1] - first_bg.shape[1]
                        crop_top = crop_num // 2
                        crop_bottom = crop_num - crop_top
                        img = np.delete(img, range(crop_top), axis=1)
                        img = np.delete(img, range(img.shape[1] - crop_bottom, img.shape[1]), axis=1)
                    elif img.shape[1] < first_bg.shape[1]:
                        fill_num = first_bg.shape[1] - img.shape[1]
                        fill_top = fill_num // 2
                        fill_bottom = fill_num - fill_top
                        img = np.concatenate([np.zeros([fill_top, width, 3]), img, np.zeros([fill_bottom, width, 3])], axis=1)
                new_bg_imgs.append(img)
            self.bg_imgs = new_bg_imgs
            assert all(img.shape[0] == first_bg.shape[0] and img.shape[1] == first_bg.shape[1] for img in self.bg_imgs), "背景图片宽和高不一致"
            self.frame_width = self.bg_imgs[0].shape[1]
            self.frame_height = self.bg_imgs[0].shape[0]
        else:
            self.frame_width = frame_width  # 窗口宽度
            self.frame_height = frame_height  # 窗口高度
        self.center_x = self.frame_width / 2
        self.center_y = self.frame_height / 2
        self.main_curve_width = -1
        self.main_curve_height = -1

        self.frame_points = []  # 每帧动态点坐标
        self.frame_num = frame_num  # 帧数
        self.seed_num = seed_num  # 伪随机种子，设置以后除光晕外粒子相对位置不动（减少内部闪烁感）
        self.seed_points_num = seed_points_num  # 主图粒子数
        self.scale = scale  # 缩放比例
        self.wait = wait

    def curve_function(self, curve):
        if "star" in curve:
            return self.star_function
        curve_dict = {
            "heart": self.heart_function,
            "butterfly": self.butterfly_function,
        }
        return curve_dict[curve]

    def heart_function(self, t, frame_idx=0, scale=5.20):
        """
        图形方程
        :param frame_idx: 帧的索引，根据帧数变换心形
        :param scale: 放大比例
        :param t: 参数
        :return: 坐标
        """
        trans = 3 - (1 + self.periodic_func(frame_idx, self.frame_num)) * 0.5  # 改变心形饱满度度的参数

        x = 15 * (np.sin(t) ** 3)
        t = np.where((pi < t) & (t < 2 * pi), 2 * pi - t, t)  # 翻转x > 0部分的图形到3、4象限
        y = -(14 * np.cos(t) - 4 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(trans * t))

       # --- BẮT ĐẦU THAY THẾ ---
        # Cơ chế "làm mỏng" đường kẻ giữa trái tim một cách nhất quán
        ign_area = 0.3  # Khu vực được coi là trung tâm
        center_ids = np.where((x > -ign_area) & (x < ign_area))
        
        # Thay vì xóa tất cả, chúng ta chỉ xóa một phần lớn (ví dụ: 85%) các điểm trên đường kẻ đó
        # để phá vỡ cấu trúc đường thẳng nhưng không tạo ra khoảng trống.
        num_to_delete = int(len(center_ids[0]) * 0.85) 
        
        # Chỉ thực hiện nếu có điểm để xóa
        if num_to_delete > 0:
            # Chọn ngẫu nhiên các chỉ số của các điểm cần xóa từ danh sách các điểm ở giữa
            indices_to_delete = np.random.choice(center_ids[0], num_to_delete, replace=False)
            # Xóa các điểm đã chọn
            x, y = np.delete(x, indices_to_delete), np.delete(y, indices_to_delete)
        # --- KẾT THÚC THAY THẾ ---

        # 放大
        x *= scale
        y *= scale

        # 移到画布中央
        x += self.center_x
        y += self.center_y

        # 原心形方程
        # x = 15 * (sin(t) ** 3)
        # y = -(14 * cos(t) - 4 * cos(2 * t) - 2 * cos(3 * t) - cos(3 * t))
        return x.astype(int), y.astype(int)

    def butterfly_function(self, t, frame_idx=0, scale=5.2):
        """
        图形函数
        :param frame_idx:
        :param scale: 放大比例
        :param t: 参数
        :return: 坐标
        """
        # 基础函数
        # t = t * pi
        p = np.exp(np.sin(t)) - 2.5 * np.cos(4 * t) + np.sin(t) ** 5
        x = 5 * p * np.cos(t)
        y = - 5 * p * np.sin(t)

        # 放大
        x *= scale
        y *= scale

        # 移到画布中央
        x += self.center_x
        y += self.center_y

        return x.astype(int), y.astype(int)

    def star_function(self, t, frame_idx=0, scale=5.2):
        n = self.n_star / self.m_star
        p = np.cos(pi / n) / np.cos(pi / n - (t % (2 * pi / n)))

        y = - 15 * p * np.cos(t)
        x = 15 * p * np.sin(t)

        if self.m_star > 1 and self.n_star % self.m_star == 0:
            x = np.concatenate([x, x], axis=0)
            y = np.concatenate([y, -y], axis=0)

        # 放大
        x *= scale
        y *= scale

        # 移到画布中央
        x += self.center_x
        y += self.center_y

        return x.astype(int), y.astype(int)

    def shrink(self, x, y, ratio, offset=1, p=0.5, dist_func="uniform"):
        """
        带随机位移的抖动
        :param x: 原x
        :param y: 原y
        :param ratio: 缩放比例
        :param p:
        :param offset:
        :return: 转换后的x,y坐标
        """
        x_ = (x - self.center_x)
        y_ = (y - self.center_y)
        force = 1 / ((x_ ** 2 + y_ ** 2) ** p + 1e-30)

        dx = ratio * force * x_
        dy = ratio * force * y_

        def d_offset(x):
            if dist_func == "uniform":
                return x + np.random.uniform(-offset, offset, size=x.shape)
            elif dist_func == "norm":
                return x + offset * np.random.normal(0, 1, size=x.shape)

        dx, dy = d_offset(dx), d_offset(dy)

        return x - dx, y - dy

    def scatter(self, x, y, alpha=0.75, beta=0.15):
        """
        随机内部扩散的坐标变换
        :param alpha: 扩散因子 - 松散
        :param x: 原x
        :param y: 原y
        :param beta: 扩散因子 - 距离
        :return: x,y 新坐标
        """

        ratio_x = - beta * np.log(np.random.random(x.shape) * alpha)
        ratio_y = - beta * np.log(np.random.random(y.shape) * alpha)
        dx = ratio_x * (x - self.center_x)
        dy = ratio_y * (y - self.center_y)

        return x - dx, y - dy

    def periodic_func(self, x, x_num):
        t = (x / x_num) * 1.8 * pi
        amplitude = 1.2  # <-- giảm biên độ phồng, giá trị từ 0.5 đến 0.8 là hợp lý
        if t > pi:
            t_norm = (t + pi) / pi
            eased_t_norm = 1 + np.sin(t_norm * pi / 2)
            final_t = pi + eased_t_norm * pi
            # Thêm -1 * để đảo ngược giá trị trả về
            return -1 * amplitude * cos(final_t)
        else:
            t_norm = t / pi
            eased_t_norm = 6 - np.cos(t_norm * pi / 2)
            final_t = eased_t_norm * pi
            # Thêm -1 * để đảo ngược giá trị trả về
            return -1 * amplitude * cos(final_t)

    def gen_points(self, points_num, frame_idx, shape_func):
        # Độ nở
        cy = self.periodic_func(frame_idx, self.frame_num)
        ratio = (self.scale * 1) * cy

        # Đồ họa
        period = 2 * pi * self.m_star if "star" in self.curve else 2 * pi
        seed_points = np.linspace(0, period, points_num)
        seed_x, seed_y = shape_func(seed_points, frame_idx, scale=self.scale)
        x, y = self.shrink(seed_x, seed_y, ratio, offset=2)
        num_outline_points = int(points_num * 0.2) # Chỉ dùng 25% số hạt cho viền
        outline_indices = np.random.choice(np.arange(len(seed_x)), num_outline_points, replace=False)
        outline_x, outline_y = seed_x[outline_indices], seed_y[outline_indices]
        x, y = self.shrink(outline_x, outline_y, ratio, offset=2)
        curve_width, curve_height = int(x.max() - x.min()), int(y.max() - y.min())
        self.main_curve_width = max(self.main_curve_width, curve_width)
        self.main_curve_height = max(self.main_curve_height, curve_height)
        point_size = np.random.choice([2, 3], x.shape, replace=True, p=[0.7, 0.3])
        tag = np.ones_like(x)

        def delete_points(x_, y_, ign_area, ign_prop):
            ign_area = ign_area
            center_ids = np.where((x_ > self.center_x - ign_area) & (x_ < self.center_x + ign_area))
            center_ids = center_ids[0]
            np.random.shuffle(center_ids)
            del_num = round(len(center_ids) * ign_prop)
            del_ids = center_ids[:del_num]
            x_, y_ = np.delete(x_, del_ids), np.delete(y_, del_ids)
            return x_, y_

        # Khuyếch tán
        for idx, beta in enumerate(np.linspace(0.1, 0.2, 7)):
            alpha = 0.8 + beta
            x_, y_ = self.scatter(seed_x, seed_y, alpha, beta)
            x_, y_ = self.shrink(x_, y_, ratio, offset=round(beta * 50))
            x = np.concatenate((x, x_), 0)
            y = np.concatenate((y, y_), 0)
            p_size = np.random.choice([1, 2], x_.shape, replace=True, p=[0.55 + beta, 0.45 - beta])
            point_size = np.concatenate((point_size, p_size), 0)
            tag_ = np.ones_like(x_) * 2
            tag = np.concatenate((tag, tag_), 0)

        # Phần rìa trong
        halo_ratio = int(10 + 5 * abs(cy))

        # Phần rìa ngoài
        x_, y_ = shape_func(seed_points, frame_idx, scale=self.scale + 2.5)
        x_1, y_1 = self.shrink(x_, y_, halo_ratio, offset=50, dist_func="uniform")
        x_1, y_1 = delete_points(x_1, y_1, 30, 0.5)
        x = np.concatenate((x, x_1), 0)
        y = np.concatenate((y, y_1), 0)

        # Số lượng rìa
        halo_number = int(points_num * 2 - points_num * abs(cy))
        seed_points = np.random.uniform(2, 10 * pi, halo_number)
        x_, y_ = shape_func(seed_points, frame_idx, scale=self.scale + 3.9)
        x_2, y_2 = self.shrink(x_, y_, halo_ratio, offset=int(10 + 15 * abs(cy)), dist_func="norm")
        x_2, y_2 = delete_points(x_2, y_2, 20, 0.5)
        x = np.concatenate((x, x_2), 0)
        y = np.concatenate((y, y_2), 0)

        # Rìa mở
        x_3, y_3 = shape_func(np.linspace(0, 2 * pi, int(points_num * .4)),
                                             frame_idx, scale=self.scale + 0.2)
        x_3, y_3 = self.shrink(x_3, y_3, ratio * 2, offset=6)
        x = np.concatenate((x, x_3), 0)
        y = np.concatenate((y, y_3), 0)

        halo_len = x_1.shape[0] + x_2.shape[0] + x_3.shape[0]
        p_size = np.random.choice([1, 2, 3], halo_len, replace=True, p=[0.7, 0.2, 0.1])
        point_size = np.concatenate((point_size, p_size), 0)
        tag_ = np.ones(halo_len) * 2 * 3
        tag = np.concatenate((tag, tag_), 0)

        x_y = np.around(np.stack([x, y], axis=1), 0)
        x, y = x_y[:, 0], x_y[:, 1]
        return x, y, point_size, tag

    def get_frames(self, shape_func):
        for frame_idx in range(self.frame_num):
            np.random.seed(self.seed_num)
            self.frame_points.append(self.gen_points(self.seed_points_num, frame_idx, shape_func))

        frames = []

        def add_points(frame, x, y, size, tag):
            highlight1 = np.array(self.highlight_points_color_1, dtype='uint8')
            highlight2 = np.array(self.highlight_points_color_2, dtype='uint8')
            base_col = np.array(self.base_color, dtype='uint8')

            x, y = x.astype(int), y.astype(int)
            frame[y, x] = base_col

            size_2 = np.int64(size == 2)
            frame[y, x + size_2] = base_col
            frame[y + size_2, x] = base_col

            size_3 = np.int64(size == 3)
            frame[y + size_3, x] = base_col
            frame[y - size_3, x] = base_col
            frame[y, x + size_3] = base_col
            frame[y, x - size_3] = base_col
            frame[y + size_3, x + size_3] = base_col
            frame[y - size_3, x - size_3] = base_col
            # frame[y - size_3, x + size_3] = color
            # frame[y + size_3, x - size_3] = color

            # ???
            random_sample = np.random.choice([1, 0], size=tag.shape, p=[self.highlight_rate, 1 - self.highlight_rate])

            # tag2_size1 = np.int64((tag <= 2) & (size == 1) & (random_sample == 1))
            # frame[y * tag2_size1, x * tag2_size1] = highlight2

            tag2_size2 = np.int64((tag <= 2) & (size == 2) & (random_sample == 1))
            frame[y * tag2_size2, x * tag2_size2] = highlight1
            # frame[y * tag2_size2, (x + 1) * tag2_size2] = highlight2
            # frame[(y + 1) * tag2_size2, x * tag2_size2] = highlight2
            frame[(y + 1) * tag2_size2, (x + 1) * tag2_size2] = highlight2

        for x, y, size, tag in self.frame_points:
            frame = np.zeros([self.frame_height, self.frame_width, 3], dtype="uint8")
            add_points(frame, x, y, size, tag)
            frames.append(frame)

        return frames

    # Xóa hàm draw() cũ, vì chúng ta không dùng cv2.imshow() nữa
    # def draw(self, times=10):
    #     ...
    
# === BẮT ĐẦU MÃ FLASK ===
from flask import Flask, Response, render_template, jsonify
import yaml

# Khởi tạo ứng dụng Flask
app = Flask(__name__, template_folder='.')

# Đọc cấu hình từ file settings.yaml
try:
    with open("settings.yaml", "r", encoding="utf-8") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'settings.yaml'. Vui lòng đảm bảo file tồn tại.")
    settings = {} # Dùng cấu hình mặc định nếu không có file

# Bỏ các khóa không cần thiết cho lớp HeartSignal
if "times" in settings:
    del settings["times"]
if settings.get("wait") == -1:
    settings["wait"] = int(settings.get("period_time", 1000) / settings.get("frame_num", 20))
if "period_time" in settings:
    del settings["period_time"]

# Tạo một thực thể (instance) của HeartSignal
heart = HeartSignal(seed_num=5201314, **settings)
# Tạo trước tất cả các khung hình
animation_frames = heart.get_frames(heart.curve_function(heart.curve))
current_frame_index = 0

def generate_video_stream():
    """Hàm tạo luồng video, liên tục lặp lại các khung hình."""
    # --- BẮT ĐẦU SỬA LỖI ---
    # Lấy giá trị chờ (từ miligiây sang giây) trực tiếp từ đối tượng heart
    # và lưu vào một biến cục bộ của hàm này.
    wait_in_seconds = heart.wait / 1000.0
    # --- KẾT THÚC SỬA LỖI ---

    current_frame_index = 0
    while True:
        frame = animation_frames[current_frame_index]
        current_frame_index = (current_frame_index + 1) % len(animation_frames)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if len(heart.bg_imgs) > 0 and heart.set_bg_imgs:
            bg_index = current_frame_index % len(heart.bg_imgs)
            frame_bgr = cv2.addWeighted(heart.bg_imgs[bg_index], heart.bg_weight, frame_bgr, heart.curve_weight, 0)
        
        ret, jpeg = cv2.imencode('.jpg', frame_bgr)
        if not ret:
            continue
        
        # Gửi khung hình dưới dạng một phần của multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
        # Chờ một khoảng thời gian tương ứng với wait time
        # Sử dụng biến cục bộ đã được định nghĩa ở trên
        time.sleep(wait_in_seconds)

@app.route('/')
def index():
    """Phục vụ file index.html."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route cung cấp luồng video."""
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Máy chủ đang chạy tại http://localhost:3000/")
    print("Mở trình duyệt của bạn và truy cập địa chỉ này.")
    # Sử dụng một WSGI server tốt hơn cho streaming, ví dụ waitress
    # app.run(host='0.0.0.0', port=3000, debug=False, threaded=True)
    from waitress import serve
    serve(app, host='0.0.0.0', port=3000)