from PIL import Image, ImageTk
import os
import math
import threading
from collections import Counter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ================= 【固定参数配置（一般无需修改）】 =================
# 自动识别近红/近白的距离阈值，仅影响基底图的颜色识别，不影响后续判定
RED_RECOGNIZE_THRESHOLD = 200
WHITE_RECOGNIZE_THRESHOLD = 150
# 纯基准色定义
PURE_RED = (255, 0, 0)
PURE_WHITE = (255, 255, 255)
PURE_GRAY = (128, 128, 128)
# 界面默认尺寸
WIN_WIDTH = 1000
WIN_HEIGHT = 700
# 预览图最大尺寸
PREVIEW_MAX_W = 600
PREVIEW_MAX_H = 600
# 阈值默认值
DEFAULT_THRESHOLD = 95

# ================= 【核心工具函数】 =================
def color_distance(c1, c2):
    """计算两个RGB颜色的欧氏距离"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def get_dominant_near_color(img, target_rgb, distance_threshold):
    """从图片中识别像素最多的目标近色"""
    pixels = list(img.getdata())
    rgb_counter = Counter(pixels)
    candidate_rgbs = []
    for rgb, count in rgb_counter.items():
        dist = color_distance(rgb, target_rgb)
        if dist < distance_threshold:
            candidate_rgbs.append((-count, dist, rgb))
    if not candidate_rgbs:
        return None
    candidate_rgbs.sort()
    return candidate_rgbs[0][2]

# ================= 【主软件界面类】 =================
class ImageProcessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("实验室图像批量处理工具")
        self.root.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}")
        self.root.resizable(True, True)

        # 核心状态变量
        self.base_img_path = ""  # 基底图片路径
        self.base_img = None  # 基底原图对象
        self.near_red = None  # 自动识别的近红色
        self.near_white = None  # 自动识别的近白色
        self.current_threshold = tk.IntVar(value=DEFAULT_THRESHOLD)  # 当前红色判定阈值
        self.processed_base_img = None  # 处理后的基底预览图
        self.exp_folder_path = ""  # 实验图像文件夹路径
        self.save_root_path = ""  # 输出保存根路径

        # 界面布局初始化
        self._init_ui()
        # 初始状态锁定
        self._lock_ui("init")

    def _init_ui(self):
        """初始化界面布局"""
        # ========== 顶部：基底文件选择区 ==========
        self.top_frame = tk.Frame(self.root, padx=10, pady=10)
        self.top_frame.pack(fill=tk.X, anchor=tk.N)

        self.select_base_btn = tk.Button(
            self.top_frame, text="选择/更换基底图片", command=self.select_base_file,
            font=("微软雅黑", 11), width=20, bg="#409EFF", fg="white"
        )
        self.select_base_btn.grid(row=0, column=0, padx=5)

        self.base_path_label = tk.Label(
            self.top_frame, text="未选择基底图片", font=("微软雅黑", 10),
            fg="#666666", anchor=tk.W
        )
        self.base_path_label.grid(row=0, column=1, padx=10, sticky=tk.W)

        # ========== 中间主体区 ==========
        self.main_frame = tk.Frame(self.root, padx=10, pady=5)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧：预览区
        self.preview_frame = tk.LabelFrame(self.main_frame, text="基底图处理预览", font=("微软雅黑", 10))
        self.preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.preview_canvas = tk.Canvas(self.preview_frame, bg="#F5F5F5")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 右侧：阈值控制区
        self.control_frame = tk.LabelFrame(self.main_frame, text="阈值调整控制", font=("微软雅黑", 10), width=280)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=0)
        self.control_frame.pack_propagate(False)

        # 阈值调整内容
        tk.Label(self.control_frame, text="红色判定阈值", font=("微软雅黑", 11)).pack(pady=(30, 10))

        self.threshold_frame = tk.Frame(self.control_frame)
        self.threshold_frame.pack(pady=5)

        self.minus_btn = tk.Button(
            self.threshold_frame, text="-", font=("微软雅黑", 12), width=3,
            command=self.threshold_minus
        )
        self.minus_btn.grid(row=0, column=0, padx=5)

        self.threshold_entry = tk.Entry(
            self.threshold_frame, textvariable=self.current_threshold,
            font=("微软雅黑", 12), width=8, justify=tk.CENTER
        )
        self.threshold_entry.grid(row=0, column=1, padx=5)

        self.plus_btn = tk.Button(
            self.threshold_frame, text="+", font=("微软雅黑", 12), width=3,
            command=self.threshold_plus
        )
        self.plus_btn.grid(row=0, column=2, padx=5)

        # 阈值提示
        tk.Label(
            self.control_frame, text="提示：数值越大，越多像素会被判定为红色\n建议范围：30-200",
            font=("微软雅黑", 9), fg="#666666", justify=tk.LEFT
        ).pack(pady=(20, 30))

        # 确认阈值按钮
        self.confirm_threshold_btn = tk.Button(
            self.control_frame, text="确认阈值，处理下一批",
            font=("微软雅黑", 11), width=20, bg="#67C23A", fg="white",
            command=self.confirm_threshold
        )
        self.confirm_threshold_btn.pack(pady=20)

        # ========== 底部：状态与进度区 ==========
        self.bottom_frame = tk.Frame(self.root, padx=10, pady=10)
        self.bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(
            self.bottom_frame, text="就绪：请先选择基底图片",
            font=("微软雅黑", 9), fg="#333333", anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, side=tk.LEFT)

        self.progress_bar = ttk.Progressbar(self.bottom_frame, length=300)
        self.progress_bar.pack(side=tk.RIGHT, padx=10)

        # 绑定阈值输入框回车事件
        self.threshold_entry.bind("<Return>", lambda e: self.refresh_preview())
        self.threshold_entry.bind("<FocusOut>", lambda e: self.refresh_preview())

    def _lock_ui(self, status):
        """UI状态锁定控制"""
        if status == "init":
            # 初始状态：只有选基底按钮可用
            self.threshold_entry.config(state="disabled")
            self.minus_btn.config(state="disabled")
            self.plus_btn.config(state="disabled")
            self.confirm_threshold_btn.config(state="disabled")
        elif status == "base_loaded":
            # 基底加载完成：所有控制可用
            self.select_base_btn.config(state="normal")
            self.threshold_entry.config(state="normal")
            self.minus_btn.config(state="normal")
            self.plus_btn.config(state="normal")
            self.confirm_threshold_btn.config(state="normal")
        elif status == "processing":
            # 处理中：所有按钮禁用
            self.select_base_btn.config(state="disabled")
            self.threshold_entry.config(state="disabled")
            self.minus_btn.config(state="disabled")
            self.plus_btn.config(state="disabled")
            self.confirm_threshold_btn.config(state="disabled")
        elif status == "batch_done":
            # 单批处理完成：恢复到可继续处理状态
            self.select_base_btn.config(state="normal")
            self.threshold_entry.config(state="normal")
            self.minus_btn.config(state="normal")
            self.plus_btn.config(state="normal")
            self.confirm_threshold_btn.config(state="normal")

    # ================= 基底图片处理相关函数 =================
    def select_base_file(self):
        """选择/更换基底图片文件"""
        file_path = filedialog.askopenfilename(
            title="选择基底图片",
            filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp"), ("所有文件", "*.*")]
        )
        if not file_path:
            return

        # 更换基底时，询问是否重置阈值
        if self.base_img_path:
            reset_choice = messagebox.askyesno(
                "更换基底",
                "检测到您正在更换新基底，是否将阈值重置为默认95？\n\n是：重置为默认值，适配新基底\n否：保留当前阈值"
            )
            if reset_choice:
                self.current_threshold.set(DEFAULT_THRESHOLD)

        # 更新基底路径
        self.base_img_path = file_path
        self.base_path_label.config(text=os.path.basename(file_path))
        self.status_label.config(text="正在加载并识别新基底图片...")
        self.preview_canvas.delete("all")  # 清空旧预览图
        self.root.update()

        # 加载并识别新基底图
        try:
            self.base_img = Image.open(file_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("错误", f"无法打开基底图片：{str(e)}")
            self.status_label.config(text="加载失败，请重新选择基底图片")
            return

        # 自动识别近红/近白色
        self.near_red = get_dominant_near_color(self.base_img, PURE_RED, RED_RECOGNIZE_THRESHOLD)
        self.near_white = get_dominant_near_color(self.base_img, PURE_WHITE, WHITE_RECOGNIZE_THRESHOLD)

        if not self.near_red or not self.near_white:
            messagebox.showerror(
                "识别失败",
                "未识别到有效的近红色/近白色，请调大识别阈值后重试\n（代码开头RED_RECOGNIZE_THRESHOLD/WHITE_RECOGNIZE_THRESHOLD）"
            )
            self.status_label.config(text="颜色识别失败，请重新选择图片")
            return

        # 识别成功，刷新预览，解锁UI
        self.status_label.config(
            text=f"✅ 新基底加载成功！近红色：{self.near_red} | 近白色：{self.near_white}"
        )
        self.refresh_preview()
        self._lock_ui("base_loaded")
        messagebox.showinfo("加载成功", "新基底已加载完成，可调整阈值预览效果后继续处理")

    def refresh_preview(self):
        """根据当前阈值刷新预览图"""
        if not self.base_img:
            return

        # 校验阈值输入
        try:
            threshold = self.current_threshold.get()
            if threshold < 0:
                threshold = 0
                self.current_threshold.set(threshold)
        except:
            messagebox.showwarning("输入错误", "请输入有效的数字阈值")
            self.current_threshold.set(DEFAULT_THRESHOLD)
            return

        # 生成红/白分离图
        width, height = self.base_img.size
        processed_img = Image.new("RGB", (width, height))
        base_pixels = self.base_img.load()
        processed_pixels = processed_img.load()

        for y in range(height):
            for x in range(width):
                current = base_pixels[x, y]
                dist = color_distance(current, self.near_red)
                processed_pixels[x, y] = PURE_RED if dist <= threshold else PURE_WHITE

        # 保存处理后的基底图（用于后续生成A.png）
        self.processed_base_img = processed_img

        # 缩放图片适配预览框
        scale = min(PREVIEW_MAX_W / width, PREVIEW_MAX_H / height, 1)
        new_w = int(width * scale)
        new_h = int(height * scale)
        preview_img = processed_img.resize((new_w, new_h), Image.Resampling.NEAREST)

        # 转换为Tkinter可显示的格式
        self.tk_preview_img = ImageTk.PhotoImage(preview_img)
        self.preview_canvas.delete("all")
        # 居中显示
        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()
        if canvas_w < 10:
            canvas_w = PREVIEW_MAX_W
            canvas_h = PREVIEW_MAX_H
        x_pos = (canvas_w - new_w) // 2
        y_pos = (canvas_h - new_h) // 2
        self.preview_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_preview_img)

    def threshold_minus(self):
        """阈值-1"""
        current = self.current_threshold.get()
        if current > 0:
            self.current_threshold.set(current - 1)
            self.refresh_preview()

    def threshold_plus(self):
        """阈值+1"""
        current = self.current_threshold.get()
        self.current_threshold.set(current + 1)
        self.refresh_preview()

    # ================= 后续流程处理函数 =================
    def confirm_threshold(self):
        """确认阈值，进入下一步选择实验文件夹"""
        if not self.processed_base_img:
            messagebox.showwarning("警告", "无有效的基底处理结果")
            return
        # 选择实验图像文件夹
        self.exp_folder_path = filedialog.askdirectory(title="选择实验图像文件夹（批量处理）")
        if not self.exp_folder_path:
            return
        # 校验文件夹是否有图片
        img_files = [f for f in os.listdir(self.exp_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not img_files:
            messagebox.showerror("错误", "所选文件夹内没有可处理的图片文件")
            return
        # 选择保存路径
        self.save_root_path = filedialog.askdirectory(title="选择结果保存根路径")
        if not self.save_root_path:
            return
        # 开始批量处理
        self._lock_ui("processing")
        self.status_label.config(text="正在准备批量处理...")
        self.progress_bar["maximum"] = len(img_files)
        self.progress_bar["value"] = 0
        # 子线程执行处理，避免UI卡死
        threading.Thread(target=self.batch_process, daemon=True).start()

    def batch_process(self):
        """批量处理核心逻辑（子线程执行）"""
        try:
            # 1. 创建带时间戳的输出文件夹，彻底避免覆盖
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            base_output_dir = os.path.join(self.save_root_path, f"01-基底处理结果_{timestamp}")
            batch_output_dir = os.path.join(self.save_root_path, f"02-批量对比结果_{timestamp}")
            os.makedirs(base_output_dir, exist_ok=True)
            os.makedirs(batch_output_dir, exist_ok=True)

            # 2. 保存基底处理结果
            self.root.after(0, lambda: self.status_label.config(text="正在保存基底处理结果..."))
            # 保存红/白分离图
            red_white_path = os.path.join(base_output_dir, "red_white_result.png")
            self.processed_base_img.save(red_white_path, "PNG")
            # 生成并保存A.png（红变灰）
            width, height = self.processed_base_img.size
            A_img = Image.new("RGB", (width, height))
            A_pixels = A_img.load()
            processed_pixels = self.processed_base_img.load()
            for y in range(height):
                for x in range(width):
                    A_pixels[x, y] = PURE_GRAY if processed_pixels[x, y] == PURE_RED else PURE_WHITE
            A_path = os.path.join(base_output_dir, "A.png")
            A_img.save(A_path, "PNG")

            # 3. 批量处理实验图像
            self.root.after(0, lambda: self.status_label.config(text="开始批量处理实验图像..."))
            # 读取A.png作为底图
            base_A = Image.open(A_path).convert("RGB")
            base_pixels = base_A.load()
            # 获取所有实验图片
            img_files = [f for f in os.listdir(self.exp_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            threshold = self.current_threshold.get()
            success_count = 0

            for idx, filename in enumerate(img_files):
                file_path = os.path.join(self.exp_folder_path, filename)
                try:
                    img_target = Image.open(file_path).convert("RGB")
                except Exception as e:
                    self.root.after(0, lambda f=filename: self.status_label.config(text=f"跳过无法读取的文件：{f}"))
                    continue
                # 校验尺寸
                if img_target.size != (width, height):
                    self.root.after(0, lambda f=filename: self.status_label.config(text=f"跳过尺寸不匹配的文件：{f}"))
                    continue
                # 处理图像
                target_pixels = img_target.load()
                result_img = Image.new("RGB", (width, height))
                result_pixels = result_img.load()
                # 复制底图
                for y in range(height):
                    for x in range(width):
                        result_pixels[x, y] = base_pixels[x, y]
                # 标记红色区域
                for y in range(height):
                    for x in range(width):
                        current_color = target_pixels[x, y]
                        dist = color_distance(current_color, self.near_red)
                        if dist <= threshold:
                            result_pixels[x, y] = PURE_RED
                # 保存结果
                save_name = os.path.splitext(filename)[0] + ".png"
                save_path = os.path.join(batch_output_dir, save_name)
                result_img.save(save_path, "PNG")
                success_count += 1
                # 更新进度
                self.root.after(0, lambda v=idx+1, f=filename: (
                    self.progress_bar.config(value=v),
                    self.status_label.config(text=f"已处理：{f}")
                ))

            # 处理完成
            self.root.after(0, lambda: (
                self.progress_bar.config(value=0),  # 重置进度条
                self.status_label.config(text=f"✅ 本批处理完成！共成功 {success_count} 张。可继续处理下一批或更换基底。"),
                self._lock_ui("batch_done"),
                messagebox.showinfo(
                    "本批处理完成",
                    f"✅ 共成功处理 {success_count} 张图片\n\n"
                    f"📁 基底处理结果：{base_output_dir}\n"
                    f"📁 批量对比结果：{batch_output_dir}\n\n"
                    f"💡 提示：程序已就绪，可直接点击「确认阈值」处理下一批，或更换基底图片。"
                )
            ))

        except Exception as e:
            self.root.after(0, lambda: (
                self.status_label.config(text=f"处理失败：{str(e)}"),
                self._lock_ui("batch_done"),
                messagebox.showerror("处理失败", f"出现错误：{str(e)}")
            ))

# ================= 程序入口 =================
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessApp(root)
    root.mainloop()