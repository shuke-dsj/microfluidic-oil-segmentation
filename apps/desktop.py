import sys
import os
from html import escape
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QFileDialog, QMessageBox, QGridLayout, QSizePolicy,
                             QTextEdit, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor, QPalette, QFont
import torchvision.transforms.functional as TF
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.amp import autocast

# ===================== 新增：Matplotlib 嵌入 PyQt5 组件 =====================
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ===================== 0. 全局配置与科研绘图设定 =====================
CLASS_NAMES = ["岩石", "孔道", "非均质剩余油", "孔喉剩余油", "壁面剩余油", "盲端剩余油"]
COLOR_MATCH_CONFIG = [
    (255, 255, 255, 80), (128, 128, 128, 40), (255, 0, 255, 50),
    (0, 255, 0, 60), (0, 0, 255, 70), (255, 255, 0, 50)
]
COLORS = [(r, g, b) for r, g, b, _ in COLOR_MATCH_CONFIG]

# 优化图表底色为更深的色调，凸显黄色盲端剩余油
DARK_CHART_BG = '#CFCCC0'


def set_plot_style():
    """统一下游绘图的科研级做旧风格"""
    plt.rcParams['figure.facecolor'] = DARK_CHART_BG
    plt.rcParams['axes.facecolor'] = DARK_CHART_BG
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False


set_plot_style()


from oilseg.model import AttentionResUNet
from oilseg.inference import pad_to_multiple, crop_to_original
from oilseg.labels import rgb_to_labels

# ===================== 2. 离线报告生成器 =====================
class ReportGenerator:
    def __init__(self):
        self.cmap = ListedColormap([(r / 255, g / 255, b / 255) for r, g, b in COLORS])
        self.save_dir = os.path.join(os.getcwd(), "outputs", "reports")
        os.makedirs(self.save_dir, exist_ok=True)

    def get_content_stats(self, pred_np):
        total_pixels = pred_np.size
        stats = {}
        for i, name in enumerate(CLASS_NAMES):
            stats[name] = (np.sum(pred_np == i) / total_pixels) * 100
        return stats

    def generate_no_gt_report(self, img_np, pred_np, filename):
        stats = self.get_content_stats(pred_np)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        fig.patch.set_facecolor('#F5F4EE')

        axes[0].imshow(img_np)
        axes[0].set_title("原始图像", fontsize=14, fontweight='bold')
        axes[1].imshow(pred_np, cmap=self.cmap, vmin=0, vmax=5, interpolation='nearest')
        axes[1].set_title("预测结果 (Prediction)", fontsize=14, fontweight='bold')
        for ax in axes: ax.axis('off')

        oil_names = [name for name in CLASS_NAMES if "油" in name]
        total_oil = sum(stats[name] for name in oil_names)

        text_str = "【各类剩余油相对占比】\n\n"
        if total_oil > 0:
            for name in oil_names:
                relative_pct = (stats[name] / total_oil) * 100
                text_str += f" 💧 {name:<8}: {relative_pct:.2f}%\n"
        else:
            text_str += " 未检测到有效剩余油\n"

        props = dict(boxstyle='round,pad=1', facecolor='#E8E5D8', alpha=0.8, edgecolor='#A09D90')
        fig.text(0.85, 0.5, text_str, fontsize=12, fontweight='bold', verticalalignment='center', bbox=props,
                 fontfamily='SimHei')

        plt.subplots_adjust(right=0.8)
        save_path = os.path.join(self.save_dir, f"{filename}_NoGT_Report.png")
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()

        set_plot_style()
        return save_path

    def generate_with_gt_report(self, img_np, gt_np, pred_np, miou, ious_list, filename):
        stats = self.get_content_stats(pred_np)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        fig.patch.set_facecolor('#F5F4EE')

        axes[0].imshow(img_np)
        axes[0].set_title("原始图像", fontsize=14, fontweight='bold')
        axes[1].imshow(gt_np, cmap=self.cmap, vmin=0, vmax=5, interpolation='nearest')
        axes[1].set_title("真实标签 (Ground Truth)", fontsize=14, fontweight='bold')
        axes[2].imshow(pred_np, cmap=self.cmap, vmin=0, vmax=5, interpolation='nearest')
        axes[2].set_title("预测结果 (Prediction)", fontsize=14, fontweight='bold')
        for ax in axes: ax.axis('off')

        text_str = f"【分割评估报告】\n⭐ 图像 mIoU: {miou:.2f}%\n\n【类别交并比 (IoU)】\n"
        for name, iou in zip(CLASS_NAMES, ious_list):
            text_str += f" 🎯 {name:<8}: {f'{iou:.2f}%' if iou is not None else '--'}\n"

        # 【核心同步】：非盲测同样只输出各种油的相对占比
        oil_names = [name for name in CLASS_NAMES if "油" in name]
        total_oil = sum(stats[name] for name in oil_names)

        text_str += "\n【各类剩余油相对占比】\n"
        if total_oil > 0:
            for name in oil_names:
                relative_pct = (stats[name] / total_oil) * 100
                text_str += f" 💧 {name:<8}: {relative_pct:.2f}%\n"
        else:
            text_str += " 未检测到有效剩余油\n"

        props = dict(boxstyle='round,pad=1', facecolor='#E8E5D8', alpha=0.8, edgecolor='#A09D90')
        fig.text(0.92, 0.5, text_str, fontsize=12, fontweight='bold', verticalalignment='center', bbox=props,
                 fontfamily='SimHei')
        plt.subplots_adjust(right=0.9)
        save_path = os.path.join(self.save_dir, f"{filename}_WithGT_Report.png")
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()

        set_plot_style()
        return save_path


# ===================== 3. 现代化 UI 组件 =====================
class DynamicImageLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("font-size: 14px; color: #888; background-color: transparent;")
        self._original_pixmap = None
        self.setMinimumSize(250, 250)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def set_original_pixmap(self, pixmap):
        self._original_pixmap = pixmap
        self.update_image()

    def resizeEvent(self, event):
        if self._original_pixmap is not None: self.update_image()
        super().resizeEvent(event)

    def update_image(self):
        if self._original_pixmap and not self._original_pixmap.isNull():
            scaled_pixmap = self._original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)


class DragDropLabel(DynamicImageLabel):
    def __init__(self, text, parent_app):
        super().__init__(text)
        self.setAcceptDrops(True)
        self.parent_app = parent_app

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        url = event.mimeData().urls()[0]
        file_path = url.toLocalFile()
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            self.parent_app.load_image(file_path)


def create_card_container(title_text, widget):
    container = QFrame()
    container.setStyleSheet("""
        QFrame { background-color: #FFFFFF; border-radius: 8px; border: 1px solid #E0E0E0; }
    """)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(15, 15, 15, 15)

    title = QLabel(title_text)
    title.setStyleSheet("font-size: 15px; font-weight: bold; color: #333; border: none;")
    title.setAlignment(Qt.AlignCenter)

    widget.setStyleSheet("border: 2px dashed #D0D0D0; border-radius: 4px; background-color: #FAFAFA;")

    layout.addWidget(title)
    layout.addSpacing(5)
    layout.addWidget(widget, stretch=1)
    return container


# ===================== 4. 主窗口程序 =====================
class SegMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("微流控剩余油智能分析系统 ")
        self.resize(1400, 950)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#F0F0F0"))
        self.setPalette(palette)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.current_img_tensor = None
        self.current_img_np_raw = None
        self.current_mask_path = None
        self.original_size = None
        self.reporter = ReportGenerator()
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        toolbar_container = QFrame()
        toolbar_container.setStyleSheet(
            "QFrame { background-color: #FFFFFF; border-radius: 8px; border: 1px solid #E0E0E0; }")
        tool_layout = QHBoxLayout(toolbar_container)
        tool_layout.setContentsMargins(15, 10, 15, 10)

        self.btn_load_model = QPushButton("📂 导入核心模型")
        self.btn_load_model.clicked.connect(self.load_model_dialog)
        self.btn_load_image = QPushButton("🖼️ 选择单张图像")
        self.btn_load_image.clicked.connect(self.load_image_dialog)
        self.btn_load_mask = QPushButton("🏷️ 导入验证GT标签")
        self.btn_load_mask.clicked.connect(self.load_mask_dialog)

        self.btn_batch = QPushButton("📁 执行批量处理")
        self.btn_batch.setObjectName("BatchBtn")
        self.btn_batch.clicked.connect(self.process_batch)

        self.lbl_status = QLabel("系统状态: 准备就绪")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #424242; font-size: 14px; border: none;")

        self.btn_process = QPushButton("🚀 启动智能分析")
        self.btn_process.setObjectName("ProcessBtn")
        self.btn_process.clicked.connect(self.process_single_image)
        self.btn_process.setEnabled(False)
        self.btn_process.setMinimumWidth(150)

        tool_layout.addWidget(self.btn_load_model)
        tool_layout.addWidget(self.btn_load_image)
        tool_layout.addWidget(self.btn_load_mask)
        tool_layout.addWidget(self.btn_batch)
        tool_layout.addStretch()
        tool_layout.addWidget(self.lbl_status)
        tool_layout.addSpacing(30)
        tool_layout.addWidget(self.btn_process)
        main_layout.addWidget(toolbar_container)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)

        self.lbl_input = DragDropLabel("将图像拖拽至此\n或通过上方工具栏导入", self)
        card_input = create_card_container("🔬 原始显微图像", self.lbl_input)

        self.lbl_output = DynamicImageLabel("等待模型推理...")
        card_output = create_card_container("🤖 模型推理结果", self.lbl_output)

        self.text_report = QTextEdit()
        self.text_report.setReadOnly(True)
        self.text_report.setStyleSheet("""
            QTextEdit {
                background-color: #F8F9FA;
                border: none;
                font-family: 'SimHei';
                font-size: 14px;
                color: #2C3E50;
                padding: 10px;
            }
        """)
        self.text_report.setPlaceholderText("系统将在分析完成后，自动提炼并展示关键数据指标...")
        card_report = create_card_container("📑 多维度智能评估报告", self.text_report)

        self.fig = Figure(dpi=100)
        self.fig.patch.set_facecolor(DARK_CHART_BG)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(DARK_CHART_BG)
        self.ax.axis('off')
        card_chart = create_card_container("📊 可视化渲染图像", self.canvas)

        grid_layout.addWidget(card_input, 0, 0)
        grid_layout.addWidget(card_output, 0, 1)
        grid_layout.addWidget(card_report, 1, 0)
        grid_layout.addWidget(card_chart, 1, 1)

        grid_layout.setRowStretch(0, 40)
        grid_layout.setRowStretch(1, 60)
        grid_layout.setColumnStretch(0, 40)
        grid_layout.setColumnStretch(1, 60)

        main_layout.addLayout(grid_layout, stretch=1)

        self.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                background-color: #3949AB;
                color: #FFFFFF;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
                border: none;
            }
            QPushButton:hover { background-color: #283593; }
            QPushButton:disabled { background-color: #BDBDBD; color: #FAFAFA; }
            QPushButton#ProcessBtn { background-color: #00897B; font-size: 14px;}
            QPushButton#ProcessBtn:hover { background-color: #00695C; }
            QPushButton#BatchBtn { background-color: #7B1FA2; }
            QPushButton#BatchBtn:hover { background-color: #4A148C; }
        """)

    def rgb2label(self, mask_rgb):
        return rgb_to_labels(mask_rgb)

    def compute_all_ious(self, pred, target):
        ious_list = []
        valid_ious = []
        for cls in range(6):
            inter = ((pred == cls) & (target == cls)).float().sum()
            union = ((pred == cls) | (target == cls)).float().sum()
            if union > 0:
                iou_val = (inter / union).item() * 100
                ious_list.append(iou_val)
                valid_ious.append(iou_val)
            else:
                ious_list.append(None)
        return (sum(valid_ious) / len(valid_ious) if valid_ious else 0.0), ious_list

    def load_model_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch Models (*.pth)")
        if file_path:
            try:
                self.lbl_status.setText("正在加载深度学习模型...")
                QApplication.processEvents()
                self.model = AttentionResUNet(num_classes=6).to(self.device)
                self.model.load_state_dict(torch.load(file_path, map_location=self.device, weights_only=True))
                self.model.eval()
                self.lbl_status.setText("✅ 核心模型就绪")
            except Exception as e:
                self.lbl_status.setText("❌ 模型加载失败")
                QMessageBox.critical(self, "错误", f"加载模型时发生异常:\n{str(e)}")

    def load_image_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择原始图像", "", "Images (*.png *.jpg *.jpeg *.tif *.bmp)")
        if file_path: self.load_image(file_path)

    def load_mask_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择GT真实标签", "", "Images (*.png *.jpg *.tif *.bmp)")
        if file_path:
            self.current_mask_path = file_path
            self.lbl_status.setText("✅ 标签已导入，将激活精度评估")

    def load_image(self, file_path):
        self.filename = os.path.basename(file_path)
        self.lbl_input.set_original_pixmap(QPixmap(file_path))
        img_pil = Image.open(file_path).convert('RGB')
        self.original_size = img_pil.size
        self.current_img_np_raw = np.array(img_pil)
        self.current_img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(self.device)
        self.btn_process.setEnabled(True)
        self.current_mask_path = None
        self.lbl_status.setText("✅ 图像加载成功")

    def infer_tensor(self, img_tensor):
        padded_tensor, pad_h, pad_w = pad_to_multiple(img_tensor, 32)
        with torch.no_grad():
            with autocast(self.device.type, enabled=self.device.type == 'cuda'):
                pred_logits = self.model(padded_tensor)
            pred_cls = torch.argmax(pred_logits, dim=1).squeeze(0)
        return crop_to_original(pred_cls, pad_h, pad_w)

    def update_ui_pie_chart(self, stats):
        self.ax.clear()
        labels, sizes, pie_colors, explodes = [], [], [], []

        oil_names = [name for name in CLASS_NAMES if "油" in name]
        total_oil_percentage = sum(stats[name] for name in oil_names)

        if total_oil_percentage > 0:
            for name in oil_names:
                if stats[name] > 0.001:
                    labels.append(name)
                    relative_size = (stats[name] / total_oil_percentage) * 100
                    sizes.append(relative_size)

                    idx = CLASS_NAMES.index(name)
                    pie_colors.append((COLORS[idx][0] / 255, COLORS[idx][1] / 255, COLORS[idx][2] / 255))
                    explodes.append(0.06)

            wedges, texts = self.ax.pie(
                sizes, explode=explodes, colors=pie_colors,
                shadow=False, startangle=190, radius=0.6
            )

            kw = dict(arrowprops=dict(arrowstyle="-", color='#555555', lw=1.2), zorder=0, va="center")

            for i, p in enumerate(wedges):
                ang = (p.theta2 - p.theta1) / 2. + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))

                horizontalalignment = "left" if x >= 0 else "right"

                x_text = np.sign(x) * 1.05
                y_text = y * 1.05

                connectionstyle = f"angle,angleA=0,angleB={ang}"
                kw["arrowprops"]["connectionstyle"] = connectionstyle

                self.ax.annotate(f"{labels[i]}\n{sizes[i]:.1f}%",
                                 xy=(x * 0.6, y * 0.6),
                                 xytext=(x_text, y_text),
                                 horizontalalignment=horizontalalignment, **kw,
                                 fontweight='bold', fontsize=11, color='#2C3E50')

            self.ax.set_title("各类型剩余油内部相对占比", fontweight='bold', fontsize=14, pad=15)
        else:
            self.ax.text(0.5, 0.5, "未检测到有效剩余油", ha='center', va='center', fontsize=14, color='gray')

        self.ax.axis('equal')
        self.fig.subplots_adjust(left=0.20, right=0.80, top=0.85, bottom=0.1)
        self.canvas.draw()

    def update_ui_bar_chart(self, avg_stats, num_images):
        self.ax.clear()
        self.ax.axis('on')
        oil_names = [name for name in CLASS_NAMES if "油" in name]

        total_oil = sum(avg_stats[name] for name in oil_names)

        # 【核心修改】：批处理柱状图也同样换算为油的内部相对占比
        if total_oil > 0:
            oil_values = [(avg_stats[name] / total_oil) * 100 for name in oil_names]
        else:
            oil_values = [0] * len(oil_names)

        bar_colors = [(COLORS[CLASS_NAMES.index(name)][0] / 255, COLORS[CLASS_NAMES.index(name)][1] / 255,
                       COLORS[CLASS_NAMES.index(name)][2] / 255) for name in oil_names]

        bars = self.ax.bar(oil_names, oil_values, color=bar_colors, edgecolor='#555555', width=0.55)
        self.ax.set_title(f"批处理平均相对占比 ({num_images}张图)", fontweight='bold', fontsize=13)
        self.ax.set_ylabel("内部相对占比 (%)", fontsize=11, fontweight='bold')
        self.ax.grid(axis='y', linestyle='--', alpha=0.5)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        self.ax.set_xticks(range(len(oil_names)))
        self.ax.set_xticklabels(oil_names, rotation=15, fontweight='bold', fontsize=10)

        for bar in bars:
            yval = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, f"{yval:.2f}%",
                         ha='center', va='bottom', fontsize=11, fontweight='bold', color='#111111')

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()

    def process_single_image(self):
        if self.model is None or self.current_img_tensor is None: return

        self.lbl_status.setText("⏳ 正在进行智能推断...")
        self.btn_process.setEnabled(False)
        QApplication.processEvents()

        try:
            pred_tensor = self.infer_tensor(self.current_img_tensor)
            pred_np = pred_tensor.cpu().numpy().astype(np.uint8)

            rgb_pred = np.array(COLORS, dtype=np.uint8)[pred_np]
            h, w, c = rgb_pred.shape
            self.lbl_output.set_original_pixmap(
                QPixmap.fromImage(QImage(rgb_pred.data, w, h, c * w, QImage.Format_RGB888)))

            stats = self.reporter.get_content_stats(pred_np)

            report_text = f"<div style='line-height: 1.6;'>"
            report_text += f"<span style='color: #1565C0; font-weight: bold;'>► 分析目标:</span> {escape(self.filename)}<hr style='border: 1px solid #E0E0E0;'>"

            base_name = os.path.splitext(self.filename)[0]

            # 先定义一个复用的相对占比文本渲染块，提取出来保持代码整洁
            oil_report_text = "<br><b>💧 各类型剩余油相对占比:</b><br>"
            oil_names = [name for name in CLASS_NAMES if "油" in name]
            total_oil = sum(stats[name] for name in oil_names)
            if total_oil > 0:
                for name in oil_names:
                    rel_pct = (stats[name] / total_oil) * 100
                    oil_report_text += f"&nbsp;&nbsp;&nbsp;• {name}: <b>{rel_pct:.2f}%</b><br>"
            else:
                oil_report_text += "&nbsp;&nbsp;&nbsp;• <span style='color:gray;'>未检测到有效剩余油</span><br>"

            if self.current_mask_path and os.path.exists(self.current_mask_path):
                gt_pil = Image.open(self.current_mask_path).convert('RGB')
                if gt_pil.size != self.original_size:
                    raise ValueError('图像与真实标签尺寸不一致，请使用已对齐的标签')
                gt_label_tensor = torch.from_numpy(self.rgb2label(np.array(gt_pil))).to(self.device)

                miou, all_ious = self.compute_all_ious(pred_tensor, gt_label_tensor)
                saved_path = self.reporter.generate_with_gt_report(self.current_img_np_raw,
                                                                   gt_label_tensor.cpu().numpy(), pred_np,
                                                                   miou, all_ious, base_name)

                report_text += f"<span style='color:#C62828; font-size: 15px; font-weight: bold;'>🏆 模型评估 mIoU: {miou:.2f}%</span><br><br>"
                report_text += "<b>🎯 各类别识别交并比 (IoU):</b><br>"
                for name, iou in zip(CLASS_NAMES, all_ious):
                    val_str = f"{iou:.2f}%" if iou is not None else "<span style='color:#9E9E9E;'>N/A</span>"
                    report_text += f"&nbsp;&nbsp;&nbsp;• {name}: <b>{val_str}</b><br>"

                # 【核心同步】：非盲测 UI 文本也只显示油的相对占比
                report_text += oil_report_text
                report_text += f"<br><span style='color:#616161;'>📝 报告已保存至: {os.path.basename(saved_path)}</span><br>"

            else:
                saved_path = self.reporter.generate_no_gt_report(self.current_img_np_raw, pred_np, base_name)

                report_text += "<span style='color:#2E7D32; font-weight: bold;'>✅ 未知图像盲测完成 (无需GT比对)</span><br>"

                report_text += oil_report_text
                report_text += f"<br><span style='color:#616161;'>📝 报告已保存至: {os.path.basename(saved_path)}</span><br><br>"

            report_text += "</div>"

            self.text_report.setHtml(report_text)
            self.update_ui_pie_chart(stats)
            self.lbl_status.setText("✅ 视图渲染完成")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"推理过程出错: {str(e)}")
        finally:
            self.btn_process.setEnabled(True)

    def process_batch(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先在上方加载模型！")
            return

        dir_path = QFileDialog.getExistingDirectory(self, "选择包含待测图像的目录")
        if not dir_path: return

        img_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        if not img_files: return

        self.btn_batch.setEnabled(False)
        self.btn_process.setEnabled(False)
        batch_stats = []
        batch_save_dir = os.path.join(self.reporter.save_dir, "batch_predictions")
        os.makedirs(batch_save_dir, exist_ok=True)

        try:
            for idx, filename in enumerate(img_files):
                self.lbl_status.setText(f"批处理执行中... ({idx + 1}/{len(img_files)})")
                QApplication.processEvents()

                img_pil = Image.open(os.path.join(dir_path, filename)).convert('RGB')
                pred_np = self.infer_tensor(TF.to_tensor(img_pil).unsqueeze(0).to(self.device)).cpu().numpy().astype(
                    np.uint8)
                batch_stats.append(self.reporter.get_content_stats(pred_np))
                Image.fromarray(np.array(COLORS, dtype=np.uint8)[pred_np]).save(
                    os.path.join(batch_save_dir, f"{os.path.splitext(filename)[0]}_pred.png"))

            avg_stats = {name: sum(s[name] for s in batch_stats) / len(img_files) for name in CLASS_NAMES}

            report_text = f"<div style='line-height: 1.6;'>"
            report_text += f"<span style='color: #6A1B9A; font-weight: bold;'>📁 批量处理任务完成</span><br>"
            report_text += f"总处理文件数: <b>{len(img_files)}</b> 张<hr style='border: 1px solid #E0E0E0;'>"

            # 【核心同步】：批处理文本同样只显示油的相对占比
            report_text += "<b>📊 批次各类型剩余油平均相对占比:</b><br>"
            oil_names = [name for name in CLASS_NAMES if "油" in name]
            batch_total_oil = sum(avg_stats[name] for name in oil_names)
            if batch_total_oil > 0:
                for name in oil_names:
                    rel_pct = (avg_stats[name] / batch_total_oil) * 100
                    report_text += f"&nbsp;&nbsp;&nbsp;• {name}: <b>{rel_pct:.2f}%</b><br>"
            else:
                report_text += "&nbsp;&nbsp;&nbsp;• <span style='color:gray;'>未检测到有效剩余油</span><br>"

            report_text += "</div>"

            self.text_report.setHtml(report_text)
            self.update_ui_bar_chart(avg_stats, len(img_files))
            self.lbl_status.setText("✅ 批处理全部圆满完成")

        except Exception as e:
            QMessageBox.critical(self, "执行异常", f"批处理中止: {str(e)}")
        finally:
            self.btn_batch.setEnabled(True)
            self.btn_process.setEnabled(True)


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    window = SegMainWindow()
    window.show()
    sys.exit(app.exec_())
