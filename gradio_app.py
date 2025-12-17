import gradio as gr
import cv2
import os
import tempfile
from config import Config
from modules import HybridAnomalyDetector
import time

def process_video(video_path, enable_running, enable_falling, enable_gathering, 
                  conf_threshold, flow_threshold, fall_ratio, gather_eps, gather_min_samples):
    """
    Hàm xử lý video cho Gradio Interface.
    Chạy detector trên video đầu vào và trả về đường dẫn video kết quả.
    """
    if video_path is None:
        return None

    output_filename = f"output_{int(time.time())}.webm"
    output_path = os.path.join(tempfile.gettempdir(), output_filename)

    # Cấu hình hệ thống
    config = Config()
    config.VIDEO_PATH = video_path
    config.OUTPUT_PATH = output_path
    config.ENABLE_RUNNING_DETECTION = enable_running
    config.ENABLE_FALLING_DETECTION = enable_falling
    config.ENABLE_GATHERING_DETECTION = enable_gathering
    config.SAVE_OUTPUT = True
    config.DISPLAY_FPS = False
    
    # Cập nhật tham số từ UI
    config.YOLO_CONF_THRESHOLD = conf_threshold
    config.OPTICAL_FLOW_THRESHOLD = flow_threshold
    config.FALL_RATIO_THRESHOLD = fall_ratio
    config.GATHERING_EPS = gather_eps
    config.GATHERING_MIN_SAMPLES = int(gather_min_samples)
    
    detector = HybridAnomalyDetector(config)
    
    # Mở video input
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")

    # Lấy thông số video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'vp09')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print("Codec vp09 không khả dụng, chuyển sang mp4v (.mp4)...")
        # Đổi đuôi file sang .mp4
        output_filename = f"output_{int(time.time())}.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Vòng lặp xử lý
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            annotated_frame, stats = detector.process_frame(frame)
            
            # Vẽ thông tin
            if config.DISPLAY_FPS:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_process_fps = 0
                annotated_frame = detector.visualizer.draw_stats(
                    annotated_frame, frame_count, total_frames, current_process_fps, stats
                )
            
            # Ghi frame vào file
            writer.write(annotated_frame)
            
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý: {str(e)}")
    finally:
        cap.release()
        writer.release()
        
    return output_path

# Định nghĩa giao diện Gradio
def create_ui():
    with gr.Blocks(title="Hệ thống Phát hiện Bất thường Video") as demo:
        gr.Markdown("# 🎥 Hệ thống Phát hiện Bất thường Video")
        gr.Markdown("Tải lên video và chọn các chế độ phát hiện mong muốn. Kết quả sẽ được hiển thị trực tiếp (streaming).")
        
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="Video Đầu vào")
                
                gr.Markdown("### Cấu hình Phát hiện")
                cb_running = gr.Checkbox(label="Phát hiện CHẠY (Running)", value=True)
                cb_falling = gr.Checkbox(label="Phát hiện NGÃ (Falling)", value=True)
                cb_gathering = gr.Checkbox(label="Phát hiện TỤ TẬP (Gathering)", value=True)
                
                gr.Markdown("### Tham số Nâng cao")
                # Khởi tạo config để lấy giá trị mặc định
                default_config = Config()
                
                slider_conf = gr.Slider(minimum=0.1, maximum=1.0, value=default_config.YOLO_CONF_THRESHOLD, step=0.05, label="Ngưỡng tin cậy YOLO")
                slider_flow = gr.Slider(minimum=1.0, maximum=10.0, value=default_config.OPTICAL_FLOW_THRESHOLD, step=0.5, label="Ngưỡng Optical Flow (Chạy)")
                slider_fall = gr.Slider(minimum=0.5, maximum=3.0, value=default_config.FALL_RATIO_THRESHOLD, step=0.1, label="Ngưỡng tỷ lệ khung hình (Ngã)")
                slider_eps = gr.Slider(minimum=10, maximum=200, value=default_config.GATHERING_EPS, step=10, label="Khoảng cách Tụ tập (pixel)")
                slider_samples = gr.Slider(minimum=2, maximum=10, value=default_config.GATHERING_MIN_SAMPLES, step=1, label="Số người Tụ tập tối thiểu")
                
                btn_process = gr.Button("🚀 Bắt đầu Xử lý", variant="primary")
            
            with gr.Column():
                # Dùng Video component để hiển thị kết quả
                output_video = gr.Video(label="Video Kết quả")
        
        btn_process.click(
            fn=process_video,
            inputs=[
                input_video, cb_running, cb_falling, cb_gathering,
                slider_conf, slider_flow, slider_fall, slider_eps, slider_samples
            ],
            outputs=[output_video],
            show_progress=True
        )
        
    return demo

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False)
