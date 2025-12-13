import gradio as gr
import cv2
import os
import tempfile
from config import Config
from modules import HybridAnomalyDetector
import time

def process_video(video_path, enable_running, enable_falling, enable_gathering):
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
    config.DISPLAY_FPS = True
    
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
                
                btn_process = gr.Button("🚀 Bắt đầu Xử lý", variant="primary")
            
            with gr.Column():
                # Dùng Video component để hiển thị kết quả
                output_video = gr.Video(label="Video Kết quả")
        
        btn_process.click(
            fn=process_video,
            inputs=[input_video, cb_running, cb_falling, cb_gathering],
            outputs=[output_video],
            show_progress=True
        )
        
    return demo

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False)
