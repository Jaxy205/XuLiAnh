"""
Hệ thống Phát hiện Bất thường Video Lai - Chương trình Chính (Phiên bản PyTorch)
Kết hợp: YOLOv8 (Nhận diện), SORT (Theo dõi), Optical Flow (Chạy), DBSCAN (Tụ tập), 
CNN-LSTM (Ngã), và Autoencoder (Bất thường chung)
"""

import argparse
from config import Config
from modules import HybridAnomalyDetector


def main():
    """Hàm chính của chương trình"""
    
    print("="*80)
    print(" HỆ THỐNG PHÁT HIỆN BẤT THƯỜNG TRONG VIDEO (PyTorch)")
    print(" Chức năng: Phát hiện Chạy | Tụ tập | Ngã")
    print("="*80 + "\n")
    
    parser = argparse.ArgumentParser(description='Hệ thống phát hiện bất thường video')
    parser.add_argument('--video', type=str, default=None, help='Đường dẫn video (hoặc 0 cho webcam)')
    parser.add_argument('--running', type=bool, default=None, help='Bật phát hiện chạy')
    parser.add_argument('--falling', type=bool, default=None, help='Bật phát hiện ngã')
    parser.add_argument('--gathering', type=bool, default=None, help='Bật phát hiện tụ tập')
    parser.add_argument('--output', type=str, default=None, help='Đường dẫn file xuất ra')
    
    args = parser.parse_args()
    
    config = Config()
    
    if args.video is not None:
        config.VIDEO_PATH = args.video
    if args.output is not None:
        config.OUTPUT_PATH = args.output
    if args.running is not None:
        config.ENABLE_RUNNING_DETECTION = args.running
    if args.falling is not None:
        config.ENABLE_FALLING_DETECTION = args.falling
    if args.gathering is not None:
        config.ENABLE_GATHERING_DETECTION = args.gathering
    
    detector = HybridAnomalyDetector(config)
    detector.run()


if __name__ == "__main__":
    main()