import numpy as np
from .simple_tracker import CentroidTracker


class TrackerWrapper:
    """Lớp bao đóng bộ theo dõi Centroid đơn giản"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Khởi tạo bộ theo dõi
        
        Tham số:
            max_age: Số frame tối đa giữ ID khi không phát hiện được đối tượng
            min_hits: (Không dùng trong bản đơn giản này, giữ lại để tương thích)
            iou_threshold: (Không dùng, thay bằng khoảng cách pixel)
        """
        # Trong bản đơn giản, ta dùng khoảng cách pixel thay vì IoU
        # max_distance=50 pixel tương đương với việc di chuyển không quá xa giữa 2 frame
        self.tracker = CentroidTracker(max_disappeared=max_age, max_distance=100)
    
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Cập nhật bộ theo dõi với các phát hiện mới
        
        Tham số:
            detections: Mảng numpy shape (N, 5) -> [x1, y1, x2, y2, conf]
        
        Trả về:
            tracked_objects: Mảng numpy shape (N, 5) -> [x1, y1, x2, y2, id]
        """
        # Chuyển đổi format detections cho CentroidTracker
        # CentroidTracker cần list các rects [x1, y1, x2, y2]
        rects = []
        if len(detections) > 0:
            rects = detections[:, :4].astype("int").tolist()
            
        # Cập nhật tracker
        # Trả về dictionary {object_id: centroid}
        objects = self.tracker.update(rects)
        
        # Chuyển đổi lại format đầu ra để khớp với code cũ
        # Cần trả về [x1, y1, x2, y2, id]
        # Vì CentroidTracker chỉ lưu tâm (centroid), ta cần map lại với box gốc
        # Cách đơn giản: Tìm box gốc gần nhất với centroid của ID đó
        
        tracked_objects = []
        
        for (object_id, centroid) in objects.items():
            # Tìm box gốc tương ứng (đơn giản nhất là tìm box có tâm gần nhất)
            # Lưu ý: Đây là cách làm đơn giản hóa.
            
            best_box = None
            min_dist = 99999
            
            if len(rects) > 0:
                input_centroids = np.zeros((len(rects), 2), dtype="int")
                for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
                    c_x = int((start_x + end_x) / 2.0)
                    c_y = int((start_y + end_y) / 2.0)
                    dist = np.linalg.norm(centroid - np.array([c_x, c_y]))
                    if dist < min_dist:
                        min_dist = dist
                        best_box = [start_x, start_y, end_x, end_y]
            
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                tracked_objects.append([x1, y1, x2, y2, object_id])
            else:
                # Nếu không tìm thấy box mới (đang bị mất dấu nhưng chưa xóa)
                # Ta có thể bỏ qua hoặc dùng centroid cũ để ước lượng box (khó chính xác)
                # Ở đây ta bỏ qua để tránh vẽ box sai
                pass
                
        return np.array(tracked_objects)
