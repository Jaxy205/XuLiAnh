"""
Các Module Logic Toán Học cho Phát Hiện Bất Thường
Cài đặt: Optical Flow (Luồng quang học) cho phát hiện chạy và Khoảng cách Euclid cho phát hiện tụ tập
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False



def check_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray, 
                       bbox: Tuple[int, int, int, int], 
                       threshold: float = 3.0) -> Tuple[bool, float]:
    """
    Phát hiện hành động CHẠY sử dụng Dense Optical Flow (Phương pháp Farneback)
    
    Nguyên lý toán học:
    - Optical Flow ước tính chuyển động của các pixel giữa 2 khung hình liên tiếp.
    - Phương pháp Farneback: Dùng đa thức để xấp xỉ cường độ sáng của ảnh.
    - Với mỗi pixel: I(x,y,t) ≈ I(x+dx, y+dy, t+dt)
    - Độ lớn (Magnitude) = sqrt(dx² + dy²) đại diện cho cường độ chuyển động.
    
    Tham số:
        prev_gray: Ảnh xám của frame trước
        curr_gray: Ảnh xám của frame hiện tại
        bbox: Khung bao (x1, y1, x2, y2) của người cần kiểm tra
        threshold: Ngưỡng độ lớn trung bình để coi là đang chạy
    
    Trả về:
        (is_running: bool, magnitude_score: float) - Có chạy hay không và điểm số
    """
    
    x1, y1, x2, y2 = bbox
    
    # Đảm bảo khung bao nằm trong kích thước ảnh
    h, w = prev_gray.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Kiểm tra kích thước khung bao hợp lệ
    if x2 <= x1 or y2 <= y1:
        return False, 0.0
    
    # Cắt vùng ảnh (ROI) chứa người
    prev_roi = prev_gray[y1:y2, x1:x2]
    curr_roi = curr_gray[y1:y2, x1:x2]
    
    if prev_roi.size == 0 or curr_roi.size == 0:
        return False, 0.0
    
    # Tính toán Optical Flow dày đặc (Dense) bằng thuật toán Farneback
    # Các tham số OpenCV:
    # - pyr_scale=0.5: Giảm kích thước ảnh 50% mỗi tầng tháp (pyramid)
    # - levels=3: Số tầng tháp
    # - winsize=15: Kích thước cửa sổ trung bình
    # - iterations=3: Số lần lặp tại mỗi tầng
    # - poly_n=5: Kích thước vùng lân cận để xấp xỉ đa thức
    # - poly_sigma=1.2: Độ lệch chuẩn Gaussian
    flow = cv2.calcOpticalFlowFarneback(
        prev_roi, curr_roi,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    
    # Tính độ lớn (magnitude) và góc (angle) của các vector chuyển động
    # flow[..., 0] là dx (chuyển động ngang)
    # flow[..., 1] là dy (chuyển động dọc)
    # magnitude = căn bậc hai(dx^2 + dy^2)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Tính giá trị trung bình của độ lớn chuyển động trong vùng ROI
    mean_magnitude = np.mean(magnitude)
    
    # So sánh với ngưỡng để quyết định
    is_running = mean_magnitude > threshold
    
    return is_running, float(mean_magnitude)


def check_gathering(trajectories: Dict[int, List[Tuple[int, int]]], 
                    eps: float = 100.0, 
                    min_samples: int = 3) -> List[List[int]]:
    """
    Phát hiện TỤ TẬP sử dụng Trajectory Clustering (Vị trí + Vận tốc)
    
    Nguyên lý cải tiến:
    - Không chỉ xét khoảng cách vị trí (Spatial) mà còn xét sự tương đồng chuyển động.
    - Feature vector: [x, y, weight * vx, weight * vy]
    - Nhóm người đi cùng nhau sẽ có vị trí gần và vận tốc tương đương.
    
    Tham số:
        trajectories: Dict {id: list of points}
        eps: Khoảng cách tối đa (đã cân bằng trọng số)
        min_samples: Số người tối thiểu
    
    Trả về:
        Danh sách các nhóm tụ tập.
    """
    if len(trajectories) < min_samples:
        return []
        
    ids = list(trajectories.keys())
    features = []
    
    velocity_weight = 15.0 # Trọng số cho vận tốc (quy đổi ra pixel)
    
    for obj_id in ids:
        trace = trajectories[obj_id]
        if not trace:
            continue
            
        curr_pos = trace[-1]
        
        # Tính vận tốc trung bình 5 frame cuối
        vx, vy = 0.0, 0.0
        if len(trace) >= 5:
            past_pos = trace[-5]
            vx = (curr_pos[0] - past_pos[0]) / 5.0
            vy = (curr_pos[1] - past_pos[1]) / 5.0
        elif len(trace) >= 2:
            past_pos = trace[0]
            dt = len(trace) - 1
            vx = (curr_pos[0] - past_pos[0]) / dt
            vy = (curr_pos[1] - past_pos[1]) / dt
            
        # Feature: [x, y, w*vx, w*vy]
        features.append([
            curr_pos[0], 
            curr_pos[1], 
            vx * velocity_weight, 
            vy * velocity_weight
        ])
        
    if not features:
        return []
        
    X = np.array(features)
    groups = []
    
    # Sử dụng DBSCAN nếu có sklearn (Ưu tiên)
    if SKLEARN_AVAILABLE:
        try:
            from sklearn.cluster import DBSCAN
            # eps ở đây áp dụng cho cả vector [x,y, vx, vy]
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            labels = clustering.labels_
            
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1: # Outlier
                    continue
                
                group_indices = np.where(labels == label)[0]
                if len(group_indices) >= min_samples:
                    group_ids = [ids[i] for i in group_indices]
                    groups.append(group_ids)
            return groups
            
        except ImportError:
            pass # Fallback to manual
            
    # Fallback: Manual Clustering (Simplified, only Position)
    # Vì viết lại DBSCAN đầy đủ hơi dài, ta dùng logic cũ cải tiến nhẹ
    n = len(X)
    visited = [False] * n
    
    for i in range(n):
        if visited[i]:
            continue
            
        current_group = [ids[i]]
        visited[i] = True
        
        for j in range(n):
            if i == j or visited[j]:
                continue
            
            # Tính khoảng cách trên không gian feature mở rộng
            dist = np.linalg.norm(X[i] - X[j])
            
            if dist <= eps:
                current_group.append(ids[j])
                visited[j] = True
                
        if len(current_group) >= min_samples:
            groups.append(current_group)

    return groups
    



def check_fall_simple(bbox: Tuple[int, int, int, int], threshold: float = 1.2) -> bool:
    """
    Phát hiện NGÃ đơn giản dựa trên tỷ lệ khung hình (Aspect Ratio)
    
    Nguyên lý:
    - Người đứng: Cao > Rộng (Width / Height < 1)
    - Người ngã: Rộng > Cao (Width / Height > 1)
    
    Tham số:
        bbox: Khung bao (x1, y1, x2, y2)
        threshold: Ngưỡng tỷ lệ W/H để coi là ngã
        
    Trả về:
        True nếu phát hiện ngã, False nếu không.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    if height == 0:
        return False
        
    aspect_ratio = width / height
    
    return aspect_ratio > threshold


# ============================================================================
# Utility Functions
# ============================================================================

def get_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """
    Calculate centroid (center point) of bounding box
    
    Args:
        bbox: (x1, y1, x2, y2)
    
    Returns:
        (cx, cy) centroid coordinates
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def euclidean_distance(point1: Tuple[float, float], 
                       point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points
    d = sqrt((x2-x1)² + (y2-y1)²)
    
    Args:
        point1: (x1, y1)
        point2: (x2, y2)
    
    Returns:
        Distance in pixels
    """
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def analyze_trajectories(trajectories: Dict[int, List[Tuple[int, int]]], 
                         min_length: int = 10,
                         n_clusters: int = 3,
                         anomaly_threshold: float = 2.0) -> Dict[int, str]:
    """
    Phân tích quỹ đạo chuyển động để phát hiện bất thường dựa trên Clustering (Gom nhóm).
    
    Nguyên lý:
    - Trích xuất đặc trưng từ quỹ đạo: [startX, startY, endX, endY, mean_angle]
    - Sử dụng K-Means để gom nhóm các quỹ đạo "bình thường" (đi theo luồng phổ biến).
    - Các quỹ đạo xa tâm cụm (outliers) được coi là bất thường.
    
    Tham số:
        trajectories: Dict {id: list of points}
        min_length: Độ dài tối thiểu của quỹ đạo để phân tích
        n_clusters: Số lượng luồng di chuyển chính (giả định)
        anomaly_threshold: Hệ số khoảng cách để coi là bất thường (so với std dev)
        
    Trả về:
        results: Dict {id: 'NORMAL_TRAJECTORY' | 'ANOMALY_TRAJECTORY' | 'UNKNOWN'}
    """
    if not SKLEARN_AVAILABLE:
        print("Cảnh báo: Scikit-learn chưa được cài đặt. Bỏ qua phân tích quỹ đạo.")
        return {}
        
    # 1. Lọc và chuẩn bị dữ liệu
    valid_ids = []
    features_list = []
    
    for obj_id, trace in trajectories.items():
        if len(trace) < min_length:
            continue
            
        # Trích xuất đặc trưng
        feat = extract_trajectory_features(trace)
        valid_ids.append(obj_id)
        features_list.append(feat)
        
    # Cần ít nhất (n_clusters + 1) mẫu để chạy clustering hiệu quả
    if len(features_list) <= n_clusters + 1:
        return {}
        
    X = np.array(features_list)
    
    # 2. Chạy K-Means Clustering
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        
        # 3. Tính khoảng cách đến tâm cụm gần nhất
        # dists[i] là khoảng cách từ mẫu i đến tâm cụm của nó
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        dists = []
        for i, label in enumerate(labels):
            center = cluster_centers[label]
            dist = np.linalg.norm(X[i] - center)
            dists.append(dist)
        dists = np.array(dists)
        
        # 4. Xác định ngưỡng bất thường (Dựa trên thống kê khoảng cách)
        # Những điểm có khoảng cách lớn hơn mean + k * std là bất thường
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        threshold_val = mean_dist + anomaly_threshold * std_dist
        
        # 5. Gán nhãn kết quả
        results = {}
        for i, obj_id in enumerate(valid_ids):
            if dists[i] > threshold_val:
                results[obj_id] = "ANOMALY_TRAJECTORY"
            else:
                results[obj_id] = "NORMAL_TRAJECTORY"
                
        return results
        
    except Exception as e:
        print(f"Lỗi khi phân tích quỹ đạo: {e}")
        return {}


def extract_trajectory_features(trace: List[Tuple[int, int]]) -> List[float]:
    """
    Trích xuất feature vector từ quỹ đạo.
    Feature: [StartX, StartY, EndX, EndY, DirectX, DirectY]
    """
    path = np.array(trace)
    
    start_pt = path[0]
    end_pt = path[-1]
    
    # Vector hướng tổng quát (từ đầu đến cuối)
    direction = end_pt - start_pt
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    else:
        direction = np.array([0, 0])
        
    # Feature vector: [sx, sy, ex, ey, dx, dy]
    # Lưu ý: Nên chuẩn hóa tọa độ theo kích thước ảnh nếu có thể, 
    # nhưng ở đây ta dùng tọa độ thô (pixel) giả định camera cố định.
    features = [
        start_pt[0], start_pt[1],
        end_pt[0], end_pt[1],
        direction[0] * 100, direction[1] * 100 # Scale direction lên để cân bằng trọng số với tọa độ
    ]
    
    return features


if __name__ == "__main__":
    # Test the modules
    print("Testing Logic Modules...")
    
    # Test 1: Optical Flow (mock data)
    print("\n1. Testing Optical Flow:")
    prev_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    curr_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    test_bbox = (100, 100, 200, 300)
    is_running, score = check_optical_flow(prev_frame, curr_frame, test_bbox)
    print(f"   Running detected: {is_running}, Magnitude: {score:.2f}")
    
    # Test 2: Gathering Detection
    print("\n2. Testing Gathering Detection:")
    test_centroids = [
        (1, (100, 100)),
        (2, (110, 105)),
        (3, (105, 110)),  # Cluster 1: IDs 1,2,3
        (4, (500, 500)),
        (5, (510, 505)),
        (6, (505, 510)),  # Cluster 2: IDs 4,5,6
        (7, (1000, 1000))  # Outlier
    ]
    groups = check_gathering(test_centroids, eps=50.0, min_samples=3)
    print(f"   Detected {len(groups)} gathering groups:")
    for i, group in enumerate(groups):
        print(f"   Group {i+1}: Person IDs {group}")
    
    print("\n✓ Logic modules test completed!")
