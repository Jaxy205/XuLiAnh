"""
Các Module Logic Toán Học cho Phát Hiện Bất Thường
Cài đặt: Optical Flow (Luồng quang học) cho phát hiện chạy và Khoảng cách Euclid cho phát hiện tụ tập
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict


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


def check_gathering(centroid_list: List[Tuple[int, Tuple[float, float]]], 
                    eps: float = 100.0, 
                    min_samples: int = 3) -> List[List[int]]:
    """
    Phát hiện TỤ TẬP sử dụng Khoảng cách Euclid (Đơn giản hóa)
    
    Nguyên lý:
    - Tính khoảng cách giữa từng cặp người.
    - Nếu khoảng cách < eps (ngưỡng), coi là họ đang đứng gần nhau.
    - Nếu một nhóm có số lượng người >= min_samples -> Tụ tập.
    
    Tham số:
        centroid_list: Danh sách (id, (x, y))
        eps: Khoảng cách tối đa để cùng nhóm (tương đương distance_threshold)
        min_samples: Kích thước nhóm tối thiểu (tương đương min_group_size)
    
    Trả về:
        Danh sách các nhóm tụ tập.
    """
    
    if len(centroid_list) < min_samples:
        return []
    
    person_ids = [item[0] for item in centroid_list]
    centroids = np.array([item[1] for item in centroid_list])
    
    n = len(centroids)
    visited = [False] * n
    groups = []
    
    for i in range(n):
        if visited[i]:
            continue
        
        # Bắt đầu một nhóm mới với người thứ i
        current_group = [person_ids[i]]
        visited[i] = True
        
        # Tìm tất cả những người khác gần người này
        for j in range(n):
            if i == j or visited[j]:
                continue
            
            # Tính khoảng cách Euclid: sqrt((x2-x1)² + (y2-y1)²)
            distance = np.linalg.norm(centroids[i] - centroids[j])
            
            if distance <= eps:
                current_group.append(person_ids[j])
                visited[j] = True
        
        # Chỉ thêm nhóm nếu đủ số lượng người tối thiểu
        if len(current_group) >= min_samples:
            groups.append(current_group)
    
    return groups
    
    if len(centroid_list) < min_group_size:
        return []
    
    person_ids = [item[0] for item in centroid_list]
    centroids = np.array([item[1] for item in centroid_list])
    
    n = len(centroids)
    visited = [False] * n
    groups = []
    
    for i in range(n):
        if visited[i]:
            continue
        
        # Bắt đầu một nhóm mới với người thứ i
        current_group = [person_ids[i]]
        visited[i] = True
        
        # Tìm tất cả những người khác gần người này
        for j in range(n):
            if i == j or visited[j]:
                continue
            
            # Tính khoảng cách Euclid: sqrt((x2-x1)² + (y2-y1)²)
            distance = np.linalg.norm(centroids[i] - centroids[j])
            
            if distance <= distance_threshold:
                current_group.append(person_ids[j])
                visited[j] = True
        
        # Chỉ thêm nhóm nếu đủ số lượng người tối thiểu
        if len(current_group) >= min_group_size:
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
