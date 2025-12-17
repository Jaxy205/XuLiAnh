import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


def iou_batch(bb_test, bb_gt):
    """
    Tính Intersection over Union (IoU) giữa hai tập hợp boxes.
    
    Args:
        bb_test: Set boxes dự đoán (N, 4)
        bb_gt: Set boxes ground truth (M, 4)
        
    Returns:
        Mảng (N, M) chứa giá trị IoU
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)  


def convert_bbox_to_z(bbox):
    """
    Chuyển đổi bbox [x1, y1, x2, y2] sang dạng vector trạng thái [x, y, s, r]
    trong đó x, y là tâm, s là diện tích (scale), r là tỷ lệ khung hình (ratio).
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Chuyển đổi vector trạng thái [x, y, s, r] ngược lại thành bbox [x1, y1, x2, y2]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))


class KalmanBoxTracker(object):
    """
    Lớp theo dõi bounding box đơn lẻ sử dụng Kalman Filter.
    """
    count = 0
    count = 0
    def __init__(self, bbox):
        # Định nghĩa mô hình trạng thái: [x, y, s, r, dx, dy, ds]
        # x, y: Tọa độ tâm của bounding box
        # s: Diện tích (scale) của bounding box
        # r: Tỷ lệ khung hình (aspect ratio) = rộng / cao
        # dx, dy, ds: Vận tốc biến thiên của x, y, s
        
        # Khởi tạo Kalman Filter từ thư viện filterpy
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # Ma trận Chuyển đổi Trạng thái (State Transition Matrix) F
        self.kf.F = np.eye(7)
        self.kf.F[0, 4] = 1 # x += dx
        self.kf.F[1, 5] = 1 # y += dy
        self.kf.F[2, 6] = 1 # s += ds
        
        # Ma trận Đo lường (Measurement Matrix) H
        self.kf.H = np.zeros((4, 7))
        self.kf.H[0, 0] = 1
        self.kf.H[1, 1] = 1
        self.kf.H[2, 2] = 1
        self.kf.H[3, 3] = 1
        
        # Ma trận Nhiễu Đo lường (Measurement Noise) R
        self.kf.R = np.eye(4) * 1.0
        self.kf.R[2, 2] *= 10.
        self.kf.R[3, 3] *= 10.
        
        # Ma trận Hiệp phương sai (Covariance Matrix) P
        self.kf.P *= 1000.
        self.kf.P[4:, 4:] *= 1000. # Give high uncertainty to the unobservable initial velocities
        
        # Ma trận Nhiễu Quá trình (Process Noise) Q
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        # Khởi tạo trạng thái ban đầu từ detection đầu tiên
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Cập nhật trạng thái bộ lọc Kalman với bounding box mới phát hiện được.
        Đây là bước "Correction" trong Kalman Filter.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Update logic using filterpy
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Dự đoán trạng thái tại bước thời gian tiếp theo (Frame tiếp theo).
        Đây là bước "Prediction" trong Kalman Filter.
        """
        # Đảm bảo diện tích không âm (trường hợp hiếm gặp do nhiễu)
        if((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Trả về bounding box ước lượng hiện tại từ trạng thái Kalman.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Gán các detection mới cho các tracker hiện có sử dụng IoU và thuật toán Hungarian.
    
    Args:
        detections: Mảng các bounding box mới phát hiện
        trackers: Mảng các bounding box dự đoán từ tracker
        iou_threshold: Ngưỡng IoU tối thiểu để chấp nhận ghép cặp
        
    Returns:
        matches: Các cặp index (detection_idx, tracker_idx) đã ghép thành công
        unmatched_detections: Các detection không tìm thấy tracker tương ứng (đối tượng mới)
        unmatched_trackers: Các tracker không tìm thấy detection tương ứng (mất dấu/bị che khuất)
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    # Tính ma trận IoU giữa mọi cặp (detection, tracker)
    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        # Trường hợp đơn giản: Mỗi detection khớp chính xác 1 tracker và ngược lại
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # Sử dụng Thuật toán Hungarian (Munkres assignment) để giải quyết bài toán ghép cặp tối ưu
            # Hàm linear_sum_assignment tìm cách ghép sao cho tổng cost là nhỏ nhất.
            # Vì ta muốn tối đa hóa IoU, nên ta dùng Cost = -IoU
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.stack((row_ind, col_ind), axis=1)
    else:
        matched_indices = np.empty((0,2))

    # Tìm các detection không được ghép
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
            
    # Tìm các tracker không được ghép
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    # Lọc lại các cặp đã ghép nhưng có IoU < threshold (ghép sai do ảo giác hoặc nhiễu)
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
            
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Quản lý theo dõi đa đối tượng (SORT - Simple Online and Realtime Tracking)
        
        Args:
            max_age (int): Số frame tối đa giữ ID khi mất dấu (không có detection match)
            min_hits (int): Số frame liên tiếp phải xuất hiện trước khi coi là tracks "chính thức" 
                            (giúp loại bỏ nhiễu detection chập chờn)
            iou_threshold (float): Ngưỡng IoU tối thiểu để ghép detection với tracker
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
        # Lưu trữ trajectories: {id: [point1, point2, ...]}
        self.trajectories = {}
        self.max_traj_length = 30

    def update(self, dets=np.empty((0, 5))):
        """
        Cập nhật trạng thái tracker với detections mới từ frame hiện tại.
        
        Quy trình:
        1. Dự đoán vị trí mới của các tracker hiện tại (Kalman Predict).
        2. Ghép cặp (Associate) các detection mới với các tracker đã dự đoán.
        3. Cập nhật trạng thái của các tracker đã ghép (Kalman Update).
        4. Tạo tracker mới cho các detection chưa được ghép.
        5. Xóa các tracker đã "chết" (quá thời hạn max_age không thấy).
        
        Args:
            dets: numpy array [[x1,y1,x2,y2,score],...]
        Returns:
            ret: numpy array [[x1,y1,x2,y2,id],...]
        """
        self.frame_count += 1
        
        # 1. Dự đoán vị trí mới của các tracker hiện tại
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Loại bỏ các tracker có dự đoán bị lỗi (NaN)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # 2. Ghép cặp detections với trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # 3. Cập nhật tracker đã match
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # 4. Tạo tracker mới cho unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
            
        # 5. Xử lý đầu ra và dọn dẹp
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            
            # Chỉ trả về kết quả nếu tracker đã "sống" đủ lâu hoặc vừa mới được tạo nhưng có detection tốt
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1)) # +1 để ID bắt đầu từ 1
            
            i -= 1
            # Xóa tracker quá cũ (mất dấu quá lâu)
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                
        if(len(ret)>0):
            ret = np.concatenate(ret)
            
            # Cập nhật trajectories với kết quả final (dùng cho tính toán gathering/movement)
            for item in ret:
                x1, y1, x2, y2, obj_id = item
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                obj_id = int(obj_id)
                
                if obj_id not in self.trajectories:
                    self.trajectories[obj_id] = []
                self.trajectories[obj_id].append((cx, cy))
                
                if len(self.trajectories[obj_id]) > self.max_traj_length:
                    self.trajectories[obj_id].pop(0)

        return ret
