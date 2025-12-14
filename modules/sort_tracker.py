import numpy as np
from scipy.optimize import linear_sum_assignment


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
    def __init__(self, bbox):
        # Định nghĩa mô hình trạng thái: [x, y, s, r, dx, dy, ds]
        # x, y: tâm
        # s: diện tích
        # r: tỷ lệ (aspect ratio)
        # dx, dy, ds: vận tốc biến thiên
        
        # Self-implemented standard Kalman Filter setup manually to avoid filterpy dependency if needed,
        # but filterpy is standard. I'll implement a basic one using numpy only for portability.
        # Actually, for robustness, I will try to implement the matrices directly.
        
        # State vector [u, v, s, r, u', v', s']
        self.kf_x = np.zeros((7, 1))
        self.kf_P = np.eye(7) * 1000.  # Covariance matrix
        
        # State Transition Matrix F
        self.kf_F = np.eye(7)
        self.kf_F[0, 4] = 1
        self.kf_F[1, 5] = 1
        self.kf_F[2, 6] = 1
        
        # Measurement Matrix H (chúng ta đo lường x, y, s, r)
        self.kf_H = np.zeros((4, 7))
        self.kf_H[0, 0] = 1
        self.kf_H[1, 1] = 1
        self.kf_H[2, 2] = 1
        self.kf_H[3, 3] = 1
        
        # Measurement Noise R
        self.kf_R = np.eye(4) * 1.0
        self.kf_R[2, 2] *= 10.
        self.kf_R[3, 3] *= 10.
        
        # Process Noise Q
        self.kf_Q = np.eye(7) * 1.0
        self.kf_Q[4:, 4:] *= 0.01
        self.kf_Q[6, 6] *= 0.0001 # Area changes slowly
        
        # Initialize state
        self.kf_x[:4] = convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Cập nhật trạng thái với quan sát mới (bbox)
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Measurement step
        z = convert_bbox_to_z(bbox)
        
        # Calculate Kalman Gain
        # S = H * P * H.T + R
        S = np.dot(self.kf_H, np.dot(self.kf_P, self.kf_H.T)) + self.kf_R
        # K = P * H.T * inv(S)
        K = np.dot(np.dot(self.kf_P, self.kf_H.T), np.linalg.inv(S))
        
        # Update State: x = x + K * (z - H * x)
        y = z - np.dot(self.kf_H, self.kf_x)
        self.kf_x = self.kf_x + np.dot(K, y)
        
        # Update Covariance: P = (I - K * H) * P
        I = np.eye(self.kf_x.shape[0])
        self.kf_P = np.dot((I - np.dot(K, self.kf_H)), self.kf_P)

    def predict(self):
        """
        Dự đoán trạng thái tại bước tiếp theo
        """
        # Tránh diện tích âm
        if((self.kf_x[6] + self.kf_x[2]) <= 0):
            self.kf_x[6] *= 0.0
            
        # Predict State: x = F * x
        self.kf_x = np.dot(self.kf_F, self.kf_x)
        
        # Predict Covariance: P = F * P * F.T + Q
        self.kf_P = np.dot(np.dot(self.kf_F, self.kf_P), self.kf_F.T) + self.kf_Q
        
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf_x))
        return self.history[-1]

    def get_state(self):
        """
        Lấy bounding box ước lượng hiện tại
        """
        return convert_x_to_bbox(self.kf_x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Gán các detection mới cho các tracker hiện có sử dụng IoU
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # Sử dụng Hungarian Algorithm (Munkres assignment)
            # scipy.optimize.linear_sum_assignment tìm min cost, nên ta dùng -iou
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.stack((row_ind, col_ind), axis=1)
    else:
        matched_indices = np.empty((0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
            
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    # Lọc các cặp có IoU thấp hơn threshold
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
        SORT Tracker
        Args:
            max_age (int): Số frame tối đa giữ ID khi mất dấu
            min_hits (int): Số lần hit tối thiểu để coi là active (dùng để lọc nhiễu ban đầu)
            iou_threshold (float): Ngưỡng IoU tối thiểu để match
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
        Cập nhật trạng thái tracker với detections mới
        Args:
            dets: numpy array [[x1,y1,x2,y2,score],...]
        Returns:
            ret: numpy array [[x1,y1,x2,y2,id],...]
        """
        self.frame_count += 1
        
        # Dự đoán vị trí mới của các tracker hiện tại
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Ghép cặp detections với trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Cập nhật tracker đã match
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Tạo tracker mới cho unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
            
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            
            # Chỉ trả về kết quả nếu tracker đã "sống" đủ lâu hoặc vừa mới được tạo nhưng có detection tốt
            # Logic gốc của SORT: (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)
            # Điều chỉnh nhẹ để nhạy hơn
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1)) # +1 để ID bắt đầu từ 1
            
            i -= 1
            # Xóa tracker quá cũ
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                
        if(len(ret)>0):
            ret = np.concatenate(ret)
            
            # Cập nhật trajectories với kết quả final
            for item in ret:
                x1, y1, x2, y2, obj_id = item
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                obj_id = int(obj_id)
                
                if obj_id not in self.trajectories:
                    self.trajectories[obj_id] = []
                self.trajectories[obj_id].append((cx, cy))
                
                if len(self.trajectories[obj_id]) > self.max_traj_length:
                    self.trajectories[obj_id].pop(0)

        # Clean trajectories of dead IDs
        active_ids = [int(item[4]) for item in ret] if len(ret) > 0 else []
        # Có thể dọn dẹp trajectories cũ nếu muốn tiết kiệm bộ nhớ, 
        # nhưng logic gốc giữ lại một chút.
        # Ở đây ta sẽ giữ đơn giản: Chỉ giữ traj của active object và những object vừa mất
        # Nhưng để tránh xóa nhầm khi bị flick, ta cứ giữ, và clear định kỳ nếu dictionary quá lớn
        
        return ret
