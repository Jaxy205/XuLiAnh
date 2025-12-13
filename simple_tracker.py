import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        """
        Khởi tạo bộ theo dõi
        
        Tham số:
            max_disappeared: Số frame tối đa giữ ID khi đối tượng biến mất
            max_distance: Khoảng cách tối đa (pixel) để coi là cùng một người
        """
        # ID tiếp theo sẽ được gán (bắt đầu từ 0)
        self.next_object_id = 0
        
        # Dictionary lưu trữ: ID -> Centroid (tâm)
        self.objects = OrderedDict()
        
        # Dictionary lưu trữ: ID -> Số frame đã biến mất
        self.disappeared = OrderedDict()
        
        # Số frame tối đa cho phép biến mất trước khi xóa ID
        self.max_disappeared = max_disappeared
        
        # Khoảng cách tối đa để ghép cặp
        self.max_distance = max_distance

    def register(self, centroid):
        """Đăng ký một đối tượng mới với ID mới"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Xóa đối tượng khỏi danh sách theo dõi"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        Cập nhật vị trí các đối tượng dựa trên các khung bao (rects) mới phát hiện
        
        Tham số:
            rects: List các khung bao [x1, y1, x2, y2]
            
        Trả về:
            objects: Dictionary chứa ID và centroid hiện tại
        """
        # Kiểm tra nếu không có phát hiện nào
        if len(rects) == 0:
            # Tăng biến đếm "biến mất" cho tất cả các ID hiện có
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Nếu biến mất quá lâu thì xóa luôn
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.objects

        # Tính toán centroid cho tất cả các khung bao mới
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (c_x, c_y)

        # Nếu chưa theo dõi ai, đăng ký tất cả centroid mới
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])

        # Nếu đang theo dõi, cần ghép cặp centroid cũ và mới
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Tính khoảng cách giữa tất cả centroid cũ và mới
            # D = ma trận khoảng cách (số cũ x số mới)
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Tìm giá trị nhỏ nhất trong mỗi hàng (người cũ)
            rows = D.min(axis=1).argsort()
            
            # Tìm giá trị nhỏ nhất trong mỗi cột (người mới)
            cols = D.argmin(axis=1)[rows]

            # Tập hợp các hàng và cột đã xử lý
            used_rows = set()
            used_cols = set()

            # Duyệt qua các cặp đã ghép (theo thứ tự khoảng cách nhỏ nhất)
            for (row, col) in zip(rows, cols):
                # Nếu đã xử lý rồi thì bỏ qua
                if row in used_rows or col in used_cols:
                    continue
                
                # Nếu khoảng cách quá xa, không ghép cặp
                if D[row][col] > self.max_distance:
                    continue

                # Cập nhật centroid mới cho ID cũ
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                # Đánh dấu là đã xử lý
                used_rows.add(row)
                used_cols.add(col)

            # Xử lý các ID cũ không tìm thấy cặp mới (đã biến mất)
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Xử lý các centroid mới không tìm thấy cặp cũ (đối tượng mới xuất hiện)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects
