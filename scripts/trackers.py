import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from deep_sort_realtime.deepsort_tracker import DeepSort  
import cjm_byte_track  

class SORTTracker:
    """Simple SORT Tracker"""
    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.25):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.track_id = 0

    def update(self, detections):
        self.frame_count += 1
        if len(self.trackers) == 0:
            for det in detections:
                self._init_tracker(det)
            return self._get_tracks()

        # Predict
        for trk in self.trackers:
            trk['kf'].predict()
            trk['bbox'] = self._tlwh_to_xyxy(trk['kf'].x[:4])

        # Associate
        cost_matrix = self._iou_batch(np.array([trk['bbox'] for trk in self.trackers]), np.array([det[0] for det in detections]))
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        unmatched_trackers = set(range(len(self.trackers)))
        unmatched_dets = set(range(len(detections)))
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > self.iou_threshold:
                continue
            matches.append((r, c))
            unmatched_trackers.remove(r)
            unmatched_dets.remove(c)

        # Update matched
        for r, c in matches:
            self.trackers[r]['kf'].update(self._xyxy_to_tlwh(detections[c][0]))
            self.trackers[r]['bbox'] = detections[c][0]
            self.trackers[r]['conf'] = detections[c][1]
            self.trackers[r]['hits'] += 1
            self.trackers[r]['time_since_update'] = 0

        # Init new tracks
        for c in unmatched_dets:
            self._init_tracker(detections[c])

        # Delete old tracks
        self.trackers = [trk for trk in self.trackers if trk['time_since_update'] <= self.max_age]

        return self._get_tracks()

    def _init_tracker(self, det):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        kf.R[2:,2:] *= 10.
        kf.P[4:,4:] *= 1000.
        kf.P *= 10.
        kf.Q[-1,-1] *= 0.01
        kf.Q[4:,4:] *= 0.01
        kf.x[:4] = self._xyxy_to_tlwh(det[0]).reshape(4,1)
        self.trackers.append({'kf': kf, 'id': self.track_id, 'bbox': det[0], 'conf': det[1], 'hits': 1, 'time_since_update': 0})
        self.track_id += 1

    def _get_tracks(self):
        return [(trk['bbox'], trk['conf'], trk['id']) for trk in self.trackers if trk['hits'] >= self.min_hits and trk['time_since_update'] < 1]

    @staticmethod
    def _xyxy_to_tlwh(bbox):
        return np.array([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])

    @staticmethod
    def _tlwh_to_xyxy(tlwh):
        return np.array([tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]])

    @staticmethod
    def _iou_batch(bb_test, bb_gt):
        iou = np.zeros((len(bb_test), len(bb_gt)))
        for i, t in enumerate(bb_test):
            for j, g in enumerate(bb_gt):
                xx1 = np.maximum(t[0], g[0])
                yy1 = np.maximum(t[1], g[1])
                xx2 = np.minimum(t[2], g[2])
                yy2 = np.minimum(t[3], g[3])
                w = np.maximum(0., xx2 - xx1)
                h = np.maximum(0., yy2 - yy1)
                wh = w * h
                o = wh / ((t[2]-t[0])*(t[3]-t[1]) + (g[2]-g[0])*(g[3]-g[1]) - wh)
                iou[i,j] = o
        return iou

class DeepSortTracker:
    """DeepSort Tracker"""
    def __init__(self, max_age=30, nn_budget=100):
        self.tracker = DeepSort(max_age=max_age, nn_budget=nn_budget)

    def update(self, detections, frame):
        # Format detections for DeepSort: [[x1,y1,x2,y2,conf]] -> list of (bbox, conf, emb) but emb=None for default
        dets = [(det[0], det[1], None) for det in detections]
        tracks = self.tracker.update_tracks(dets, frame=frame)
        return [(trk.tlbr.tolist(), 1.0, trk.track_id) for trk in tracks if trk.is_confirmed()]

class ByteTrackTracker:
    """ByteTrack Tracker"""
    def __init__(self, track_thresh=0.5, match_thresh=0.8):
        from cjm_byte_track.basetrack import BaseTrack
        BaseTrack._count = 0  # Reset if needed
        self.tracker = cjm_byte_track.BYTETracker(track_thresh=track_thresh, match_thresh=match_thresh)

    def update(self, detections):
        # Format: detections as numpy array [x1,y1,x2,y2,conf]
        dets = np.array([det[0] + [det[1]] for det in detections]) if detections else np.empty((0,5))
        tracks = self.tracker.update(dets)
        return [(trk.tlbr.tolist(), trk.score, trk.track_id) for trk in tracks]