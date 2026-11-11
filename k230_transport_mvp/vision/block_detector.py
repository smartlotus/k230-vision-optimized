"""Native multi-color components and fresh-frame tracking; no inferred millimeters."""
from .common import elapsed_ms


class BlockDetector:
    def __init__(self, parameters):
        self.p = parameters
        self.color_ids = tuple(c for c in sorted(parameters['thresholds']) if parameters['thresholds'][c] is not None)
        self.thresholds = [parameters['thresholds'][c] for c in self.color_ids]
        self.codes = {1 << i:c for i,c in enumerate(self.color_ids)}
        self.reset()

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.last_frame = self.last_time = None

    def detect(self, image):
        p = self.p
        if (image.width(),image.height()) != (p['width'],p['height']):
            raise ValueError('Block profile resolution mismatch')
        if not self.thresholds:
            return []
        blobs = image.find_blobs(self.thresholds, roi=p['roi'], pixels_threshold=p['min_pixels'],
            area_threshold=p['min_area'], merge=False, x_stride=2, y_stride=2)
        result = []
        for blob in blobs:
            color_id = self.codes.get(blob.code())
            if color_id is None:
                continue
            x,y,w,h = blob.rect()
            if w <= 0 or h <= 0:
                continue
            area, count = w*h, blob.pixels()
            density = count/area
            shape=p.get('class_filters',{}).get(color_id,{})
            if not (shape.get('min_area',p['min_area']) <= area <= shape.get('max_area',p['max_area'])
                    and count >= shape.get('min_pixels',p['min_pixels'])
                    and density >= shape.get('min_density',p['min_density'])
                    and shape.get('min_aspect',p['min_aspect']) <= w/h <= shape.get('max_aspect',p['max_aspect'])):
                continue
            rx,ry,rw,rh = p['roi']
            margin = p['edge_margin']
            clipped = (x <= rx+margin or y <= ry+margin or x+w >= rx+rw-margin or y+h >= ry+rh-margin)
            result.append({'target_id':color_id,'bbox':(x,y,w,h),'center':(x+w/2,y+h/2),
                'pixels':count,'density':density,'clipped':clipped,'overlapping':False,'stable':False,
                'eligible':not clipped,'score_hint':min(100,int(density*100)),
                'profile_id':p['profile_id'],'units':'PX','x_mm':None,'y_mm':None})
        result.sort(key=lambda c:c['pixels'],reverse=True)
        result = result[:p['max_candidates']]
        # Conflicting overlapping colors are not independent physical objects.
        ambiguous = set()
        for i,a in enumerate(result):
            for j in range(i+1,len(result)):
                b = result[j]
                ax,ay,aw,ah=a['bbox']; bx,by,bw,bh=b['bbox']
                overlap=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
                if overlap > 0.5*min(aw*ah,bw*bh):
                    ambiguous.update((i,j))
                elif overlap > 0.15*min(aw*ah,bw*bh):
                    # Boxes alone cannot prove occlusion; conservatively withhold completeness.
                    a['overlapping']=b['overlapping']=True
                    a['eligible']=b['eligible']=False
        return [c for i,c in enumerate(result) if i not in ambiguous]

    def update(self, candidates, frame_id, timestamp_ms):
        if self.last_frame is not None:
            gap=elapsed_ms(timestamp_ms,self.last_time)
            if frame_id <= self.last_frame or gap <= 0 or gap > self.p['max_gap_ms']:
                self.tracks=[]
        self.last_frame,self.last_time=frame_id,timestamp_ms
        old=self.tracks
        used=set()
        updated=[]
        for candidate in candidates:
            current=dict(candidate)
            center=current['center']
            matches=[]
            for i,track in enumerate(old):
                if i in used or track['target_id']!=current['target_id']:
                    continue
                distance=sum((center[k]-track['center'][k])**2 for k in (0,1))
                if distance <= self.p['track_distance_px']**2:
                    matches.append((distance,i,track))
            matches.sort(key=lambda m:m[0])
            if matches and (len(matches)==1 or matches[1][0]-matches[0][0] > 16):
                _,index,previous=matches[0]
                used.add(index)
                history=previous['_points']
                if any(sum((center[k]-point[k])**2 for k in (0,1)) > self.p['stable_motion_px']**2 for point in history):
                    history=[]
                current['track_id']=previous['track_id']
            else:
                current['track_id']=self.next_id
                self.next_id+=1
                history=[]
            current['_points']=(history+[center])[-self.p['stable_frames']:]
            current['stable']=len(current['_points']) >= self.p['stable_frames'] and current['eligible']
            current['frame_id']=frame_id
            current['timestamp_ms']=timestamp_ms
            updated.append(current)
        # No stale ghosts: absent tracks are removed immediately.
        self.tracks=updated
        return [{k:v for k,v in t.items() if k!='_points'} for t in updated]

    def process(self, image, frame_id, timestamp_ms):
        return self.update(self.detect(image),frame_id,timestamp_ms)
