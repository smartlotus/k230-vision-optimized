"""Same-frame material snapshot producer; no image, board or UART imports."""
from ports.protocol import _canonical_json, encode_material_snapshot
from vision.material_tracker import MaterialTracker
from .schema import validate, clone, image_profile
from .store import sha

ALL_COLORS = (1,2,3,4,5,6)
ALGORITHM_ID = 'NATIVE_MASK_MATERIAL_SNAPSHOT_V1'


def scan_state(colors, rows, selected, overflow=False):
    """Completion is relative to an explicit scan, never a claim of accuracy."""
    if overflow:
        return False, 'SCENE_LIMIT'
    if not selected or not colors:
        return False, 'MATERIAL_SCAN_NOT_RUN'
    if (not isinstance(colors,(list,tuple)) or not isinstance(rows,(list,tuple)) or
            any(not isinstance(row,dict) for row in rows)):
        return False, 'SCAN_EVIDENCE_MISSING'
    if len(rows) != len(colors) or [r.get('color_id') for r in rows] != list(colors):
        return False, 'SCAN_EVIDENCE_MISSING'
    for row in rows:
        if row.get('scan_complete') is not True:
            return False, row.get('scan_reason') or 'SCAN_INCOMPLETE'
    return True, 'MATERIAL_SCAN_COMPLETE'


class SourceClock:
    """Extend the declared power-of-two device ticks to a uint32 source clock.

    Observe every captured frame. More than half a raw period between observations
    is ambiguous and requires a new explicit session, not an inferred wrap.
    """
    def __init__(self, period=4294967296):
        if type(period) is not int or not 2 <= period <= 4294967296 or period & (period-1):
            raise ValueError('Power-of-two tick period required')
        self.period, self.raw, self.extended = period, None, None

    def update(self, raw):
        if type(raw) is not int or not 0 <= raw < self.period:
            raise ValueError('Invalid native ticks')
        if self.raw is None:
            self.raw = raw
            self.extended = raw
        else:
            delta = (raw-self.raw) % self.period
            if not delta or delta >= self.period//2:
                raise ValueError('Nonforward or ambiguous source clock')
            self.extended = (self.extended+delta) & 0xffffffff
            self.raw = raw
        return self.extended


class MaterialSnapshotProducer:
    def __init__(self, active, draft, source_session, colors, boundary_backend,
                 mode='RING', station='PROCESS'):
        self.active, self.draft = validate(clone(active)), validate(clone(draft))
        if (not isinstance(source_session,str) or len(source_session) != 32 or
                any(c not in '0123456789abcdef' for c in source_session)):
            raise ValueError('Explicit source session required')
        if (not isinstance(colors,(list,tuple)) or len(colors)>6 or
                any(type(v) is not int or not 1<=v<=6 for v in colors) or len(set(colors))!=len(colors)):
            raise ValueError('Distinct material colors required')
        if boundary_backend not in ('SCALAR_FULL_BOUNDARY','ULAB_FULL_BOUNDARY'):
            raise ValueError('Declared native boundary backend required')
        self.source_session, self.colors = source_session, list(colors)
        self.mode, self.station = mode, station
        self.binding = image_profile(self.active)
        self.record = {'schema_version':1,'algorithm_id':ALGORITHM_ID,
            'capture_profile_sha256':sha(_canonical_json(self.active).encode()),
            'algorithm_profile_sha256':sha(_canonical_json(self.draft).encode()),
            'active_revision':self.active['revision'],'draft_revision':self.draft['revision'],
            'camera_id':'CSI2','image_profile':clone(self.binding),
            'pixel_format':self.active['binding']['pixel_format'],
            'mode':mode,'station':station,'scan_colors':list(colors),
            'maximum_materials':3,'close_kernel':3,'boundary_backend':boundary_backend,
            'tracking':dict(self.draft['stability'],match_px=30),
            'timestamp_encoding':'EXTENDED_NATIVE_TICKS_UINT32',
            'units':{'center':'SOURCE_IMAGE_PX','semiaxes':'PX','angle':'RAD'}}
        self.profile_id = sha(_canonical_json(self.record).encode())
        p = self.draft['stability']
        self.tracker = MaterialTracker(p['frames'],p['motion_px'],p['gap_ms'])
        self.last_frame = self.last_time = None

    def invalidate(self):
        self.tracker.invalidate()

    def contract(self):
        return dict(clone(self.record), profile_id=self.profile_id, source_session=self.source_session)

    def build(self, frame, geometry):
        """Caller measures using this producer's frozen draft on this source image.

        Source timestamps are extended uint32 ticks, not arbitrary native ticks.
        Legacy/incomplete evidence is UNKNOWN. This method never sends data.
        """
        if not isinstance(frame,dict):
            self.invalidate()
            raise ValueError('Explicit source metadata required')
        fid, stamp = frame.get('frame_id'), frame.get('timestamp_ms')
        if (type(fid) is not int or not 0 <= fid <= 4294967295 or type(stamp) is not int or
                not 0 <= stamp <= 4294967295):
            self.invalidate()
            raise ValueError('Fresh explicit material frame required')
        if self.last_frame is not None:
            delta = (stamp-self.last_time) & 0xffffffff
            if fid <= self.last_frame or not delta or delta > 0x7fffffff:
                self.invalidate()
                raise ValueError('Repeated or backward material source')
        self.last_frame, self.last_time = fid, stamp
        simulated = frame.get('is_simulated')
        if type(simulated) is not bool:
            self.invalidate()
            raise ValueError('Explicit source simulation flag required')
        value = dict(version=1,status='UNKNOWN',camera_id='CSI2',frame_id=fid,timestamp_ms=stamp,
                     profile_id=self.profile_id,source_session=self.source_session,
                     width=self.binding['width'],height=self.binding['height'],coord_frame='IMAGE_PX',
                     is_simulated=simulated,complete=False,objects=[],reason='SCAN_EVIDENCE_MISSING')

        def unknown(reason):
            self.invalidate()
            value['reason'] = reason
            return value

        if (frame.get('camera_id') != 2 or frame.get('image_profile') != self.binding or
                frame.get('active_revision') != self.active['revision'] or
                frame.get('width') != self.binding['width'] or frame.get('height') != self.binding['height']):
            return unknown('SOURCE_BINDING_MISMATCH')
        if self.draft['binding'] != self.active['binding']:
            return unknown('DRAFT_CAPTURE_GEOMETRY_MISMATCH')
        if self.mode != 'RING' or sorted(self.colors) != list(ALL_COLORS):
            return unknown('FULL_SIX_COLOR_SCAN_REQUIRED')
        if any(self.draft['contexts']['TOP_BUFFER']['thresholds'].get(str(c)) is None for c in self.colors):
            return unknown('MATERIAL_THRESHOLD_REQUIRED')
        if not isinstance(geometry,dict):
            return unknown('SCAN_EVIDENCE_MISSING')
        if (geometry.get('material_algorithm_revision') != self.draft['revision'] or
                geometry.get('material_scan_colors') != self.colors):
            return unknown('ALGORITHM_BINDING_MISMATCH')
        fits = geometry.get('material_fits')
        raw_count = geometry.get('raw_material_fit_count')
        if not isinstance(fits,list) or type(raw_count) is not int or not 0 <= raw_count <= 65535:
            return unknown('MATERIAL_COUNT_EVIDENCE_REQUIRED')
        if geometry.get('material_limit_exceeded') is True or raw_count > 3 or len(fits)>3:
            return unknown('SCENE_LIMIT:%d' % max(raw_count,len(fits)))
        if type(geometry.get('material_limit_exceeded')) is not bool:
            return unknown('MATERIAL_COUNT_EVIDENCE_REQUIRED')
        if (geometry.get('material_complete') is not True or raw_count != len(fits)):
            return unknown(geometry.get('material_scan_reason') or 'SCAN_INCOMPLETE')
        complete, reason = scan_state(self.colors,geometry.get('material_by_color',[]),True)
        if not complete:
            return unknown(reason)
        counts = {c:0 for c in self.colors}
        for row in fits:
            if not isinstance(row,dict) or type(row.get('color_id')) is not int or row['color_id'] not in counts:
                return unknown('MALFORMED_MATERIAL_GEOMETRY')
            counts[row['color_id']] += 1
        for row in geometry['material_by_color']:
            if row.get('threshold_available') is not True or type(row.get('top_count')) is not int or row['top_count'] != counts[row['color_id']]:
                return unknown('MATERIAL_COUNT_EVIDENCE_MISMATCH')
        objects = []
        try:
            for row in fits:
                if not isinstance(row,dict):
                    return unknown('MALFORMED_MATERIAL_GEOMETRY')
                if row.get('accepted') is not True or row.get('reason') != 'GEOMETRY_ONLY':
                    return unknown('MEASURED_TOP_GEOMETRY_REQUIRED')
                center, axes = row['center'], row['semiaxes']
                if len(center)!=2 or len(axes)!=2 or type(row['color_id']) is not int or not 1<=row['color_id']<=6:
                    return unknown('MALFORMED_MATERIAL_GEOMETRY')
                objects.append(dict(color_id=row['color_id'],u_px=center[0],v_px=center[1],
                    semimajor_px=axes[0],semiminor_px=axes[1],angle_rad=row['angle_rad'],
                    score_hint=None,stable=False,geometry_valid=True))
            # Reuse wire validation of finite/full-source ellipse bounds before tracking.
            value.update(status='OK' if objects else 'NOT_FOUND',objects=objects,complete=True,reason=None)
            encode_material_snapshot(0,value)
            for i,a in enumerate(objects):
                for b in objects[i+1:]:
                    dx,dy=a['u_px']-b['u_px'],a['v_px']-b['v_px']
                    if dx*dx+dy*dy < max(2,.5*min(a['semiminor_px'],b['semiminor_px']))**2:
                        value.update(status='UNKNOWN',objects=[],complete=False)
                        return unknown('AMBIGUOUS_DUPLICATE_TOP')
            if simulated:
                self.invalidate()
            else:
                value['objects'] = self.tracker.update((self.source_session,self.profile_id),fid,stamp,objects)
        except (KeyError,TypeError,ValueError,OverflowError):
            value.update(status='UNKNOWN',objects=[],complete=False)
            return unknown('MALFORMED_MATERIAL_GEOMETRY')
        return value
