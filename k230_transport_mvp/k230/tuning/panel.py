"""Chinese touch panel; renderer and event state are testable without a board."""
from .schema import CONTEXTS,SIZES,context_key,clone
from .geometry import Viewport
from k230.camera_roles import MODE_CAMERA

MODE_NAMES={'COLOR':'物块','TOP':'顶面','RING':'圆环','QR':'二维码','LANE':'车道'}
SITE_NAMES={'RAW':'原料','PROCESS':'粗加工','BUFFER':'暂存','FRONT':'前视通用'}
COLOR_NAMES={1:'红色',2:'黄色',3:'蓝色',4:'绿色',5:'黑色',6:'浅蓝'}
FIELD_NAMES={'min_area':'最小框面积','max_area':'最大框面积','min_pixels':'最少有效像素',
    'min_density':'最小填充率','min_aspect':'最小宽高比','max_aspect':'最大宽高比','edge_margin':'触边余量',
    'r_min':'最小半径','r_max':'最大半径','circle_threshold':'圆检测阈值','r_step':'半径步长',
    'center_tolerance':'同心圆容差','gray_min':'灰度下限','gray_max':'灰度上限',
    'lane_min_width':'车道最小宽度','lane_max_width':'车道最大宽度','scan_rows':'扫描行数',
    'sample_step':'扫描步长','center_spread':'行间中心偏差','frames':'稳定帧数',
    'motion_px':'稳定运动上限','gap_ms':'确认最大间隔','confirmations':'二维码确认帧数',
    'timeout_ms':'扫码超时毫秒','exposure_us':'手动曝光微秒'}


def fields(session,group):
    key=context_key(session.mode,session.station);root=['contexts',key];c=session.context();mode=session.mode
    if group=='STABILITY':return [(['stability',k],FIELD_NAMES[k],1) for k in ('frames','motion_px','gap_ms')]
    if group=='CAMERA':return [(['camera','exposure_us'],FIELD_NAMES['exposure_us'],100)]
    if group=='FILTER':
        names=('min_area','max_area','min_density','min_aspect','max_aspect') if mode=='TOP' else (
            ('min_area','max_area','min_pixels','min_density','min_aspect','max_aspect','edge_margin') if mode=='COLOR' else ())
    elif mode in ('COLOR','TOP'):
        if c['thresholds'][str(session.target_id)] is None:return []
        return [(root+['thresholds',str(session.target_id),i],name,1)
            for i,name in enumerate(('L 下限','L 上限','A 下限','A 上限','B 下限','B 上限'))]
    elif mode=='RING':names=('r_min','r_max','circle_threshold','r_step','center_tolerance')
    elif mode=='LANE':names=('gray_min','gray_max','lane_min_width','lane_max_width','scan_rows','sample_step','center_spread')
    else:return [(['qr',k],FIELD_NAMES[k],1 if k=='confirmations' else 100) for k in ('confirmations','timeout_ms','gap_ms')]
    return [(root+[k],FIELD_NAMES[k],0.05 if k in ('min_density','min_aspect','max_aspect') else (100 if k in ('min_area','max_area','circle_threshold') else 1)) for k in names]


def value_at(profile,path):
    for key in path:profile=profile[key]
    return profile


class Panel:
    def __init__(self,session):
        self.session=session;self.page='PREVIEW';self.group='PARAM';self.offset=0
        self.buttons=[];self.message='手动调参；保存不代表实物标定通过'
        self.gesture=None;self.roi_kind='ROI';self.live_requested=False;self.frozen=False
        self.key_field=None;self.key_text='';self.return_page='PARAM'
        self.confirm=None;self.touch_sources=[];self.viewport=None
        self.render_signature=None;self.gesture_owner=None
        self.capture_requested=False
        if session.store.warnings:self.message='最新配置损坏，已回退有效版本'

    def image_signature(self):
        s=self.session
        return (s.slot,s.mode,s.station,s.target_id,s.apply_serial,
            tuple((k,repr(s.active['binding'][k])) for k in sorted(s.active['binding'])),
            tuple((k,repr(s.view[k])) for k in sorted(s.view)))

    def button(self,canvas,rect,label,action,arg=None,active=False):
        x,y,w,h=rect
        canvas.draw_rectangle(x,y,w,h,color=(38,101,128) if active else (40,48,58),fill=True)
        canvas.draw_string_advanced(x+10,y+max(5,(h-22)//2),20,label,color=(245,248,250))
        self.buttons.append((rect,action,arg))

    def hit(self,x,y):
        return next((b for b in reversed(self.buttons) if b[0][0]<=x<b[0][0]+b[0][2] and b[0][1]<=y<b[0][1]+b[0][3]),None)

    def set_field(self,path,value):
        def edit(profile):
            node=profile
            for key in path[:-1]:node=node[key]
            node[path[-1]]=value
            if path[0]=='contexts':profile['contexts'][path[1]]['sampling_status']='MANUAL_UNVERIFIED'
        self.session.change(edit)

    def dispatch(self,action,arg=None):
        s=self.session
        try:
            if action=='page':
                self.page=arg;self.offset=0;self.gesture=None
                if arg=='PARAM':self.group='PARAM'
            elif action=='context':s.select(*arg);self.page='PREVIEW';self.gesture=None;self.roi_kind='ROI'
            elif action=='target':
                if s.mode not in ('COLOR','TOP','RING') or s.running:raise ValueError('Target controlled by runtime')
                s.target_id=arg;self.page='PREVIEW';self.gesture=None;s.observation=None;s.adapter.reset_session()
            elif action=='group':self.group=arg;self.page='PARAM';self.offset=0
            elif action=='step':
                field,direction=arg;path,label,step=field
                value=value_at(s.draft,path)+step*direction
                self.set_field(path,int(value) if type(step) is int else round(value,4));self.message=label+' 已修改，待应用'
            elif action=='keypad':
                self.key_field=arg;self.key_text=str(value_at(s.draft,arg[0]));self.return_page=self.page;self.page='KEYPAD'
            elif action=='key':self.key(arg)
            elif action=='next':self.offset=max(0,self.offset+arg)
            elif action=='enable_lab':s.enable_threshold();self.message='已建立全通草稿，请收窄LAB；尚未应用'
            elif action=='apply':s.apply();self.message='已应用；检测与稳定历史已清空'
            elif action=='save':
                record=s.save();self.message='已读回保存：配置%d / 版本%d' % (s.slot,record['generation'])
            elif action=='restore':
                self.confirm=('restore_yes',arg);self.page='CONFIRM';self.message='恢复会替换当前草稿，是否继续？'
            elif action=='restore_yes':s.restore(arg);self.page='PREVIEW';self.message='已恢复有效版本'
            elif action=='lock':s.set_locked(not s.draft['locked']);self.message='参数已锁定' if s.draft['locked'] else '已解锁，允许编辑'
            elif action=='slot':
                s.editable()
                if s.dirty:raise ValueError('Unsaved draft')
                if s.running:raise ValueError('Controller running')
                s.store.select_slot(arg)
                s.__init__(s.store,arg,s.demo);self.message='已切换配置%d' % arg
            elif action=='size':s.set_resolution(*arg);self.message='尺寸草稿已改；ROI已重置，原认可失效'
            elif action=='fit':
                mode='FIT' if s.view['mode']=='NATIVE' else 'NATIVE'
                s.set_display(mode=mode);self.gesture=None
            elif action=='pan':
                b=s.active['binding'];d=s.view;dx,dy=arg
                s.set_display(pan_x=max(0,min(b['width']-1,d['pan_x']+dx)),
                    pan_y=max(0,min(b['height']-1,d['pan_y']+dy)))
                self.gesture=None
            elif action=='roi_kind':self.roi_kind=arg;self.gesture=None
            elif action=='pose':self.set_field(['binding','pose_id'],'POSE_1');self.message='已手动登记固定机位1；不生成毫米标定'
            elif action=='review_ring':
                self.set_field(['contexts',context_key(s.mode,s.station),'ring_id_reviewed'],True)
                self.message='已确认手动编号范围；仍无毫米标定'
            elif action=='auto_exposure':self.set_field(['camera','auto_exposure'],not s.draft['camera']['auto_exposure'])
            elif action=='fps':self.set_field(['camera','fps'],60 if s.draft['camera']['fps']==30 else 30)
            elif action=='live':
                if s.running:raise ValueError('Controller running')
                if s.demo:raise ValueError('Demo camera disabled')
                self.live_requested=not self.live_requested;self.frozen=False;self.message='正在开启原始画面' if self.live_requested else '已关闭摄像头'
            elif action=='freeze':
                if s.running:raise ValueError('Controller running')
                self.frozen=not self.frozen;s.observation=None
            elif action=='capture':
                if s.running or s.demo:raise ValueError('Sample unavailable in demo or controller run')
                self.capture_requested=True;self.message='已请求原生样本，写入时短暂停顿'
            elif action=='cancel':self.page='PREVIEW';self.confirm=None;self.gesture=None
        except (ValueError,OSError,TypeError) as error:
            s.last_error=str(error)
            text=str(error)
            self.message=('参数已锁定，请先解锁' if 'locked' in text else
                '请先保存草稿再切换配置' if 'Unsaved' in text else
                '模拟界面不打开摄像头' if 'Demo' in text else
                '保存失败，上一有效版本仍保留' if isinstance(error,OSError) else '参数无效，已保留原值：'+text[:34])

    def key(self,key):
        if key=='取消':self.page=self.return_page;return
        if key=='确定':
            value=float(self.key_text)
            if self.key_field[2]>=1:
                if value!=int(value):raise ValueError('Integer required')
                value=int(value)
            self.set_field(self.key_field[0],value);self.page=self.return_page;self.message='数值已改，待应用';return
        if key=='退格':self.key_text=self.key_text[:-1]
        elif key=='清空':self.key_text=''
        elif key=='+/-':self.key_text=self.key_text[1:] if self.key_text.startswith('-') else '-'+self.key_text
        elif len(self.key_text)<14:self.key_text+=key

    def handle_touch(self,event,x,y,owner=None):
        s=self.session
        if event=='down':
            if self.gesture is not None and self.gesture_owner!=owner:return
            self.gesture_owner=owner
            hit=self.hit(x,y)
            if hit:self.gesture=('button',hit);return
            if self.page=='ROI' and self.viewport and self.viewport.image_point(x,y):
                if s.draft['binding']!=s.active['binding'] or self.image_signature()!=self.render_signature:
                    self.message='尺寸或画面已变化：先应用并等待重绘';return
                self.gesture=('roi',(x,y),self.render_signature)
        elif event=='up':
            if self.gesture_owner!=owner:return
            gesture=self.gesture;self.gesture=None
            if not gesture:return
            if gesture[0]=='button':
                if self.hit(x,y)==gesture[1]:self.dispatch(gesture[1][1],gesture[1][2])
            elif (gesture[2]==self.image_signature()==self.render_signature
                    and s.draft['binding']==s.active['binding']):
                point=self.viewport.image_point(x,y)
                if point is None:return
                if self.roi_kind=='REFERENCE':
                    try:self.set_field(['contexts',context_key(s.mode,s.station),'reference'],[int(point[0]),int(point[1])]);self.message='参考点已改，待应用'
                    except ValueError as error:self.message='编辑被拒绝：'+str(error)
                else:
                    roi=self.viewport.drag_roi(gesture[1],(x,y))
                    if roi:
                        try:s.set_roi(roi,self.roi_kind=='NUMBERED');self.message='范围已改，待应用'
                        except ValueError as error:self.message='编辑被拒绝：'+str(error)

    def render(self,canvas,frame=None,status='未开启摄像头'):
        s=self.session;self.buttons=[];canvas.clear()
        canvas.draw_rectangle(0,0,800,480,color=(15,20,28),fill=True)
        title=MODE_NAMES[s.mode]+' / '+SITE_NAMES[s.station]
        self.button(canvas,(0,0,256,44),title,'page','MODE')
        label=('环号%d' % s.target_id if s.mode=='RING' else COLOR_NAMES.get(s.target_id,'前视通用'))
        self.button(canvas,(264,0,144,44),label,'page','MODE' if s.mode in ('QR','LANE') else 'COLOR')
        self.button(canvas,(648,0,152,44),'解锁参数' if s.draft['locked'] else '锁定参数','lock',active=s.draft['locked'])
        canvas.draw_string_advanced(424,12,18,'DEMO' if s.demo else '配置%d%s' % (s.slot,' *' if s.dirty else ''),color=(255,210,80))
        if self.page in ('PREVIEW','ROI'):
            b=s.active['binding'];d=s.view;self.viewport=Viewport(b['width'],b['height'],mode=d['mode'],pan=(d['pan_x'],d['pan_y']))
            self.render_signature=self.image_signature()
            v=self.viewport
            if frame is not None:canvas.draw_image(frame,v.x,v.y,roi=(v.ox,v.oy,v.sw,v.sh),x_scale=v.scale,y_scale=v.scale)
            else:
                for x in range(v.x,v.x+v.w,40):canvas.draw_line(x,v.y,x,v.y+v.h-1,color=(37,46,58))
                for y in range(v.y,v.y+v.h,40):canvas.draw_line(v.x,y,v.x+v.w-1,y,color=(37,46,58))
                canvas.draw_string_advanced(v.x+12,v.y+v.h//2,20,'模拟格栅：不进行物块识别' if s.demo else status,color=(180,200,220))
            if s.draft['binding']['width']==b['width'] and s.draft['binding']['height']==b['height']:
                roi=s.context()['id_rois'].get(str(s.target_id)) if self.roi_kind=='NUMBERED' else s.context()['roi']
                box=v.screen_roi(roi) if roi else None
                if box:canvas.draw_rectangle(*box,color=(255,210,65),thickness=2)
                point=v.screen_point(*s.context()['reference'])
                if 0<=point[0]<640 and 48<=point[1]<408:canvas.draw_cross(*point,color=(0,230,220),size=8)
            obs=s.observation
            if obs and obs.get('bbox') and obs.get('u_px') is not None:
                box=v.screen_roi(obs['bbox'])
                if box:canvas.draw_rectangle(*box,color=(0,255,100) if obs.get('stable') else (255,150,40),thickness=2)
            canvas.draw_rectangle(640,48,160,360,color=(20,28,39),fill=True)
            for i,text in enumerate(('CSI%d' % MODE_CAMERA[s.mode],
                    '%dx%d' % (b['width'],b['height']), '输出像素1:1' if d['mode']=='NATIVE' else '等比适配',
                    '草稿/应用:%d/%d' % (s.draft['revision'],s.active['revision']),
                    '仅PX，未标定',obs['status'] if obs else '未检测',
                    ('视窗%d,%d' % (v.ox,v.oy)) if b['width']>640 else ('稳定:是' if obs and obs.get('stable') else '稳定:否'))):
                canvas.draw_string_advanced(650,50+i*21,16,text,color=(220,230,240))
            if self.page=='ROI':
                self.button(canvas,(648,198,144,48),'观察窗口','roi_kind','ROI')
                if s.mode in ('COLOR','TOP'):self.button(canvas,(648,254,144,48),'选择参考点','roi_kind','REFERENCE')
                if s.mode=='RING':self.button(canvas,(648,310,144,48),'当前环号区','roi_kind','NUMBERED')
            else:
                self.button(canvas,(648,198,144,48),'显示方式','fit')
                self.button(canvas,(648,254,144,48),'解冻' if self.frozen else '冻结画面','freeze')
                self.button(canvas,(648,310,144,48),'关闭画面' if self.live_requested else '开启画面','live')
                if b['width']>640 and d['mode']=='NATIVE':
                    for j,(label,delta) in enumerate((('←',(-160,0)),('→',(160,0)),('↑',(0,-90)),('↓',(0,90)))):
                        self.button(canvas,(j*100,350,92,48),label,'pan',delta)
        elif self.page=='MODE':
            for i,(mode,station) in enumerate(CONTEXTS):
                self.button(canvas,(8+(i%2)*400,56+(i//2)*82,384,68),MODE_NAMES[mode]+' / '+SITE_NAMES[station],'context',(mode,station))
        elif self.page=='COLOR':
            for i,cid in enumerate(range(1,4) if s.mode=='RING' else range(1,7)):
                self.button(canvas,(32+(i%2)*392,72+(i//2)*104,344,80),'环号%d' % cid if s.mode=='RING' else '%d %s' % (cid,COLOR_NAMES[cid]),'target',cid)
        elif self.page=='MORE':
            options=[('尺寸 / 机位','page','GEOMETRY'),('稳定条件','group','STABILITY'),('过滤条件','group','FILTER'),
                ('曝光设置','group','CAMERA'),('自动曝光：'+('开' if s.draft['camera']['auto_exposure'] else '关'),'auto_exposure',None),
                ('输入FPS：'+str(s.draft['camera']['fps']),'fps',None),('应用当前草稿','apply',None),('保存原生样本','capture',None)]
            for i,(label,action,arg) in enumerate(options):self.button(canvas,(8+(i%2)*400,56+(i//2)*82,384,68),label,action,arg)
        elif self.page=='GEOMETRY':
            for i,size in enumerate(SIZES):self.button(canvas,(12+(i%2)*400,60+(i//2)*84,376,68),'%d×%d / 16:9' % size,'size',size)
            self.button(canvas,(12,240,376,60),'我已固定机位，登记机位1','pose')
            if s.mode=='RING':self.button(canvas,(412,240,376,60),'确认三个手动环号区','review_ring')
            canvas.draw_string_advanced(16,322,20,'改采集尺寸后必须重新核对ROI与几何；不自动迁移旧H',color=(255,210,80))
        elif self.page=='PARAM':
            entries=fields(s,self.group)
            if not entries:
                if s.mode in ('COLOR','TOP') and self.group=='PARAM':self.button(canvas,(100,150,600,80),'建立该色LAB草稿（初始全通）','enable_lab')
                else:canvas.draw_string_advanced(32,150,22,'本模式没有此类参数，请切换参数页',color=(200,220,240))
            self.offset=min(self.offset,max(0,((len(entries)-1)//4)*4))
            for i,field in enumerate(entries[self.offset:self.offset+4]):
                path,label,step=field;y=58+i*76
                canvas.draw_string_advanced(16,y+16,20,label,color=(225,235,245))
                self.button(canvas,(294,y,68,60),'-','step',(field,-1))
                self.button(canvas,(372,y,244,60),str(value_at(s.draft,path)),'keypad',field)
                self.button(canvas,(626,y,68,60),'+','step',(field,1))
            self.button(canvas,(16,348,128,42),'上一页','next',-4)
            self.button(canvas,(152,348,128,42),'下一页','next',4)
            self.button(canvas,(480,348,296,42),'应用当前草稿','apply')
        elif self.page=='STORAGE':
            for slot in range(1,7):self.button(canvas,(8+(slot-1)*132,60,124,56),'配置%d' % slot,'slot',slot,slot==s.slot)
            options=[('保存当前草稿','save',None),('恢复最新保存','restore',False),('应用当前草稿','apply',None),('回退上一版本','restore',True)]
            for i,(label,action,arg) in enumerate(options):self.button(canvas,(12+(i%2)*400,144+(i//2)*94,376,76),label,action,arg)
            canvas.draw_string_advanced(16,350,20,'当前保存版本：'+str(s.last_commit['generation'] if s.last_commit else '无'),color=(220,230,240))
        elif self.page=='KEYPAD':
            canvas.draw_string_advanced(48,54,26,self.key_field[1]+': '+self.key_text,color=(255,230,150))
            for row,keys in enumerate((('7','8','9','退格'),('4','5','6','清空'),('1','2','3','+/-'),('0','.','取消','确定'))):
                for col,key in enumerate(keys):self.button(canvas,(72+col*166,108+row*72,154,60),key,'key',key)
        elif self.page=='CONFIRM':
            canvas.draw_string_advanced(48,140,24,'是否用已保存版本替换当前草稿？',color=(255,220,120))
            self.button(canvas,(80,230,280,80),'取消','cancel')
            self.button(canvas,(440,230,280,80),'确认恢复',self.confirm[0],self.confirm[1])
        if self.page not in ('KEYPAD','CONFIRM'):
            for i,(label,page) in enumerate((('预览','PREVIEW'),('范围','ROI'),('参数','PARAM'),('更多','MORE'),('保存','STORAGE'))):
                self.button(canvas,(i*160,416,154,60),label,'page',page,self.page==page)
        if self.page not in ('PREVIEW','ROI','KEYPAD'):
            canvas.draw_string_advanced(8,394,16,self.message[:48],color=(255,210,100))
        elif self.page in ('PREVIEW','ROI'):
            canvas.draw_string_advanced(648,365,14,self.message[:10],color=(255,220,110))
            canvas.draw_string_advanced(648,385,14,self.message[10:20],color=(255,220,110))
        else:
            canvas.draw_string_advanced(8,400,16,self.message[:48],color=(255,210,100))
