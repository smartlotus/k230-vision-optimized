"""One uniform transform for pixels, overlays and touch; no stretched axes."""
import math


class Viewport:
    def __init__(self,width,height,rect=(0,48,640,360),mode='NATIVE',pan=(0,0)):
        if type(width) is not int or type(height) is not int or min(width,height)<=0:
            raise ValueError('Positive image dimensions required')
        if mode not in ('NATIVE','FIT'):raise ValueError('Unknown view mode')
        self.width,self.height=width,height
        x,y,w,h=rect
        if min(w,h)<=0:raise ValueError('Positive viewport required')
        self.scale=min(w/width,h/height) if mode=='FIT' else 1.0
        self.sw=min(width,int(w/self.scale));self.sh=min(height,int(h/self.scale))
        self.ox=max(0,min(int(pan[0]),width-self.sw)) if mode=='NATIVE' else 0
        self.oy=max(0,min(int(pan[1]),height-self.sh)) if mode=='NATIVE' else 0
        self.x=x+int((w-self.sw*self.scale)/2);self.y=y+int((h-self.sh*self.scale)/2)
        self.w=int(self.sw*self.scale);self.h=int(self.sh*self.scale)

    def image_point(self,x,y):
        if not self.x<=x<self.x+self.w or not self.y<=y<self.y+self.h:return None
        return (self.ox+(x-self.x)/self.scale,self.oy+(y-self.y)/self.scale)

    def screen_point(self,u,v):
        return (int(self.x+(u-self.ox)*self.scale),int(self.y+(v-self.oy)*self.scale))

    def drag_roi(self,a,b):
        p,q=self.image_point(*a),self.image_point(*b)
        if p is None or q is None:return None
        x=max(0,int(math.floor(min(p[0],q[0]))));y=max(0,int(math.floor(min(p[1],q[1]))))
        right=min(self.width,int(math.ceil(max(p[0],q[0])))+1)
        bottom=min(self.height,int(math.ceil(max(p[1],q[1])))+1)
        return [x,y,right-x,bottom-y] if right-x>=8 and bottom-y>=8 else None

    def screen_roi(self,roi):
        x,y,w,h=roi
        left=max(x,self.ox);top=max(y,self.oy)
        right=min(x+w,self.ox+self.sw);bottom=min(y+h,self.oy+self.sh)
        if left>=right or top>=bottom:return None
        sx,sy=self.screen_point(left,top);ex,ey=self.screen_point(right,bottom)
        return (sx,sy,max(1,ex-sx),max(1,ey-sy))
