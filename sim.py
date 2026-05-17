"""
Depo Teslimat Robotu - Otonom Navigasyon Simülasyonu
Mobil Robotlar Ödevi

Kullanılan yapay zeka: Claude (claude-sonnet-4-6)
Kullanılan bölümler: Kod iskeleti, algoritma implementasyonu, arayüz tasarımı
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.widgets import RadioButtons, Button
import heapq
import random
import math
from typing import List, Tuple

# ─── RENKLER ────────────────────────────────────────────────────────────────
DARK  = "#0d1117"
PANEL = "#0f3460"
ACC   = "#e94560"
SHELF_COLORS = [
    "#5C3317","#7B4B2A","#8B5E3C",
    "#5C3317","#7B4B2A","#8B5E3C",
    "#2C4A6E","#1A6B3C","#1A6B3C","#1A6B3C","#1A6B3C","#1A6B3C"
]


# ─── PROFESYONEL AYARLAR ────────────────────────────────────────────────────
@dataclass
class SimConfig:
    dt: float = 0.1
    steps_per_tick: int = 4
    timer_ms: int = 50
    lidar_ray_stride: int = 6
    lidar_update_every: int = 1
    loc_update_every: int = 1
    err_update_every: int = 1
    fps_ema_alpha: float = 0.12


# ─── ORTAM ──────────────────────────────────────────────────────────────────
class Environment:
    def __init__(self):
        self.width  = 20.0
        self.height = 15.0
        self.start  = (1.0, 1.0)
        self.goal   = (19.0, 14.0)
        # Üç yatay bariyer — geçit: SAĞ → ORTA → SOL (zigzag)
        self.obstacles = [
            ( 2.0,  4.0, 12.0,  5.5),
            (14.0,  4.0, 19.0,  5.5),
            ( 2.0,  8.0,  8.5,  9.5),
            (11.0,  8.0, 19.0,  9.5),
            ( 2.0, 11.5,  5.5, 13.0),
            ( 8.0, 11.5, 19.0, 13.0),
            ( 9.0,  1.5, 11.0,  3.0),
            ( 5.0,  6.0,  7.0,  7.5),
            (13.5,  6.0, 15.5,  7.5),
            (10.5, 10.5, 12.5, 11.5),
            ( 3.0, 10.0,  4.5, 11.0),
            (16.5, 10.5, 18.0, 11.5),
        ]
        self.labels = [
            "Raf A-Sol","Raf A-Sag","Raf B-Sol","Raf B-Sag",
            "Raf C-Sol","Raf C-Sag","Blok","Kutu1","Kutu2",
            "Kutu3","Kutu4","Kutu5",
        ]

    def in_obstacle(self, x: float, y: float, margin: float = 0.32) -> bool:
        for (x1,y1,x2,y2) in self.obstacles:
            if x1-margin<=x<=x2+margin and y1-margin<=y<=y2+margin:
                return True
        return False

    def in_bounds(self, x: float, y: float, margin: float = 0.32) -> bool:
        return margin<=x<=self.width-margin and margin<=y<=self.height-margin

    def is_free(self, x: float, y: float, margin: float = 0.32) -> bool:
        return self.in_bounds(x,y,margin) and not self.in_obstacle(x,y,margin)

    def segment_free(self, x1: float, y1: float, x2: float, y2: float, steps: int = 25) -> bool:
        for i in range(steps+1):
            t=i/steps
            if not self.is_free(x1+t*(x2-x1),y1+t*(y2-y1)):
                return False
        return True


# ─── SENSÖRLER ──────────────────────────────────────────────────────────────
class LiDAR:
    def __init__(self, env: Environment, num_beams: int = 36, max_range: float = 8.0, noise_std: float = 0.10):
        self.env=env; self.num_beams=num_beams
        self.max_range=max_range; self.noise_std=noise_std
        self.angles=np.linspace(0,2*np.pi,num_beams,endpoint=False)

    def scan(self, x: float, y: float, theta: float) -> Tuple[np.ndarray, np.ndarray]:
        raw, filt = [], []
        for a in self.angles:
            d=self._cast(x,y,theta+a)
            raw.append(max(0.05, d+np.random.normal(0,self.noise_std)))
            filt.append(max(0.05,d))
        return np.array(raw), np.array(filt)

    def _cast(self, x: float, y: float, angle: float) -> float:
        cos_a=math.cos(angle); sin_a=math.sin(angle)
        for d in np.arange(0.1, self.max_range, 0.08):
            if not self.env.is_free(x+d*cos_a, y+d*sin_a, margin=0.0):
                return d
        return self.max_range

    def to_points(self, x: float, y: float, theta: float, ranges: np.ndarray) -> List[Tuple[float,float]]:
        return [(x+r*math.cos(theta+a), y+r*math.sin(theta+a))
                for a,r in zip(self.angles,ranges)]


class IMU:
    def __init__(self, noise_std: float = 0.012, bias: float = 0.003):
        self.noise_std=noise_std; self.bias=bias

    def measure(self, omega: float) -> float:
        return omega+self.bias+np.random.normal(0,self.noise_std)


class Encoder:
    def __init__(self, slip: float = 0.01):
        self.slip=slip

    def measure(self, dl: float, dr: float) -> Tuple[float,float]:
        return (dl*(1+np.random.uniform(-self.slip,self.slip)),
                dr*(1+np.random.uniform(-self.slip,self.slip)))


# ─── ROBOT ÇİZİM YARDIMCISI ─────────────────────────────────────────────────
def _rect_poly(cx: float, cy: float, theta: float, w: float, h: float):
    c,s=math.cos(theta),math.sin(theta)
    hw,hh=w/2,h/2
    pts=[(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)]
    return [(cx+c*px-s*py, cy+s*px+c*py) for px,py in pts]

def _add_rect(ax, cx,cy,theta,w,h, fc,ec="white",lw=1.2,zorder=7,alpha=1.0):
    p=patches.Polygon(_rect_poly(cx,cy,theta,w,h),closed=True,
                      fc=fc,ec=ec,lw=lw,zorder=zorder,alpha=alpha)
    ax.add_patch(p); return p


# ─── ROBOT MODELLERİ ────────────────────────────────────────────────────────
class DiffDriveRobot:
    name="Differential"; L=0.5; holonomic=False

    def __init__(self,x: float,y: float,theta: float = 0.0):
        self.x=x; self.y=y; self.theta=theta
        self.L=self.__class__.L

    def control(self,tx: float,ty: float) -> Tuple[float,float]:
        dx=tx-self.x; dy=ty-self.y; dist=math.hypot(dx,dy)
        des=math.atan2(dy,dx)
        aerr=math.atan2(math.sin(des-self.theta),math.cos(des-self.theta))
        v=min(0.7,0.5*dist)
        omega=float(np.clip(2.5*aerr,-2.5,2.5))
        return v,omega

    def true_omega(self, v: float, omega_cmd: float) -> float:
        return omega_cmd

    def step(self,v: float,omega: float,dt: float = 0.1) -> Tuple[float,float]:
        dl=(v-omega*self.L/2)*dt; dr=(v+omega*self.L/2)*dt
        dS=(dl+dr)/2; dth=(dr-dl)/self.L
        mid=self.theta+dth/2
        self.x+=dS*math.cos(mid); self.y+=dS*math.sin(mid); self.theta+=dth
        return dl,dr

    def draw(self,ax):
        arts=[]
        c,s=math.cos(self.theta),math.sin(self.theta)
        lf=np.array([-s,c])
        pos=np.array([self.x,self.y])
        arts.append(_add_rect(ax,self.x,self.y,self.theta,0.55,0.35,
                               fc="#2471a3",ec="#85c1e9",lw=1.5,zorder=7))
        for side in [-1,1]:
            wc=pos+lf*side*0.22
            arts.append(_add_rect(ax,wc[0],wc[1],self.theta,0.20,0.08,
                                   fc="#1c1c1c",ec="#7f8c8d",lw=1,zorder=8))
        ex=self.x+0.38*c; ey=self.y+0.38*s
        arts.append(ax.annotate("",xy=(ex,ey),xytext=(self.x,self.y),
            arrowprops=dict(arrowstyle="->",color="#f1c40f",lw=2.0),zorder=9))
        return arts


class AckermannRobot:
    name="Ackermann"; L=0.70; max_steer=math.radians(35); holonomic=False

    def __init__(self,x: float,y: float,theta: float = 0.0):
        self.x=x; self.y=y; self.theta=theta; self.steer=0.0
        self.L=self.__class__.L

    def control(self,tx: float,ty: float) -> Tuple[float,float]:
        dx=tx-self.x; dy=ty-self.y; dist=math.hypot(dx,dy)
        des=math.atan2(dy,dx)
        aerr=math.atan2(math.sin(des-self.theta),math.cos(des-self.theta))
        v=min(0.7,0.5*dist)
        omega=float(np.clip(2.5*aerr,-2.5,2.5))
        self.steer=float(np.clip(
            math.atan2(omega*self.L,v) if abs(v)>0.01 else 0.0,
            -self.max_steer, self.max_steer))
        return v,omega

    def true_omega(self, v: float, omega_cmd: float) -> float:
        return v*math.tan(self.steer)/self.L

    def step(self,v: float,omega: float,dt: float = 0.1) -> Tuple[float,float]:
        dth=v*math.tan(self.steer)/self.L*dt
        mid=self.theta+dth/2
        self.x+=v*math.cos(mid)*dt; self.y+=v*math.sin(mid)*dt; self.theta+=dth
        dl=(v-omega*self.L/2)*dt; dr=(v+omega*self.L/2)*dt
        return dl,dr

    def draw(self,ax):
        arts=[]
        c,s=math.cos(self.theta),math.sin(self.theta)
        fw=np.array([c,s]); lf=np.array([-s,c])
        pos=np.array([self.x,self.y])
        # Body
        arts.append(_add_rect(ax,self.x,self.y,self.theta,0.78,0.34,
                               fc="#922b21",ec="#f1948a",lw=1.5,zorder=7))
        # Roof
        rpos=pos+fw*0.05
        arts.append(_add_rect(ax,rpos[0],rpos[1],self.theta,0.32,0.22,
                               fc="#7b241c",ec="#f1948a",lw=1,zorder=8,alpha=0.9))
        # Rear wheels (fixed)
        rax=pos-fw*0.26
        for side in [-1,1]:
            wc=rax+lf*side*0.22
            arts.append(_add_rect(ax,wc[0],wc[1],self.theta,0.16,0.08,
                                   fc="#1c1c1c",ec="#7f8c8d",lw=1,zorder=9))
        # Front wheels (steered)
        fax=pos+fw*0.26
        fth=self.theta+self.steer
        for side in [-1,1]:
            wc=fax+lf*side*0.22
            arts.append(_add_rect(ax,wc[0],wc[1],fth,0.16,0.08,
                                   fc="#1c1c1c",ec="#bdc3c7",lw=1,zorder=9))
        ex=self.x+0.5*c; ey=self.y+0.5*s
        arts.append(ax.annotate("",xy=(ex,ey),xytext=(self.x,self.y),
            arrowprops=dict(arrowstyle="->",color="#f39c12",lw=2.0),zorder=10))
        return arts


class MecanumRobot:
    name="Mecanum"; L=0.5; holonomic=True

    def __init__(self,x: float,y: float,theta: float = 0.0):
        self.x=x; self.y=y; self.theta=theta
        self.vx_w=0.0; self.vy_w=0.0
        self.L=self.__class__.L

    def control(self,tx: float,ty: float) -> Tuple[float,float]:
        dx=tx-self.x; dy=ty-self.y; dist=math.hypot(dx,dy)
        speed=min(0.8,0.5*dist)
        if dist>0.01:
            self.vx_w=speed*dx/dist; self.vy_w=speed*dy/dist
        else:
            self.vx_w=self.vy_w=0.0
        des=math.atan2(dy,dx)
        aerr=math.atan2(math.sin(des-self.theta),math.cos(des-self.theta))
        omega=float(np.clip(1.5*aerr,-2.0,2.0))
        return speed,omega

    def true_omega(self, v: float, omega_cmd: float) -> float:
        return omega_cmd

    def step(self,v: float,omega: float,dt: float = 0.1) -> Tuple[float,float]:
        nx=self.x+self.vx_w*dt; ny=self.y+self.vy_w*dt
        self.x=float(np.clip(nx,0.35,19.65)); self.y=float(np.clip(ny,0.35,14.65))
        self.theta+=omega*dt
        dl=(v-omega*self.L/2)*dt; dr=(v+omega*self.L/2)*dt
        return dl,dr

    def draw(self,ax):
        arts=[]
        c,s=math.cos(self.theta),math.sin(self.theta)
        fw=np.array([c,s]); lf=np.array([-s,c])
        pos=np.array([self.x,self.y])
        arts.append(_add_rect(ax,self.x,self.y,self.theta,0.46,0.46,
                               fc="#7d3c98",ec="#d2b4de",lw=1.5,zorder=7))
        # Mecanum wheels with roller angle
        wheel_offsets=[ fw*0.17+lf*0.27,  fw*0.17-lf*0.27,
                       -fw*0.17+lf*0.27, -fw*0.17-lf*0.27]
        roller_off=[45,-45,-45,45]
        for wp,ra in zip(wheel_offsets,roller_off):
            wc=pos+wp
            arts.append(_add_rect(ax,wc[0],wc[1],self.theta+math.radians(ra),
                                   0.16,0.07,fc="#1c1c1c",ec="#d2b4de",lw=1,zorder=8))
        # World velocity arrow
        vm=math.hypot(self.vx_w,self.vy_w)
        if vm>0.05:
            ex=self.x+0.5*self.vx_w/vm; ey=self.y+0.5*self.vy_w/vm
            arts.append(ax.annotate("",xy=(ex,ey),xytext=(self.x,self.y),
                arrowprops=dict(arrowstyle="->",color="#a9cce3",lw=2.0),zorder=9))
        # Heading arrow (smaller)
        arts.append(ax.annotate("",xy=(self.x+0.3*c,self.y+0.3*s),
            xytext=(self.x,self.y),
            arrowprops=dict(arrowstyle="->",color="#d7bde2",lw=1.2),zorder=9))
        return arts


class UnicycleRobot:
    name="Unicycle"; L=0.5; holonomic=False

    def __init__(self,x: float,y: float,theta: float = 0.0):
        self.x=x; self.y=y; self.theta=theta
        self.L=self.__class__.L

    def control(self,tx: float,ty: float) -> Tuple[float,float]:
        dx=tx-self.x; dy=ty-self.y; dist=math.hypot(dx,dy)
        des=math.atan2(dy,dx)
        aerr=math.atan2(math.sin(des-self.theta),math.cos(des-self.theta))
        v=min(0.7,0.5*dist)
        omega=float(np.clip(3.0*aerr,-3.0,3.0))
        return v,omega

    def true_omega(self, v: float, omega_cmd: float) -> float:
        return omega_cmd

    def step(self,v: float,omega: float,dt: float = 0.1) -> Tuple[float,float]:
        self.x+=v*math.cos(self.theta)*dt; self.y+=v*math.sin(self.theta)*dt
        self.theta+=omega*dt
        dl=(v-omega*self.L/2)*dt; dr=(v+omega*self.L/2)*dt
        return dl,dr

    def draw(self,ax):
        arts=[]
        c,s=math.cos(self.theta),math.sin(self.theta)
        lf=np.array([-s,c])
        circ=patches.Circle((self.x,self.y),0.27,fc="#16a085",ec="#a2d9ce",lw=2,zorder=7)
        ax.add_patch(circ); arts.append(circ)
        hub=patches.Circle((self.x,self.y),0.07,fc="#0e6655",ec="#a2d9ce",lw=1,zorder=8)
        ax.add_patch(hub); arts.append(hub)
        # Wheel plane line
        w1=np.array([self.x,self.y])+lf*0.27
        w2=np.array([self.x,self.y])-lf*0.27
        ln,=ax.plot([w1[0],w2[0]],[w1[1],w2[1]],"-",color="#1c1c1c",lw=4,zorder=9)
        arts.append(ln)
        arts.append(ax.annotate("",xy=(self.x+0.38*c,self.y+0.38*s),
            xytext=(self.x,self.y),
            arrowprops=dict(arrowstyle="->",color="#f8c471",lw=2.5),zorder=10))
        return arts


ROBOT_TYPES={"Differential":DiffDriveRobot,"Ackermann":AckermannRobot,
             "Mecanum":MecanumRobot,"Unicycle":UnicycleRobot}


# ─── LOKALİZASYON ───────────────────────────────────────────────────────────
class DeadReckoning:
    def __init__(self,x: float,y: float,theta: float,L: float = 0.5):
        self.x=x; self.y=y; self.theta=theta; self.L=L
        self.history=[(x,y)]

    def update(self,dl: float,dr: float):
        dS=(dl+dr)/2; dth=(dr-dl)/self.L
        self.x+=dS*math.cos(self.theta+dth/2)
        self.y+=dS*math.sin(self.theta+dth/2)
        self.theta+=dth
        self.history.append((self.x,self.y))


class EKF:
    """EKF — durum: [x, y, θ]. IMU + LiDAR tarama eşleşmesi ile güncelleme."""
    def __init__(self,x: float,y: float,theta: float,env: Environment,L: float = 0.5):
        self.mu=np.array([x,y,theta],dtype=float)
        self.P=np.eye(3)*0.05
        self.L=L; self.env=env
        self.Q=np.diag([0.002,0.002,0.001])   # küçük süreç gürültüsü
        self.R_imu=np.array([[8e-4]])           # IMU açısal ölçüm gürültüsü
        self.R_lid=0.03                         # LiDAR menzil gürültüsü varyansı
        self._theta_prev=theta
        self._lid_ctr=0
        self.history=[(x,y)]

    # ── Tahmin adımı (v, ω) ──
    def predict_vw(self,v: float,omega: float,dt: float):
        self._theta_prev=self.mu[2]
        dth=omega*dt
        dS=v*dt
        mid=self.mu[2]+dth/2
        F=np.array([[1,0,-dS*math.sin(mid)],
                    [0,1, dS*math.cos(mid)],
                    [0,0,1]])
        self.mu[0]+=dS*math.cos(mid)
        self.mu[1]+=dS*math.sin(mid)
        self.mu[2]+=dth
        self.P=F@self.P@F.T+self.Q

    # ── IMU ile açı düzeltmesi ──
    def update_imu(self,omega_meas: float,dt: float):
        z_th=self._theta_prev+omega_meas*dt   # IMU'nun tahmin ettiği yeni açı
        H=np.array([[0,0,1]])
        S=H@self.P@H.T+self.R_imu
        K=(self.P@H.T)/S[0,0]
        innov=math.atan2(math.sin(z_th-self.mu[2]),math.cos(z_th-self.mu[2]))
        self.mu+=K.flatten()*innov
        self.P=(np.eye(3)-np.outer(K,H))@self.P

    # ── LiDAR tarama eşleşmesi ile konum düzeltmesi ──
    def update_lidar(self,lidar_raw: np.ndarray,lidar_angles: np.ndarray):
        self._lid_ctr+=1
        self.history.append((float(self.mu[0]),float(self.mu[1])))
        if self._lid_ctr%4!=0:   # her 4 adımda bir LiDAR güncellemesi
            return
        step=9   # 36 ışın / 9 = 4 ışın kullan
        for i in range(0,len(lidar_raw),step):
            z_m=lidar_raw[i]
            if z_m>7.4: continue   # duvar yok
            aw=lidar_angles[i]+self.mu[2]   # dünya çerçevesinde açı
            z_e=self._cast(self.mu[0],self.mu[1],aw)
            if z_e>7.4: continue
            innov=z_m-z_e
            if abs(innov)>1.2: continue   # aykırı değeri reddet
            eps=0.05
            h_x=(self._cast(self.mu[0]+eps,self.mu[1],aw)-z_e)/eps
            h_y=(self._cast(self.mu[0],self.mu[1]+eps,aw)-z_e)/eps
            h_t=(self._cast(self.mu[0],self.mu[1],aw+eps)-z_e)/eps
            H=np.array([[h_x,h_y,h_t]])
            S_val=float((H@self.P@H.T)[0,0])+self.R_lid
            if abs(S_val)<1e-9: continue
            K=(self.P@H.T)/S_val
            self.mu+=K.flatten()*innov
            self.P=(np.eye(3)-np.outer(K.flatten(),H))@self.P

    def _cast(self,x: float,y: float,angle: float) -> float:
        ca=math.cos(angle); sa=math.sin(angle)
        for d in np.arange(0.1,8.0,0.12):
            if not self.env.is_free(x+d*ca,y+d*sa,margin=0.0):
                return d
        return 8.0


# ─── NAVİGASYON ALGORİTMALARI ───────────────────────────────────────────────
class AStarPlanner:
    name="A*"
    def __init__(self,env: Environment,res: float = 0.4): self.env=env; self.res=res

    def plan(self,start,goal):
        g=lambda p:(int(round(p[0]/self.res)),int(round(p[1]/self.res)))
        w=lambda c:(c[0]*self.res,c[1]*self.res)
        s=g(start); heap=[(0.0,s)]; came={}; gs={s:0.0}; cl=set()
        dirs=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        while heap:
            _,cur=heapq.heappop(heap)
            if cur in cl: continue
            cl.add(cur); wx,wy=w(cur)
            if math.hypot(wx-goal[0],wy-goal[1])<self.res*1.5:
                path=[]; c=cur
                while c in came: path.append(w(c)); c=came[c]
                return list(reversed(path))+[goal]
            for dx,dy in dirs:
                nb=(cur[0]+dx,cur[1]+dy)
                if nb in cl: continue
                wx2,wy2=w(nb)
                if not self.env.is_free(wx2,wy2): continue
                ng=gs[cur]+math.hypot(dx,dy)*self.res
                if ng<gs.get(nb,1e18):
                    came[nb]=cur; gs[nb]=ng
                    heapq.heappush(heap,(ng+math.hypot(wx2-goal[0],wy2-goal[1]),nb))
        return [start,goal]


class DijkstraPlanner:
    name="Dijkstra"
    def __init__(self,env: Environment,res: float = 0.4): self.env=env; self.res=res

    def plan(self,start,goal):
        g=lambda p:(int(round(p[0]/self.res)),int(round(p[1]/self.res)))
        w=lambda c:(c[0]*self.res,c[1]*self.res)
        s=g(start); heap=[(0.0,s)]; came={}; dist={s:0.0}; cl=set()
        dirs=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        while heap:
            d,cur=heapq.heappop(heap)
            if cur in cl: continue
            cl.add(cur); wx,wy=w(cur)
            if math.hypot(wx-goal[0],wy-goal[1])<self.res*1.5:
                path=[]; c=cur
                while c in came: path.append(w(c)); c=came[c]
                return list(reversed(path))+[goal]
            for dx,dy in dirs:
                nb=(cur[0]+dx,cur[1]+dy)
                if nb in cl: continue
                wx2,wy2=w(nb)
                if not self.env.is_free(wx2,wy2): continue
                nd=d+math.hypot(dx,dy)*self.res
                if nd<dist.get(nb,1e18):
                    came[nb]=cur; dist[nb]=nd
                    heapq.heappush(heap,(nd,nb))
        return [start,goal]


class APFPlanner:
    name="APF"
    def __init__(self,env: Environment,step: float = 0.22,k_att: float = 1.0,k_rep: float = 4.0,rep_r: float = 2.0):
        self.env=env; self.step=step; self.k_att=k_att
        self.k_rep=k_rep; self.rep_r=rep_r

    def plan(self,start,goal):
        path=[start]; x,y=start; stuck=[]
        for _ in range(4000):
            if math.hypot(x-goal[0],y-goal[1])<0.5:
                path.append(goal); break
            fx=self.k_att*(goal[0]-x); fy=self.k_att*(goal[1]-y)
            for (x1,y1,x2,y2) in self.env.obstacles:
                cx=max(x1,min(x,x2)); cy=max(y1,min(y,y2))
                d=math.hypot(x-cx,y-cy)
                if 0<d<self.rep_r:
                    f=self.k_rep*(1/d-1/self.rep_r)/d**2
                    fx+=f*(x-cx)/d; fy+=f*(y-cy)/d
            for dd,nx,ny in [(x,-1,0),(self.env.width-x,1,0),
                              (y,0,-1),(self.env.height-y,0,1)]:
                if 0<dd<self.rep_r:
                    f=self.k_rep*(1/dd-1/self.rep_r)/dd**2
                    fx+=f*nx; fy+=f*ny
            mag=math.hypot(fx,fy)
            if mag<1e-6:
                fx,fy=random.uniform(-1,1),random.uniform(-1,1)
                mag=math.hypot(fx,fy)
            x=float(np.clip(x+self.step*fx/mag,0.4,self.env.width-0.4))
            y=float(np.clip(y+self.step*fy/mag,0.4,self.env.height-0.4))
            stuck.append((x,y))
            if len(stuck)>40:
                stuck.pop(0)
                if max(math.hypot(p[0]-x,p[1]-y) for p in stuck)<0.2:
                    x+=random.uniform(-2,2); y+=random.uniform(-2,2)
                    x=float(np.clip(x,0.4,self.env.width-0.4))
                    y=float(np.clip(y,0.4,self.env.height-0.4))
                    stuck.clear()
            path.append((x,y))
        return path


class RRTPlanner:
    name="RRT"
    def __init__(self,env: Environment,step: float = 0.6,max_iter: int = 6000,goal_bias: float = 0.12):
        self.env=env; self.step=step
        self.max_iter=max_iter; self.goal_bias=goal_bias

    def plan(self,start,goal):
        nodes=[start]; parent={0:-1}
        for _ in range(self.max_iter):
            samp=(goal if random.random()<self.goal_bias else
                  (random.uniform(0.4,self.env.width-0.4),
                   random.uniform(0.4,self.env.height-0.4)))
            dists=[math.hypot(n[0]-samp[0],n[1]-samp[1]) for n in nodes]
            ni=int(np.argmin(dists)); nn=nodes[ni]; d=dists[ni]
            t=min(self.step/d,1.0) if d>0 else 1.0
            nw=(nn[0]+t*(samp[0]-nn[0]),nn[1]+t*(samp[1]-nn[1]))
            if not self.env.is_free(*nw): continue
            if not self.env.segment_free(nn[0],nn[1],nw[0],nw[1]): continue
            idx=len(nodes); nodes.append(nw); parent[idx]=ni
            if math.hypot(nw[0]-goal[0],nw[1]-goal[1])<self.step:
                gi=len(nodes); nodes.append(goal); parent[gi]=idx
                path=[]; c=gi
                while c!=-1: path.append(nodes[c]); c=parent[c]
                return list(reversed(path))
        return [start,goal]


PLANNERS={"A*":AStarPlanner,"Dijkstra":DijkstraPlanner,
          "APF":APFPlanner,"RRT":RRTPlanner}


# ─── SİMÜLASYON ─────────────────────────────────────────────────────────────
class Simulation:
    def __init__(self, config: SimConfig):
        self.cfg=config
        self.env=Environment()
        self.lidar=LiDAR(self.env)
        self.imu=IMU(); self.encoder=Encoder()
        self.nav_algo="A*"; self.loc_algo="EKF"
        self.robot_type="Differential"
        self.reset()

    def reset(self):
        sx,sy=self.env.start
        self.robot=ROBOT_TYPES[self.robot_type](sx,sy,0.0)
        self.dr=DeadReckoning(sx,sy,0.0, L=self.robot.L)
        self.ekf=EKF(sx,sy,0.0,self.env, L=self.robot.L)
        self.true_path=[(sx,sy)]
        self.lidar_raw=[]; self.lidar_filt=[]
        self.errors_ekf=[]; self.errors_dr=[]
        self.times=[]; self.t=0.0
        self.path_idx=0; self.done=False
        self.plan_path=PLANNERS[self.nav_algo](self.env).plan(
            self.env.start,self.env.goal)

    def step(self,dt: float = 0.1):
        if self.done: return
        if self.path_idx>=len(self.plan_path):
            self.done=True; return
        target=self.plan_path[self.path_idx]
        dist=math.hypot(target[0]-self.robot.x,target[1]-self.robot.y)
        if dist<0.35:
            self.path_idx+=1; return

        v_cmd,omega_cmd=self.robot.control(target[0],target[1])
        omega_used=self.robot.true_omega(v_cmd, omega_cmd)
        v_for_ekf=v_cmd
        if getattr(self.robot, "holonomic", False):
            v_for_ekf=math.hypot(getattr(self.robot, "vx_w", 0.0), getattr(self.robot, "vy_w", 0.0))

        dl_t,dr_t=self.robot.step(v_cmd,omega_cmd,dt)
        dl_e,dr_e=self.encoder.measure(dl_t,dr_t)
        om_i=self.imu.measure(omega_used)
        raw,filt=self.lidar.scan(self.robot.x,self.robot.y,self.robot.theta)
        self.lidar_raw=self.lidar.to_points(self.robot.x,self.robot.y,self.robot.theta,raw)
        self.lidar_filt=self.lidar.to_points(self.robot.x,self.robot.y,self.robot.theta,filt)

        self.dr.update(dl_e,dr_e)
        self.ekf.predict_vw(v_for_ekf, omega_used, dt)
        self.ekf.update_imu(om_i,dt)
        self.ekf.update_lidar(raw,self.lidar.angles)

        self.true_path.append((self.robot.x,self.robot.y))
        self.errors_ekf.append(math.hypot(self.ekf.mu[0]-self.robot.x,
                                          self.ekf.mu[1]-self.robot.y))
        self.errors_dr.append(math.hypot(self.dr.x-self.robot.x,
                                         self.dr.y-self.robot.y))
        self.t+=dt; self.times.append(self.t)
        if math.hypot(self.robot.x-self.env.goal[0],
                      self.robot.y-self.env.goal[1])<0.5:
            self.done=True


# ─── GRAFİK ARAYÜZÜ ─────────────────────────────────────────────────────────
class GUI:
    def __init__(self, config: SimConfig):
        self.cfg=config
        self.sim=Simulation(config)
        self.timer=None
        self._dyn_arts=[]
        self._lidar_lc: LineCollection | None = None
        self._last_frame_t=time.perf_counter()
        self._fps=0.0
        self._build()

    # ── İnşa ────────────────────────────────────────────────────────────────
    def _build(self):
        self.fig=plt.figure(figsize=(17,9),facecolor=DARK)
        self.fig.suptitle(
            "Depo Teslimat Robotu  —  Otonom Navigasyon Simülasyonu",
            color="white",fontsize=13,fontweight="bold",y=0.98)
        gs=self.fig.add_gridspec(3,3,left=0.17,right=0.97,
                                  top=0.93,bottom=0.05,hspace=0.40,wspace=0.32)
        self.ax_map  =self.fig.add_subplot(gs[0:2,0:2])
        self.ax_lidar=self.fig.add_subplot(gs[0,2])
        self.ax_loc  =self.fig.add_subplot(gs[1,2])
        self.ax_err  =self.fig.add_subplot(gs[2,:])
        self._style(self.ax_map,self.ax_lidar,self.ax_loc,self.ax_err)
        self._build_controls()
        self._draw_map()

    def _style(self,*axes):
        for ax in axes:
            ax.set_facecolor("#0d1b2a")
            ax.tick_params(colors="#aaaaaa",labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor("#2c3e50")

    # ── Kontrol paneli ───────────────────────────────────────────────────────
    def _build_controls(self):
        # Nav algoritması
        ax_n=self.fig.add_axes((0.01,0.74,0.14,0.20),facecolor=PANEL)
        self.radio_nav=RadioButtons(ax_n,list(PLANNERS.keys()),activecolor=ACC)
        ax_n.set_title("Navigasyon\nAlg.",color="white",fontsize=8,pad=3)
        for l in self.radio_nav.labels: l.set_color("white"); l.set_fontsize(9)

        # Robot türü
        ax_rt=self.fig.add_axes((0.01,0.55,0.14,0.17),facecolor=PANEL)
        self.radio_robot=RadioButtons(ax_rt,list(ROBOT_TYPES.keys()),
                                      activecolor="#27ae60")
        ax_rt.set_title("Robot Türü",color="white",fontsize=8,pad=3)
        for l in self.radio_robot.labels: l.set_color("white"); l.set_fontsize(9)

        # Lokalizasyon
        ax_l=self.fig.add_axes((0.01,0.43,0.14,0.10),facecolor=PANEL)
        self.radio_loc=RadioButtons(ax_l,["EKF","Dead Reckoning"],activecolor=ACC)
        ax_l.set_title("Lokalizasyon",color="white",fontsize=8,pad=3)
        for l in self.radio_loc.labels: l.set_color("white"); l.set_fontsize(9)

        # Butonlar
        ax_s=self.fig.add_axes((0.02,0.35,0.11,0.06))
        self.btn_start=Button(ax_s,"BASLA ▶",color=ACC,hovercolor="#c0392b")
        self.btn_start.label.set_color("white"); self.btn_start.label.set_fontweight("bold")

        ax_r=self.fig.add_axes((0.02,0.27,0.11,0.06))
        self.btn_reset=Button(ax_r,"SIFIRLA ↺",color="#533483",hovercolor="#6a44a0")
        self.btn_reset.label.set_color("white")

        # Metrik paneli
        self.ax_met=self.fig.add_axes((0.01,0.05,0.14,0.20),facecolor=PANEL)
        self.ax_met.axis("off")
        self.txt_met=self.ax_met.text(0.05,0.95,self._metric_str(),
            transform=self.ax_met.transAxes,color="white",
            fontsize=7.5,va="top",fontfamily="monospace")

        self.radio_nav.on_clicked(self._on_nav)
        self.radio_robot.on_clicked(self._on_robot)
        self.radio_loc.on_clicked(self._on_loc)
        self.btn_start.on_clicked(self._on_start)
        self.btn_reset.on_clicked(self._on_reset)

    # ── Statik harita (sadece reset/değişim sırasında çizilir) ──────────────
    def _draw_map(self):
        ax=self.ax_map; ax.clear(); self._style(ax)
        env=self.sim.env
        ax.set_xlim(-0.5,env.width+0.5); ax.set_ylim(-0.5,env.height+0.5)
        ax.set_aspect("equal")
        ax.set_title(
            f"Ortam Haritası  |  Nav: {self.sim.nav_algo}  "
            f"|  Robot: {self.sim.robot_type}  |  Lok: {self.sim.loc_algo}",
            color="white",fontsize=8.5,pad=4)
        ax.set_xlabel("X (m)",color="#aaa",fontsize=8)
        ax.set_ylabel("Y (m)",color="#aaa",fontsize=8)
        ax.grid(True,alpha=0.05,color="white",lw=0.4)
        ax.add_patch(patches.Rectangle((0,0),env.width,env.height,
                     lw=2,edgecolor="#4a4a8a",facecolor="none",zorder=1))

        for i,(x1,y1,x2,y2) in enumerate(env.obstacles):
            ax.add_patch(patches.FancyBboxPatch(
                (x1,y1),x2-x1,y2-y1,boxstyle="round,pad=0.04",
                lw=1,edgecolor="#FFD700",facecolor=SHELF_COLORS[i],alpha=0.90,zorder=2))
            ax.text((x1+x2)/2,(y1+y2)/2,env.labels[i],
                ha="center",va="center",color="white",fontsize=4.5,fontweight="bold",zorder=3)

        for gx,gy,lbl in [(13.0,4.75,"→"),(9.75,8.75,"↑"),(6.75,12.25,"←")]:
            ax.text(gx,gy,lbl,ha="center",va="center",color="#00ffcc",
                    fontsize=10,fontweight="bold",zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15",facecolor="#003333",
                              alpha=0.75,edgecolor="#00ffcc",lw=1))

        ax.plot(*env.start,"o",color="#2ecc71",ms=10,zorder=5)
        ax.plot(*env.goal, "*",color="#e74c3c",ms=13,zorder=5)
        ax.annotate("START",env.start,xytext=(4,6),textcoords="offset points",
                    color="#2ecc71",fontsize=7,fontweight="bold")
        ax.annotate("GOAL", env.goal, xytext=(4,6),textcoords="offset points",
                    color="#e74c3c",fontsize=7,fontweight="bold")

        if len(self.sim.plan_path)>1:
            pp=np.array(self.sim.plan_path)
            ax.plot(pp[:,0],pp[:,1],"--",color="#00bcd4",lw=1,alpha=0.4,
                    label="Plan",zorder=2)

        # Dinamik path line nesneleri — her tick'te set_data ile güncellenir
        self._ln_true,=ax.plot([],[],"-",color="#2ecc71",lw=1.8,zorder=4,label="Gercek")
        self._ln_ekf, =ax.plot([],[],"-",color="#ff6b35",lw=1.0,alpha=0.85,zorder=3,label="EKF")
        self._ln_dr,  =ax.plot([],[],"--",color="#ffd166",lw=0.9,alpha=0.75,zorder=3,label="DR")
        ax.legend(loc="upper left",fontsize=6.5,facecolor=PANEL,
                  labelcolor="white",framealpha=0.8)

        # LiDAR çizgileri (tek LineCollection ile, performans için)
        self._lidar_lc = LineCollection([], colors="#e74c3c", linewidths=0.5, alpha=0.10, zorder=2)
        ax.add_collection(self._lidar_lc)
        self._dyn_arts=[]

    # ── Hızlı güncelleme (her tick) ────────────────────────────���────────────
    def _update_map(self):
        if len(self.sim.true_path)>1:
            tp=np.array(self.sim.true_path)
            self._ln_true.set_data(tp[:,0],tp[:,1])
        if len(self.sim.ekf.history)>1:
            eh=np.array(self.sim.ekf.history)
            self._ln_ekf.set_data(eh[:,0],eh[:,1])
        if len(self.sim.dr.history)>1:
            dh=np.array(self.sim.dr.history)
            self._ln_dr.set_data(dh[:,0],dh[:,1])

        # Önceki dinamik nesneleri kaldır
        for a in self._dyn_arts:
            try: a.remove()
            except Exception: pass
        self._dyn_arts=[]

        # Robot çiz
        self._dyn_arts.extend(self.sim.robot.draw(self.ax_map))

        # LiDAR ışınları (LineCollection)
        if self._lidar_lc is not None:
            r=self.sim.robot
            segs=[]
            for i,(px,py) in enumerate(self.sim.lidar_raw):
                if i%self.cfg.lidar_ray_stride==0:
                    segs.append([(r.x,r.y),(px,py)])
            self._lidar_lc.set_segments(segs)

    def _update_lidar(self):
        ax=self.ax_lidar; ax.clear(); self._style(ax)
        ax.set_title("LiDAR — Ham vs Filtrelenmis",color="white",fontsize=8)
        ax.set_xlabel("X (m)",color="#aaa",fontsize=7)
        ax.set_ylabel("Y (m)",color="#aaa",fontsize=7)
        ax.grid(True,alpha=0.07,color="white")
        if not self.sim.lidar_raw: return
        r=self.sim.robot
        raw=np.array(self.sim.lidar_raw)-[r.x,r.y]
        flt=np.array(self.sim.lidar_filt)-[r.x,r.y]
        ax.scatter(raw[:,0],raw[:,1],s=5,c="#e74c3c",alpha=0.5,label="Ham",zorder=3)
        ax.scatter(flt[:,0],flt[:,1],s=7,c="#2ecc71",alpha=0.85,label="Filtrelenmis",zorder=4)
        ax.plot(0,0,"w^",ms=9,zorder=5)
        ax.set_aspect("equal")
        ax.legend(fontsize=6.5,facecolor=PANEL,labelcolor="white",framealpha=0.8)

    def _update_loc(self):
        ax=self.ax_loc; ax.clear(); self._style(ax)
        ax.set_title("Lokalizasyon Karsilastirmasi",color="white",fontsize=8)
        ax.set_xlabel("X (m)",color="#aaa",fontsize=7)
        ax.set_ylabel("Y (m)",color="#aaa",fontsize=7)
        ax.grid(True,alpha=0.07,color="white")
        if len(self.sim.true_path)<2: return
        tp=np.array(self.sim.true_path)
        ax.plot(tp[:,0],tp[:,1],"-",color="#2ecc71",lw=1.8,label="Gercek")
        if len(self.sim.ekf.history)>1:
            eh=np.array(self.sim.ekf.history)
            ax.plot(eh[:,0],eh[:,1],"-",color="#ff6b35",lw=1,alpha=0.9,label="EKF")
        if len(self.sim.dr.history)>1:
            dh=np.array(self.sim.dr.history)
            ax.plot(dh[:,0],dh[:,1],"--",color="#ffd166",lw=1,alpha=0.85,label="DR")
        ax.legend(fontsize=6.5,facecolor=PANEL,labelcolor="white",framealpha=0.8)

    def _update_err(self):
        ax=self.ax_err; ax.clear(); self._style(ax)
        ax.set_title("Konum Hatasi Analizi — Zaman Serisi",color="white",fontsize=9)
        ax.set_xlabel("Zaman (s)",color="#aaa",fontsize=8)
        ax.set_ylabel("Hata (m)",color="#aaa",fontsize=8)
        ax.grid(True,alpha=0.07,color="white")
        t=self.sim.times
        if len(t)<2: return
        ax.plot(t,self.sim.errors_ekf,"-",color="#ff6b35",lw=1.5,alpha=0.9,label="EKF hatasi")
        ax.plot(t,self.sim.errors_dr,"--",color="#ffd166",lw=1.2,alpha=0.9,label="DR hatasi")
        if len(t)>5:
            re=math.sqrt(np.mean(np.array(self.sim.errors_ekf)**2))
            rd=math.sqrt(np.mean(np.array(self.sim.errors_dr)**2))
            ax.axhline(re,color="#ff6b35",ls=":",lw=1.2,
                       label=f"RMSE EKF={re:.3f}m",alpha=0.7)
            ax.axhline(rd,color="#ffd166",ls=":",lw=1.2,
                       label=f"RMSE DR ={rd:.3f}m",alpha=0.7)
        ax.legend(fontsize=7,facecolor=PANEL,labelcolor="white",framealpha=0.8,ncol=2)

    def _update_metrics(self):
        self.txt_met.set_text(self._metric_str())

    def _metric_str(self):
        sim=self.sim
        if len(sim.true_path)<2:
            return ("Metrikler:\n──────────\n"
                    "Yol: —\nSure: —\nRMSE EKF: —\nRMSE DR:  —\n"
                    f"Nav: {sim.nav_algo}\nRobot: {sim.robot_type}\n"
                    f"Lok: {sim.loc_algo}\nFPS: {self._fps:.1f}\n"
                    "──────────\nHazir")
        pl=sum(math.hypot(sim.true_path[i][0]-sim.true_path[i-1][0],
                          sim.true_path[i][1]-sim.true_path[i-1][1])
               for i in range(1,len(sim.true_path)))
        re=math.sqrt(np.mean(np.array(sim.errors_ekf)**2)) if sim.errors_ekf else 0
        rd=math.sqrt(np.mean(np.array(sim.errors_dr)**2))  if sim.errors_dr  else 0
        st="TAMAMLANDI" if sim.done else "Calisiyor"
        return (f"Metrikler:\n──────────\n"
                f"Yol: {pl:.1f} m\nSure: {sim.t:.1f} s\n"
                f"RMSE EKF: {re:.3f} m\nRMSE DR:  {rd:.3f} m\n"
                f"Nav: {sim.nav_algo}\nRobot: {sim.robot_type}\n"
                f"Lok: {sim.loc_algo}\nFPS: {self._fps:.1f}\n"
                f"──────────\n{st}")

    # ── Tick ────────────────────────────────────────────────────────────────
    def _tick(self):
        try:
            for _ in range(self.cfg.steps_per_tick):
                if not self.sim.done:
                    self.sim.step(dt=self.cfg.dt)
            self._update_map()
            if self.cfg.lidar_update_every == 1 or (self.sim.t / self.cfg.dt) % self.cfg.lidar_update_every == 0:
                self._update_lidar()
            if self.cfg.loc_update_every == 1 or (self.sim.t / self.cfg.dt) % self.cfg.loc_update_every == 0:
                self._update_loc()
            if self.cfg.err_update_every == 1 or (self.sim.t / self.cfg.dt) % self.cfg.err_update_every == 0:
                self._update_err()
            self._update_metrics()

            now=time.perf_counter()
            dt=now-self._last_frame_t
            if dt>1e-6:
                inst_fps=1.0/dt
                self._fps=(1-self.cfg.fps_ema_alpha)*self._fps + self.cfg.fps_ema_alpha*inst_fps
            self._last_frame_t=now

            self.fig.canvas.draw_idle()
            if self.sim.done and self.timer is not None:
                self.timer.stop()
        except Exception:
            import traceback; traceback.print_exc()
            if self.timer is not None:
                self.timer.stop()

    # ── Callback'ler ────────────────────────────────────────────────────────
    def _on_start(self,_):
        self._stop_timer()
        self.timer=self.fig.canvas.new_timer(interval=self.cfg.timer_ms)
        self.timer.add_callback(self._tick)
        self.timer.start()

    def _on_reset(self,_):
        self._stop_timer(); self.sim.reset(); self._draw_map()
        for ax in (self.ax_lidar,self.ax_loc,self.ax_err):
            ax.clear(); self._style(ax)
        self._update_metrics(); self.fig.canvas.draw_idle()

    def _on_nav(self,label):
        self._stop_timer(); self.sim.nav_algo=label
        self.sim.reset(); self._draw_map()
        for ax in (self.ax_lidar,self.ax_loc,self.ax_err):
            ax.clear(); self._style(ax)
        self._update_metrics(); self.fig.canvas.draw_idle()

    def _on_robot(self,label):
        self._stop_timer(); self.sim.robot_type=label
        self.sim.reset(); self._draw_map()
        for ax in (self.ax_lidar,self.ax_loc,self.ax_err):
            ax.clear(); self._style(ax)
        self._update_metrics(); self.fig.canvas.draw_idle()

    def _on_loc(self,label):
        self._stop_timer(); self.sim.loc_algo=label
        self.sim.reset(); self._draw_map()
        self._update_metrics(); self.fig.canvas.draw_idle()

    def _stop_timer(self):
        if self.timer is not None:
            self.timer.stop(); self.timer=None

    def run(self):
        plt.show()


# ─── GİRİŞ ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg=SimConfig()
    gui=GUI(cfg)
    gui.run()
