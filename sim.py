"""
Depo Teslimat Robotu - Otonom Navigasyon Simülasyonu
Mobil Robotlar Ödevi

Kullanılan yapay zeka: Claude (claude-sonnet-4-6)
Kullanılan bölümler: Kod iskeleti, algoritma implementasyonu, arayüz tasarımı
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RadioButtons, Button
import heapq, random, math

# ─── RENKLER ────────────────────────────────────────────────────────────────
DARK  = "#0d1117"
PANEL = "#0f3460"
ACC   = "#e94560"
SHELF_COLORS = [
    "#5C3317","#7B4B2A","#8B5E3C",
    "#5C3317","#7B4B2A","#8B5E3C",
    "#2C4A6E","#1A6B3C","#1A6B3C","#1A6B3C","#1A6B3C","#1A6B3C"
]


# ─── ORTAM ──────────────────────────────────────────────────────────────────
class Environment:
    def __init__(self):
        self.width  = 20.0
        self.height = 15.0
        self.start  = (1.0, 1.0)
        self.goal   = (19.0, 14.0)
        # Üç yatay bariyer — geçit: SAĞ → ORTA → SOL (zigzag)
        self.obstacles = [
            ( 1.0,  4.0, 12.0,  5.5),
            (14.0,  4.0, 19.0,  5.5),
            ( 1.0,  8.0,  8.5,  9.5),
            (11.0,  8.0, 19.0,  9.5),
            ( 1.0, 11.5,  5.5, 13.0),
            ( 8.0, 11.5, 19.0, 13.0),
            ( 9.0,  1.5, 11.0,  2.2),
            ( 5.0,  6.0,  7.0,  7.5),
            (15.5,  6.5, 17.5,  8.0),
            (12.0, 10.5, 14.0, 11.5),
            ( 3.0, 10.0,  4.5, 11.0),
            (16.5, 10.5, 18.0, 11.5),
        ]
        self.labels = [
            "Raf A-Sol","Raf A-Sag","Raf B-Sol","Raf B-Sag",
            "Raf C-Sol","Raf C-Sag","Blok","Kutu1","Kutu2",
            "Kutu3","Kutu4","Kutu5",
        ]

    def in_obstacle(self, x, y, margin=0.32):
        for (x1,y1,x2,y2) in self.obstacles:
            if x1-margin<=x<=x2+margin and y1-margin<=y<=y2+margin:
                return True
        return False

    def in_bounds(self, x, y, margin=0.32):
        return margin<=x<=self.width-margin and margin<=y<=self.height-margin

    def is_free(self, x, y, margin=0.32):
        return self.in_bounds(x,y,margin) and not self.in_obstacle(x,y,margin)

    def segment_free(self, x1,y1,x2,y2,steps=28,margin=0.32):
        for i in range(steps+1):
            t=i/steps
            if not self.is_free(x1+t*(x2-x1),y1+t*(y2-y1),margin):
                return False
        return True

    def randomize(self):
        """Rastgele engel konumları — başlangıç/hedef çevresi korunur."""
        new_obs=[]
        attempts=0
        while len(new_obs)<12 and attempts<8000:
            attempts+=1
            x1=random.uniform(1.2,15.0)
            y1=random.uniform(1.2,11.5)
            w=random.uniform(0.8,3.5)
            h=random.uniform(0.6,1.8)
            x2=min(x1+w,19.2); y2=min(y1+h,13.8)
            cx=(x1+x2)/2; cy=(y1+y2)/2
            # Başlangıç ve hedeften uzak tut
            if math.hypot(cx-self.start[0],cy-self.start[1])<2.8: continue
            if math.hypot(cx-self.goal[0], cy-self.goal[1])<2.8:  continue
            # Mevcut engellerle çok fazla örtüşme olmasın
            ok=True
            for (ox1,oy1,ox2,oy2) in new_obs:
                if x1<ox2+0.5 and x2>ox1-0.5 and y1<oy2+0.5 and y2>oy1-0.5:
                    ok=False; break
            if ok:
                new_obs.append((x1,y1,x2,y2))
        self.obstacles=new_obs
        self.labels=[f"Engel{i+1}" for i in range(len(new_obs))]
        self.is_random=True

    # Varsayılan map mi?
    is_random=False


# ─── SENSÖRLER ──────────────────────────────────────────────────────────────
class LiDAR:
    def __init__(self, env, num_beams=36, max_range=8.0, noise_std=0.10):
        self.env=env; self.num_beams=num_beams
        self.max_range=max_range; self.noise_std=noise_std
        self.angles=np.linspace(0,2*np.pi,num_beams,endpoint=False)

    def scan(self, x, y, theta):
        raw, filt = [], []
        for a in self.angles:
            d=self._cast(x,y,theta+a)
            raw.append(max(0.05, d+np.random.normal(0,self.noise_std)))
            filt.append(max(0.05,d))
        return np.array(raw), np.array(filt)

    def _cast(self, x, y, angle):
        cos_a=math.cos(angle); sin_a=math.sin(angle)
        for d in np.arange(0.1, self.max_range, 0.08):
            if not self.env.is_free(x+d*cos_a, y+d*sin_a, margin=0.0):
                return d
        return self.max_range

    def to_points(self, x, y, theta, ranges):
        return [(x+r*math.cos(theta+a), y+r*math.sin(theta+a))
                for a,r in zip(self.angles,ranges)]


class IMU:
    def __init__(self, noise_std=0.012, bias=0.003):
        self.noise_std=noise_std; self.bias=bias

    def measure(self, omega):
        return omega+self.bias+np.random.normal(0,self.noise_std)


class Encoder:
    def __init__(self, slip=0.01):
        self.slip=slip

    def measure(self, dl, dr):
        return (dl*(1+np.random.uniform(-self.slip,self.slip)),
                dr*(1+np.random.uniform(-self.slip,self.slip)))


# ─── ROBOT ÇİZİM YARDIMCISI ─────────────────────────────────────────────────
def _rect_poly(cx, cy, theta, w, h):
    c,s=math.cos(theta),math.sin(theta)
    hw,hh=w/2,h/2
    pts=[(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)]
    return [(cx+c*px-s*py, cy+s*px+c*py) for px,py in pts]

def _add_rect(ax, cx,cy,theta,w,h, fc,ec="white",lw=1.2,zorder=7,alpha=1.0):
    p=patches.Polygon(_rect_poly(cx,cy,theta,w,h),closed=True,
                      fc=fc,ec=ec,lw=lw,zorder=zorder,alpha=alpha)
    ax.add_patch(p); return p


# ─── ROBOT MODELLERİ ────────────────────────────────────────────────────────
_WR = 0.038   # animasyon için teker yarıçapı (m)

class DiffDriveRobot:
    """Diferansiyel sürüş — her teker bağımsız döner; dönüşlerde iç teker yavaşlar."""
    name="Differential"; L=0.5; holonomic=False

    def __init__(self,x,y,theta=0.0):
        self.x=x; self.y=y; self.theta=theta
        self.wl=0.0; self.wr=0.0   # teker dönme açıları (animasyon)

    def control(self,tx,ty):
        dx=tx-self.x; dy=ty-self.y; dist=math.hypot(dx,dy)
        des=math.atan2(dy,dx)
        aerr=math.atan2(math.sin(des-self.theta),math.cos(des-self.theta))
        v=min(0.7,0.5*dist)
        omega=float(np.clip(2.5*aerr,-2.5,2.5))
        return v,omega

    def step(self,v,omega,dt=0.1):
        dl=(v-omega*self.L/2)*dt; dr=(v+omega*self.L/2)*dt
        dS=(dl+dr)/2; dth=(dr-dl)/self.L
        mid=self.theta+dth/2
        self.x+=dS*math.cos(mid); self.y+=dS*math.sin(mid); self.theta+=dth
        self.wl+=dl/_WR; self.wr+=dr/_WR   # farklı hızda dönerler
        return dl,dr

    def draw(self,ax):
        arts=[]
        c,s=math.cos(self.theta),math.sin(self.theta)
        fw=np.array([c,s]); lf=np.array([-s,c])
        pos=np.array([self.x,self.y])
        # Gövde
        arts.append(_add_rect(ax,self.x,self.y,self.theta,0.55,0.35,
                               fc="#2471a3",ec="#85c1e9",lw=1.5,zorder=7))
        # Tekerlekler + dönen spoke
        for side,wa in [(-1,self.wl),(1,self.wr)]:
            wc=pos+lf*side*0.22
            arts.append(_add_rect(ax,wc[0],wc[1],self.theta,0.20,0.08,
                                   fc="#1c1c1c",ec="#7f8c8d",lw=1,zorder=8))
            # Spoke — teker dönüşünü 2D'de göster (fw eksenine projeksiyon)
            sp_c=math.cos(wa); sp_s=math.sin(wa)
            sp_fw=fw*0.09*sp_c; sp_lf=lf*side*0.04*sp_s
            sx,sy=wc[0]+sp_fw[0]+sp_lf[0], wc[1]+sp_fw[1]+sp_lf[1]
            ln,=ax.plot([wc[0],sx],[wc[1],sy],"-",color="#bdc3c7",lw=1.5,zorder=9)
            arts.append(ln)
        # Yön oku
        arts.append(ax.annotate("",xy=(self.x+0.38*c,self.y+0.38*s),
            xytext=(self.x,self.y),
            arrowprops=dict(arrowstyle="->",color="#f1c40f",lw=2.0),zorder=10))
        return arts


class AckermannRobot:
    """Ackermann (araba) sürüşü — bisiklet modeli kinematiği, min dönüş yarıçapı kısıtı."""
    name="Ackermann"; L=0.70; max_steer=math.radians(35); holonomic=False

    def __init__(self,x,y,theta=0.0):
        self.x=x; self.y=y; self.theta=theta
        self.steer=0.0; self.wangle=0.0   # tüm tekerlekler aynı hızda döner

    def control(self,tx,ty):
        """Pure Pursuit — daire çizmeden yolu izler, min dönüş yarıçapına uyar."""
        dx=tx-self.x; dy=ty-self.y; dist=math.hypot(dx,dy)
        alpha=math.atan2(dy,dx)-self.theta
        alpha=math.atan2(math.sin(alpha),math.cos(alpha))   # [-π,π]
        # Pure Pursuit: steer = atan(2L·sin(α) / L_d)
        ld=max(dist,1.2)
        steer_des=math.atan2(2*self.L*math.sin(alpha),ld)
        self.steer=float(np.clip(steer_des,-self.max_steer,self.max_steer))
        cos_a=math.cos(alpha)
        v=min(0.65,0.45*dist)*max(0.15, cos_a if cos_a>0 else 0.15)
        omega=v*math.tan(self.steer)/self.L
        return v,omega

    def step(self,v,omega,dt=0.1):
        # Gerçek bisiklet modeli kinematiği
        dth=v*math.tan(self.steer)/self.L*dt
        dS=v*dt
        mid=self.theta+dth/2
        self.x+=dS*math.cos(mid); self.y+=dS*math.sin(mid); self.theta+=dth
        self.wangle+=dS/_WR   # animasyon
        # Gerçek bisiklet hareketine eşdeğer dl/dr (EKF için doğru odometri)
        dl=dS-dth*self.L/2
        dr=dS+dth*self.L/2
        return dl,dr

    def draw(self,ax):
        arts=[]
        c,s=math.cos(self.theta),math.sin(self.theta)
        fw=np.array([c,s]); lf=np.array([-s,c])
        pos=np.array([self.x,self.y])
        # Gövde
        arts.append(_add_rect(ax,self.x,self.y,self.theta,0.78,0.34,
                               fc="#922b21",ec="#f1948a",lw=1.5,zorder=7))
        rpos=pos+fw*0.05
        arts.append(_add_rect(ax,rpos[0],rpos[1],self.theta,0.32,0.22,
                               fc="#7b241c",ec="#f1948a",lw=1,zorder=8,alpha=0.9))
        # Arka tekerlekler (sabit) + spoke
        rax=pos-fw*0.26
        sp_c=math.cos(self.wangle); sp_s=math.sin(self.wangle)
        for side in [-1,1]:
            wc=rax+lf*side*0.22
            arts.append(_add_rect(ax,wc[0],wc[1],self.theta,0.16,0.08,
                                   fc="#1c1c1c",ec="#7f8c8d",lw=1,zorder=9))
            sx=wc[0]+fw[0]*0.07*sp_c+lf[0]*side*0.035*sp_s
            sy=wc[1]+fw[1]*0.07*sp_c+lf[1]*side*0.035*sp_s
            ln,=ax.plot([wc[0],sx],[wc[1],sy],"-",color="#aaa",lw=1.5,zorder=10)
            arts.append(ln)
        # Ön tekerlekler (dönen) + spoke
        fax=pos+fw*0.26; fth=self.theta+self.steer
        fc2=math.cos(fth); fs2=math.sin(fth)
        fw2=np.array([fc2,fs2]); lf2=np.array([-fs2,fc2])
        for side in [-1,1]:
            wc=fax+lf*side*0.22
            arts.append(_add_rect(ax,wc[0],wc[1],fth,0.16,0.08,
                                   fc="#1c1c1c",ec="#bdc3c7",lw=1,zorder=9))
            sx=wc[0]+fw2[0]*0.07*sp_c+lf2[0]*side*0.035*sp_s
            sy=wc[1]+fw2[1]*0.07*sp_c+lf2[1]*side*0.035*sp_s
            ln,=ax.plot([wc[0],sx],[wc[1],sy],"-",color="#eee",lw=1.5,zorder=10)
            arts.append(ln)
        # Dönüş yarıçapı yayı — Ackermann kısıtını görsel göster
        if abs(self.steer)>0.06:
            R=self.L/math.tan(self.steer)
            abs_R=abs(R)
            tc_x=self.x-R*math.sin(self.theta)
            tc_y=self.y+R*math.cos(self.theta)
            ang_rob=math.degrees(math.atan2(self.y-tc_y,self.x-tc_x))
            span=75
            t1,t2=(ang_rob,ang_rob+span) if R>0 else (ang_rob-span,ang_rob)
            arc=patches.Arc((tc_x,tc_y),2*abs_R,2*abs_R,angle=0,
                             theta1=t1,theta2=t2,
                             color="#00e5ff",lw=1.2,ls="--",zorder=5,alpha=0.75)
            ax.add_patch(arc); arts.append(arc)
            dot,=ax.plot([tc_x],[tc_y],"+",color="#00e5ff",ms=7,zorder=5,alpha=0.6)
            arts.append(dot)
        # Hız oku
        arts.append(ax.annotate("",xy=(self.x+0.5*c,self.y+0.5*s),
            xytext=(self.x,self.y),
            arrowprops=dict(arrowstyle="->",color="#f39c12",lw=2.0),zorder=11))
        return arts


class MecanumRobot:
    """Mecanum / Omni sürüş — holonomik; herhangi bir yönde hareket edebilir."""
    name="Mecanum"; L=0.5; holonomic=True

    def __init__(self,x,y,theta=0.0):
        self.x=x; self.y=y; self.theta=theta
        self.vx_w=0.0; self.vy_w=0.0
        self.roller_spin=0.0   # roller animasyon açısı

    def control(self,tx,ty):
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

    def step(self,v,omega,dt=0.1):
        nx=self.x+self.vx_w*dt; ny=self.y+self.vy_w*dt
        self.x=float(np.clip(nx,0.35,19.65)); self.y=float(np.clip(ny,0.35,14.65))
        self.theta+=omega*dt
        vm=math.hypot(self.vx_w,self.vy_w)
        self.roller_spin+=vm*dt/_WR   # roller dönüşü hız büyüklüğüne göre
        dl=(v-omega*self.L/2)*dt; dr=(v+omega*self.L/2)*dt
        return dl,dr

    def draw(self,ax):
        arts=[]
        c,s=math.cos(self.theta),math.sin(self.theta)
        fw=np.array([c,s]); lf=np.array([-s,c])
        pos=np.array([self.x,self.y])
        # Gövde
        arts.append(_add_rect(ax,self.x,self.y,self.theta,0.46,0.46,
                               fc="#7d3c98",ec="#d2b4de",lw=1.5,zorder=7))
        # 4 mecanum tekerleği + dönen roller çizgileri
        wheel_offsets=[ fw*0.17+lf*0.27,  fw*0.17-lf*0.27,
                       -fw*0.17+lf*0.27, -fw*0.17-lf*0.27]
        roller_signs=[1,-1,-1,1]   # roller 45° yönü (FL,FR,RL,RR)
        for wp,rsign in zip(wheel_offsets,roller_signs):
            wc=pos+wp
            wth=self.theta+rsign*math.pi/4
            arts.append(_add_rect(ax,wc[0],wc[1],wth,0.17,0.07,
                                   fc="#1c1c1c",ec="#d2b4de",lw=1,zorder=8))
            # Dönen roller işareti: teker üzerinde kayan nokta
            r_ang=self.roller_spin*rsign
            rc=math.cos(wth); rs=math.sin(wth)
            rfx=np.array([rc,rs]); rlf=np.array([-rs,rc])
            dot_pos=np.array([wc[0],wc[1]])+rfx*0.06*math.cos(r_ang)+rlf*0.025*math.sin(r_ang)
            d,=ax.plot([dot_pos[0]],[dot_pos[1]],"o",
                       color="#a9cce3",ms=3,zorder=10,alpha=0.9)
            arts.append(d)
        # Dünya hız vektörü (belirgin ok)
        vm=math.hypot(self.vx_w,self.vy_w)
        if vm>0.05:
            ex=self.x+0.55*self.vx_w/vm; ey=self.y+0.55*self.vy_w/vm
            arts.append(ax.annotate("",xy=(ex,ey),xytext=(self.x,self.y),
                arrowprops=dict(arrowstyle="-|>",color="#00e5ff",lw=2.5,
                                mutation_scale=12),zorder=11))
        # Heading oku (ince)
        arts.append(ax.annotate("",xy=(self.x+0.28*c,self.y+0.28*s),
            xytext=(self.x,self.y),
            arrowprops=dict(arrowstyle="->",color="#d7bde2",lw=1.0),zorder=9))
        return arts


class UnicycleRobot:
    """Tekerlekli denge robotu — tek eksen sürüşü, spoke dönüşüyle görselleştirilir."""
    name="Unicycle"; L=0.5; holonomic=False

    def __init__(self,x,y,theta=0.0):
        self.x=x; self.y=y; self.theta=theta
        self.wangle=0.0   # teker dönme açısı

    def control(self,tx,ty):
        dx=tx-self.x; dy=ty-self.y; dist=math.hypot(dx,dy)
        des=math.atan2(dy,dx)
        aerr=math.atan2(math.sin(des-self.theta),math.cos(des-self.theta))
        v=min(0.7,0.5*dist)
        omega=float(np.clip(3.0*aerr,-3.0,3.0))
        return v,omega

    def step(self,v,omega,dt=0.1):
        self.x+=v*math.cos(self.theta)*dt; self.y+=v*math.sin(self.theta)*dt
        self.theta+=omega*dt
        dl=(v-omega*self.L/2)*dt; dr=(v+omega*self.L/2)*dt
        self.wangle+=v*dt/_WR   # teker ileri dönüşü
        return dl,dr

    def draw(self,ax):
        arts=[]
        c,s=math.cos(self.theta),math.sin(self.theta)
        fw=np.array([c,s]); lf=np.array([-s,c])
        # Dış çember
        circ=patches.Circle((self.x,self.y),0.27,fc="#16a085",ec="#a2d9ce",lw=2,zorder=7)
        ax.add_patch(circ); arts.append(circ)
        # Merkez göbek
        hub=patches.Circle((self.x,self.y),0.07,fc="#0e6655",ec="#a2d9ce",lw=1,zorder=8)
        ax.add_patch(hub); arts.append(hub)
        # Teker düzlemi (siyah çizgi — axle yönü)
        w1=np.array([self.x,self.y])+lf*0.27
        w2=np.array([self.x,self.y])-lf*0.27
        ln,=ax.plot([w1[0],w2[0]],[w1[1],w2[1]],"-",color="#1c1c1c",lw=4,zorder=9)
        arts.append(ln)
        # Dönen spoke — ileri hıza göre döner, dönüş yönünü gösterir
        sp_c=math.cos(self.wangle); sp_s=math.sin(self.wangle)
        sx=self.x+fw[0]*0.25*sp_c+lf[0]*0.25*sp_s
        sy=self.y+fw[1]*0.25*sp_c+lf[1]*0.25*sp_s
        sp,=ax.plot([self.x,sx],[self.y,sy],"-",color="#f8c471",lw=2.2,zorder=10)
        arts.append(sp)
        # Dönen spoke (zıt uç)
        sx2=self.x-fw[0]*0.25*sp_c-lf[0]*0.25*sp_s
        sy2=self.y-fw[1]*0.25*sp_c-lf[1]*0.25*sp_s
        sp2,=ax.plot([self.x,sx2],[self.y,sy2],"-",color="#f8c471",lw=2.2,zorder=10)
        arts.append(sp2)
        # Hareket yönü oku
        arts.append(ax.annotate("",xy=(self.x+0.38*c,self.y+0.38*s),
            xytext=(self.x,self.y),
            arrowprops=dict(arrowstyle="->",color="#f8c471",lw=2.5),zorder=11))
        return arts


ROBOT_TYPES={"Differential":DiffDriveRobot,"Ackermann":AckermannRobot,
             "Mecanum":MecanumRobot,"Unicycle":UnicycleRobot}


# ─── LOKALİZASYON ───────────────────────────────────────────────────────────
class DeadReckoning:
    def __init__(self,x,y,theta,L=0.5):
        self.x=x; self.y=y; self.theta=theta; self.L=L
        self.history=[(x,y)]

    def update(self,dl,dr):
        dS=(dl+dr)/2; dth=(dr-dl)/self.L
        self.x+=dS*math.cos(self.theta+dth/2)
        self.y+=dS*math.sin(self.theta+dth/2)
        self.theta+=dth
        self.history.append((self.x,self.y))


class EKF:
    """EKF — durum: [x, y, θ]. IMU + LiDAR tarama eşleşmesi ile güncelleme."""
    def __init__(self,x,y,theta,env,L=0.5):
        self.mu=np.array([x,y,theta],dtype=float)
        self.P=np.eye(3)*0.05
        self.L=L; self.env=env
        self.Q=np.diag([0.002,0.002,0.001])   # küçük süreç gürültüsü
        self.R_imu=np.array([[8e-4]])           # IMU açısal ölçüm gürültüsü
        self.R_lid=0.03                         # LiDAR menzil gürültüsü varyansı
        self._theta_prev=theta
        self._lid_ctr=0
        self.history=[(x,y)]

    # ── Tahmin adımı (enkoder) ──
    def predict(self,dl,dr):
        self._theta_prev=self.mu[2]
        dS=(dl+dr)/2; dth=(dr-dl)/self.L
        mid=self.mu[2]+dth/2
        F=np.array([[1,0,-dS*math.sin(mid)],
                    [0,1, dS*math.cos(mid)],
                    [0,0,1]])
        self.mu[0]+=dS*math.cos(mid)
        self.mu[1]+=dS*math.sin(mid)
        self.mu[2]+=dth
        self.P=F@self.P@F.T+self.Q

    # ── IMU ile açı düzeltmesi ──
    def update_imu(self,omega_meas,dt):
        z_th=self._theta_prev+omega_meas*dt   # IMU'nun tahmin ettiği yeni açı
        H=np.array([[0,0,1]])
        S=H@self.P@H.T+self.R_imu
        K=(self.P@H.T)/S[0,0]
        innov=math.atan2(math.sin(z_th-self.mu[2]),math.cos(z_th-self.mu[2]))
        self.mu+=K.flatten()*innov
        self.P=(np.eye(3)-np.outer(K,H))@self.P

    # ── LiDAR tarama eşleşmesi ile konum düzeltmesi ──
    def update_lidar(self,lidar_raw,lidar_angles):
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

    def _cast(self,x,y,angle):
        ca=math.cos(angle); sa=math.sin(angle)
        for d in np.arange(0.1,8.0,0.12):
            if not self.env.is_free(x+d*ca,y+d*sa,margin=0.0):
                return d
        return 8.0


# ─── NAVİGASYON ALGORİTMALARI ───────────────────────────────────────────────
class AStarPlanner:
    name="A*"
    def __init__(self,env,res=0.4,margin=0.32): self.env=env; self.res=res; self.mg=margin

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
                if not self.env.is_free(wx2,wy2,self.mg): continue
                ng=gs[cur]+math.hypot(dx,dy)*self.res
                if ng<gs.get(nb,1e18):
                    came[nb]=cur; gs[nb]=ng
                    heapq.heappush(heap,(ng+math.hypot(wx2-goal[0],wy2-goal[1]),nb))
        return [start,goal]


class DijkstraPlanner:
    name="Dijkstra"
    def __init__(self,env,res=0.4,margin=0.32): self.env=env; self.res=res; self.mg=margin

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
                if not self.env.is_free(wx2,wy2,self.mg): continue
                nd=d+math.hypot(dx,dy)*self.res
                if nd<dist.get(nb,1e18):
                    came[nb]=cur; dist[nb]=nd
                    heapq.heappush(heap,(nd,nb))
        return [start,goal]


class APFPlanner:
    name="APF"
    def __init__(self,env,res=None,step=0.22,k_att=1.0,k_rep=4.0,rep_r=2.0,margin=0.32):
        self.env=env; self.step=step; self.k_att=k_att
        self.k_rep=k_rep; self.rep_r=rep_r; self.mg=margin

    def plan(self,start,goal):
        bd=self.mg+0.05   # kenar tamponu
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
            x=float(np.clip(x+self.step*fx/mag,bd,self.env.width-bd))
            y=float(np.clip(y+self.step*fy/mag,bd,self.env.height-bd))
            stuck.append((x,y))
            if len(stuck)>40:
                stuck.pop(0)
                # Hem sabit takılı hem de salınım (oscillation) tespiti
                progress=math.hypot(stuck[0][0]-goal[0],stuck[0][1]-goal[1])-math.hypot(x-goal[0],y-goal[1])
                truly_stuck=max(math.hypot(p[0]-x,p[1]-y) for p in stuck)<0.3
                no_progress=progress<0.25
                if truly_stuck or no_progress:
                    x+=random.uniform(-3,3); y+=random.uniform(-3,3)
                    x=float(np.clip(x,bd,self.env.width-bd))
                    y=float(np.clip(y,bd,self.env.height-bd))
                    stuck.clear()
            path.append((x,y))
        return path


class RRTPlanner:
    name="RRT"
    def __init__(self,env,res=None,step=0.6,max_iter=6000,goal_bias=0.12,margin=0.32):
        self.env=env; self.step=step
        self.max_iter=max_iter; self.goal_bias=goal_bias; self.mg=margin

    def plan(self,start,goal):
        bd=self.mg+0.05
        nodes=[start]; parent={0:-1}
        for _ in range(self.max_iter):
            samp=(goal if random.random()<self.goal_bias else
                  (random.uniform(bd,self.env.width-bd),
                   random.uniform(bd,self.env.height-bd)))
            dists=[math.hypot(n[0]-samp[0],n[1]-samp[1]) for n in nodes]
            ni=int(np.argmin(dists)); nn=nodes[ni]; d=dists[ni]
            t=min(self.step/d,1.0) if d>0 else 1.0
            nw=(nn[0]+t*(samp[0]-nn[0]),nn[1]+t*(samp[1]-nn[1]))
            if not self.env.is_free(*nw,self.mg): continue
            if not self.env.segment_free(nn[0],nn[1],nw[0],nw[1],margin=self.mg): continue
            idx=len(nodes); nodes.append(nw); parent[idx]=ni
            if math.hypot(nw[0]-goal[0],nw[1]-goal[1])<self.step:
                gi=len(nodes); nodes.append(goal); parent[gi]=idx
                path=[]; c=gi
                while c!=-1: path.append(nodes[c]); c=parent[c]
                return list(reversed(path))
        return [start,goal]


PLANNERS={"A*":AStarPlanner,"Dijkstra":DijkstraPlanner,
          "APF":APFPlanner,"RRT":RRTPlanner}

# Robota özgü planlama parametreleri
ROBOT_PLAN_PARAMS={
    "Differential": {"res":0.40,"margin":0.51},  # margin>0.5 → x=0.5 sol sinir disi, gecitlerden gec
    "Ackermann":    {"res":0.50,"margin":0.51},  # x=0.5 sol sinir disi, gecitler acik
    "Mecanum":      {"res":0.38,"margin":0.51},  # margin>0.5 → zigzag gecit rotasi
    "Unicycle":     {"res":0.40,"margin":0.51},  # margin>0.5 → zigzag gecit rotasi
}

def _smooth_path(path, env, margin, steps=30):
    """Açgözlü kısayol düzleştirme — Ackermann için keskin dönüşleri azaltır."""
    if len(path)<3: return path
    result=[path[0]]; i=0
    while i<len(path)-1:
        best=i+1
        for j in range(len(path)-1,i+1,-1):
            if env.segment_free(result[-1][0],result[-1][1],
                                path[j][0],path[j][1],steps=steps,margin=margin):
                best=j; break
        result.append(path[best]); i=best
    return result


# ─── SİMÜLASYON ─────────────────────────────────────────────────────────────
class Simulation:
    def __init__(self):
        self.env=Environment()
        self.lidar=LiDAR(self.env)
        self.imu=IMU(); self.encoder=Encoder()
        self.nav_algo="A*"; self.loc_algo="EKF"
        self.robot_type="Differential"
        self.reset()

    def reset(self):
        sx,sy=self.env.start
        self.robot=ROBOT_TYPES[self.robot_type](sx,sy,0.0)
        self.dr=DeadReckoning(sx,sy,0.0,L=self.robot.L)
        self.ekf=EKF(sx,sy,0.0,self.env,L=self.robot.L)
        self.true_path=[(sx,sy)]
        self.lidar_raw=[]; self.lidar_filt=[]
        self.errors_ekf=[]; self.errors_dr=[]
        self.times=[]; self.t=0.0
        self.path_idx=0; self.done=False
        pp=ROBOT_PLAN_PARAMS[self.robot_type]
        raw=PLANNERS[self.nav_algo](self.env,**pp).plan(self.env.start,self.env.goal)
        # Tüm robotlar için yol düzleştirme — waypoint sayısını azaltır
        self.plan_path=_smooth_path(raw,self.env,pp["margin"],steps=50)

    def step(self,dt=0.1):
        if self.done: return
        path=self.plan_path; n=len(path); r=self.robot
        if self.path_idx>=n:
            self.done=True; return

        if self.robot_type=="Ackermann":
            # Arkada kalan veya çok yakın waypoint'leri atla
            while self.path_idx<n-1:
                wx,wy=path[self.path_idx]
                dx,dy=wx-r.x,wy-r.y
                fwd=dx*math.cos(r.theta)+dy*math.sin(r.theta)
                if math.hypot(dx,dy)<0.5 or fwd<-0.15:
                    self.path_idx+=1
                else: break
            if self.path_idx>=n: self.done=True; return
            target=self._lookahead_target()
        else:
            target=path[self.path_idx]
            if math.hypot(target[0]-r.x,target[1]-r.y)<0.35:
                self.path_idx+=1; return

        v,omega=self.robot.control(target[0],target[1])
        dl_t,dr_t=self.robot.step(v,omega,dt)
        r.x=float(np.clip(r.x,0.1,self.env.width-0.1))
        r.y=float(np.clip(r.y,0.1,self.env.height-0.1))
        dl_e,dr_e=self.encoder.measure(dl_t,dr_t)
        om_i=self.imu.measure(omega)
        raw,filt=self.lidar.scan(self.robot.x,self.robot.y,self.robot.theta)
        self.lidar_raw=self.lidar.to_points(self.robot.x,self.robot.y,self.robot.theta,raw)
        self.lidar_filt=self.lidar.to_points(self.robot.x,self.robot.y,self.robot.theta,filt)

        self.dr.update(dl_e,dr_e)
        self.ekf.predict(dl_e,dr_e)
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

    def _lookahead_target(self,L_d=0.9):
        """Ackermann Pure Pursuit: yol üzerinde L_d mesafedeki noktayı döndürür."""
        path=self.plan_path; r=self.robot; n=len(path)
        if self.path_idx>=n-1: return path[-1]
        px,py=path[self.path_idx]
        cum=math.hypot(px-r.x,py-r.y)
        if cum>=L_d: return (px,py)
        for i in range(self.path_idx,n-1):
            seg=math.hypot(path[i+1][0]-path[i][0],path[i+1][1]-path[i][1])
            if cum+seg>=L_d:
                t=(L_d-cum)/seg if seg>0 else 0.0
                return (path[i][0]+t*(path[i+1][0]-path[i][0]),
                        path[i][1]+t*(path[i+1][1]-path[i][1]))
            cum+=seg
        return path[-1]


# ─── GRAFİK ARAYÜZÜ ─────────────────────────────────────────────────────────
class GUI:
    def __init__(self):
        self.sim=Simulation()
        self.timer=None
        self._dyn_arts=[]   # çerçeveden çerçeveye silinen dinamik sanatçılar
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
        ax_s=self.fig.add_axes((0.02,0.36,0.11,0.055))
        self.btn_start=Button(ax_s,"BASLA ▶",color=ACC,hovercolor="#c0392b")
        self.btn_start.label.set_color("white"); self.btn_start.label.set_fontweight("bold")

        ax_r=self.fig.add_axes((0.02,0.29,0.11,0.055))
        self.btn_reset=Button(ax_r,"SIFIRLA ↺",color="#533483",hovercolor="#6a44a0")
        self.btn_reset.label.set_color("white")

        ax_rnd=self.fig.add_axes((0.02,0.22,0.11,0.055))
        self.btn_rand=Button(ax_rnd,"RASTGELE ⟳",color="#117a65",hovercolor="#1abc9c")
        self.btn_rand.label.set_color("white"); self.btn_rand.label.set_fontsize(8)

        # Metrik paneli
        self.ax_met=self.fig.add_axes((0.01,0.05,0.14,0.15),facecolor=PANEL)
        self.ax_met.axis("off")
        self.txt_met=self.ax_met.text(0.05,0.95,self._metric_str(),
            transform=self.ax_met.transAxes,color="white",
            fontsize=7.5,va="top",fontfamily="monospace")

        self.radio_nav.on_clicked(self._on_nav)
        self.radio_robot.on_clicked(self._on_robot)
        self.radio_loc.on_clicked(self._on_loc)
        self.btn_start.on_clicked(self._on_start)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_rand.on_clicked(self._on_rand)

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
                lw=1,edgecolor="#FFD700",
                facecolor=SHELF_COLORS[i%len(SHELF_COLORS)],alpha=0.90,zorder=2))
            ax.text((x1+x2)/2,(y1+y2)/2,env.labels[i],
                ha="center",va="center",color="white",fontsize=4.5,fontweight="bold",zorder=3)

        # Zigzag geçit oklarını sadece varsayılan haritada göster
        if not env.is_random:
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
        self._dyn_arts=[]

    # ── Hızlı güncelleme (her tick) ─────────────────────────────────────────
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

        # LiDAR ışınları (her 6. ışın)
        r=self.sim.robot
        for i,(px,py) in enumerate(self.sim.lidar_raw):
            if i%6==0:
                ln,=self.ax_map.plot([r.x,px],[r.y,py],"-",
                                     color="#e74c3c",alpha=0.10,lw=0.5,zorder=2)
                self._dyn_arts.append(ln)

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
                    f"Lok: {sim.loc_algo}\n──────────\nHazir")
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
                f"Lok: {sim.loc_algo}\n──────────\n{st}")

    # ── Tick ────────────────────────────────────────────────────────────────
    def _tick(self):
        try:
            for _ in range(4):
                if not self.sim.done:
                    self.sim.step(dt=0.1)
            self._update_map()
            self._update_lidar()
            self._update_loc()
            self._update_err()
            self._update_metrics()
            self.fig.canvas.draw_idle()
            if self.sim.done and self.timer is not None:
                self.timer.stop()
        except Exception as e:
            import traceback; traceback.print_exc()
            if self.timer is not None:
                self.timer.stop()

    # ── Callback'ler ────────────────────────────────────────────────────────
    def _on_start(self,_):
        self._stop_timer()
        self.timer=self.fig.canvas.new_timer(interval=50)
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

    def _on_rand(self,_):
        """Engelleri rastgele yenile ve yeniden planla."""
        self._stop_timer()
        self.sim.env.randomize()
        self.sim.reset(); self._draw_map()
        for ax in (self.ax_lidar,self.ax_loc,self.ax_err):
            ax.clear(); self._style(ax)
        self._update_metrics(); self.fig.canvas.draw_idle()

    def _stop_timer(self):
        if self.timer is not None:
            self.timer.stop(); self.timer=None

    def run(self):
        plt.show()


# ─── GİRİŞ ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gui=GUI()
    gui.run()
