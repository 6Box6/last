# -*- coding: utf-8 -*-
"""
Monkey Ball風 3Dゲーム (カメラ操作実装版)
Controls:
 - Arrow Keys: Tilt Stage
 - WASD: Move Camera (X/Z)
 - QE: Move Camera (Y)
 - IJKL: Rotate Camera
"""
import sys
import pygame
import numpy as np

# 画面サイズ
WIDTH, HEIGHT = 1024, 768
scrcentr = np.array([WIDTH/2, HEIGHT/2])

f = 750

# --- 行列演算関数 ---
def RotateMat(ori):
    c0, s0 = np.cos(ori[0]), np.sin(ori[0])
    c1, s1 = np.cos(ori[1]), np.sin(ori[1])
    c2, s2 = np.cos(ori[2]), np.sin(ori[2])
    RX = np.array([[ 1,  0,   0,  0],
                   [ 0,  c0, -s0,  0],
                   [ 0,  s0,  c0,  0],
                   [ 0,  0,   0,  1]])
    RY = np.array([[ c1,  0,  s1,  0],
                   [ 0,  1,   0,  0],
                   [-s1,  0,  c1,  0],
                   [ 0,  0,   0,  1]])
    RZ = np.array([[ c2, -s2,  0,  0],
                   [ s2,  c2,  0,  0],
                   [ 0,   0,   1,  0],
                   [ 0,   0,   0,  1]])
    return RX @ RY @ RZ

def TransMat(pos):
    return np.c_[np.r_[np.eye(3),[[0.0, 0.0, 0.0]]], np.r_[pos, 1]]

# --- クラス定義 ---

class Object3D:
    def __init__(self):
        self.vertices = np.array([])
        self.edges = np.array([])
        self.pos = np.array([0.0, 0.0, 0.0])
        self.ori = np.array([0.0, 0.0, 0.0]) 
        self.color = (255, 255, 255)

    def draw(self, buf, ViewMat, ParentMat=None):
        # 1. ローカル -> ワールド
        ModelMat = TransMat(self.pos) @ RotateMat(self.ori)
        
        if ParentMat is not None:
            WorldMat = ParentMat @ ModelMat
        else:
            WorldMat = ModelMat

        # 2. ワールド -> カメラ座標 (View)
        MV = ViewMat @ WorldMat

        # 頂点変換
        verts_homo = np.c_[self.vertices, np.ones(len(self.vertices))]
        camera_coords = MV @ verts_homo.T
        
        # 3. 透視投影
        x = camera_coords[0, :]
        y = camera_coords[1, :]
        z = camera_coords[2, :]

        # クリッピング (Zがカメラより後ろなら描画しない)
        valid_mask = z > 1.0
        
        # 射影計算
        scale = f / z
        
        # スクリーン座標変換
        # (Y軸はflipで反転させるため、ここではそのまま加算)
        sx = x * scale + scrcentr[0]
        sy = y * scale + scrcentr[1] 

        screen_points = np.stack([sx, sy], axis=1)

        # エッジ描画
        for edge in self.edges:
            p1_idx, p2_idx = edge
            if valid_mask[p1_idx] and valid_mask[p2_idx]:
                p1 = screen_points[p1_idx]
                p2 = screen_points[p2_idx]
                pygame.draw.line(buf, self.color, p1, p2, 2)
        
        return WorldMat

class Stage(Object3D):
    def __init__(self, size=250.0, div=10):
        super().__init__()
        self.color = (0, 255, 191)
        self.size = size
        verts = []
        edges = []
        step = size * 2 / div
        
        idx = 0
        # 床
        for i in range(div + 1):
            z = -size + i * step
            verts.append([-size, 0, z]); verts.append([ size, 0, z])
            edges.append([idx, idx+1]); idx += 2
        for i in range(div + 1):
            x = -size + i * step
            verts.append([x, 0, -size]); verts.append([x, 0,  size])
            edges.append([idx, idx+1]); idx += 2
            
        # 壁
        wall_h = 30.0
        corners = [[-size, -size], [ size, -size], [ size,  size], [-size,  size]]
        start_idx = len(verts)
        for i in range(4):
            x, z = corners[i]
            verts.append([x, 0, z])      
            verts.append([x, wall_h, z]) 
            edges.append([start_idx + i*2, start_idx + i*2 + 1])
            curr_top = start_idx + i*2 + 1
            next_top = start_idx + ((i+1)%4)*2 + 1
            edges.append([curr_top, next_top])

        self.vertices = np.array(verts)
        self.edges = np.array(edges)
        self.MAX_TILT = 0.5 

    def update(self):
        keys = pygame.key.get_pressed()
        target_x, target_z = 0.0, 0.0
        
        # ステージ操作（矢印キー）
        if keys[pygame.K_RIGHT]:  target_z = -self.MAX_TILT
        if keys[pygame.K_LEFT]: target_z =  self.MAX_TILT
        if keys[pygame.K_UP]:    target_x =  self.MAX_TILT 
        if keys[pygame.K_DOWN]:  target_x = -self.MAX_TILT 
        
        self.ori[0] = self.ori[0] * 0.9 + target_x * 0.1
        self.ori[2] = self.ori[2] * 0.9 + target_z * 0.1

class Player(Object3D):
    def __init__(self, radius=30.0):
        super().__init__()
        self.color = (255, 100, 100)
        self.radius = radius
        self.create_sphere(radius, 8, 8)
        self.velocity = np.array([0.0, 0.0, 0.0])

    def create_sphere(self, r, stacks, slices):
        verts = []
        edges = []
        for i in range(stacks + 1):
            lat = np.pi * i / stacks
            y = np.cos(lat) * r
            rs = np.sin(lat) * r
            for j in range(slices):
                lon = 2 * np.pi * j / slices
                x = np.cos(lon) * rs
                z = np.sin(lon) * rs
                verts.append([x, y, z])
        self.vertices = np.array(verts)
        for i in range(stacks):
            for j in range(slices):
                curr = i * slices + j
                next_slice = i * slices + (j + 1) % slices
                next_stack = (i + 1) * slices + j
                edges.append([curr, next_slice])
                edges.append([curr, next_stack])
        self.edges = np.array(edges)

    def update(self, stage):
        GRAVITY = 0.5 
        FRICTION = 0.98
        
        acc_x = -GRAVITY * np.sin(stage.ori[2]) 
        acc_z =  GRAVITY * np.sin(stage.ori[0])
        
        self.velocity[0] += acc_x
        self.velocity[2] += acc_z
        self.velocity *= FRICTION
        self.pos += self.velocity
        
        limit = stage.size - self.radius
        if self.pos[0] > limit:  self.pos[0] = limit;  self.velocity[0] *= -0.6
        if self.pos[0] < -limit: self.pos[0] = -limit; self.velocity[0] *= -0.6
        if self.pos[2] > limit:  self.pos[2] = limit;  self.velocity[2] *= -0.6
        if self.pos[2] < -limit: self.pos[2] = -limit; self.velocity[2] *= -0.6

        self.ori[2] -= self.velocity[0] / self.radius
        self.ori[0] += self.velocity[2] / self.radius

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    buf = pygame.Surface((WIDTH, HEIGHT))
    pygame.display.set_caption("WASD:Move Camera | IJKL:Rotate Camera | Arrows:Tilt Stage") 

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    stage = Stage(size=250.0, div=10)
    player = Player(radius=25.0)
    player.pos[1] = player.radius 

    # --- カメラの変数を初期化 ---
    cam_pos = np.array([0.0, 200.0, -1000.0]) # 位置
    cam_ori = np.array([np.radians(-30), 0.0, 0.0]) # 回転 (Pitch, Yaw, Roll)
    
    # カメラ移動速度
    CAM_SPEED = 10.0
    CAM_ROT_SPEED = 0.02

    running = True
    while running:     
        buf.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:  
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_r: # リセット
                    player.pos = np.array([0.0, player.radius, 0.0])
                    player.velocity = np.array([0.0, 0.0, 0.0])
                    stage.ori = np.array([0.0, 0.0, 0.0])
                    # カメラもリセットしたければここで cam_pos 等を初期値に戻す

        # --- カメラ操作処理 (WASD / QE / IJKL) ---
        keys = pygame.key.get_pressed()
        
        # 移動 (World軸に対して移動)
        if keys[pygame.K_w]: cam_pos[2] += CAM_SPEED # 前 (Z+)
        if keys[pygame.K_s]: cam_pos[2] -= CAM_SPEED # 後 (Z-)
        if keys[pygame.K_a]: cam_pos[0] -= CAM_SPEED # 左 (X-)
        if keys[pygame.K_d]: cam_pos[0] += CAM_SPEED # 右 (X+)
        if keys[pygame.K_q]: cam_pos[1] -= CAM_SPEED # 上 (Y-) ※座標系注意
        if keys[pygame.K_e]: cam_pos[1] += CAM_SPEED # 下 (Y+)

        # 回転 (傾き)
        if keys[pygame.K_i]: cam_ori[0] += CAM_ROT_SPEED # 上を向く
        if keys[pygame.K_k]: cam_ori[0] -= CAM_ROT_SPEED # 下を向く
        if keys[pygame.K_j]: cam_ori[1] += CAM_ROT_SPEED # 左を向く
        if keys[pygame.K_l]: cam_ori[1] -= CAM_ROT_SPEED # 右を向く

        # 更新処理
        stage.update()
        player.update(stage)

        # ビュー行列の構築 (動的な cam_pos, cam_ori を使用)
        ViewT = TransMat(-cam_pos) 
        ViewR = RotateMat(cam_ori) 
        ViewMat = ViewT @ ViewR 

        # 描画
        stage_matrix = stage.draw(buf, ViewMat, ParentMat=None)
        player.draw(buf, ViewMat, ParentMat=stage_matrix)

        # 画像反転
        flippedbuf = pygame.transform.flip(buf, 0, 1) 
        screen.blit(flippedbuf, (0,0))
        
        # 操作説明テキスト
        info = font.render(f"Cam Pos: {cam_pos[0]:.0f},{cam_pos[1]:.0f},{cam_pos[2]:.0f}", True, (255, 255, 255))
        screen.blit(info, (10, 10))

        pygame.display.update()
        clock.tick(60)

if __name__ == "__main__":
    main()