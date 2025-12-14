# -*- coding: utf-8 -*-
"""
2025/11/15
最初のコード例
"""
import sys
import pygame
import numpy as np

# 定数の設定
WIDTH, HEIGHT = 1024,768                        # 画面の幅と高さ
scrcentr = np.array([WIDTH/2, HEIGHT/2]) # 画面の中心の座標
scale = (WIDTH-1)/2                            # ピクセル単位の座標に変換する際の拡大率
print(scrcentr)
print(scale)
fps = 600                                      # フレーム/秒
#dt = 1.0/fps                                   # 1フレームあたりの経過時間
dt = 1.0/600                                  # 1フレームあたりの経過時間
f = 1000    # 焦点距離
g = 9.8     # 重力

def RotateMat(ori):

    c0, s0 = np.cos(ori[0]), np.sin(ori[0])
    c1, s1 = np.cos(ori[1]), np.sin(ori[1])
    c2, s2 = np.cos(ori[2]), np.sin(ori[2])
    
    RX = np.array([[ 1,  0,  0,  0],
                   [ 0, c0, -s0,  0],
                   [ 0, s0, c0,  0],
                   [ 0,  0,  0,  1]])
    RY = np.array([[ c1,  0, s1,  0],
                   [ 0,  1,  0,  0],
                   [-s1,  0, c1,  0],
                   [ 0,  0,  0,  1]])
    RZ = np.array([[ c2, -s2,  0,  0],
                   [ s2, c2,  0,  0],
                   [ 0,  0,  1,  0],
                   [ 0,  0,  0,  1]])
    
    return RX @ RY @ RZ

def TransMat(pos):
    return np.c_[np.r_[np.eye(3),[[0.0, 0.0, 0.0]]], np.r_[pos, 1]]

def ProjMat(f):

    A = np.array([[ f, 0,  WIDTH/2, 0],
                  [ 0, f, HEIGHT/2, 0],
                  [ 0, 0,        1, 0]])
    
    return A


class Stage:
    def __init__(self, size, div, pos, color):
        self.size = size
        self.pos = np.array(pos, dtype=float)
        self.color = color
        
        # 傾きパラメータ
        self.tilt_x = 0.0
        self.tilt_z = 0.0
        self.MAX_TILT = 0.5 
        self.rot_matrix = np.eye(3)

        lin = np.linspace(-size, size, div + 1)
        # メッシュグリッド
        X, Z = np.meshgrid(lin, lin)
        Y = np.zeros_like(X)
        
        # 床の頂点
        floor_verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        n_floor_pts = len(floor_verts)

        wall_h = 30.0
        

        idxs_2d = np.arange(n_floor_pts).reshape(div+1, div+1)
        corner_idxs = np.array([
            idxs_2d[0, 0],   idxs_2d[0, -1], 
            idxs_2d[-1, -1], idxs_2d[-1, 0]
        ])

        # 壁の上の頂点
        wall_top_verts = floor_verts[corner_idxs].copy()
        wall_top_verts[:, 1] = wall_h
        
        # 全頂点を結合
        self.init_verts = np.vstack([floor_verts, wall_top_verts])
        
        # 壁の上の頂点が配列
        wall_top_idxs = np.arange(n_floor_pts, n_floor_pts + 4)

        # 床のエッジ
        # 横線: (i, j) と (i, j+1) を結ぶ
        h_edges = np.column_stack([idxs_2d[:, :-1].ravel(), idxs_2d[:, 1:].ravel()])
        # 縦線: (i, j) と (i+1, j) を結ぶ
        v_edges = np.column_stack([idxs_2d[:-1, :].ravel(), idxs_2d[1:, :].ravel()])
        
        # 柱
        pillars = np.column_stack([corner_idxs, wall_top_idxs])
        
        # 壁上の辺
        beams = np.column_stack([wall_top_idxs, np.roll(wall_top_idxs, -1)])
        
        # 全エッジを結合
        self.edges = np.vstack([h_edges, v_edges, pillars, beams])

        # 現在の頂点用バッファ初期化
        self.curr_verts = np.zeros_like(self.init_verts)
        self.transform_verts()

    def update(self, dt, mouse_mode):
        target_x, target_z = 0.0, 0.0

        if mouse_mode:
            # --- マウス操作モード ---
            mx, my = pygame.mouse.get_pos()
            
            # 画面中心基準、感度調整
            norm_x = (mx - WIDTH / 2) / (WIDTH / 4)
            norm_y = (my - HEIGHT / 2) / (HEIGHT / 4)
            
            norm_x = max(-1.0, min(1.0, norm_x))
            norm_y = max(-1.0, min(1.0, norm_y))

            target_z = -norm_x * self.MAX_TILT
            target_x =  norm_y * self.MAX_TILT
            
        else:
            # --- キーボード操作モード ---
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RIGHT]: target_z = -self.MAX_TILT
            if keys[pygame.K_LEFT]:  target_z =  self.MAX_TILT
            if keys[pygame.K_UP]:    target_x =  self.MAX_TILT 
            if keys[pygame.K_DOWN]:  target_x = -self.MAX_TILT 
        
        # 共通の補間処理 (Lerp)
        lerp_speed = 5.0 * dt 
        self.tilt_x = self.tilt_x * (1 - lerp_speed) + target_x * lerp_speed
        self.tilt_z = self.tilt_z * (1 - lerp_speed) + target_z * lerp_speed
        
        rx = RotateMat([self.tilt_x, 0, 0])[0:3, 0:3]
        rz = RotateMat([0, 0, self.tilt_z])[0:3, 0:3]
        self.rot_matrix = rx @ rz
        self.transform_verts()

    def transform_verts(self):
        rotated_verts = self.init_verts @ self.rot_matrix.T
        self.curr_verts = rotated_verts + self.pos

    def draw(self, buf, Rc, Tc):
        verts_homo = np.c_[self.curr_verts, np.ones(len(self.curr_verts))]
        P = ProjMat(f)
        VP = P @ Rc @ Tc
        ps_all = (verts_homo @ VP.T)
        
        valid_mask = ps_all[:, 2] > 1.0
        screen_points = np.zeros((len(self.curr_verts), 2))
        
        if np.any(valid_mask):
            z = ps_all[valid_mask, 2:3]
            screen_points[valid_mask] = ps_all[valid_mask, 0:2] / z
            
        for e in self.edges:
            p1_idx, p2_idx = e
            if valid_mask[p1_idx] and valid_mask[p2_idx]:
                p1 = screen_points[p1_idx]
                p2 = screen_points[p2_idx]
                pygame.draw.line(buf, self.color, p1, p2, 2)


class Player:
    def __init__(self, radius=25.0, pos=np.array([0.0, 0.0, 0.0])):
        
        self.radius = radius 
        self.pos = np.array(pos)
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.color = (255, 0, 0)
        self.rot_matrix = np.eye(3)

        # 正20面体
        phi = (1 + np.sqrt(5)) / 2
        points = [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
        ]
        faces = [
             [0, 1, 8], [0, 8, 4], [0, 4, 5], [0, 5, 9], [0, 9, 1],
             [1, 6, 8], [8, 6, 10], [8, 10, 4], [4, 10, 2], [4, 2, 5],
             [5, 2, 11], [5, 11, 9], [9, 11, 7], [9, 7, 1], [1, 7, 6],
             [3, 6, 7], [3, 7, 11], [3, 11, 2], [3, 2, 10], [3, 10, 6]
        ]
        self.vertices_local = np.array(points, dtype=float)
        norms = np.linalg.norm(self.vertices_local, axis=1)
        self.vertices_local = (self.vertices_local / norms[:, np.newaxis]) * radius

        edge_set = set()
        for face in faces:
            for i in range(len(face)):
                idx1, idx2 = face[i], face[(i + 1) % len(face)]
                if idx1 < idx2: edge_set.add((idx1, idx2))
                else:           edge_set.add((idx2, idx1))
        self.edges = np.array(list(edge_set))
        self.curr_verts = np.zeros_like(self.vertices_local)
        self.transform_verts()

    def update(self, stage, dt):
        g_pix = g * scale
        g_vec = np.array([0.0, -g_pix, 0.0]) 
        
        mu = 0.6
        
        n = np.array([0.0, 1.0, 0.0])
        normal_vec = stage.rot_matrix @ n
        
        projection_mag = np.dot(g_vec, normal_vec)
        force_perpendicular = projection_mag * normal_vec
        force_parallel = g_vec - force_perpendicular
        
        self.velocity[0] += force_parallel[0] * dt
        self.velocity[2] += force_parallel[2] * dt
        
        if self.velocity[0] > 0.0:
            friction_factor = mu * normal_vec
            self.velocity -= friction_factor
        
        self.pos += self.velocity * dt
        
        rel_pos = self.pos - stage.pos 
        limit = stage.size - self.radius
        bounce = -0.6
        
        if rel_pos[0] > limit:  self.pos[0] = stage.pos[0] + limit;  self.velocity[0] *= bounce
        if rel_pos[0] < -limit: self.pos[0] = stage.pos[0] - limit;  self.velocity[0] *= bounce
        if rel_pos[2] > limit:  self.pos[2] = stage.pos[2] + limit;  self.velocity[2] *= bounce
        if rel_pos[2] < -limit: self.pos[2] = stage.pos[2] - limit;  self.velocity[2] *= bounce

        nx, ny, nz = normal_vec
        if abs(ny) > 0.001:
            dx = self.pos[0] - stage.pos[0]
            dz = self.pos[2] - stage.pos[2]
            dy = -(nx * dx + nz * dz) / ny
            self.pos[1] = stage.pos[1] + dy + self.radius

        d_angle_z = -self.velocity[0] * dt / self.radius 
        d_angle_x =  self.velocity[2] * dt / self.radius
        
        rot_delta = RotateMat([d_angle_x, 0, d_angle_z])[0:3, 0:3]
        self.rot_matrix = self.rot_matrix @ rot_delta
        
        self.transform_verts()

    def transform_verts(self):
        rotated_verts = self.vertices_local @ self.rot_matrix.T
        self.curr_verts = rotated_verts + self.pos

    def draw(self, buf, Rc, Tc):
        verts_homo = np.c_[self.curr_verts, np.ones(len(self.curr_verts))]
        P = ProjMat(f)
        VP = P @ Rc @ Tc
        ps_all = (verts_homo @ VP.T)
        
        valid_mask = ps_all[:, 2] > 1.0
        screen_points = np.zeros((len(self.curr_verts), 2))
        
        if np.any(valid_mask):
            z = ps_all[valid_mask, 2:3]
            screen_points[valid_mask] = ps_all[valid_mask, 0:2] / z
            
        for e in self.edges:
            p1_idx, p2_idx = e
            if valid_mask[p1_idx] and valid_mask[p2_idx]:
                p1 = screen_points[p1_idx]
                p2 = screen_points[p2_idx]
                pygame.draw.line(buf, self.color, p1, p2, 2)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    buf = pygame.Surface((WIDTH, HEIGHT))
    
    mouse_mode = False

    clock = pygame.time.Clock()

    stage_pos = np.array([0.0, -80.0, 600.0])
    stage = Stage(size=250.0, div=10, pos=stage_pos, color=(0, 255, 191))
    
    player_start_pos = stage_pos + np.array([0.0, 30.0, 0.0])
    player = Player(radius=30.0, pos=player_start_pos)

    pos_c = np.array([0.0, 10.0, 1]) 
    ori_c = np.array([0.0, 0.0, 0.0])
    
    TSTEP = 1.0 
    RSTEP = np.pi/96    

    running = True
    while running:     
        buf.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:  
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_r: 
                    player.pos = stage.pos + np.array([0.0, player.radius, 0.0])
                    player.velocity = np.array([0.0, 0.0, 0.0])
                    player.rot_matrix = np.eye(3)
                    stage.tilt_x = 0.0
                    stage.tilt_z = 0.0
                    pos_c = np.array([0.0, 10.0, 1])
                    ori_c = np.array([0.0, 0.0, 0.0])
                    mouse_mode = False
                    player.transform_verts()
                # Mキーでマウスモード切替
                if event.key == pygame.K_m:
                    mouse_mode = not mouse_mode

        keys = pygame.key.get_pressed()

        # カメラ移動(x軸)
        if keys[pygame.K_a]: pos_c[0] -= TSTEP # →
        if keys[pygame.K_d]: pos_c[0] += TSTEP # ←

        # カメラ移動(y軸)
        if keys[pygame.K_w]: pos_c[1] += TSTEP # ↑
        if keys[pygame.K_s]: pos_c[1] -= TSTEP # ↓


        
        # カメラ移動(z軸)
        if keys[pygame.K_z]: pos_c[2] += TSTEP # +z
        if keys[pygame.K_x]: pos_c[2] -= TSTEP # -z



        # カメラ回転
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            if keys[pygame.K_s]: ori_c[0] += RSTEP # ↑
            if keys[pygame.K_w]: ori_c[0] -= RSTEP # ↓
            if keys[pygame.K_d]: ori_c[1] += RSTEP # →
            if keys[pygame.K_a]: ori_c[1] -= RSTEP # ←

        # モードフラグを渡して更新
        stage.update(dt, mouse_mode)
        player.update(stage, dt)

        Rc = RotateMat(ori_c).T 
        Tc = TransMat(-pos_c)

        stage.draw(buf, Rc, Tc)
        player.draw(buf, Rc, Tc)

        flippedbuf = pygame.transform.flip(buf, 0, 1) 
        screen.blit(flippedbuf, (0,0))

        pygame.display.update()
        clock.tick(fps)

if __name__ == "__main__":
    main()