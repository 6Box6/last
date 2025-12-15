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
fps = 120                                      # フレーム/秒
#dt = 1.0/fps                                   # 1フレームあたりの経過時間
dt = 1.0/120                                  # 1フレームあたりの経過時間
f = 1000    # 焦点距離


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


class stage:
    def __init__(self, size, div, pos, color):
        self.size = size
        self.div = div
        self.pos = np.array(pos, dtype=float)
        self.color = color

        # 傾き
        self.tilt_x = 0.0
        self.tilt_z = 0.0
        self.max_tilt = 0.5
        self.rot_matrix = np.eye(3)

        # 床生成
        lin = np.linspace(-self.size, self.size, div + 1)

        self.vertex = np.zeros((div + 1, div + 1, 3), dtype=float)
        for i in range(div + 1):
            for j in range(div + 1):
                self.vertex[i, j] = [lin[j], 0.0, lin[i]]

        # 壁上の頂点
        wall_h = 30.0
        
        self.wall_vertex = np.array([
            self.vertex[0, 0],
            self.vertex[0, div],
            self.vertex[div, div],
            self.vertex[div, 0]
        ], dtype=float)

        self.wall_vertex[:, 1] = wall_h

        # エッジ
        self.edges = []

        # 横
        for i in range(div + 1):
            for j in range(div):
                self.edges.append(((i, j), (i, j + 1)))

        # 縦
        for i in range(div):
            for j in range(div + 1):
                self.edges.append(((i, j), (i + 1, j)))
        
        print(self.edges)

        # 壁上
        self.wall_edges = [(0,1),(1,2),(2,3),(3,0)]
        print(self.wall_edges)

        self.transform_verts()

    def update(self, dt, mouse_mode):
        target_x = 0.0
        target_z = 0.0

        if mouse_mode:
            
            mx, my = pygame.mouse.get_pos()
            nx = np.clip((mx - WIDTH/2) / (WIDTH/4), -1.0, 1.0)
            ny = np.clip((my - HEIGHT/2) / (HEIGHT/4), -1.0, 1.0)
            target_z =  -nx * self.max_tilt
            target_x =  -ny * self.max_tilt

        else:
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RIGHT]: target_z = -self.max_tilt
            if keys[pygame.K_LEFT]:  target_z =  self.max_tilt
            if keys[pygame.K_UP]:    target_x =  self.max_tilt
            if keys[pygame.K_DOWN]:  target_x = -self.max_tilt

        # 線形補完
        self.tilt_x = self.tilt_x * (1 - dt * 10.0) + target_x * dt * 10.0
        self.tilt_z = self.tilt_z * (1 - dt * 10.0) + target_z * dt * 10.0

        # 傾き
        rx = RotateMat([self.tilt_x, 0, 0])[0:3, 0:3]
        rz = RotateMat([0, 0, self.tilt_z])[0:3, 0:3]
        self.rot_matrix = rx @ rz

        self.transform_verts()

    def transform_verts(self):
        # 現在床
        self.cfloor_vertex = self.vertex @ self.rot_matrix.T + self.pos
        # 現在壁
        self.cwall_vertex = self.wall_vertex @ self.rot_matrix.T + self.pos

    def draw(self, buf, Rc, Tc):
        P = ProjMat(f)
        VP = P @ Rc @ Tc

        # 床
        h, w, _ = self.cfloor_vertex.shape
        flat = self.cfloor_vertex.reshape(-1, 3)
        homo = np.c_[flat, np.ones(len(flat))]
        ps = homo @ VP.T

        valid = ps[:, 2] > 1.0
        screen = np.zeros((len(ps), 2))
        screen[valid] = ps[valid, :2] / ps[valid, 2:3]

        screen = screen.reshape(h, w, 2)
        valid = valid.reshape(h, w)

        for (i1,j1),(i2,j2) in self.edges:
            if valid[i1,j1] and valid[i2,j2]:
                pygame.draw.line(buf, self.color, screen[i1,j1], screen[i2,j2], 2)

        # 壁
        w_homo = np.c_[self.cwall_vertex, np.ones(4)]
        psw = w_homo @ VP.T
        vw = psw[:,2] > 1.0
        sw = np.zeros((4,2))
        sw[vw] = psw[vw,:2] / psw[vw,2:3]

        # 柱
        corners = [(0,0),(0,-1),(-1,-1),(-1,0)]
        for k,(i,j) in enumerate(corners):
            if valid[i,j] and vw[k]:
                pygame.draw.line(buf, self.color, screen[i,j], sw[k], 2)

        # 壁上
        for i,j in self.wall_edges:
            if vw[i] and vw[j]:
                pygame.draw.line(buf, self.color, sw[i], sw[j], 2)



class player:
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

        self.icosahedron_vertex = np.array(points, dtype=float)
        norms = np.linalg.norm(self.icosahedron_vertex, axis=1)
        self.icosahedron_vertex = (self.icosahedron_vertex / norms[:, np.newaxis]) * radius

        edge_set = set()
        for face in faces:
            for i in range(len(face)):
                idx1, idx2 = face[i], face[(i + 1) % len(face)]
                if idx1 < idx2: edge_set.add((idx1, idx2))
                else:           edge_set.add((idx2, idx1))
        self.edges = np.array(list(edge_set))
        self.curr_verts = np.zeros_like(self.icosahedron_vertex)
        self.transform_verts()

    def update(self, stage, dt):
        g = 9.8 # 重力
        g_pix = g * scale
        g_vec = np.array([0.0, -g_pix, 0.0]) 
        
        mu = 0.2
        
        n = np.array([0.0, 1.0, 0.0])

        # 法線ベクトル
        normal_vec = stage.rot_matrix @ n
        
        # 垂直抗力
        normal_force = np.dot(g_vec, normal_vec)
        # print(normal_force)

        # 垂直抗力の反力
        normal_force_yvec = normal_force * normal_vec
        # print(normal_force_yvec)

        # 推進力　= 重力 - 垂直抗力
        force = g_vec - normal_force_yvec
        
        self.velocity[0] += force[0] * dt
        self.velocity[2] += force[2] * dt
        
        speed = np.linalg.norm(self.velocity[[0,2]])
        # print(speed)

        if speed > 0.000001:
            # 摩擦の向き
            friction_dir = -self.velocity[[0,2]] / speed
            normal_mag = abs(normal_force)

            # 動摩擦による速度変化
            friction = mu * normal_mag * dt
            self.velocity[0] += friction * friction_dir[0]
            self.velocity[2] += friction * friction_dir[1]
        
        self.pos += self.velocity * dt
        
        # 床からの距離
        rel_pos = self.pos - stage.pos 
        limit = stage.size - self.radius
        bounce = -0.6
        
        # 反射処理
        if rel_pos[0] > limit:  
            self.pos[0] = stage.pos[0] + limit
            self.velocity[0] *= bounce
        if rel_pos[0] < -limit: 
            self.pos[0] = stage.pos[0] - limit
            self.velocity[0] *= bounce
        if rel_pos[2] > limit:
            self.pos[2] = stage.pos[2] + limit
            self.velocity[2] *= bounce
        if rel_pos[2] < -limit:
            self.pos[2] = stage.pos[2] - limit
            self.velocity[2] *= bounce

        nx, ny, nz = normal_vec

        # 床を離さない
        if abs(ny) > 0.000001:
            dx = self.pos[0] - stage.pos[0]
            dz = self.pos[2] - stage.pos[2]
            dy = -(nx * dx + nz * dz) / ny
            self.pos[1] = stage.pos[1] + dy + self.radius

        d_angle_z = self.velocity[0] * dt / self.radius 
        d_angle_x = self.velocity[2] * dt / self.radius
        
        rot_delta = RotateMat([d_angle_x, 0, d_angle_z])[0:3, 0:3]
        self.rot_matrix = self.rot_matrix @ rot_delta
        
        self.transform_verts()

    def transform_verts(self):
        rotated_vertex = self.icosahedron_vertex @ self.rot_matrix.T
        self.cvertex = rotated_vertex + self.pos

    def draw(self, buf, Rc, Tc):
        verts_homo = np.c_[self.cvertex, np.ones(len(self.cvertex))]
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
    stage1 = stage(size=250.0, div=10, pos=stage_pos, color=(0, 255, 191))
    
    player_start_pos = stage_pos + np.array([0.0, 30.0, 0.0])
    player1 = player(radius=30.0, pos=player_start_pos)

    pos_c = np.array([0.0, 10.0, 1]) 
    ori_c = np.array([0.0, 0.0, 0.0])
    
    TSTEP = 1.0 
    RSTEP = np.pi / 240  

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
                    player1.pos = stage1.pos + np.array([0.0, player1.radius, 0.0])
                    player1.velocity[:] = 0.0
                    player1.rot_matrix = np.eye(3)
                    stage1.tilt_x = 0.0
                    stage1.tilt_z = 0.0
                    pos_c = np.array([0.0, 10.0, 1])
                    ori_c = np.array([0.0, 0.0, 0.0])
                    mouse_mode = False

                    player1.transform_verts()

                if event.key == pygame.K_m:
                    mouse_mode = not mouse_mode

        keys = pygame.key.get_pressed()

        # カメラ移動
        if keys[pygame.K_a]: pos_c[0] -= TSTEP
        if keys[pygame.K_d]: pos_c[0] += TSTEP
        if keys[pygame.K_w]: pos_c[1] += TSTEP
        if keys[pygame.K_s]: pos_c[1] -= TSTEP
        if keys[pygame.K_z]: pos_c[2] += TSTEP
        if keys[pygame.K_x]: pos_c[2] -= TSTEP

        # カメラ回転
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            if keys[pygame.K_s]: ori_c[0] += RSTEP
            if keys[pygame.K_w]: ori_c[0] -= RSTEP
            if keys[pygame.K_d]: ori_c[1] += RSTEP
            if keys[pygame.K_a]: ori_c[1] -= RSTEP

        # 更新
        stage1.update(dt, mouse_mode)
        player1.update(stage1, dt)

        Rc = RotateMat(ori_c).T 
        Tc = TransMat(-pos_c)

        stage1.draw(buf, Rc, Tc)
        player1.draw(buf, Rc, Tc)

        screen.blit(pygame.transform.flip(buf, 0, 1), (0, 0))
        pygame.display.update()
        clock.tick(fps)


if __name__ == "__main__":
    main()