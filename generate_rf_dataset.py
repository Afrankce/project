import os

# ==========================================
# 🛑 强制使用 CPU 模式
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # 减少 TF 警告
print("🔒 已屏蔽 GPU,强制使用 CPU 运行...")

import numpy as np
import tensorflow as tf
import mitsuba as mi

# 设置 Mitsuba 为 CPU 模式
try:
    mi.set_variant('llvm_ad_rgb')
except:
    mi.set_variant('scalar_rgb')

import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, RadioMaterial
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time

# ==========================================
#               CONFIG 配置
# ==========================================
SCENE_FILE = "scene.xml"
OUTPUT_DIR = "dataset_cpu"  
TX_POSITION = [3.0, 0.0, 0.8]
BOUNDS = [-3.0, 3.0, -2.0, 2.0, -1.5, 1.5]
FREQUENCY = 2.4e9
NUM_SAMPLES = 50  # 数量
# ==========================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_colmap_structure(base_dir):
    sparse_dir = os.path.join(base_dir, "sparse", "0")
    images_dir = os.path.join(base_dir, "images")
    ensure_dir(sparse_dir)
    ensure_dir(images_dir)
    return sparse_dir, images_dir

def save_cameras_txt(path, num_images, width=512, height=512):
    focal = width * 1.0 
    cx, cy = width / 2, height / 2
    with open(os.path.join(path, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera.")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]")
        f.write(f"1 PINHOLE {width} {height} {focal} {focal} {cx} {cy}")

def save_images_txt(path, positions, orientations, filenames):
    with open(os.path.join(path, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image.")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME")
        for i, (pos, rot, name) in enumerate(zip(positions, orientations, filenames)):
            img_id = i + 1
            qx, qy, qz, qw = rot.as_quat()
            tx, ty, tz = pos
            f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {name}")
            f.write("")

def save_points3d_txt(path):
    with open(os.path.join(path, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point.")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)")

def main():
    print(f"🚀 Loading scene: {SCENE_FILE}")
    if not os.path.exists(SCENE_FILE):
        print(f"❌ 错误: 找不到文件 {SCENE_FILE}")
        return

    # 1. 加载场景
    scene = load_scene(SCENE_FILE)
    scene.frequency = FREQUENCY

    # 2. 设置天线 (包含修复后的 spacing 参数)
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V", 
                                 vertical_spacing=0.5, horizontal_spacing=0.5)
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V", 
                                 vertical_spacing=0.5, horizontal_spacing=0.5)

    # 3. 材质修复
    rm_concrete = RadioMaterial("rm_concrete", relative_permittivity=5.31, conductivity=0.0326)
    if "rm_concrete" not in scene.radio_materials:
        scene.add(rm_concrete)
    for name, obj in scene.objects.items():
        if obj.radio_material is None:
            obj.radio_material = rm_concrete

    # 4. 添加发射机
    tx = Transmitter(name="tx", position=TX_POSITION)
    scene.add(tx)

    # 5. 生成随机坐标
    rx_pos_np = np.random.uniform(
        low=[BOUNDS[0], BOUNDS[2], BOUNDS[4]],
        high=[BOUNDS[1], BOUNDS[3], BOUNDS[5]],
        size=(NUM_SAMPLES, 3)
    ).astype(np.float32)

    # 准备存储结果
    all_powers = []
    
    print(f"⚡ 开始逐点计算 (共 {NUM_SAMPLES} 个点)...")
    start_total = time.time()

    # ==========================================
    # 🚨 核心修改: 循环逐个计算，避免形状报错
    # ==========================================
    for i in range(NUM_SAMPLES):
        # 移除旧的 rx (如果存在)
        if "rx" in scene.receivers:
            scene.remove("rx")
        
        # 添加当前位置的 rx
        # 注意: position 必须是 [3] 或 [1, 3]
        current_pos = rx_pos_np[i]
        rx = Receiver(name="rx", position=current_pos)
        scene.add(rx)

        # 计算路径
        # 降低 max_depth 和 num_samples 以提高 CPU 速度
        try:
            paths = scene.compute_paths(
                max_depth=3,
                num_samples=10000, 
                method="fibonacci",
                diffraction=False,
                scattering=False,
                check_scene=False
            )
            
            # 提取能量
            a, tau = paths.cir()
            # 形状通常是 [1, 1, 1, path_count] -> 需要求和
            power_val = tf.reduce_sum(tf.abs(a)**2).numpy()
            all_powers.append(power_val)

        except Exception as e:
            print(f"⚠️ 点 {i} 计算失败: {e}")
            all_powers.append(0.0)

        # 打印进度
        if (i + 1) % 5 == 0:
            print(f"   进度: {i + 1}/{NUM_SAMPLES} | 耗时: {time.time() - start_total:.1f}s")

    print("✅ 计算完成！")

    # --- 后处理与导出 ---
    print("💾 正在导出数据...")
    sparse_dir, images_dir = create_colmap_structure(OUTPUT_DIR)

    # 转换为 dB 并归一化
    rx_power = np.array(all_powers)
    rx_power_db = 10 * np.log10(rx_power + 1e-16)
    
    min_p, max_p = np.min(rx_power_db), np.max(rx_power_db)
    print(f"📊 信号强度: Min={min_p:.2f} dB, Max={max_p:.2f} dB")

    if max_p > min_p:
        norm_power = (rx_power_db - min_p) / (max_p - min_p)
    else:
        norm_power = np.zeros_like(rx_power_db)

    filenames = []
    orientations = []

    for i in range(NUM_SAMPLES):
        filename = f"{i:05d}.png"
        filenames.append(filename)

        # 生成图像
        plt.figure(figsize=(4, 4), dpi=128)
        plt.axis('off')
        viz = np.random.normal(loc=norm_power[i], scale=0.05, size=(64, 64))
        plt.imshow(viz, cmap='magma', vmin=0, vmax=1)
        plt.savefig(os.path.join(images_dir, filename), bbox_inches='tight', pad_inches=0)
        plt.close()

        # 计算朝向
        direction = np.array(TX_POSITION) - rx_pos_np[i] 
        dist = np.linalg.norm(direction)
        if dist > 0: direction /= dist
        else: direction = np.array([1, 0, 0])
        
        up = np.array([0, 0, 1])
        right = np.cross(direction, up)
        if np.linalg.norm(right) < 1e-5: right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        new_up = np.cross(right, direction)
        
        rot_mat = np.column_stack((right, new_up, -direction))
        orientations.append(R.from_matrix(rot_mat))

    save_cameras_txt(sparse_dir, NUM_SAMPLES)
    save_images_txt(sparse_dir, rx_pos_np, orientations, filenames)
    save_points3d_txt(sparse_dir)

    print(f"🎉 成功! 数据集已保存至: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
