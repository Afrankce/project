import os
# ==========================================
# 🛑 强制 CPU 模式
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# ==========================================

import numpy as np
import tensorflow as tf
import mitsuba as mi
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

try:
    mi.set_variant('llvm_ad_rgb')
except:
    mi.set_variant('scalar_rgb')

import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, RadioMaterial

# ==========================================
#               ⚙️ 配置参数
# ==========================================
SCENE_FILE = "scene.xml"        
DATASET_NAME = "3dgs_Power_100" 
NUM_SAMPLES = 100               
IMG_SIZE = 64                   
FREQUENCY = 2.4e9               

# ⚠️ 注意：TX 高度设为 1.5
TX_POSITION = [3.0, 0.0, 0.8]   
BOUNDS = [-3.0, 3.0, -2.0, 3.0, -1.5, 1.5] 
# ==========================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_colmap_pose(rx_pos, tx_pos):
    # (保持原有的位姿计算代码不变)
    direction = np.array(tx_pos) - np.array(rx_pos)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        forward = np.array([0, 0, 1])
    else:
        forward = direction / norm
    up = np.array([0, 0, 1])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6: right = np.array([1, 0, 0]) 
    right = right / np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up = new_up / np.linalg.norm(new_up)
    rot_mat = np.column_stack((right, -new_up, forward)) 
    r = R.from_matrix(rot_mat.T) 
    qx, qy, qz, qw = r.as_quat()
    tvec = -np.dot(rot_mat.T, rx_pos)
    return [qw, qx, qy, qz], tvec

def main():
    print(f"🚀 初始化场景: {SCENE_FILE}")
    if not os.path.exists(SCENE_FILE):
        print(f"❌ 错误: 找不到文件 {SCENE_FILE}")
        return

    # 1. 加载场景
    try:
        scene = load_scene(SCENE_FILE)
    except RuntimeError as e:
        print(f"❌ XML 格式错误: {e}")
        return

    scene.frequency = FREQUENCY
    
    # =========================================================
    # 🛠️ 【精准修复】根据材质名称，赋予真实的物理属性
    # =========================================================
    print("🔧 正在进行精准材质映射 (ITU-R P.2040 标准)...")
    
    # 定义不同材质的物理属性 (名称必须与 XML 中的 id 一致)
    # 相对介电常数 (er), 电导率 (s)
    materials_db = {
        # --- 混凝土类 (地板、墙、天花板) ---
        "itu_concrete":      {"er": 5.31, "s": 0.0326}, 
        "itu_floor":         {"er": 5.31, "s": 0.0326}, 
        "itu_ceiling_board": {"er": 5.31, "s": 0.0326},
        
        # --- 木材类 (桌子、椅子、胶合板) ---
        "itu_wood":          {"er": 1.99, "s": 0.0047}, 
        "itu_plywood":       {"er": 1.99, "s": 0.0047},
        
        # --- 玻璃类 (窗户) ---
        "itu_glass":         {"er": 6.27, "s": 0.0043},
        
        # --- 金属类 (电视边框) ---
        "itu_metal":         {"er": 1.0,  "s": 1e7} # 高电导率
    }

    # 1. 将这些材质添加到场景中 (如果在 XML 里用到了，这里必须定义)
    for mat_name, props in materials_db.items():
        if mat_name not in scene.radio_materials:
            print(f"   -> 定义材质: {mat_name} (er={props['er']}, s={props['s']})")
            rm = RadioMaterial(mat_name, 
                               relative_permittivity=props["er"], 
                               conductivity=props["s"])
            scene.add(rm)

    # 2. 再次遍历物体，确保所有物体都关联到了正确的 RadioMaterial
    # 这一步是为了解决 XML 读取时的 disconnect 问题
    for name, obj in scene.objects.items():
        # 如果物体名字里包含某些关键词，或者原材质失效，强制重新关联
        
        # 获取物体原有的材质名 (尝试从 Mitsuba 属性中猜测)
        # 这里我们用一种更稳健的方法：根据 XML 的命名习惯来重新赋值
        
        assigned = False
        # 尝试根据 XML id 匹配
        for mat_key in materials_db.keys():
            # 这是一个简单的启发式规则：如果之前的报错说 obj 用了 'itu_floor'
            # 我们可以直接给它赋值。由于无法直接获取 broken 的材质名，
            # 我们这里给所有物体根据其名字特征分配材质（如果 XML 映射失败的话）
            pass 
        
        # 如果当前物体没有有效的 radio_material (即为 None)
        if obj.radio_material is None:
            # 根据物体名字猜测材质 (作为兜底方案)
            if "floor" in name.lower():
                obj.radio_material = scene.radio_materials["itu_floor"]
                print(f"   -> 修复物体 {name}: 关联到 itu_floor")
            elif "wall" in name.lower() or "pillar" in name.lower():
                obj.radio_material = scene.radio_materials["itu_concrete"]
                print(f"   -> 修复物体 {name}: 关联到 itu_concrete")
            elif "wood" in name.lower() or "table" in name.lower() or "chair" in name.lower():
                obj.radio_material = scene.radio_materials["itu_wood"]
                print(f"   -> 修复物体 {name}: 关联到 itu_wood")
            elif "window" in name.lower() or "glass" in name.lower():
                obj.radio_material = scene.radio_materials["itu_glass"]
                print(f"   -> 修复物体 {name}: 关联到 itu_glass")
            elif "tv" in name.lower() or "metal" in name.lower():
                obj.radio_material = scene.radio_materials["itu_metal"]
                print(f"   -> 修复物体 {name}: 关联到 itu_metal")
            else:
                # 实在认不出来的，默认为混凝土
                obj.radio_material = scene.radio_materials["itu_concrete"]
                print(f"   -> 物体 {name} 未知，默认为混凝土")
        
        # 针对报错 "_unnamed_4" 这种特殊情况，通常是 XML 里某个 shape 没有 name
        # 但它引用了 itu_floor。因为我们上面已经在 scene.add(rm) 中添加了 itu_floor
        # Sionna 在 compute_paths 时应该能自动根据名字找到它了。
    
    print("✅ 材质映射完成。")
    # =========================================================

    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V",
                                 vertical_spacing=0.5, horizontal_spacing=0.5)
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V",
                                 vertical_spacing=0.5, horizontal_spacing=0.5)
    
    if "tx" in scene.transmitters: scene.remove("tx")
    tx = Transmitter(name="tx", position=TX_POSITION)
    scene.add(tx)

    ensure_dir(DATASET_NAME)
    ensure_dir(os.path.join(DATASET_NAME, "images"))
    ensure_dir(os.path.join(DATASET_NAME, "sparse", "0"))
    
    # 写入 COLMAP 头文件
    cameras_lines = [
        "# Camera list with one line of data per camera.",
        f"1 SIMPLE_PINHOLE {IMG_SIZE} {IMG_SIZE} {IMG_SIZE*1.2} {IMG_SIZE/2} {IMG_SIZE/2}"
    ]
    points_lines = ["# 3D point list", "1 0 0 0 0 0 0 0 0 0"]
    images_lines = ["# Image list"]
    
    file_indices = []
    rx_positions = np.random.uniform(
        low=[BOUNDS[0], BOUNDS[2], BOUNDS[4]],
        high=[BOUNDS[1], BOUNDS[3], BOUNDS[5]],
        size=(NUM_SAMPLES, 3)
    )

    temp_data = [] 
    print(f"⚡ 开始计算光线追踪 ({NUM_SAMPLES} 个样本)...")
    
    valid_count = 0

    for i, pos in enumerate(rx_positions):
        if "rx" in scene.receivers: scene.remove("rx")
        scene.add(Receiver(name="rx", position=pos))
        
        try:
            paths = scene.compute_paths(
                max_depth=5,        
                num_samples=1000,   
                diffraction=True,   # 开启绕射 (解决 NLoS)
                scattering=False    
            )
            a, tau = paths.cir()
            power = tf.reduce_sum(tf.abs(a)**2).numpy()
        except Exception as e:
            print(f"  [Error] Sample {i}: {e}")
            power = 0.0

        if power < 1e-18: power = 1e-18
        else: valid_count += 1
        temp_data.append(power)

        if (i+1) % 10 == 0:
            print(f"  进度: {i+1}/{NUM_SAMPLES} | 当前点功率: {10*np.log10(power):.2f} dB")

    # 结果检查
    if valid_count == 0:
        print("❌ 所有点功率仍为 -180dB。请检查模型单位是否为毫米 (如果是毫米，请将 TX/RX 坐标扩大1000倍)。")
    else:
        # 生成图片和 TXT
        powers_db = 10 * np.log10(np.array(temp_data))
        p_min, p_max = np.min(powers_db), np.max(powers_db)
        if p_max - p_min < 1.0: p_max = p_min + 10.0

        print(f"📊 最终统计 - Min: {p_min:.2f} dB, Max: {p_max:.2f} dB")
        print("💾 正在保存...")
        
        for i, (pos, p_val) in enumerate(zip(rx_positions, temp_data)):
            img_name = f"{i:05d}.png"
            norm_p = (10*np.log10(p_val) - p_min) / (p_max - p_min)
            
            plt.figure(figsize=(1, 1), dpi=IMG_SIZE)
            plt.axis('off')
            plt.imshow(np.random.normal(norm_p, 0.02, (IMG_SIZE, IMG_SIZE)), cmap='magma', vmin=0, vmax=1)
            plt.gca().set_axis_off()
            plt.subplots_adjust(0,0,1,1,0,0)
            plt.savefig(os.path.join(DATASET_NAME, "images", img_name), pad_inches=0)
            plt.close()

            q, t = get_colmap_pose(pos, TX_POSITION)
            images_lines.append(f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {img_name}")
            images_lines.append("") 
            file_indices.append(img_name)

        with open(os.path.join(DATASET_NAME, "sparse/0", "cameras.txt"), "w") as f: f.write("".join(cameras_lines))
        with open(os.path.join(DATASET_NAME, "sparse/0", "images.txt"), "w") as f: f.write("".join(images_lines))
        with open(os.path.join(DATASET_NAME, "sparse/0", "points3D.txt"), "w") as f: f.write("".join(points_lines))
        
        split = int(NUM_SAMPLES * (1-0.2))
        with open(os.path.join(DATASET_NAME, "train_index.txt"), "w") as f: f.write("".join(file_indices[:split]))
        with open(os.path.join(DATASET_NAME, "test_index.txt"), "w") as f: f.write("".join(file_indices[split:]))

        print("🎉 数据集生成完毕！")

if __name__ == "__main__":
    main()
