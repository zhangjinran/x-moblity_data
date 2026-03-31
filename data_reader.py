import os
import io
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

# ==================== 配置 ====================
# 显示选项（用于调试）
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 文件路径（根据实际修改）
test_file_path = os.path.join(os.getcwd(), 'data', 'train', 'sceneblox_warehouse_style_1_id_2', 'goal_0000_0hz.pqt')

# ==================== 颜色映射（16类） ====================
SEMANTIC_COLORS_16 = np.array([
    [128, 128, 128],   # 0: BACKGROUND
    [0, 0, 0],         # 1: unlabeled
    [139, 69, 19],     # 2: STRUCTURE
    [0, 255, 0],       # 3: FLOOR
    [0, 255, 0],       # 4: FLOOR
    [255, 165, 0],     # 5: PILE
    [0, 128, 128],     # 6: SHELF
    [128, 0, 128],     # 7: RACK
    [255, 0, 0],       # 8: FENCE
    [255, 255, 0],     # 9: CONE
    [255, 0, 255],     # 10: HAZARD_SIGN
    [0, 0, 255],       # 11: PALLET
    [0, 255, 255],     # 12: CRATE
    [128, 128, 0],     # 13: LINE
    [0, 128, 0],       # 14: PALLET
    [0, 0, 128],       # 15: PALLET
    [255, 128, 0]      # 16: FORKLIFT
], dtype=np.uint8)

CLASS_NAMES_16 = {
    0: "BACKGROUND", 1: "unlabeled", 2: "STRUCTURE", 3: "FLOOR", 4: "FLOOR",
    5: "PILE", 6: "SHELF", 7: "RACK", 8: "FENCE", 9: "CONE", 10: "HAZARD_SIGN",
    11: "PALLET", 12: "CRATE", 13: "LINE", 14: "PALLET", 15: "PALLET", 16: "FORKLIFT"
}

# ==================== 颜色映射（7类） ====================
SEMANTIC_COLORS_7 = np.array([
    [128, 128, 128],  # 0: Background
    [0, 255, 0],      # 1: NavigableSurface
    [255, 165, 0],    # 2: Forklift
    [0, 0, 255],      # 3: Pallet
    [255, 255, 0],    # 4: Cone
    [255, 0, 255],    # 5: Sign
    [255, 0, 0]       # 6: Fence
], dtype=np.uint8)

CLASS_NAMES_7 = {
    0: "BACKGROUND", 1: "NAVIGABLE", 2: "FORKLIFT", 3: "PALLET",
    4: "CONE", 5: "SIGN", 6: "FENCE"
}

# ==================== 辅助函数 ====================
def show_semantic_image(data, col_name, color_map, class_names, img_size=(640, 960)):
    """
    显示语义分割图像，并添加图例。
    
    参数:
        data: 扁平化的语义标签数组 (H*W,)
        col_name: 列名（用于标题）
        color_map: 颜色查找表（shape (num_classes, 3)）
        class_names: 字典 {label: name}
        img_size: (height, width) 图像尺寸
    """
    h, w = img_size
    semantic_map = data.reshape(h, w)
    colored = color_map[semantic_map]  # (h, w, 3)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(colored)
    ax.set_title(f"Semantic Image from column: {col_name}")
    ax.axis('off')
    
    # 创建图例（只显示在颜色映射中定义的类别）
    legend_handles = []
    for idx, name in class_names.items():
        if idx < len(color_map):
            color = color_map[idx] / 255.0
            legend_handles.append(Patch(color=color, label=name))
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1),
              loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def show_path_poses(data, col_name):
    """
    显示路径点序列（二维点）。
    
    参数:
        data: 一维数组，长度必须是偶数，每两个数组成一个点 (x, y)
        col_name: 列名（用于标题）
    """
    if len(data) % 2 != 0:
        print(f"警告: {col_name} 长度 {len(data)} 不是偶数，无法解析为二维点")
        return
    points = data.reshape(-1, 2)   # (N, 2)
    
    plt.figure(figsize=(8, 6))
    plt.plot(points[:, 0], points[:, 1], 'o-', linewidth=2, markersize=5, color='blue')
    plt.scatter(points[0, 0], points[0, 1], color='green', s=100, label='start')
    plt.scatter(points[-1, 0], points[-1, 1], color='red', s=100, label='end')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'{col_name} ({len(points)} points)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def show_binary_image(data, col_name):
    """
    将二进制数据解码为图像并显示。
    
    参数:
        data: bytes 对象，包含图像数据（如 JPEG）
        col_name: 列名（用于标题）
    """
    try:
        img = Image.open(io.BytesIO(data))
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title(f"Image from column: {col_name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"无法解码图像 (列: {col_name})：{e}")

# ==================== 主流程 ====================
def main():
##########################################
    row_idx = 187  # 要展示的行索引，可以修改为其它行
##########################################

    # 读取 Parquet 文件
    table = pq.read_table(test_file_path)
    df = table.to_pandas()

    print("*********************** 数据预览 ***********************")
    # 打印基本信息
    print("DataFrame info:")
    print(df.info())
    print("\n" + "="*60 + "\n")
    
    # 定义列名与处理函数的映射
    column_handlers = {
        'perspective_semantic_image': lambda val: show_semantic_image(
            val, 'perspective_semantic_image', SEMANTIC_COLORS_16, CLASS_NAMES_16),
        'semantic_labels': lambda val: show_semantic_image(
            val, 'semantic_labels', SEMANTIC_COLORS_7, CLASS_NAMES_7),
        'route_poses': lambda val: show_path_poses(val, 'route_poses'),
        'path': lambda val: show_path_poses(val, 'path'),
        # 如果还有其它需要特殊处理的列，可以继续添加
    }
    
    # 遍历每一列，展示第 0 行的数据
    
    print(f"*********************** 展示第 {row_idx} 行的数据 ***********************\n")
    for col in df.columns:
        val = df[col].iloc[row_idx]
        print(f"Column: {col}")
        print("Type:", type(val))
        
        # 处理有形状的数组
        if hasattr(val, "shape"):
            print("Shape:", val.shape)
            print("Dtype:", getattr(val, "dtype", "N/A"))
            print("Value:", val)
            
            # 根据列名调用对应的处理函数
            if col in column_handlers:
                column_handlers[col](val)
            else:
                # 对于其它数组，可以在这里添加通用处理（例如绘制 hist 等）
                pass
        
        # 处理列表
        elif isinstance(val, list):
            print("Length:", len(val))
            print("Value:", val)
            if len(val) > 0:
                print("Element type:", type(val[0]))
        
        # 处理字典
        elif isinstance(val, dict):
            print("Keys:", val.keys())
            for k, v in val.items():
                print(f"Key: {k}, Value: {v}")
        
        # 处理标量（如 int, float, bytes 等）
        else:
            # 如果是 bytes 且可能是图像，尝试显示
            if isinstance(val, bytes) and len(val) > 100:  # 简单判断
                print("Value: bytes (possible image)")
                show_binary_image(val, col)
            else:
                print("Value:", val)
        
        print("-"*40 + "\n")

if __name__ == "__main__":
    main()