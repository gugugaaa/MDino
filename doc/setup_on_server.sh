#!/bin/bash

set -e
set -u
set -o pipefail

echo "🚀 (1/10) 开始自动化部署..."
echo "当前工作目录: $(pwd)"

echo "🛠️ (2/10) 安装系统编译依赖 (build-essential, ninja-build)..."
sudo apt-get update
sudo apt-get install -y build-essential ninja-build

echo "⚙️ (3/10) 检查并配置 /root/.bashrc 环境..."
BASHRC_FILE="/root/.bashrc"
CUDA_LINE="export CUDA_HOME=/usr/local/cuda-11.8"
MASKDINO_LINE="export PYTHONPATH=/root/MaskDINO:\$PYTHONPATH"

if ! grep -qF "$CUDA_LINE" "$BASHRC_FILE"; then
    echo "" >> "$BASHRC_FILE"
    echo "# --- 自动添加的 CUDA 11.8 配置 ---" >> "$BASHRC_FILE"
    echo "export CUDA_HOME=/usr/local/cuda-11.8" >> "$BASHRC_FILE"
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> "$BASHRC_FILE"
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> "$BASHRC_FILE"
    echo "  -> CUDA 路径添加完成。"
else
    echo "  -> CUDA 环境变量已存在，跳过。"
fi

if ! grep -qF "$MASKDINO_LINE" "$BASHRC_FILE"; then
    echo "" >> "$BASHRC_FILE"
    echo "# --- 自动添加的 MaskDINO PYTHONPATH ---" >> "$BASHRC_FILE"
    echo "export PYTHONPATH=/root/MaskDINO:\$PYTHONPATH" >> "$BASHRC_FILE"
    echo "  -> MaskDINO PYTHONPATH 添加完成。"
else
    echo "  -> MaskDINO PYTHONPATH 已存在，跳过。"
fi

echo "🔄 (4/10) 立即加载环境变量..."
if [ -f "$BASHRC_FILE" ]; then
    . "$BASHRC_FILE"
fi

if [ -z "${CUDA_HOME:-}" ]; then
    echo "❌ 错误: CUDA_HOME 未能成功加载。请检查 .bashrc 文件！"
    exit 1
fi
echo "  -> CUDA_HOME 已加载: $CUDA_HOME"

echo "🐍 (5/10) 安装/升级 OpenCV 和 NumPy..."
pip install -U opencv-python numpy

echo "📦 (6/10) 安装 Detectron2..."
cd /root
if [ ! -d "detectron2" ]; then
    git clone https://github.com/facebookresearch/detectron2.git
else
    echo "  -> detectron2 目录已存在，跳过克隆。"
fi
cd detectron2
pip install -e .
echo "  -> Detectron2 安装完成。"

echo "📦 (7/10) 安装 MaskDINO 依赖 (panopticapi, cityscapesScripts)..."
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

echo "📦 (8/10) 安装 MaskDINO..."
cd /root
if [ ! -d "MaskDINO" ]; then
    git clone https://github.com/IDEA-Research/MaskDINO.git
else
    echo "  -> MaskDINO 目录已存在，跳过克隆。"
fi
cd MaskDINO
pip install -r requirements.txt
echo "  -> MaskDINO Python 需求安装完成。"

echo "⚙️ (9/10) 编译 MaskDINO 自定义 Ops..."
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
echo "  -> Ops 编译完成。"

echo "🧪 (10/10) 创建并运行最终测试脚本..."
cd /root
cat << 'EOF' > test_imports.py
import os
import sys
print("--- 自动化环境测试 ---")
print(f"Python Executable: {sys.executable}")
print(f"CUDA_HOME (from env): {os.environ.get('CUDA_HOME')}")
print(f"PYTHONPATH (from env): {os.environ.get('PYTHONPATH')}")

try:
    import detectron2
    print(f"✅ [成功] 导入 detectron2 (版本: {detectron2.__version__})")
except ImportError as e:
    print(f"❌ [失败] 导入 detectron2: {e}")
    sys.exit(1)

try:
    import maskdino
    print(f"✅ [成功] 导入 maskdino")
except ImportError as e:
    print(f"❌ [失败] 导入 maskdino: {e}")
    print("    -> 提示: 请确保 /root/MaskDINO 在 PYTHONPATH 中！")
    sys.exit(1)

print("--------------------------")
print("🎉 恭喜！Detectron2 和 MaskDINO 均可成功导入！")
EOF

python test_imports.py

echo "--------------------------------------------------------"
echo "✅ 自动化脚本执行完毕！"
echo "环境已配置。您现在应该可以直接在 Python 中 'import maskdino'。"
echo "注意: 如果您打开新的 shell/终端，.bashrc 将自动生效。"
echo "--------------------------------------------------------"