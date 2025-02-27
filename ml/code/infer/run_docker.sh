#!/bin/bash

# 定义主机上的 ml 目录路径
ML_DIR="/home/ec2-user/sylvia/HighPotentialCustomers/ml"



# 定义 infer 目录路径
INFER_DIR="$ML_DIR/code/infer"

# 定义 Docker 镜像名称
IMAGE_NAME="acm_high_potential_customers_infer:latest"

# 定义需要挂载的数据目录
DATA_DIR="$ML_DIR/input/data/infer"

# 定义输出目录
OUTPUT_DIR="$ML_DIR/output/data/audiences"

# 定义模型目录
MODEL_DIR="$ML_DIR/model"

# 切换到 infer 目录
cd "$INFER_DIR" || { echo "无法切换到目录 $INFER_DIR"; exit 1; }

# 构建 Docker 镜像，使用默认的 Dockerfile
echo "正在构建 Docker 镜像: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

# 检查 Docker 镜像是否构建成功
if [ $? -ne 0 ]; then
    echo "Docker 镜像构建失败，请检查 Dockerfile 和构建环境。"
    exit 1
fi

# 返回到原目录（可选）
cd - >/dev/null

# 创建输出目录和模型目录（如果不存在）
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MODEL_DIR"

# # 确保模型文件存在
# if [ ! -f "$MODEL_DIR/high_potential_model.pkl" ]; then
#     echo "模型文件 predictor.pkl 不存在于 $MODEL_DIR。请确保模型已训练并保存。"
#     exit 1
# fi

# 运行 Docker 容器，仅挂载必要的子目录，避免覆盖容器内的代码
echo "正在运行 Docker 容器..."
docker run --rm \
    -v "$DATA_DIR":/opt/ml/data \
    -v "$OUTPUT_DIR":/opt/ml/output \
    -v "$MODEL_DIR":/opt/ml/model \
    $IMAGE_NAME

# 检查容器是否成功运行
if [ $? -ne 0 ]; then
    echo "Docker 容器运行失败，请检查推理脚本和数据路径。"
    exit 1
fi

echo "推理任务完成，结果保存在主机的 $OUTPUT_DIR 目录下。"
