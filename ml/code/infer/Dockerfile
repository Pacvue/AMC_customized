# 使用官方 Python 3.10 基础镜像
FROM python:3.10

# 设置环境变量，防止 Python 写入 .pyc 文件，并使输出即时刷新
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 设置工作目录为 /app
WORKDIR /app

# 将 requirements.txt 复制到工作目录
COPY requirements.txt .

# 升级 pip 并安装依赖包
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 复制推理脚本到工作目录
COPY infer.py .


# Set environment variables
# ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code/infer
# ENV SAGEMAKER_PROGRAM=infer.py


# 定义容器启动时执行的命令
ENTRYPOINT ["python", "infer.py"]


# # Use an official Python base image
# FROM python:3.10

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# # Copy requirements.txt into the container
# COPY requirements.txt .

# # Install dependencies
# RUN pip install --no-cache-dir --upgrade pip

# # Install the required Python packages via pip
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the training and inference scripts inside the container
# COPY infer.py /opt/ml/code/infer/infer.py

# # Set environment variables
# ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code/infer
# ENV SAGEMAKER_PROGRAM=infer.py

# # Define the entry point script
# ENTRYPOINT ["python", "/opt/ml/code/infer/infer.py"]

