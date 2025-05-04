FROM python:3.11-slim

WORKDIR /app

# 安装 OpenCV 和 MediaPipe 相关依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# 复制 mediapipe 的 whl 文件并安装
COPY mediapipe-0.10.21-cp311-cp311-manylinux_2_28_x86_64.whl .
RUN pip install --no-cache-dir mediapipe-0.10.21-cp311-cp311-manylinux_2_28_x86_64.whl

# 直接安装 Python 库
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    Flask==3.1.0 \
    numpy==1.26.4 \
    opencv-python==4.11.0.86 \
    requests==2.31.0 \
    Werkzeug==3.1.3

# 复制项目代码
COPY . .

# 启动服务
CMD ["python", "app.py"]













#2.0
# FROM python:3.11-slim

# WORKDIR /app

# # 安装 OpenCV 和 MediaPipe 相关依赖
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0

# # 直接安装 Python 库
# RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
#     Flask==3.1.0 \
#     numpy==1.26.4 \
#     opencv-python==4.11.0.86 \
#     requests==2.31.0 \
#     Werkzeug==3.1.3

# # 复制本地的 mediapipe 的 whl 文件到容器中并安装
# COPY mediapipe-0.10.21-cp311-cp311-manylinux_2_28_x86_64.whl .
# RUN pip install --no-cache-dir mediapipe-0.10.21-cp311-cp311-manylinux_2_28_x86_64.whl

# # 复制项目代码
# COPY . .

# # 启动服务
# CMD ["python", "app.py"]











# FROM python:3.11-slim

# WORKDIR /app

# # 安装 OpenCV 和 MediaPipe 相关依赖
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0

# # 复制 requirements.txt
# COPY requirements.txt .
# RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# # 复制 mediapipe 的 whl 文件到容器
# COPY mediapipe-0.10.21-cp311-cp311-manylinux_2_28_x86_64.whl .

# # 安装本地的 mediapipe
# RUN pip install --no-cache-dir mediapipe-0.10.21-cp311-cp311-manylinux_2_28_x86_64.whl

# # 复制项目文件
# COPY . .

# # 启动服务
# CMD ["python", "app.py"]
