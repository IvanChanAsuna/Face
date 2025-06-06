# 人脸识别及低光人脸检测系统  
<p align="left">支持复杂光照条件下的人脸识别登录与自适应低光检测</p>  

## ✨ 核心特性  
- **高精度人脸识别**：基于深度特征匹配的活体检测登录系统  
- **低光增强检测**：集成DAI-Net算法，暗光环境识别准确率提升30%  
- **全栈架构**：Flask后端RESTful API + Vue3前端动态交互  
- **GPU加速**：深度优化支持RTX 4090的Tensor Core计算  
## 🎯 系统概览
![登录](demo.png)
![人脸检测](demo2.png)
## 🚀 快速部署  
### 后端环境安装（Python 3.8）  
```bash
# 创建Conda环境（推荐）  
conda create -n face_system python=3.8  
conda activate face_system  

# 安装PyTorch（CUDA 11.3）  
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html  

# 安装其他依赖  
pip install -r requirements.txt  
```

### 安装前端依赖环境
```bash
npm install
```

### 后端启动命令
```bash
python app.py
```

### 前端启动命令
```bash
npm run serve
```

如果想要自己训练模型，训练识别是否是自己，你可以在user_photos里装入你自己的照片，然后启用一下命令：
```bash
python run.py
```

### 训练环境
- **Python  3.8(ubuntu20.04)**
- **PyTorch  1.10.0**
- **CUDA  11.3**
- **GPU：RTX 4090(24GB) * 1**
- **CPU：16 vCPU Intel(R) Xeon(R) Gold 6430**

