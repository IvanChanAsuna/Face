<template>
  <div class="face-detect-container">
    <!-- 简化的背景装饰 -->
    <div class="background-decorations">
      <div class="decoration-circle circle-1"></div>
      <div class="decoration-circle circle-2"></div>
      <div class="decoration-circle circle-3"></div>
    </div>

    <div class="page-content">
      <!-- 去掉重复的标题区域，直接进入主要内容 -->
      <div class="main-content">
        <!-- 图片显示卡片 -->
        <div class="image-card">
          <div class="card-header">
            <h3 class="card-title">
              <el-icon class="card-icon">
                <Picture />
              </el-icon>
              检测结果展示
            </h3>
            <div class="status-indicator">
              <div class="status-dot" :class="{ 'processing': state.isProcessing }"></div>
              <span>{{ state.isProcessing ? '处理中...' : '准备就绪' }}</span>
            </div>
          </div>

          <div class="image-display-area">
            <div class="image-container">
              <!-- Element Plus Image组件 -->
              <el-image v-if="state.imageSrc && !state.useNativeImg" class="detection-image" fit="contain"
                :src="state.imageSrc" :preview-src-list="[state.imageSrc]" loading="lazy" @error="handleImageError"
                @load="handleImageLoad">
                <template #placeholder>
                  <div class="image-placeholder">
                    <el-icon class="placeholder-icon">
                      <Loading />
                    </el-icon>
                    <span>处理中...</span>
                  </div>
                </template>
                <template #error>
                  <div class="image-error">
                    <el-icon class="error-icon">
                      <Warning />
                    </el-icon>
                    <span>图片加载失败</span>
                    <button @click="state.useNativeImg = true" class="retry-btn">尝试原生img</button>
                  </div>
                </template>
              </el-image>

              <!-- 原生img标签作为备选 -->
              <img v-else-if="state.imageSrc && state.useNativeImg" :src="state.imageSrc" class="detection-image-native"
                @error="handleNativeImageError" @load="handleNativeImageLoad" alt="检测结果" />

              <!-- 默认状态显示 -->
              <div v-else class="empty-state">
                <div class="empty-content">
                  <el-icon class="empty-icon">
                    <Picture />
                  </el-icon>
                  <h3 class="empty-title">等待上传图片</h3>
                  <p class="empty-description">上传一张包含人脸的图片，系统将自动进行检测并标注</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 上传控制区域 -->
        <div class="upload-section">
          <div class="upload-card">
            <div class="upload-header">
              <el-icon class="upload-header-icon">
                <Upload />
              </el-icon>
              <h3>上传图片进行检测</h3>
              <p>支持拖拽上传或点击选择文件</p>
            </div>

            <div class="upload-area">
              <el-upload class="custom-upload" :before-upload="beforeUpload" accept="image/png, image/jpeg, image/jpg"
                :show-file-list="false" :disabled="state.isProcessing" drag>
                <div class="upload-content">
                  <el-icon class="upload-icon">
                    <UploadFilled />
                  </el-icon>
                  <div class="upload-text">
                    <p class="upload-title">拖拽图片到此处</p>
                    <p class="upload-subtitle">或点击选择文件</p>
                    <p class="upload-hint">支持 PNG、JPG、JPEG 格式，文件大小不超过 10MB</p>
                  </div>
                </div>
              </el-upload>
            </div>

            <div class="upload-tips">
              <div class="tip-item">
                <el-icon class="tip-icon success">
                  <Select />
                </el-icon>
                <span>自动检测并标注人脸位置</span>
              </div>
              <div class="tip-item">
                <el-icon class="tip-icon success">
                  <Select />
                </el-icon>
                <span>支持多人脸同时检测</span>
              </div>
              <div class="tip-item">
                <el-icon class="tip-icon success">
                  <Select />
                </el-icon>
                <span>支持低光人脸检测</span>
              </div>
            </div>
          </div>
        </div>

        <!-- 调试面板 -->
        <div class="debug-panel" v-if="state.showDebug">
          <h4>调试信息</h4>
          <div class="debug-item">
            <strong>图片数据长度:</strong> {{ state.imageSrc.length }}
          </div>
          <div class="debug-item">
            <strong>数据前缀:</strong> {{ state.imageSrc.substring(0, 50) }}
          </div>
          <button @click="testImageInNewTab" class="debug-btn">在新窗口测试图片</button>
          <button @click="downloadImage" class="debug-btn">下载图片</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { reactive, nextTick } from 'vue'
import http from '@/utils/http'
import { ElMessage } from 'element-plus'
import {
  Picture,
  Loading,
  Upload,
  UploadFilled,
  Select,
  Warning
} from '@element-plus/icons-vue'

const state = reactive({
  imageSrc: '',
  isProcessing: false,
  useNativeImg: false,
  showDebug: true // 开启调试模式
})

const beforeUpload = (file: File) => {
  console.log('=== 开始文件上传流程 ===')
  console.log('文件信息:')
  console.log('- 文件名:', file.name)
  console.log('- 文件大小:', (file.size / 1024 / 1024).toFixed(2), 'MB')
  console.log('- 文件类型:', file.type)
  console.log('- 最后修改时间:', new Date(file.lastModified))
  
  // 验证文件大小
  const isLt10M = file.size / 1024 / 1024 < 10
  if (!isLt10M) {
    ElMessage.error('上传图片大小不能超过 10MB!')
    return false
  }

  // 验证文件类型
  const isValidType = ['image/jpeg', 'image/jpg', 'image/png'].includes(file.type)
  if (!isValidType) {
    ElMessage.error('只支持 JPG、JPEG、PNG 格式的图片!')
    return false
  }

  console.log('✅ 文件验证通过，开始处理...')
  processImage(file)
  return false // 阻止默认上传行为
}

const processImage = async (file: File) => {
  console.log('=== 开始图片处理流程 ===')
  
  state.isProcessing = true
  state.useNativeImg = false
  state.imageSrc = '' // 清空之前的图片

  try {
    // 创建 FormData
    console.log('创建 FormData...')
    const formData = new FormData()
    formData.append('image', file)
    
    // 验证 FormData 内容
    console.log('FormData 验证:')
    for (const [key, value] of formData.entries()) {
      if (value instanceof File) {
        console.log(`- ${key}: File(${value.name}, ${value.size} bytes, ${value.type})`)
      } else {
        console.log(`- ${key}:`, value)
      }
    }

    // 发送请求
    console.log('发送请求到后端...')
    console.log('请求URL: POST /detectFaces')
    
    const response = await http.post('/detectFaces', formData)
    
    console.log('=== 收到后端响应 ===')
    console.log('HTTP状态码:', response.status)
    console.log('响应头:', response.headers)
    console.log('完整响应数据:', JSON.stringify(response.data, null, 2))

    const responseData = response.data
    
    // 检查响应格式
    if (!responseData) {
      throw new Error('后端返回空响应')
    }
    
    console.log('响应数据分析:')
    console.log('- code:', responseData.code)
    console.log('- message:', responseData.message)
    console.log('- data存在:', !!responseData.data)
    
    if (responseData.data) {
      console.log('- data.image存在:', !!responseData.data.image)
      if (responseData.data.image) {
        console.log('- image数据类型:', typeof responseData.data.image)
        console.log('- image数据长度:', responseData.data.image.length)
        console.log('- image前50字符:', responseData.data.image.substring(0, 50))
      }
    }
    
    if (responseData.code === 0) {
      if (responseData.data && responseData.data.image) {
        let imageData = responseData.data.image
        
        console.log('=== 处理图片数据 ===')
        console.log('原始数据长度:', imageData.length)
        console.log('是否包含data URL前缀:', imageData.startsWith('data:image/'))
        
        // 确保数据格式正确
        if (imageData.startsWith('data:image/')) {
          console.log('✅ 数据已包含完整前缀')
        } else {
          console.log('⚠️ 添加data URL前缀')
          imageData = `data:image/jpeg;base64,${imageData}`
        }

        console.log('最终数据长度:', imageData.length)
        console.log('最终数据前缀:', imageData.substring(0, 50))

        // 设置图片数据
        state.imageSrc = imageData
        
        await nextTick()
        console.log('✅ DOM已更新，开始测试图片加载...')
        
        // 测试图片是否能加载
        testImageLoad(imageData)

        ElMessage.success(responseData.message || '人脸检测完成!')
      } else {
        console.error('❌ 后端响应中没有图片数据')
        console.log('响应结构:', responseData)
        ElMessage.warning('后端处理完成，但未返回图片数据')
      }
    } else {
      console.error('❌ 后端返回错误响应')
      console.log('错误代码:', responseData.code)
      console.log('错误信息:', responseData.message)
      ElMessage.error(responseData.message || '检测失败，请重试')
    }
    
  } catch (error: any) {
    console.error('=== 请求处理失败 ===')
    console.error('错误对象:', error)
    
    if (error.response) {
      // 服务器返回了错误响应
      console.error('服务器错误详情:')
      console.error('- 状态码:', error.response.status)
      console.error('- 状态文本:', error.response.statusText)
      console.error('- 响应头:', error.response.headers)
      console.error('- 错误数据:', error.response.data)
      
      ElMessage.error(`服务器错误 (${error.response.status}): ${error.response.data?.message || error.response.statusText}`)
    } else if (error.request) {
      // 请求发送了但没有收到响应
      console.error('网络连接错误:')
      console.error('- 请求对象:', error.request)
      console.error('- 错误信息:', error.message)
      
      ElMessage.error('无法连接到服务器，请检查网络连接和后端服务状态')
    } else {
      // 其他错误
      console.error('请求配置错误:', error.message)
      ElMessage.error(`请求错误: ${error.message}`)
    }
  } finally {
    state.isProcessing = false
    console.log('=== 图片处理流程结束 ===')
  }
}

const testImageLoad = (imageSrc: string) => {
  console.log('=== 测试图片加载 ===')
  
  const img = new Image()
  
  img.onload = () => {
    console.log('✅ 图片加载成功!')
    console.log('- 图片尺寸:', img.width, 'x', img.height)
    console.log('- 图片src长度:', img.src.length)
    ElMessage.success('图片显示成功!')
  }
  
  img.onerror = (error) => {
    console.error('❌ 图片加载失败:')
    console.error('- 错误事件:', error)
    console.error('- 图片src长度:', imageSrc.length)
    console.error('- 图片src前缀:', imageSrc.substring(0, 100))
    ElMessage.error('图片数据有问题，无法显示')
  }
  
  img.src = imageSrc
}

const handleImageError = (error: Event) => {
  console.error('Element Plus 图片组件加载错误:', error)
  console.log('当前 imageSrc 长度:', state.imageSrc.length)
  console.log('尝试切换到原生 img 标签...')
  
  ElMessage.warning('图片显示失败，正在尝试备用方案...')
  
  setTimeout(() => {
    state.useNativeImg = true
    console.log('已切换到原生 img 标签')
  }, 1000)
}

const handleImageLoad = () => {
  console.log('✅ Element Plus 图片组件加载成功')
}

const handleNativeImageError = (error: Event) => {
  console.error('原生 img 标签加载错误:', error)
  console.log('图片数据可能已损坏')
  ElMessage.error('图片无法显示，请检查数据格式')
}

const handleNativeImageLoad = () => {
  console.log('✅ 原生 img 标签加载成功')
  ElMessage.success('图片显示成功（使用备用方案）')
}

const testImageInNewTab = () => {
  if (!state.imageSrc) {
    ElMessage.warning('没有可测试的图片')
    return
  }

  console.log('在新标签页中测试图片显示')
  const newWindow = window.open()
  if (newWindow) {
    newWindow.document.write(`
      <html>
        <head><title>图片测试 - ${new Date().toLocaleString()}</title></head>
        <body style="margin:0; padding:20px; background:#f5f5f5; font-family: Arial, sans-serif;">
          <h2>图片测试</h2>
          <p><strong>数据长度:</strong> ${state.imageSrc.length}</p>
          <p><strong>数据前缀:</strong> ${state.imageSrc.substring(0, 100)}</p>
          <div style="border: 2px solid #ddd; padding: 20px; background: white; margin: 20px 0;">
            <img src="${state.imageSrc}" 
                 style="max-width:100%; height:auto; display:block; margin:0 auto;" 
                 onload="console.log('新窗口图片加载成功'); document.getElementById('status').textContent = '✅ 加载成功'" 
                 onerror="console.error('新窗口图片加载失败'); document.getElementById('status').textContent = '❌ 加载失败'" />
          </div>
          <p id="status">加载中...</p>
        </body>
      </html>
    `)
  }
}

const downloadImage = () => {
  if (!state.imageSrc) {
    ElMessage.warning('没有可下载的图片')
    return
  }

  try {
    console.log('开始下载图片...')
    
    // 解析MIME类型
    const mimeMatch = state.imageSrc.match(/data:(.*?);/)
    const mimeType = mimeMatch ? mimeMatch[1] : 'image/jpeg'
    console.log('检测到MIME类型:', mimeType)
    
    // 提取base64数据
    const base64Data = state.imageSrc.split(',')[1]
    if (!base64Data) {
      throw new Error('无法提取base64数据')
    }
    
    // 转换为二进制数据
    const byteString = atob(base64Data)
    const ab = new ArrayBuffer(byteString.length)
    const ia = new Uint8Array(ab)
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i)
    }
    
    // 创建Blob
    const blob = new Blob([ab], { type: mimeType })
    console.log('创建Blob成功，大小:', blob.size, 'bytes')

    // 生成文件名
    const extension = mimeType.split('/')[1] || 'jpg'
    const filename = `face_detect_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.${extension}`
    console.log('文件名:', filename)

    // 创建下载链接
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.style.display = 'none'
    
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    // 清理对象URL
    setTimeout(() => URL.revokeObjectURL(link.href), 100)
    
    console.log('✅ 图片下载完成')
    ElMessage.success('图片下载成功!')
  } catch (error) {
    console.error('❌ 下载失败:', error)
    ElMessage.error('下载失败')
  }
}
</script>

<style scoped>
/* 保持原有样式 */
.face-detect-container {
  min-height: 100vh;
  background: transparent;
  position: relative;
  overflow: hidden;
  padding: 0;
}

.background-decorations {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  z-index: 0;
}

.decoration-circle {
  position: absolute;
  border-radius: 50%;
  opacity: 0.05;
  animation: float 8s ease-in-out infinite;
}

.circle-1 {
  width: 300px;
  height: 300px;
  background: linear-gradient(135deg, #8b5cf6, #a855f7);
  top: -150px;
  right: -150px;
  animation-delay: 0s;
}

.circle-2 {
  width: 200px;
  height: 200px;
  background: linear-gradient(135deg, #ec4899, #be185d);
  bottom: -100px;
  left: -100px;
  animation-delay: 3s;
}

.circle-3 {
  width: 150px;
  height: 150px;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  top: 50%;
  right: 10%;
  animation-delay: 6s;
}

.page-content {
  position: relative;
  z-index: 1;
  max-width: 1400px;
  margin: 0 auto;
  padding: 30px;
}

.main-content {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 40px;
  align-items: start;
}

.image-card {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 20px;
  box-shadow:
    0 20px 40px rgba(0, 0, 0, 0.1),
    0 0 0 1px rgba(255, 255, 255, 0.5);
  backdrop-filter: blur(10px);
  overflow: hidden;
  animation: fadeInLeft 0.8s ease-out;
}

.card-header {
  padding: 25px 30px;
  background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.card-title {
  color: white;
  font-size: 18px;
  font-weight: 600;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 10px;
}

.card-icon {
  font-size: 20px;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  color: white;
  font-size: 14px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #00ff88;
  animation: pulse 2s infinite;
}

.status-dot.processing {
  background: #fbbf24;
  animation: blink 1s infinite;
}

.image-display-area {
  padding: 30px;
}

.image-container {
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  background: #faf7ff;
  min-height: 600px;
  box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.1);
}

.detection-image {
  width: 100%;
  height: 600px;
  border-radius: 16px;
}

.detection-image-native {
  width: 100%;
  height: 600px;
  object-fit: contain;
  border-radius: 16px;
  background: #faf7ff;
}

.image-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 600px;
  color: #a78bfa;
  font-size: 16px;
  gap: 15px;
}

.placeholder-icon {
  font-size: 48px;
  opacity: 0.5;
}

.image-error {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 600px;
  color: #ef4444;
  gap: 15px;
}

.error-icon {
  font-size: 48px;
}

.retry-btn {
  background: #8b5cf6;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  margin-top: 10px;
}

.retry-btn:hover {
  background: #7c3aed;
}

.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 600px;
  padding: 40px;
}

.empty-content {
  text-align: center;
  max-width: 400px;
}

.empty-icon {
  font-size: 80px;
  color: #e2e8f0;
  margin-bottom: 20px;
}

.empty-title {
  font-size: 24px;
  font-weight: 600;
  color: #475569;
  margin: 0 0 10px 0;
}

.empty-description {
  font-size: 16px;
  color: #64748b;
  margin: 0 0 30px 0;
  line-height: 1.5;
}

.upload-section {
  animation: fadeInRight 0.8s ease-out;
}

.upload-card {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 20px;
  box-shadow:
    0 20px 40px rgba(0, 0, 0, 0.1),
    0 0 0 1px rgba(255, 255, 255, 0.5);
  backdrop-filter: blur(10px);
  overflow: hidden;
  transition: all 0.3s ease;
}

.upload-card:hover {
  transform: translateY(-5px);
  box-shadow:
    0 25px 50px rgba(0, 0, 0, 0.15),
    0 0 0 1px rgba(255, 255, 255, 0.5);
}

.upload-header {
  padding: 30px;
  text-align: center;
  background: linear-gradient(135deg, rgba(139, 92, 246, 0.05) 0%, rgba(168, 85, 247, 0.05) 100%);
  border-bottom: 1px solid rgba(139, 92, 246, 0.1);
}

.upload-header-icon {
  font-size: 40px;
  color: #8b5cf6;
  margin-bottom: 15px;
}

.upload-header h3 {
  font-size: 20px;
  font-weight: 600;
  color: #1e293b;
  margin: 0 0 8px 0;
}

.upload-header p {
  font-size: 14px;
  color: #64748b;
  margin: 0;
}

.upload-area {
  padding: 30px;
}

.custom-upload {
  width: 100%;
  min-height: 300px;
}

.upload-content {
  padding: 50px 30px;
  text-align: center;
  color: #64748b;
  transition: all 0.3s ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.upload-content:hover {
  color: #8b5cf6;
  transform: scale(1.02);
}

.upload-icon {
  font-size: 60px;
  margin-bottom: 20px;
  color: #8b5cf6;
}

.upload-title {
  font-size: 18px;
  font-weight: 600;
  margin: 0 0 5px 0;
  color: #1e293b;
}

.upload-subtitle {
  font-size: 16px;
  margin: 0 0 15px 0;
  color: #475569;
}

.upload-hint {
  font-size: 14px;
  margin: 0;
  opacity: 0.8;
}

.upload-tips {
  padding: 0 30px 30px 30px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.tip-item {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
  color: #475569;
}

.tip-icon {
  font-size: 16px;
}

.tip-icon.success {
  color: #10b981;
}

.debug-panel {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 12px;
  padding: 20px;
  margin-top: 20px;
  border: 1px solid #e5e7eb;
  grid-column: 1 / -1;
}

.debug-panel h4 {
  margin: 0 0 15px 0;
  color: #374151;
  font-size: 16px;
}

.debug-item {
  margin: 8px 0;
  font-size: 14px;
  color: #6b7280;
  word-break: break-all;
}

.debug-btn {
  background: #6366f1;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  margin: 5px 10px 5px 0;
}

.debug-btn:hover {
  background: #4f46e5;
}

:deep(.el-upload-dragger) {
  width: 100%;
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  border: 2px dashed #c4b5fd !important;
  background: rgba(139, 92, 246, 0.02) !important;
}

:deep(.el-upload-dragger:hover) {
  border-color: #8b5cf6 !important;
  background: rgba(139, 92, 246, 0.05) !important;
}

/* 动画定义 */
@keyframes fadeInLeft {
  from {
    opacity: 0;
    transform: translateX(-30px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes fadeInRight {
  from {
    opacity: 0;
    transform: translateX(30px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes float {
  0%, 100% {
    transform: translateY(0px) rotate(0deg);
  }
  33% {
    transform: translateY(-20px) rotate(120deg);
  }
  66% {
    transform: translateY(10px) rotate(240deg);
  }
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.8;
    transform: scale(1.05);
  }
}

@keyframes blink {
  0%, 50% {
    opacity: 1;
  }
  51%, 100% {
    opacity: 0.3;
  }
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .main-content {
    grid-template-columns: 1fr;
    gap: 30px;
  }

  .page-content {
    padding: 20px;
  }
}

@media (max-width: 768px) {
  .card-header,
  .upload-header {
    padding: 20px;
  }

  .image-display-area,
  .upload-area {
    padding: 20px;
  }

  .detection-image,
  .detection-image-native,
  .image-placeholder,
  .empty-state {
    height: 400px;
  }

  .image-container {
    min-height: 400px;
  }
}
</style>