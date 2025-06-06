<template>
  <div class="login-container">
    <!-- 背景装饰 -->
    <div class="background-decorations">
      <div class="decoration-circle circle-1"></div>
      <div class="decoration-circle circle-2"></div>
      <div class="decoration-circle circle-3"></div>
      <div class="decoration-circle circle-4"></div>
    </div>

    <!-- 主要内容 -->
    <div class="login-content">
      <!-- 标题区域 -->
      <div class="login-header">
        <div class="logo-section">
          <div class="logo-icon">
            <svg class="icon-avatar" viewBox="0 0 1024 1024" width="40" height="40">
              <path d="M512 74.666667C270.933333 74.666667 74.666667 270.933333 74.666667 512S270.933333 949.333333 512 949.333333 949.333333 753.066667 949.333333 512 753.066667 74.666667 512 74.666667z m0 160c93.866667 0 170.666667 76.8 170.666667 170.666666s-76.8 170.666667-170.666667 170.666667-170.666667-76.8-170.666667-170.666667S418.133333 234.666667 512 234.666667z m0 650.666666c-108.8 0-206.933333-42.666667-285.866667-117.333333 14.933333-98.133333 96-174.933333 200.533334-174.933333 21.333333 8.533333 44.8 12.8 68.266666 12.8s46.933333-4.266667 68.266667-12.8c104.533333 0 185.6 76.8 200.533333 174.933333C718.933333 842.666667 620.8 885.333333 512 885.333333z" fill="currentColor"/>
            </svg>
          </div>
          <div class="logo-text">
            <h1 class="main-title">智能登录系统</h1>
            <p class="sub-title">Intelligent Login System</p>
          </div>
        </div>
        <div class="header-decoration">
          <div class="decoration-dot"></div>
          <div class="decoration-line"></div>
          <div class="decoration-dot"></div>
        </div>
      </div>

      <!-- 登录卡片 -->
      <div class="login-card">
        <!-- 登录方式切换 -->
        <div class="login-tabs">
          <div class="tab-container">
            <button 
              class="tab-button" 
              :class="{ active: loginMode === 'face' }"
              @click="switchLoginMode('face')"
            >
              <svg class="tab-icon" viewBox="0 0 1024 1024" width="18" height="18">
                <path d="M512 74.666667C270.933333 74.666667 74.666667 270.933333 74.666667 512S270.933333 949.333333 512 949.333333 949.333333 753.066667 949.333333 512 753.066667 74.666667 512 74.666667z m0 160c93.866667 0 170.666667 76.8 170.666667 170.666666s-76.8 170.666667-170.666667 170.666667-170.666667-76.8-170.666667-170.666667S418.133333 234.666667 512 234.666667z m0 650.666666c-108.8 0-206.933333-42.666667-285.866667-117.333333 14.933333-98.133333 96-174.933333 200.533334-174.933333 21.333333 8.533333 44.8 12.8 68.266666 12.8s46.933333-4.266667 68.266667-12.8c104.533333 0 185.6 76.8 200.533333 174.933333C718.933333 842.666667 620.8 885.333333 512 885.333333z" fill="currentColor"/>
              </svg>
              人脸识别
            </button>
            <button 
              class="tab-button" 
              :class="{ active: loginMode === 'password' }"
              @click="switchLoginMode('password')"
            >
              <svg class="tab-icon" viewBox="0 0 1024 1024" width="18" height="18">
                <path d="M832 464h-68V240c0-70.7-57.3-128-128-128H388c-70.7 0-128 57.3-128 128v224h-68c-17.7 0-32 14.3-32 32v384c0 17.7 14.3 32 32 32h640c17.7 0 32-14.3 32-32V496c0-17.7-14.3-32-32-32zM332 240c0-30.9 25.1-56 56-56h248c30.9 0 56 25.1 56 56v224H332V240z m460 600H232V536h560v304z" fill="currentColor"/>
              </svg>
              账号密码
            </button>
          </div>
          <div class="tab-indicator" :style="getTabIndicatorStyle()"></div>
        </div>

        <!-- 登录状态指示 -->
        <div class="login-status">
          <div class="status-indicator">
            <div class="status-dot" :class="getStatusClass()"></div>
            <span class="status-text">{{ getStatusText() }}</span>
          </div>
        </div>

        <!-- 人脸识别登录区域 -->
        <div v-if="loginMode === 'face'" class="face-login-section">
          <!-- 摄像头区域 -->
          <div class="camera-section">
            <div class="camera-container">
              <!-- 视频预览 -->
              <div class="video-container">
                <video
                  id="videoCamera"
                  ref="videoRef"
                  :width="videoWidth"
                  :height="videoHeight"
                  autoplay
                  playsinline
                  muted
                  class="camera-video"
                ></video>
                <canvas
                  id="canvasCamera"
                  ref="canvasRef"
                  :width="videoWidth"
                  :height="videoHeight"
                  style="display: none;"
                ></canvas>
                
                <!-- 人脸识别框架和特效 -->
                <div class="camera-overlay">
                  <div class="camera-hint">
                    <p>请将脸部对准框内</p>
                    <p>保持良好光线，正视摄像头</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- 控制按钮区域 -->
          <div class="face-control-section">
            <div class="button-group">
              <button
                class="control-button recognition-button"
                :class="{ 'loading': faceImgState }"
                @click="faceVef"
                :disabled="faceImgState"
              >
                <svg v-if="faceImgState" class="loading-icon" viewBox="0 0 1024 1024" width="16" height="16">
                  <path d="M512 1024c-69.1 0-136.2-13.5-199.3-40.2C251.7 958.5 197.8 921.3 152.6 876.1c-45.2-45.2-82.4-99.1-107.7-160.2C17.5 652.8 4 585.7 4 516.6s13.5-136.2 40.2-199.3C69.5 256.2 106.7 202.3 151.9 157.1c45.2-45.2 99.1-82.4 160.2-107.7C375.2 22.1 442.3 8.6 511.4 8.6s136.2 13.5 199.3 40.2c61.1 25.3 115 62.5 160.2 107.7 45.2 45.2 82.4 99.1 107.7 160.2 26.7 63.1 40.2 130.2 40.2 199.3s-13.5 136.2-40.2 199.3c-25.3 61.1-62.5 115-107.7 160.2-45.2 45.2-99.1 82.4-160.2 107.7-63.1 26.7-130.2 40.2-199.3 40.2zM511.4 77.1c-60.8 0-119.5 11.9-174.4 35.4-53.1 22.7-100.8 55.1-141.9 96.2-41.1 41.1-73.5 88.8-96.2 141.9-23.5 54.9-35.4 113.6-35.4 174.4s11.9 119.5 35.4 174.4c22.7 53.1 55.1 100.8 96.2 141.9 41.1 41.1 88.8 73.5 141.9 96.2 54.9 23.5 113.6 35.4 174.4 35.4s119.5-11.9 174.4-35.4c53.1-22.7 100.8-55.1 141.9-96.2 41.1-41.1 73.5-88.8 96.2-141.9 23.5-54.9 35.4-113.6 35.4-174.4s-11.9-119.5-35.4-174.4c-22.7-53.1-55.1-100.8-96.2-141.9-41.1-41.1-88.8-73.5-141.9-96.2-54.9-23.5-113.6-35.4-174.4-35.4z" fill="currentColor"/>
                </svg>
                <svg v-else class="camera-icon" viewBox="0 0 1024 1024" width="16" height="16">
                  <path d="M896 320H768l-25.6-76.8c-12.8-38.4-48-64-89.6-64H371.2c-41.6 0-76.8 25.6-89.6 64L256 320H128c-70.4 0-128 57.6-128 128v320c0 70.4 57.6 128 128 128h768c70.4 0 128-57.6 128-128V448c0-70.4-57.6-128-128-128z m-384 384c-105.6 0-192-86.4-192-192s86.4-192 192-192 192 86.4 192 192-86.4 192-192 192z m0-128c35.2 0 64-28.8 64-64s-28.8-64-64-64-64 28.8-64 64 28.8 64 64 64z" fill="currentColor"/>
                </svg>
                <span>{{ faceImgState ? '正在拍照识别...' : '拍照识别' }}</span>
              </button>
            </div>

            <!-- 消息显示区域 -->
            <div v-if="msg" class="message-display" :class="getMessageClass()">
              <div class="message-icon">
                <svg v-if="faceImgState" class="loading-icon" viewBox="0 0 1024 1024" width="18" height="18">
                  <path d="M512 1024c-69.1 0-136.2-13.5-199.3-40.2C251.7 958.5 197.8 921.3 152.6 876.1c-45.2-45.2-82.4-99.1-107.7-160.2C17.5 652.8 4 585.7 4 516.6s13.5-136.2 40.2-199.3C69.5 256.2 106.7 202.3 151.9 157.1c45.2-45.2 99.1-82.4 160.2-107.7C375.2 22.1 442.3 8.6 511.4 8.6s136.2 13.5 199.3 40.2c61.1 25.3 115 62.5 160.2 107.7 45.2 45.2 82.4 99.1 107.7 160.2 26.7 63.1 40.2 130.2 40.2 199.3s-13.5 136.2-40.2 199.3c-25.3 61.1-62.5 115-107.7 160.2-45.2 45.2-99.1 82.4-160.2 107.7-63.1 26.7-130.2 40.2-199.3 40.2zM511.4 77.1c-60.8 0-119.5 11.9-174.4 35.4-53.1 22.7-100.8 55.1-141.9 96.2-41.1 41.1-73.5 88.8-96.2 141.9-23.5 54.9-35.4 113.6-35.4 174.4s11.9 119.5 35.4 174.4c22.7 53.1 55.1 100.8 96.2 141.9 41.1 41.1 88.8 73.5 141.9 96.2 54.9 23.5 113.6 35.4 174.4 35.4s119.5-11.9 174.4-35.4c53.1-22.7 100.8-55.1 141.9-96.2 41.1-41.1 73.5-88.8 96.2-141.9 23.5-54.9 35.4-113.6 35.4-174.4s-11.9-119.5-35.4-174.4c-22.7-53.1-55.1-100.8-96.2-141.9-41.1-41.1-88.8-73.5-141.9-96.2-54.9-23.5-113.6-35.4-174.4-35.4z" fill="currentColor"/>
                </svg>
                <svg v-else-if="isSuccess" class="success-icon" viewBox="0 0 1024 1024" width="18" height="18">
                  <path d="M512 0C229.232 0 0 229.232 0 512c0 282.784 229.232 512 512 512 282.784 0 512-229.216 512-512C1024 229.232 794.784 0 512 0z m238.4 390.4L448 692.8c-12.8 12.8-35.2 12.8-48 0L246.4 539.2c-12.8-12.8-12.8-35.2 0-48 12.8-12.8 35.2-12.8 48 0L424 620.8l278.4-278.4c12.8-12.8 35.2-12.8 48 0 12.8 12.8 12.8 35.2 0 48z" fill="currentColor"/>
                </svg>
                <svg v-else class="error-icon" viewBox="0 0 1024 1024" width="18" height="18">
                  <path d="M512 0C229.232 0 0 229.232 0 512c0 282.784 229.232 512 512 512 282.784 0 512-229.216 512-512C1024 229.232 794.784 0 512 0z m195.2 659.2c12.8 12.8 12.8 35.2 0 48-6.4 6.4-14.4 9.6-24 9.6s-17.6-3.2-24-9.6L512 560l-147.2 147.2c-6.4 6.4-14.4 9.6-24 9.6s-17.6-3.2-24-9.6c-12.8-12.8-12.8-35.2 0-48L464 512 316.8 364.8c-12.8-12.8-12.8-35.2 0-48 12.8-12.8 35.2-12.8 48 0L512 464l147.2-147.2c12.8-12.8 35.2-12.8 48 0 12.8 12.8 12.8 35.2 0 48L560 512l147.2 147.2z" fill="currentColor"/>
                </svg>
              </div>
              <div class="message-content">
                <div class="server-msg">{{ msg }}</div>
                <div class="welcome-msg">Welcome to intelligent login</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 账号密码登录区域 -->
        <div v-if="loginMode === 'password'" class="password-login-section">
          <form @submit.prevent="passwordLogin" class="login-form">
            <div class="form-group">
              <label class="form-label">
                <svg class="label-icon" viewBox="0 0 1024 1024" width="16" height="16">
                  <path d="M512 74.666667C270.933333 74.666667 74.666667 270.933333 74.666667 512S270.933333 949.333333 512 949.333333 949.333333 753.066667 949.333333 512 753.066667 74.666667 512 74.666667z m0 160c93.866667 0 170.666667 76.8 170.666667 170.666666s-76.8 170.666667-170.666667 170.666667-170.666667-76.8-170.666667-170.666667S418.133333 234.666667 512 234.666667z m0 650.666666c-108.8 0-206.933333-42.666667-285.866667-117.333333 14.933333-98.133333 96-174.933333 200.533334-174.933333 21.333333 8.533333 44.8 12.8 68.266666 12.8s46.933333-4.266667 68.266667-12.8c104.533333 0 185.6 76.8 200.533333 174.933333C718.933333 842.666667 620.8 885.333333 512 885.333333z" fill="currentColor"/>
                </svg>
                用户名
              </label>
              <div class="input-wrapper">
                <input
                  type="text"
                  v-model="loginForm.username"
                  class="form-input"
                  placeholder="请输入用户名"
                  required
                  :disabled="passwordLoginState"
                />
                <svg class="input-icon" viewBox="0 0 1024 1024" width="18" height="18">
                  <path d="M512 74.666667C270.933333 74.666667 74.666667 270.933333 74.666667 512S270.933333 949.333333 512 949.333333 949.333333 753.066667 949.333333 512 753.066667 74.666667 512 74.666667z m0 160c93.866667 0 170.666667 76.8 170.666667 170.666666s-76.8 170.666667-170.666667 170.666667-170.666667-76.8-170.666667-170.666667S418.133333 234.666667 512 234.666667z m0 650.666666c-108.8 0-206.933333-42.666667-285.866667-117.333333 14.933333-98.133333 96-174.933333 200.533334-174.933333 21.333333 8.533333 44.8 12.8 68.266666 12.8s46.933333-4.266667 68.266667-12.8c104.533333 0 185.6 76.8 200.533333 174.933333C718.933333 842.666667 620.8 885.333333 512 885.333333z" fill="currentColor"/>
                </svg>
              </div>
            </div>

            <div class="form-group">
              <label class="form-label">
                <svg class="label-icon" viewBox="0 0 1024 1024" width="16" height="16">
                  <path d="M832 464h-68V240c0-70.7-57.3-128-128-128H388c-70.7 0-128 57.3-128 128v224h-68c-17.7 0-32 14.3-32 32v384c0 17.7 14.3 32 32 32h640c17.7 0 32-14.3 32-32V496c0-17.7-14.3-32-32-32zM332 240c0-30.9 25.1-56 56-56h248c30.9 0 56 25.1 56 56v224H332V240z m460 600H232V536h560v304z" fill="currentColor"/>
                </svg>
                密码
              </label>
              <div class="input-wrapper">
                <input
                  :type="showPassword ? 'text' : 'password'"
                  v-model="loginForm.password"
                  class="form-input"
                  placeholder="请输入密码"
                  required
                  :disabled="passwordLoginState"
                />
                <button
                  type="button"
                  class="password-toggle"
                  @click="showPassword = !showPassword"
                  :disabled="passwordLoginState"
                >
                  <svg v-if="showPassword" class="eye-icon" viewBox="0 0 1024 1024" width="18" height="18">
                    <path d="M942.2 486.2C847.4 286.5 704.1 186 512 186c-192.2 0-335.4 100.5-430.2 300.3a60.3 60.3 0 0 0 0 51.5C176.6 737.5 319.9 838 512 838c192.2 0 335.4-100.5 430.2-300.3 7.7-16.2 7.7-35 0-51.5zM512 766c-161.3 0-279.4-81.8-362.7-254C232.6 339.8 350.7 258 512 258c161.3 0 279.4 81.8 362.7 254C791.5 684.2 673.4 766 512 766z"/>
                    <path d="M508 336c-97.2 0-176 78.8-176 176s78.8 176 176 176 176-78.8 176-176-78.8-176-176-176z m0 288c-61.9 0-112-50.1-112-112s50.1-112 112-112 112 50.1 112 112-50.1 112-112 112z"/>
                  </svg>
                  <svg v-else class="eye-icon" viewBox="0 0 1024 1024" width="18" height="18">
                    <path d="M942.2 486.2C847.4 286.5 704.1 186 512 186c-192.2 0-335.4 100.5-430.2 300.3a60.3 60.3 0 0 0 0 51.5C176.6 737.5 319.9 838 512 838c192.2 0 335.4-100.5 430.2-300.3 7.7-16.2 7.7-35 0-51.5zM512 766c-161.3 0-279.4-81.8-362.7-254C232.6 339.8 350.7 258 512 258c161.3 0 279.4 81.8 362.7 254C791.5 684.2 673.4 766 512 766z"/>
                    <path d="M508 336c-97.2 0-176 78.8-176 176s78.8 176 176 176 176-78.8 176-176-78.8-176-176-176z m0 288c-61.9 0-112-50.1-112-112s50.1-112 112-112 112 50.1 112 112-50.1 112-112 112z"/>
                    <path d="M942 196h-84l-98 98-84-84-186 186-186-186-84 84-98-98H196c-17.7 0-32 14.3-32 32s14.3 32 32 32h84l98 98 84 84 186-186 186 186 84-84 98 98h84c17.7 0 32-14.3 32-32s-14.3-32-32-32z"/>
                  </svg>
                </button>
              </div>
            </div>

            <div class="form-actions">
              <button
                type="submit"
                class="login-button"
                :class="{ 'loading': passwordLoginState }"
                :disabled="passwordLoginState"
              >
                <svg v-if="passwordLoginState" class="loading-icon" viewBox="0 0 1024 1024" width="16" height="16">
                  <path d="M512 1024c-69.1 0-136.2-13.5-199.3-40.2C251.7 958.5 197.8 921.3 152.6 876.1c-45.2-45.2-82.4-99.1-107.7-160.2C17.5 652.8 4 585.7 4 516.6s13.5-136.2 40.2-199.3C69.5 256.2 106.7 202.3 151.9 157.1c45.2-45.2 99.1-82.4 160.2-107.7C375.2 22.1 442.3 8.6 511.4 8.6s136.2 13.5 199.3 40.2c61.1 25.3 115 62.5 160.2 107.7 45.2 45.2 82.4 99.1 107.7 160.2 26.7 63.1 40.2 130.2 40.2 199.3s-13.5 136.2-40.2 199.3c-25.3 61.1-62.5 115-107.7 160.2-45.2 45.2-99.1 82.4-160.2 107.7-63.1 26.7-130.2 40.2-199.3 40.2zM511.4 77.1c-60.8 0-119.5 11.9-174.4 35.4-53.1 22.7-100.8 55.1-141.9 96.2-41.1 41.1-73.5 88.8-96.2 141.9-23.5 54.9-35.4 113.6-35.4 174.4s11.9 119.5 35.4 174.4c22.7 53.1 55.1 100.8 96.2 141.9 41.1 41.1 88.8 73.5 141.9 96.2 54.9 23.5 113.6 35.4 174.4 35.4s119.5-11.9 174.4-35.4c53.1-22.7 100.8-55.1 141.9-96.2 41.1-41.1 73.5-88.8 96.2-141.9 23.5-54.9 35.4-113.6 35.4-174.4s-11.9-119.5-35.4-174.4c-22.7-53.1-55.1-100.8-96.2-141.9-41.1-41.1-88.8-73.5-141.9-96.2-54.9-23.5-113.6-35.4-174.4-35.4z" fill="currentColor"/>
                </svg>
                <svg v-else class="login-icon" viewBox="0 0 1024 1024" width="16" height="16">
                  <path d="M521.7 82c-152.5-.4-286.7 78.5-363.4 197.7-3.4 5.3.4 12.3 6.7 12.3h70.3c4.8 0 9.3-2.1 12.3-5.8 7-8.5 14.5-16.7 22.4-24.5 32.6-32.5 70.5-58.1 112.7-75.9 43.6-18.4 90-27.8 137.9-27.8 47.9 0 94.3 9.3 137.9 27.8 42.2 17.8 80.1 43.4 112.7 75.9 32.6 32.5 58.1 70.4 76 112.5C865.7 417.8 875 464.1 875 512s-9.3 94.2-27.8 137.8c-17.8 42.1-43.4 80-76 112.5s-70.5 58.1-112.7 75.9A352.8 352.8 0 0 1 520.6 866c-47.9 0-94.3-9.4-137.9-27.8A353.84 353.84 0 0 1 270 762.3c-7.9-7.9-15.3-16.1-22.4-24.5-3-3.7-7.6-5.8-12.3-5.8H165c-6.3 0-10.2 7-6.7 12.3C234.9 863.2 368.5 943.9 520.8 944c236.2.3 428.2-190.8 428.2-427S757.1 81.6 521.7 82zM395.02 624v-76h-314c-4.4 0-8-3.6-8-8v-56c0-4.4 3.6-8 8-8h314v-76c0-6.7 7.8-10.5 13-6.3l141.9 112a8 8 0 0 1 0 12.6l-141.9 112c-5.2 4.1-13 .4-13-6.3z" fill="currentColor"/>
                </svg>
                <span>{{ passwordLoginState ? '登录中...' : '登录' }}</span>
              </button>
            </div>
          </form>

          <!-- 密码登录消息显示 -->
          <div v-if="passwordMsg" class="message-display" :class="getPasswordMessageClass()">
            <div class="message-icon">
              <svg v-if="passwordLoginState" class="loading-icon" viewBox="0 0 1024 1024" width="18" height="18">
                <path d="M512 1024c-69.1 0-136.2-13.5-199.3-40.2C251.7 958.5 197.8 921.3 152.6 876.1c-45.2-45.2-82.4-99.1-107.7-160.2C17.5 652.8 4 585.7 4 516.6s13.5-136.2 40.2-199.3C69.5 256.2 106.7 202.3 151.9 157.1c45.2-45.2 99.1-82.4 160.2-107.7C375.2 22.1 442.3 8.6 511.4 8.6s136.2 13.5 199.3 40.2c61.1 25.3 115 62.5 160.2 107.7 45.2 45.2 82.4 99.1 107.7 160.2 26.7 63.1 40.2 130.2 40.2 199.3s-13.5 136.2-40.2 199.3c-25.3 61.1-62.5 115-107.7 160.2-45.2 45.2-99.1 82.4-160.2 107.7-63.1 26.7-130.2 40.2-199.3 40.2zM511.4 77.1c-60.8 0-119.5 11.9-174.4 35.4-53.1 22.7-100.8 55.1-141.9 96.2-41.1 41.1-73.5 88.8-96.2 141.9-23.5 54.9-35.4 113.6-35.4 174.4s11.9 119.5 35.4 174.4c22.7 53.1 55.1 100.8 96.2 141.9 41.1 41.1 88.8 73.5 141.9 96.2 54.9 23.5 113.6 35.4 174.4 35.4s119.5-11.9 174.4-35.4c53.1-22.7 100.8-55.1 141.9-96.2 41.1-41.1 73.5-88.8 96.2-141.9 23.5-54.9 35.4-113.6 35.4-174.4s-11.9-119.5-35.4-174.4c-22.7-53.1-55.1-100.8-96.2-141.9-41.1-41.1-88.8-73.5-141.9-96.2-54.9-23.5-113.6-35.4-174.4-35.4z" fill="currentColor"/>
              </svg>
              <svg v-else-if="passwordSuccess" class="success-icon" viewBox="0 0 1024 1024" width="18" height="18">
                <path d="M512 0C229.232 0 0 229.232 0 512c0 282.784 229.232 512 512 512 282.784 0 512-229.216 512-512C1024 229.232 794.784 0 512 0z m238.4 390.4L448 692.8c-12.8 12.8-35.2 12.8-48 0L246.4 539.2c-12.8-12.8-12.8-35.2 0-48 12.8-12.8 35.2-12.8 48 0L424 620.8l278.4-278.4c12.8-12.8 35.2-12.8 48 0 12.8 12.8 12.8 35.2 0 48z" fill="currentColor"/>
              </svg>
              <svg v-else class="error-icon" viewBox="0 0 1024 1024" width="18" height="18">
                <path d="M512 0C229.232 0 0 229.232 0 512c0 282.784 229.232 512 512 512 282.784 0 512-229.216 512-512C1024 229.232 794.784 0 512 0z m195.2 659.2c12.8 12.8 12.8 35.2 0 48-6.4 6.4-14.4 9.6-24 9.6s-17.6-3.2-24-9.6L512 560l-147.2 147.2c-6.4 6.4-14.4 9.6-24 9.6s-17.6-3.2-24-9.6c-12.8-12.8-12.8-35.2 0-48L464 512 316.8 364.8c-12.8-12.8-12.8-35.2 0-48 12.8-12.8 35.2-12.8 48 0L512 464l147.2-147.2c12.8-12.8 35.2-12.8 48 0 12.8 12.8 12.8 35.2 0 48L560 512l147.2 147.2z" fill="currentColor"/>
              </svg>
            </div>
            <div class="message-content">
              <div class="server-msg">{{ passwordMsg }}</div>
              <div class="welcome-msg">Welcome to intelligent login</div>
            </div>
          </div>
        </div>

        <!-- 使用提示 -->
        <div class="usage-tips">
          <div v-if="loginMode === 'face'" class="tip-item">
            <svg class="tip-icon" viewBox="0 0 1024 1024" width="16" height="16">
              <path d="M512 160c194.4 0 352 157.6 352 352s-157.6 352-352 352S160 706.4 160 512s157.6-352 352-352m0-64C282.256 96 96 282.256 96 512c0 229.76 186.256 416 416 416s416-186.24 416-416c0-229.744-186.256-416-416-416z m0 224c35.344 0 64 28.656 64 64s-28.656 64-64 64-64-28.656-64-64 28.656-64 64-64z m0 160c17.696 0 32 14.336 32 32v192c0 17.664-14.304 32-32 32s-32-14.336-32-32V512c0-17.664 14.304-32 32-32z" fill="currentColor"/>
            </svg>
            <span>确保光线充足，正面面向摄像头</span>
          </div>
          <div v-if="loginMode === 'face'" class="tip-item">
            <svg class="tip-icon" viewBox="0 0 1024 1024" width="16" height="16">
              <path d="M896 320H768l-25.6-76.8c-12.8-38.4-48-64-89.6-64H371.2c-41.6 0-76.8 25.6-89.6 64L256 320H128c-70.4 0-128 57.6-128 128v320c0 70.4 57.6 128 128 128h768c70.4 0 128-57.6 128-128V448c0-70.4-57.6-128-128-128z m-384 384c-105.6 0-192-86.4-192-192s86.4-192 192-192 192 86.4 192 192-86.4 192-192 192z m0-128c35.2 0 64-28.8 64-64s-28.8-64-64-64-64 28.8-64 64 28.8 64 64 64z" fill="currentColor"/>
            </svg>
            <span>点击按钮拍照，系统将自动识别身份</span>
          </div>
          <div v-if="loginMode === 'face'" class="tip-item">
            <svg class="tip-icon" viewBox="0 0 1024 1024" width="16" height="16">
              <path d="M832 464h-68V240c0-70.7-57.3-128-128-128H388c-70.7 0-128 57.3-128 128v224h-68c-17.7 0-32 14.3-32 32v384c0 17.7 14.3 32 32 32h640c17.7 0 32-14.3 32-32V496c0-17.7-14.3-32-32-32zM332 240c0-30.9 25.1-56 56-56h248c30.9 0 56 25.1 56 56v224H332V240z m460 600H232V536h560v304z" fill="currentColor"/>
            </svg>
            <span>您的面部数据将被安全处理</span>
          </div>
          
          <div v-if="loginMode === 'password'" class="tip-item">
            <svg class="tip-icon" viewBox="0 0 1024 1024" width="16" height="16">
              <path d="M832 464h-68V240c0-70.7-57.3-128-128-128H388c-70.7 0-128 57.3-128 128v224h-68c-17.7 0-32 14.3-32 32v384c0 17.7 14.3 32 32 32h640c17.7 0 32-14.3 32-32V496c0-17.7-14.3-32-32-32zM332 240c0-30.9 25.1-56 56-56h248c30.9 0 56 25.1 56 56v224H332V240z m460 600H232V536h560v304z" fill="currentColor"/>
            </svg>
            <span>请使用您的注册账号和密码登录</span>
          </div>
          <div v-if="loginMode === 'password'" class="tip-item">
            <svg class="tip-icon" viewBox="0 0 1024 1024" width="16" height="16">
              <path d="M512 160c194.4 0 352 157.6 352 352s-157.6 352-352 352S160 706.4 160 512s157.6-352 352-352m0-64C282.256 96 96 282.256 96 512c0 229.76 186.256 416 416 416s416-186.24 416-416c0-229.744-186.256-416-416-416z m0 224c35.344 0 64 28.656 64 64s-28.656 64-64 64-64-28.656-64-64 28.656-64 64-64z m0 160c17.696 0 32 14.336 32 32v192c0 17.664-14.304 32-32 32s-32-14.336-32-32V512c0-17.664 14.304-32 32-32z" fill="currentColor"/>
            </svg>
            <span>忘记密码请联系系统管理员</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import $camera from '../camera/index.js'
import { useAuthStore } from '@/stores/auth'
import { useRouter } from 'vue-router'

export default {
  name: 'Login',
  setup() {
    const authStore = useAuthStore()
    const router = useRouter()
    
    return {
      authStore,
      router
    }
  },
  data() {
    return {
      // 登录模式：'face' 或 'password'
      loginMode: 'face',
      
      // 人脸识别相关
      videoWidth: 200,
      videoHeight: 200,
      msg: '',
      faceImgState: false,
      faceOption: {},
      isSuccess: false,
      
      // 账号密码登录相关
      loginForm: {
        username: '',
        password: ''
      },
      passwordMsg: '',
      passwordLoginState: false,
      passwordSuccess: false,
      showPassword: false
    }
  },
  mounted() {
    // 检查是否已登录
    if (this.authStore.isLoggedIn) {
      this.router.push('/dashboard/faceDetect')
      return
    }
    
    // 默认初始化人脸识别
    this.initCamera()
  },
  beforeUnmount() {
    // 组件销毁时清理摄像头资源
    this.cleanupCamera()
  },
  methods: {
    // 处理登录成功
    handleLoginSuccess(token, userInfo) {
      console.log('登录成功处理:', { token, userInfo })
      
      // 使用认证store保存登录状态
      this.authStore.login(token, userInfo)
      
      // 清理摄像头资源
      this.cleanupCamera()
      
      // 跳转到主界面
      setTimeout(() => {
        this.router.push('/dashboard/faceDetect')
      }, 1500)
    },
    
    // 切换登录模式
    switchLoginMode(mode) {
      if (this.loginMode === mode) return
      
      this.loginMode = mode
      this.clearMessages()
      
      if (mode === 'face') {
        // 切换到人脸识别，初始化摄像头
        this.$nextTick(() => {
          this.initCamera()
        })
      } else {
        // 切换到密码登录，清理摄像头资源
        this.cleanupCamera()
      }
    },
    
    // 获取tab指示器样式
    getTabIndicatorStyle() {
      return {
        left: this.loginMode === 'face' ? '0%' : '50%'
      }
    },
    
    // 清理所有消息
    clearMessages() {
      this.msg = ''
      this.passwordMsg = ''
      this.isSuccess = false
      this.passwordSuccess = false
    },

    // 初始化摄像头
    initCamera() {
      try {
        this.faceOption = $camera.getCamera({
          videoWidth: this.videoWidth,
          videoHeight: this.videoHeight,
          thisCancas: null,
          thisContext: null,
          thisVideo: null,
          canvasId: 'canvasCamera',
          videoId: 'videoCamera'
        })
        
        // 设置初始消息
        setTimeout(() => {
          this.msg = '摄像头已就绪，可以开始拍照识别'
          this.isSuccess = true
          
          // 3秒后清除消息
          setTimeout(() => {
            this.msg = ''
          }, 3000)
        }, 1000)
      } catch (error) {
        console.error('摄像头初始化失败:', error)
        this.msg = '摄像头初始化失败，请检查设备权限'
        this.isSuccess = false
      }
    },
    
    // 清理摄像头资源
    cleanupCamera() {
      if (this.faceOption.thisVideo && this.faceOption.thisVideo.srcObject) {
        this.faceOption.thisVideo.srcObject.getTracks().forEach(track => track.stop())
        this.faceOption = {}
      }
    },

    // 人脸识别登录 - 拍照识别流程
    faceVef() {
      if (this.faceImgState) {
        return
      }
      
      // 第一步：拍照获取图像数据
      let imageBase = $camera.draw(this.faceOption)
      
      if (imageBase === '' || imageBase === null || imageBase === undefined) {
        this.msg = '拍照失败，请重试'
        this.faceImgState = false
        this.isSuccess = false
        return
      }

      // 第二步：设置识别状态
      this.faceImgState = true
      this.isSuccess = false
      this.msg = '照片已拍摄，正在传输到服务器进行身份识别...'
      
      // 第三步：将拍摄的照片传给后端进行识别
      this.$http.post("/face/vef", { 
        imageBase: imageBase,
        timestamp: Date.now() // 添加时间戳
      }).then(res => {
        console.log('人脸识别结果:', res)
        this.faceImgState = false
        
        // 第四步：根据后端识别结果处理登录
        if (res.data.code === 200) {
          // 识别成功
          this.msg = res.data.msg || '身份识别成功，正在登录...'
          this.isSuccess = true
          
          // 构造用户信息
          const userInfo = {
            username: res.data.name || 'face_user',
            name: res.data.name || 'Face User',
            loginType: 'face'
          }
          
          // 处理登录成功
          this.handleLoginSuccess(res.data.token, userInfo)
          
        } else if (res.data.code === 201) {
          // 识别失败，但照片有效
          this.msg = res.data.msg || '身份识别失败，未找到匹配的用户'
          this.isSuccess = false
        } else {
          // 其他错误
          this.msg = res.data.msg || '识别过程中出现错误，请重试'
          this.isSuccess = false
        }
      }).catch(error => {
        console.error('人脸识别请求失败:', error)
        this.faceImgState = false
        this.msg = '网络连接失败，请检查网络后重试'
        this.isSuccess = false
      })
    },

    // 账号密码登录
    passwordLogin() {
      if (this.passwordLoginState) {
        return
      }
      
      // 表单验证
      if (!this.loginForm.username.trim()) {
        this.passwordMsg = '请输入用户名'
        this.passwordSuccess = false
        return
      }
      
      if (!this.loginForm.password.trim()) {
        this.passwordMsg = '请输入密码'
        this.passwordSuccess = false
        return
      }

      this.passwordLoginState = true
      this.passwordSuccess = false
      this.passwordMsg = '正在验证账号密码...'
      
      // 发送登录请求
      this.$http.post("/user/login", {
        username: this.loginForm.username.trim(),
        password: this.loginForm.password.trim()
      }).then(res => {
        console.log('密码登录结果:', res)
        this.passwordLoginState = false
        
        if (res.data.code === 200) {
          // 登录成功
          this.passwordMsg = res.data.msg || '登录成功！'
          this.passwordSuccess = true
          
          // 构造用户信息
          const userInfo = {
            username: res.data.username || this.loginForm.username,
            name: res.data.name || res.data.username || this.loginForm.username,
            loginType: 'password'
          }
          
          // 处理登录成功
          this.handleLoginSuccess(res.data.token, userInfo)
          
        } else {
          // 登录失败
          this.passwordMsg = res.data.msg || '用户名或密码错误'
          this.passwordSuccess = false
        }
      }).catch(error => {
        console.error('密码登录请求失败:', error)
        this.passwordLoginState = false
        this.passwordMsg = '网络连接失败，请检查网络后重试'
        this.passwordSuccess = false
      })
    },

    // 获取状态样式类
    getStatusClass() {
      if (this.loginMode === 'face') {
        if (this.faceImgState) return 'processing'
        if (this.faceOption.thisVideo) return 'active'
        return 'idle'
      } else {
        if (this.passwordLoginState) return 'processing'
        return 'active'
      }
    },

    // 获取状态文本
    getStatusText() {
      if (this.loginMode === 'face') {
        if (this.faceImgState) return '正在识别中...'
        if (this.faceOption.thisVideo) return '摄像头已就绪'
        return '准备开始识别'
      } else {
        if (this.passwordLoginState) return '正在登录中...'
        return '请输入账号密码'
      }
    },

    // 获取人脸识别消息样式类
    getMessageClass() {
      if (this.faceImgState) return 'loading'
      if (this.isSuccess) return 'success'
      return 'error'
    },
    
    // 获取密码登录消息样式类
    getPasswordMessageClass() {
      if (this.passwordLoginState) return 'loading'
      if (this.passwordSuccess) return 'success'
      return 'error'
    }
  }
}
</script>

<style scoped>
/* 基础样式 */
.login-container {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  box-sizing: border-box;
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
  opacity: 0.1;
  animation: float 8s ease-in-out infinite;
}

.circle-1 {
  width: 300px;
  height: 300px;
  background: linear-gradient(135deg, #f093fb, #f5576c);
  top: -150px;
  right: -150px;
  animation-delay: 0s;
}

.circle-2 {
  width: 200px;
  height: 200px;
  background: linear-gradient(135deg, #4ecdc4, #44a08d);
  bottom: -100px;
  left: -100px;
  animation-delay: 2s;
}

.circle-3 {
  width: 150px;
  height: 150px;
  background: linear-gradient(135deg, #fa709a, #fee140);
  top: 20%;
  left: 10%;
  animation-delay: 4s;
}

.circle-4 {
  width: 120px;
  height: 120px;
  background: linear-gradient(135deg, #a8edea, #fed6e3);
  bottom: 30%;
  right: 15%;
  animation-delay: 6s;
}

.login-content {
  position: relative;
  z-index: 1;
  width: 100%;
  max-width: 500px;
  animation: fadeInUp 0.8s ease-out;
  margin: 0 auto;
}

.login-header {
  text-align: center;
  margin-bottom: 30px;
  color: white;
}

.logo-section {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 20px;
  margin-bottom: 20px;
}

.logo-icon {
  width: 70px;
  height: 70px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  animation: pulse 2s infinite;
  color: white;
}

.icon-avatar {
  color: inherit;
}

.logo-text {
  text-align: left;
}

.main-title {
  font-size: 32px;
  font-weight: 700;
  margin: 0;
  line-height: 1.2;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.sub-title {
  font-size: 16px;
  margin: 5px 0 0 0;
  opacity: 0.9;
  font-weight: 400;
  letter-spacing: 1px;
}

.header-decoration {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.decoration-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.8);
}

.decoration-line {
  width: 60px;
  height: 2px;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.8), transparent);
}

.login-card {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 24px;
  box-shadow: 
    0 20px 40px rgba(0, 0, 0, 0.1),
    0 0 0 1px rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(20px);
  overflow: hidden;
  transition: all 0.3s ease;
}

.login-card:hover {
  transform: translateY(-5px);
  box-shadow: 
    0 25px 50px rgba(0, 0, 0, 0.15),
    0 0 0 1px rgba(255, 255, 255, 0.2);
}

/* 登录方式切换标签 */
.login-tabs {
  position: relative;
  padding: 20px 30px 0;
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
}

.tab-container {
  display: flex;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 12px;
  padding: 4px;
  position: relative;
}

.tab-button {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 16px;
  background: none;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  color: #64748b;
  position: relative;
  z-index: 2;
}

.tab-button.active {
  color: #3b82f6;
}

.tab-icon {
  width: 18px;
  height: 18px;
}

.tab-indicator {
  position: absolute;
  top: 4px;
  bottom: 4px;
  width: calc(50% - 4px);
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: left 0.3s ease;
  z-index: 1;
}

.login-status {
  padding: 20px 30px;
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  border-bottom: 1px solid rgba(226, 232, 240, 0.5);
}

.status-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  font-size: 14px;
  font-weight: 500;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  transition: all 0.3s ease;
}

.status-dot.idle {
  background: #94a3b8;
}

.status-dot.active {
  background: #3b82f6;
  animation: pulse 2s infinite;
}

.status-dot.processing {
  background: #f59e0b;
  animation: blink 1s infinite;
}

.status-text {
  color: #475569;
}

/* 人脸识别区域样式 */
.face-login-section {
  padding: 0;
}

.camera-section {
  padding: 30px;
}

.camera-container {
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  background: #1e293b;
  height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.video-container {
  position: relative;
  width: 100%;
  height: 100%;
}

.camera-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 16px;
  background: #000;
  transform: scaleX(-1); /* 镜像效果 */
}

.camera-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  pointer-events: none;
}

.camera-hint {
  text-align: center;
  color: white;
  font-size: 14px;
  line-height: 1.5;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
}

.face-control-section {
  padding: 0 30px 30px 30px;
}

.button-group {
  display: flex;
  justify-content: center;
  margin-bottom: 20px;
}

.control-button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-width: 180px;
  height: 48px;
  font-size: 16px;
  font-weight: 600;
  border-radius: 12px;
  border: none;
  cursor: pointer;
  transition: all 0.3s ease;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  color: white;
}

.control-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
}

.control-button:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  transform: none;
}

.control-button.loading {
  background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
}

.loading-icon {
  animation: spin 1s linear infinite;
}

/* 账号密码登录区域样式 */
.password-login-section {
  padding: 30px;
}

.login-form {
  margin-bottom: 20px;
}

.form-group {
  margin-bottom: 24px;
}

.form-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 600;
  color: #374151;
  margin-bottom: 8px;
}

.label-icon {
  width: 16px;
  height: 16px;
  color: #667eea;
}

.input-wrapper {
  position: relative;
}

.form-input {
  width: 100%;
  height: 48px;
  padding: 0 48px 0 16px;
  border: 2px solid #e5e7eb;
  border-radius: 12px;
  font-size: 16px;
  background: white;
  transition: all 0.3s ease;
  box-sizing: border-box;
}

.form-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.form-input:disabled {
  background: #f9fafb;
  color: #6b7280;
  cursor: not-allowed;
}

.input-icon {
  position: absolute;
  right: 16px;
  top: 50%;
  transform: translateY(-50%);
  color: #9ca3af;
  pointer-events: none;
}

.password-toggle {
  position: absolute;
  right: 16px;
  top: 50%;
  transform: translateY(-50%);
  background: none;
  border: none;
  cursor: pointer;
  color: #6b7280;
  transition: color 0.3s ease;
}

.password-toggle:hover:not(:disabled) {
  color: #3b82f6;
}

.password-toggle:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.eye-icon {
  width: 18px;
  height: 18px;
}

.form-actions {
  display: flex;
  justify-content: center;
}

.login-button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: 100%;
  height: 48px;
  font-size: 16px;
  font-weight: 600;
  border-radius: 12px;
  border: none;
  cursor: pointer;
  transition: all 0.3s ease;
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  color: white;
}

.login-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
}

.login-button:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  transform: none;
}

.login-button.loading {
  background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
}

.login-icon,
.success-icon,
.error-icon {
  flex-shrink: 0;
}

/* 消息显示区域 */
.message-display {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 20px;
  border-radius: 12px;
  font-size: 14px;
  margin-top: 15px;
  animation: slideInUp 0.3s ease-out;
}

.message-display.loading {
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.2);
  color: #d97706;
}

.message-display.success {
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.2);
  color: #059669;
}

.message-display.error {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.2);
  color: #dc2626;
}

.message-icon {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  margin-top: 2px;
}

.message-content {
  flex: 1;
}

.server-msg {
  font-weight: 600;
  margin-bottom: 4px;
}

.welcome-msg {
  font-size: 12px;
  opacity: 0.8;
  font-style: italic;
}

.usage-tips {
  padding: 20px 30px 30px 30px;
  background: rgba(248, 250, 252, 0.5);
  border-top: 1px solid rgba(226, 232, 240, 0.3);
}

.tip-item {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  font-size: 14px;
  color: #64748b;
}

.tip-item:last-child {
  margin-bottom: 0;
}

.tip-icon {
  width: 16px;
  height: 16px;
  color: #667eea;
  flex-shrink: 0;
}

/* 动画定义 */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes slideInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
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

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

/* 响应式设计 */
@media (max-width: 768px) {
  .login-container {
    padding: 10px;
  }
  
  .login-content {
    max-width: 95%;
  }

  .logo-section {
    flex-direction: column;
    gap: 15px;
  }

  .logo-text {
    text-align: center;
  }

  .main-title {
    font-size: 28px;
  }

  .camera-container {
    height: 250px;
  }

  .control-button,
  .login-button {
    width: 100%;
    max-width: 280px;
  }

  .camera-section,
  .login-status,
  .face-control-section,
  .password-login-section {
    padding: 20px;
  }

  .usage-tips {
    padding: 15px 20px 20px 20px;
  }

  .login-tabs {
    padding: 15px 20px 0;
  }
}

@media (max-width: 480px) {
  .main-title {
    font-size: 24px;
  }

  .sub-title {
    font-size: 14px;
  }

  .camera-container {
    height: 220px;
  }

  .tab-button {
    font-size: 12px;
    padding: 10px 12px;
  }

  .tab-icon {
    width: 16px;
    height: 16px;
  }
}
</style>