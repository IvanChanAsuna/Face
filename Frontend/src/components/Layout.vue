<template>
  <div class="layout">
    <!-- 背景装饰 -->
    <div class="background-decorations">
      <div class="decoration-circle circle-1"></div>
      <div class="decoration-circle circle-2"></div>
      <div class="decoration-circle circle-3"></div>
      <div class="decoration-circle circle-4"></div>
    </div>

    <el-container v-if="state.showMenu" class="container">
      <!-- 侧边栏 -->
      <el-aside class="aside">
        <div class="sidebar-content">
          <!-- 头部区域 -->
          <div class="head">
            <div class="logo-section">
              <div class="logo-icon">
                <el-icon :size="28">
                  <Avatar />
                </el-icon>
              </div>
              <div class="logo-text">
                <span class="main-title">人脸识别系统</span>
                <span class="sub-title">Face Recognition</span>
              </div>
            </div>
          </div>

          <!-- 用户信息区域 -->
          <div class="user-info-section">
            <div class="user-info">
              <div class="user-avatar">
                <el-icon :size="20">
                  <User />
                </el-icon>
              </div>
              <div class="user-details">
                <div class="user-name">{{ authStore.userInfo?.name || '用户' }}</div>
                <div class="user-status">
                  <span class="status-dot"></span>
                  <span class="status-text">在线</span>
                </div>
              </div>
              <el-dropdown @command="handleUserCommand" class="user-dropdown">
                <el-icon class="dropdown-icon">
                  <More />
                </el-icon>
                <template #dropdown>
                  <el-dropdown-menu>
                    <el-dropdown-item command="profile">
                      <el-icon><User /></el-icon>
                      个人信息
                    </el-dropdown-item>
                    <el-dropdown-item command="settings">
                      <el-icon><Setting /></el-icon>
                      设置
                    </el-dropdown-item>
                    <el-dropdown-item divided command="logout">
                      <el-icon><SwitchButton /></el-icon>
                      退出登录
                    </el-dropdown-item>
                  </el-dropdown-menu>
                </template>
              </el-dropdown>
            </div>
          </div>

          <!-- 分割线 -->
          <div class="divider">
            <div class="divider-line"></div>
          </div>

          <!-- 导航菜单 -->
          <div class="menu-section">
            <el-menu 
              :default-openeds="state.defaultOpen" 
              :router="true" 
              :default-active="state.currentPath"
              class="custom-menu"
            >
              <el-submenu index="1">
                <template #title>
                  <div class="menu-group-title">
                    <el-icon class="group-icon">
                      <Opportunity />
                    </el-icon>
                    <span>功能模块</span>
                  </div>
                </template>

                <el-menu-item-group>
                  <el-menu-item index="/dashboard/faceDetect" class="menu-item">
                    <div class="menu-item-content">
                      <div class="menu-icon-wrapper">
                        <el-icon>
                          <Avatar />
                        </el-icon>
                      </div>
                      <span class="menu-text">人脸检测</span>
                      <div class="menu-badge"></div>
                    </div>
                  </el-menu-item>

                  <el-menu-item index="/dashboard/faceCompare" class="menu-item">
                    <!-- <div class="menu-item-content">
                      <div class="menu-icon-wrapper">
                        <el-icon>
                          <User />
                        </el-icon>
                      </div>
                      <span class="menu-text">人脸相似度</span>
                      <div class="menu-badge"></div>
                    </div> -->
                  </el-menu-item>

                  <el-menu-item index="/dashboard/faceRecognition" class="menu-item">
                    <!-- <div class="menu-item-content">
                      <div class="menu-icon-wrapper">
                        <el-icon>
                          <Cherry />
                        </el-icon>
                      </div>
                      <span class="menu-text">人脸识别</span>
                      <div class="menu-badge"></div>
                    </div> -->
                  </el-menu-item>

                  <el-menu-item index="/dashboard/faceStream" class="menu-item">
                    <!-- <div class="menu-item-content">
                      <div class="menu-icon-wrapper">
                        <el-icon>
                          <Opportunity />
                        </el-icon>
                      </div>
                      <span class="menu-text">视频流识别</span>
                      <div class="menu-badge"></div>
                    </div> -->
                  </el-menu-item>
                </el-menu-item-group>
              </el-submenu>
            </el-menu>
          </div>
        </div>
      </el-aside>

      <!-- 主内容区域 -->
      <el-container class="content">
        <!-- 顶部栏 -->
        <el-header class="header">
          <div class="header-content" style="justify-content: center; gap: 1000px">
            <div class="header-left">
              <el-breadcrumb separator="/" class="breadcrumb">
                <el-breadcrumb-item :to="{ path: '/dashboard' }">控制台</el-breadcrumb-item>
                <el-breadcrumb-item>{{ getCurrentPageTitle() }}</el-breadcrumb-item>
              </el-breadcrumb>
            </div>
            <div class="header-right">
              <div class="header-info">
                <!-- <span class="welcome-text">欢迎回来，{{ authStore.userInfo?.name || '用户' }}</span> -->
                <el-button type="text" @click="handleLogout" class="logout-btn">
                  <el-icon><SwitchButton /></el-icon>
                  退出
                </el-button>
              </div>
            </div>
          </div>
        </el-header>

        <div class="main">
          <div class="main-wrapper">
            <router-view />
          </div>
        </div>
      </el-container>
    </el-container>

    <!-- 无菜单模式 -->
    <el-container v-else class="container">
      <div class="main-wrapper">
        <router-view />
      </div>
    </el-container>
  </div>
</template>

<script lang="ts" setup>
import { onUnmounted, onMounted, reactive, computed } from 'vue'
import { NavigationGuardNext, RouteLocationNormalized, useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  User,
  Avatar,
  Opportunity,
  Cherry,
  More,
  Setting,
  SwitchButton
} from '@element-plus/icons-vue'

console.log('Layout Component Mounted')

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()

const state = reactive<any>({
  defaultOpen: ['1'],
  showMenu: true,
  currentPath: '/',
  count: {
    number: 1
  }
})

// 页面标题映射
const pageTitles: Record<string, string> = {
  '/dashboard/faceDetect': '人脸检测',
  '/dashboard/faceCompare': '人脸相似度',
  '/dashboard/faceRecognition': '人脸识别',
  '/dashboard/faceStream': '视频流识别'
}

// 获取当前页面标题
const getCurrentPageTitle = () => {
  return pageTitles[route.path] || '首页'
}

// 处理用户下拉菜单命令
const handleUserCommand = (command: string) => {
  switch (command) {
    case 'profile':
      ElMessage.info('个人信息功能开发中...')
      break
    case 'settings':
      ElMessage.info('设置功能开发中...')
      break
    case 'logout':
      handleLogout()
      break
  }
}

// 处理登出
const handleLogout = async () => {
  try {
    await ElMessageBox.confirm(
      '确定要退出登录吗？',
      '提示',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning',
      }
    )
    
    // 执行登出
    authStore.logout()
    ElMessage.success('已退出登录')
  } catch (error) {
    // 用户取消登出
    console.log('用户取消登出')
  }
}

// 路由守卫
const unwatch = router.beforeEach((to: RouteLocationNormalized, from: RouteLocationNormalized, next: NavigationGuardNext) => {
  next()
  state.currentPath = to.path
})

onMounted(() => {
  // 检查登录状态
  if (!authStore.isLoggedIn) {
    console.log('Layout: 用户未登录，重定向到登录页')
    router.push('/login')
  }
  
  // 设置当前路径
  state.currentPath = route.path
})

onUnmounted(() => {
  unwatch()
})
</script>

<style scoped>
.layout {
  min-height: 100vh;
  background: linear-gradient(135deg, #f3f0ff 0%, #d8b4fe 100%);
  position: relative;
  overflow: hidden;
}

/* 背景装饰 */
.background-decorations {
  position: fixed;
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
  opacity: 0.08;
  animation: float 10s ease-in-out infinite;
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

.circle-4 {
  width: 100px;
  height: 100px;
  background: linear-gradient(135deg, #8b5cf6, #a855f7);
  top: 20%;
  left: 5%;
  animation-delay: 9s;
}

.container {
  height: 100vh;
  position: relative;
  z-index: 1;
}

/* 侧边栏样式 */
.aside {
  width: 280px !important;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  border-right: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 4px 0 20px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  animation: slideInLeft 0.8s ease-out;
}

.sidebar-content {
  height: 100%;
  display: flex;
  flex-direction: column;
}

/* 头部区域 */
.head {
  padding: 30px 25px 20px 25px;
  background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
  color: white;
  position: relative;
  overflow: hidden;
}

.head::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(45deg, rgba(255, 255, 255, 0.1) 0%, transparent 100%);
  pointer-events: none;
}

.logo-section {
  display: flex;
  align-items: center;
  gap: 15px;
  position: relative;
  z-index: 1;
}

.logo-icon {
  width: 50px;
  height: 50px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
  animation: pulse 2s infinite;
}

.logo-text {
  display: flex;
  flex-direction: column;
}

.main-title {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 2px;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.sub-title {
  font-size: 12px;
  opacity: 0.8;
  font-weight: 400;
  letter-spacing: 0.5px;
}

/* 用户信息区域 */
.user-info-section {
  padding: 20px 25px;
  background: rgba(255, 255, 255, 0.95);
  border-bottom: 1px solid rgba(139, 92, 246, 0.1);
}

.user-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.user-avatar {
  width: 36px;
  height: 36px;
  background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.user-details {
  flex: 1;
  min-width: 0;
}

.user-name {
  font-size: 14px;
  font-weight: 600;
  color: #1e293b;
  margin-bottom: 2px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.user-status {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #64748b;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #10b981;
  animation: pulse 2s infinite;
}

.status-text {
  font-size: 12px;
}

.user-dropdown {
  cursor: pointer;
}

.dropdown-icon {
  color: #64748b;
  transition: color 0.3s ease;
}

.dropdown-icon:hover {
  color: #8b5cf6;
}

/* 分割线 */
.divider {
  padding: 0 25px;
  margin: 10px 0;
}

.divider-line {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
}

/* 菜单区域 */
.menu-section {
  flex: 1;
  padding: 0 15px;
  overflow-y: auto;
}

.menu-section::-webkit-scrollbar {
  width: 4px;
}

.menu-section::-webkit-scrollbar-track {
  background: transparent;
}

.menu-section::-webkit-scrollbar-thumb {
  background: rgba(139, 92, 246, 0.3);
  border-radius: 2px;
}

/* 菜单样式重写 */
.custom-menu {
  background: transparent !important;
  border: none !important;
}

.menu-group-title {
  display: flex;
  align-items: center;
  gap: 12px;
  font-weight: 600;
  color: #1e293b;
  font-size: 14px;
}

.group-icon {
  font-size: 18px;
  color: #8b5cf6;
}

.menu-item {
  margin: 8px 0;
  border-radius: 12px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.menu-item:hover {
  background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
  transform: translateX(5px);
}

.menu-item.is-active {
  background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
  color: white;
  box-shadow: 0 8px 16px rgba(139, 92, 246, 0.3);
}

.menu-item.is-active .menu-text {
  color: white;
}

.menu-item.is-active .menu-icon-wrapper {
  background: rgba(255, 255, 255, 0.2);
  color: white;
}

.menu-item-content {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 4px 0;
  position: relative;
}

.menu-icon-wrapper {
  width: 36px;
  height: 36px;
  background: rgba(139, 92, 246, 0.1);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #8b5cf6;
  transition: all 0.3s ease;
}

.menu-text {
  font-size: 14px;
  font-weight: 500;
  color: #1e293b;
  transition: all 0.3s ease;
}

.menu-badge {
  width: 6px;
  height: 6px;
  background: #10b981;
  border-radius: 50%;
  margin-left: auto;
  opacity: 0;
  transition: all 0.3s ease;
}

.menu-item:hover .menu-badge {
  opacity: 1;
}

/* 主内容区域 */
.content {
  display: flex;
  flex-direction: column;
  max-height: 100vh;
  overflow: hidden;
  background: transparent;
}

.header {
  height: 60px !important;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(139, 92, 246, 0.1);
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  padding: 0 30px;
  display: flex;
  align-items: center;
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
}

.header-left {
  display: flex;
  align-items: center;
  
}

.breadcrumb {
  font-size: 14px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 20px;
}

.header-info {
  display: flex;
  align-items: center;
  gap: 15px;
}

.welcome-text {
  font-size: 14px;
  color: #64748b;
}

.logout-btn {
  color: #64748b;
  transition: color 0.3s ease;
}

.logout-btn:hover {
  color: #8b5cf6;
}

.main {
  flex: 1;
  overflow: auto;
  padding: 0;
}

.main-wrapper {
  min-height: 100%;
  animation: fadeIn 0.8s ease-out;
}

/* 动画定义 */
@keyframes slideInLeft {
  from {
    opacity: 0;
    transform: translateX(-100%);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
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

/* 响应式设计 */
@media (max-width: 1024px) {
  .aside {
    width: 240px !important;
  }
  
  .head {
    padding: 25px 20px 20px 20px;
  }
  
  .main-title {
    font-size: 16px;
  }
  
  .sub-title {
    font-size: 11px;
  }
  
  .header {
    padding: 0 20px;
  }
}

@media (max-width: 768px) {
  .aside {
    width: 200px !important;
  }
  
  .logo-section {
    gap: 10px;
  }
  
  .logo-icon {
    width: 40px;
    height: 40px;
  }
  
  .main-title {
    font-size: 14px;
  }
  
  .sub-title {
    display: none;
  }
  
  .menu-text {
    font-size: 13px;
  }
  
  .header {
    padding: 0 15px;
  }
  
  .welcome-text {
    display: none;
  }
}
</style>

<style>
/* 全局样式覆盖 */
body {
  padding: 0;
  margin: 0;
  box-sizing: border-box;
}

/* Element Plus 组件样式覆盖 */
.el-menu {
  border-right: none !important;
  background: transparent !important;
}

.el-submenu {
  border: none !important;
}

.el-submenu .el-submenu__title {
  background: transparent !important;
  border: none !important;
  padding: 15px 10px !important;
  font-weight: 600 !important;
  color: #1e293b !important;
}

.el-submenu .el-submenu__title:hover {
  background: rgba(139, 92, 246, 0.05) !important;
  color: #8b5cf6 !important;
}

.el-menu-item-group {
  padding: 0 !important;
}

.el-menu-item-group__title {
  display: none !important;
}

.el-menu-item {
  background: transparent !important;
  border: none !important;
  margin: 8px 10px !important;
  border-radius: 12px !important;
  padding: 12px 15px !important;
  transition: all 0.3s ease !important;
}

.el-menu-item:hover {
  background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%) !important;
  color: #8b5cf6 !important;
}

.el-menu-item.is-active {
  background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%) !important;
  color: white !important;
  box-shadow: 0 8px 16px rgba(139, 92, 246, 0.3) !important;
}

.el-submenu__icon-arrow {
  display: none !important;
}

.el-menu--collapse .el-submenu__title {
  padding: 0 20px !important;
}

.el-header {
  padding: 0 !important;
}

.el-breadcrumb__inner {
  color: #64748b !important;
}

.el-breadcrumb__inner:hover {
  color: #8b5cf6 !important;
}

.el-dropdown-menu__item {
  padding: 8px 16px !important;
  font-size: 14px !important;
}

.el-dropdown-menu__item:hover {
  background: rgba(139, 92, 246, 0.1) !important;
  color: #8b5cf6 !important;
}

a {
  color: #8b5cf6;
  text-decoration: none;
}

.el-pagination {
  text-align: center;
  margin-top: 20px;
}

.el-popper__arrow {
  display: none;
}

/* 滚动条美化 */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.05);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: rgba(139, 92, 246, 0.3);
  border-radius: 4px;
  transition: all 0.3s ease;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(139, 92, 246, 0.5);
}
</style>