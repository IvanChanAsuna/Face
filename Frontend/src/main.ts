import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import { useAuthStore } from '@/stores/auth'
import http from '@/utils/http' // 导入我们配置的 axios 实例

const app = createApp(App)

// 全局注册 axios 实例
app.config.globalProperties.$http = http

// 使用路由和ElementPlus
app.use(router)
app.use(ElementPlus)

// 在应用挂载前初始化认证状态
const authStore = useAuthStore()
authStore.initAuth()

console.log('应用启动时的认证状态:', {
    isLoggedIn: authStore.isLoggedIn,
    token: !!authStore.token,
    userInfo: authStore.userInfo
})

// 测试后端连接
console.log('正在测试后端连接...')
http.get('/health')
  .then(response => {
    console.log('后端连接成功:', response.data)
  })
  .catch(error => {
    console.error('后端连接失败:', error.message)
    console.error('请确保：')
    console.error('1. 后端服务已启动 (python app.py)')
    console.error('2. 后端运行在 http://127.0.0.1:8089')
    console.error('3. 防火墙允许该端口')
  })

app.mount('#app')