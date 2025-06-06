// src/utils/http.js
import axios from 'axios'

// 创建 axios 实例
const http = axios.create({
  baseURL: 'http://127.0.0.1:8089', // 确保端口号正确
  timeout: 900000, // 增加超时时间，人脸检测可能需要更长时间
  // 不设置默认的 Content-Type
})

// 请求拦截器
http.interceptors.request.use(
  config => {
    console.log('=== HTTP请求拦截器 ===')
    console.log('请求URL:', config.baseURL + config.url)
    console.log('请求方法:', config.method?.toUpperCase())
    console.log('请求数据类型:', config.data instanceof FormData ? 'FormData' : typeof config.data)
    
    // 添加认证 token（如果存在）
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
      console.log('添加认证token')
    }
    
    // 关键修复：对于 FormData，不设置 Content-Type
    if (config.data instanceof FormData) {
      console.log('检测到FormData，让浏览器自动设置Content-Type')
      // 删除任何手动设置的 Content-Type
      delete config.headers['Content-Type']
      delete config.headers['content-type']
      
      // 验证FormData内容
      console.log('FormData内容:')
      for (const [key, value] of config.data.entries()) {
        if (value instanceof File) {
          console.log(`- ${key}: File(${value.name}, ${value.size}字节, ${value.type})`)
        } else {
          console.log(`- ${key}:`, value)
        }
      }
    } else if (config.data) {
      // 对于非FormData，设置JSON类型
      config.headers['Content-Type'] = 'application/json'
      console.log('设置Content-Type为application/json')
    }
    
    console.log('最终请求头:', config.headers)
    return config
  },
  error => {
    console.error('❌ 请求拦截器错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
http.interceptors.response.use(
  response => {
    console.log('=== HTTP响应拦截器 ===')
    console.log('响应状态:', response.status, response.statusText)
    console.log('响应头:', response.headers)
    console.log('响应数据类型:', typeof response.data)
    
    if (response.data) {
      console.log('响应数据结构:', {
        code: response.data.code,
        message: response.data.message,
        hasData: !!response.data.data,
        hasImage: !!(response.data.data && response.data.data.image),
        imageLength: (response.data.data && response.data.data.image) ? response.data.data.image.length : 0
      })
    }
    
    return response // 返回完整的响应对象
  },
  error => {
    console.error('=== HTTP响应错误 ===')
    console.error('错误对象:', error)
    
    if (error.response) {
      // 服务器返回了错误状态码
      console.error('服务器错误详情:')
      console.error('- 状态码:', error.response.status)
      console.error('- 状态文本:', error.response.statusText)
      console.error('- 响应头:', error.response.headers)  
      console.error('- 错误数据:', error.response.data)
      
      // 处理常见错误
      const status = error.response.status
      const errorMessages = {
        400: '请求参数错误',
        401: '未授权访问',
        403: '禁止访问',
        404: '接口不存在',
        413: '文件太大，超过10MB限制',
        415: '不支持的文件类型',
        500: '服务器内部错误',
        502: '网关错误',
        503: '服务不可用',
        504: '网关超时'
      }
      
      const errorMsg = errorMessages[status] || `HTTP错误 ${status}`
      console.error(`❌ ${errorMsg}`)
      
    } else if (error.request) {
      // 请求发送了但没有收到响应
      console.error('网络连接问题:')
      console.error('- 请求对象:', error.request)
      console.error('- 错误信息:', error.message)
      console.error('- 错误代码:', error.code)
      
      console.error('可能的原因:')
      console.error('1. 后端服务未启动')
      console.error('2. 端口号错误或被占用')
      console.error('3. CORS配置问题') 
      console.error('4. 网络连接问题')
      console.error('5. 防火墙阻止请求')
      
      // 提供解决建议
      console.error('解决建议:')
      console.error('1. 检查后端服务: http://127.0.0.1:8089/health')
      console.error('2. 检查网络连接')
      console.error('3. 检查浏览器控制台Network标签')
      
    } else {
      // 其他错误（通常是配置错误）
      console.error('请求配置错误:')
      console.error('- 错误信息:', error.message)
      console.error('- 错误堆栈:', error.stack)
    }
    
    return Promise.reject(error)
  }
)

// 导出配置好的实例
export default http

// 额外导出一些工具函数用于调试
export const testConnection = async () => {
  try {
    console.log('=== 测试后端连接 ===')
    const response = await http.get('/health')
    console.log('✅ 后端连接正常:', response.data)
    return true
  } catch (error) {
    console.error('❌ 后端连接失败:', error.message)
    return false
  }
}

export const testImageUpload = async (file) => {
  try {
    console.log('=== 测试图片上传 ===')
    const formData = new FormData()
    formData.append('image', file)
    
    const response = await http.post('/detectFaces', formData)
    console.log('✅ 图片上传测试成功:', response.data)
    return response.data
  } catch (error) {
    console.error('❌ 图片上传测试失败:', error)
    throw error
  }
}