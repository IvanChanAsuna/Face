import { reactive } from 'vue';
import type { Router } from 'vue-router';

// 用户信息接口
interface UserInfo {
  username: string;
  name?: string;
  [key: string]: any;
}

// 认证状态接口
interface AuthState {
  token: string;
  userInfo: UserInfo | null;
  isLoggedIn: boolean;
}

// 全局认证状态 - 单例模式
const authState = reactive<AuthState>({
  token: '',
  userInfo: null,
  
  // 计算属性：是否已登录
  get isLoggedIn(): boolean {
    return !!this.token && !!this.userInfo;
  }
});

// 认证管理类
class AuthManager {
  private router: Router | null = null;
  
  // 设置路由实例
  setRouter(router: Router) {
    this.router = router;
  }
  
  // 登录方法
  login(tokenValue: string, user: UserInfo) {
    console.log('执行登录:', { tokenValue, user });
    
    authState.token = tokenValue;
    authState.userInfo = user;
    
    // 保存到 localStorage
    localStorage.setItem('auth_token', tokenValue);
    localStorage.setItem('user_info', JSON.stringify(user));
    
    console.log('登录状态已更新:', authState.isLoggedIn);
  }
  
  // 登出方法
  logout() {
    console.log('执行登出');
    
    authState.token = '';
    authState.userInfo = null;
    
    // 清除所有相关的localStorage数据
    this.clearStorage();
    
    // 跳转到登录页
    if (this.router) {
      this.router.push('/login');
    }
    
    console.log('登出完成');
  }
  
  // 清除存储数据
  private clearStorage() {
    const keysToRemove = [
      'auth_token',
      'user_info', 
      'face_token',
      'user_token',
      'username'
    ];
    
    keysToRemove.forEach(key => {
      localStorage.removeItem(key);
    });
  }
  
  // 初始化认证状态
  initAuth() {
    console.log('初始化认证状态');
    
    // 检查多种可能的token存储方式（兼容旧版本）
    const authToken = localStorage.getItem('auth_token');
    const faceToken = localStorage.getItem('face_token');
    const userToken = localStorage.getItem('user_token');
    const savedUserInfo = localStorage.getItem('user_info');
    const username = localStorage.getItem('username');
    
    // 优先使用auth_token，其次是face_token或user_token
    const token = authToken || faceToken || userToken;
    
    if (token) {
      authState.token = token;
      
      // 如果有保存的用户信息就使用，否则用username创建基本信息
      if (savedUserInfo) {
        try {
          authState.userInfo = JSON.parse(savedUserInfo);
        } catch (e) {
          console.error('解析用户信息失败:', e);
          // 解析失败时，如果有username则创建基本用户信息
          if (username) {
            authState.userInfo = { username, name: username };
          }
        }
      } else if (username) {
        authState.userInfo = { username, name: username };
      }
      
      // 统一存储格式（将旧格式转换为新格式）
      if (authState.userInfo && !authToken) {
        localStorage.setItem('auth_token', token);
        localStorage.setItem('user_info', JSON.stringify(authState.userInfo));
      }
      
      console.log('从localStorage恢复登录状态:', {
        token: !!authState.token,
        userInfo: authState.userInfo,
        isLoggedIn: authState.isLoggedIn
      });
    } else {
      console.log('未找到有效的登录信息');
    }
  }
  
  // 检查token有效性
  checkTokenValidity(): boolean {
    if (!this.isLoggedIn) {
      return false;
    }
    
    // TODO: 这里可以添加token过期检查或向服务器验证token的逻辑
    // 例如：检查token的过期时间、向服务器发送验证请求等
    
    return true;
  }
  
  // 获取当前认证状态
  get isLoggedIn(): boolean {
    return authState.isLoggedIn;
  }
  
  // 获取当前token
  get token(): string {
    return authState.token;
  }
  
  // 获取当前用户信息
  get userInfo(): UserInfo | null {
    return authState.userInfo;
  }
}

// 创建认证管理器实例
const authManager = new AuthManager();

// 导出使用函数
export const useAuthStore = () => {
  return {
    // 状态
    ...authState,
    
    // 方法
    login: authManager.login.bind(authManager),
    logout: authManager.logout.bind(authManager),
    initAuth: authManager.initAuth.bind(authManager),
    checkTokenValidity: authManager.checkTokenValidity.bind(authManager),
    setRouter: authManager.setRouter.bind(authManager)
  };
};

// 初始化认证状态（应用启动时调用）
export const initializeAuth = () => {
  authManager.initAuth();
};

// 设置路由实例（在router创建后调用）
export const setAuthRouter = (router: Router) => {
  authManager.setRouter(router);
};

export default authManager;