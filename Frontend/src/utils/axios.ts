import { reactive } from 'vue';
import axios from 'axios'

// 创建响应式状态
const authState = reactive({
    token: localStorage.getItem('auth_token') || '',
    userInfo: JSON.parse(localStorage.getItem('user_info') || 'null'),
    
    // 计算属性：是否已登录
    get isLoggedIn() {
        return !!this.token && !!this.userInfo;
    },
    
    // 登录方法
    login(tokenValue: string, user: any) {
        this.token = tokenValue;
        this.userInfo = user;
        
        // 保存到 localStorage
        localStorage.setItem('auth_token', tokenValue);
        localStorage.setItem('user_info', JSON.stringify(user));
    },
    
    // 登出方法
    logout() {
        this.token = '';
        this.userInfo = null;
        
        // 清除 localStorage
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_info');
    },
    
    // 初始化认证状态
    initAuth() {
        const savedToken = localStorage.getItem('auth_token');
        const savedUserInfo = localStorage.getItem('user_info');
        
        if (savedToken && savedUserInfo) {
            this.token = savedToken;
            this.userInfo = JSON.parse(savedUserInfo);
        }
    }
});

// 导出使用函数
export const useAuthStore = () => authState;