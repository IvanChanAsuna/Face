import { createRouter, createWebHashHistory, RouteRecordRaw } from "vue-router";
import { useAuthStore, setAuthRouter } from "@/stores/auth";

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'Root',
    redirect: '/login' // 根路径直接重定向到登录页
  },
  {
    path: '/login',
    name: 'Login',
    component: () => import("@/views/Login.vue"),
    meta: { 
      requiresAuth: false,
      title: '登录'
    }
  },
  {
    path: '/dashboard',
    component: () => import("@/components/Layout.vue"),
    meta: { 
      requiresAuth: true,
      title: '控制台'
    },
    children: [
      {
        path: '',
        redirect: '/dashboard/faceDetect'
      },
      {
        path: 'faceDetect',
        name: 'FaceDetect',
        component: () => import("@/views/FaceDetect.vue"),
        meta: { 
          requiresAuth: true,
          title: '人脸检测'
        }
      },
      {
        path: 'faceCompare',
        name: 'FaceCompare',
        component: () => import("@/views/FaceCompare.vue"),
        meta: { 
          requiresAuth: true,
          title: '人脸相似度'
        }
      },
      {
        path: 'faceRecognition',
        name: 'FaceRecognition',
        component: () => import("@/views/FaceRecognition.vue"),
        meta: { 
          requiresAuth: true,
          title: '人脸识别'
        }
      },
      {
        path: 'faceStream',
        name: 'FaceStream',
        component: () => import("@/views/FaceStream.vue"),
        meta: { 
          requiresAuth: true,
          title: '视频流识别'
        }
      }
    ]
  },
  {
    // 兼容旧路径（如果有其他地方使用了/home）
    path: '/home', 
    redirect: '/dashboard/faceDetect'
  },
  {
    // 捕获所有未匹配的路由
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    redirect: '/login'
  }
];

const router = createRouter({
  history: createWebHashHistory(),
  routes: routes
});

// 路由守卫
router.beforeEach((to, from, next) => {
  const authStore = useAuthStore();
  
  // 获取路由元信息
  const requiresAuth = to.matched.some(record => record.meta.requiresAuth);
  const isLoginPage = to.path === '/login';
  
  console.log('路由守卫检查:', {
    from: from.path,
    to: to.path,
    requiresAuth,
    isLoggedIn: authStore.isLoggedIn,
    userInfo: authStore.userInfo
  });
  
  // 情况1: 访问需要认证的页面，但未登录
  if (requiresAuth && !authStore.isLoggedIn) {
    console.log('未登录用户尝试访问受保护路由，重定向到登录页');
    next('/login');
    return;
  }
  
  // 情况2: 已登录用户访问登录页，重定向到主界面
  if (isLoginPage && authStore.isLoggedIn) {
    console.log('已登录用户访问登录页，重定向到主界面');
    next('/dashboard/faceDetect');
    return;
  }
  
  // 情况3: 正常访问
  console.log('正常路由访问');
  next();
});

// 路由后置守卫（可选，用于设置页面标题等）
router.afterEach((to) => {
  // 设置页面标题
  const title = to.meta?.title as string;
  if (title) {
    document.title = `${title} - 人脸识别系统`;
  } else {
    document.title = '人脸识别系统';
  }
});

// 将路由实例传递给认证管理器
setAuthRouter(router);

export default router;