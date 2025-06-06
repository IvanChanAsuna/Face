from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import os
import jwt
import datetime
from PIL import Image
from DAI_Net_main.face_detection import FaceDetector
import numpy as np
from werkzeug.utils import secure_filename
import traceback
import hashlib
from mywork.test import DualModelFaceRecognizer
from mywork.configs import Configs
# 创建Flask应用
app = Flask(__name__)

# JWT密钥
app.config['JWT_SECRET_KEY'] = 'jpjanvnvwokd'

# 更详细的 CORS 配置
CORS(app,
     resources={
         r"/*": {
             "origins": [
                 "http://localhost:8088", "http://127.0.0.1:8088",
                 "http://localhost:8080", "http://127.0.0.1:8080",
                 "http://localhost:3000", "http://127.0.0.1:3000",   # Vue开发服务器默认端口
                 "http://localhost:5173", "http://127.0.0.1:5173",   # Vite默认端口
                 "http://localhost:8000", "http://127.0.0.1:8000"    # 其他可能的端口
             ],
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             "allow_headers": [
                 "Content-Type",
                 "Authorization",
                 "Accept",
                 "Origin",
                 "X-Requested-With",
                 "Access-Control-Allow-Origin"
             ],
             "supports_credentials": True,
             "max_age": 3600
         }
     })


# 配置文件上传
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB限制
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ==================== 用户数据和人脸数据库 ====================
# 模拟用户数据库（生产环境请使用真实数据库）
USERS_DB = {
    'admin': {
        'username': 'admin',
        'password': 'e10adc3949ba59abbe56e057f20f883e',  # 123456的MD5
        'name': '管理员',
        'role': 'admin'
    },
    'user1': {
        'username': 'user1',
        'password': '25d55ad283aa400af464c76d713c07ad',  # hello的MD5
        'name': '用户1',
        'role': 'user'
    },
    'demo': {
        'username': 'demo',
        'password': 'fe01ce2a7fbac8fafaed7c982a04e229',  # demo的MD5
        'name': '演示用户',
        'role': 'user'
    }
}




def generate_token(user_info):
    """生成JWT token"""
    payload = {
        'username': user_info['username'],
        'name': user_info['name'],
        'role': user_info.get('role', 'user'),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24),  # 24小时过期
        'iat': datetime.datetime.utcnow()
    }

    token = jwt.encode(payload, app.config['JWT_SECRET_KEY'], algorithm='HS256')
    return token


def verify_token(token):
    """验证JWT token"""
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def hash_password(password):
    """对密码进行MD5加密"""
    return hashlib.md5(password.encode()).hexdigest()


def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image):
    """将PIL Image转换为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def base64_to_image(base64_str):
    """将base64字符串转换为PIL Image"""
    # 去除data:image/jpeg;base64,前缀（如果存在）
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]

    # 解码base64
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image




def process_face_detection(image):
    """
    人脸检测算法处理函数（用于人脸检测页面）

    参数:
        image: PIL Image对象

    返回:
        processed_image: 处理后的PIL Image对象（带有anchor box标注）

    注意: 这里是你需要替换为你自己人脸检测算法的地方
    """

    # =============== 在这里调用你的人脸检测算法 ===============
    print("正在处理图片...")

    try:
        detector = FaceDetector()
        processed_image = detector.detect_single_image(image)
        return processed_image
    except Exception as e:
        print(f"人脸检测处理失败: {str(e)}")
        # 如果算法失败，返回原图
        return image


# ==================== 登录相关API ====================

@app.route('/user/login', methods=['POST', 'OPTIONS'])
def password_login():
    """账号密码登录接口"""
    # 处理 OPTIONS 预检请求
    if request.method == 'OPTIONS':
        return jsonify({'code': 200, 'msg': 'OK'}), 200

    try:
        print(f"收到登录请求: {request.method} {request.url}")
        print(f"请求头: {dict(request.headers)}")

        data = request.get_json()
        print(f"请求数据: {data}")

        if not data:
            return jsonify({
                'code': 400,
                'msg': '请求数据格式错误'
            }), 400

        username = data.get('username', '').strip()
        password = data.get('password', '').strip()

        # 验证输入
        if not username or not password:
            return jsonify({
                'code': 400,
                'msg': '用户名和密码不能为空'
            })

        # 查找用户
        user = USERS_DB.get(username)
        if not user:
            return jsonify({
                'code': 401,
                'msg': '用户名或密码错误'
            })

        # 验证密码
        hashed_password = hash_password(password)
        if user['password'] != hashed_password:
            return jsonify({
                'code': 401,
                'msg': '用户名或密码错误'
            })

        # 生成token
        user_info = {
            'username': user['username'],
            'name': user['name'],
            'role': user['role']
        }
        token = generate_token(user_info)

        return jsonify({
            'code': 200,
            'msg': '登录成功',
            'token': token,
            'username': user['username'],
            'name': user['name']
        })

    except Exception as e:
        print(f"密码登录失败: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'code': 500,
            'msg': '服务器内部错误'
        }), 500


@app.route('/face/vef', methods=['POST', 'OPTIONS'])
def face_verification():
    """人脸识别登录接口"""
    # 处理 OPTIONS 预检请求
    if request.method == 'OPTIONS':
        return jsonify({'code': 200, 'msg': 'OK'}), 200

    try:

        data = request.get_json()

        if not data or 'imageBase' not in data:
            return jsonify({
                'code': 400,
                'msg': '请上传人脸图片'
            })

        # 解析base64图片
        try:
            image_base64 = data['imageBase']
            image = base64_to_image(image_base64)

            # 确保图片是RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')


        except Exception as e:
            print(f"图片解析失败: {str(e)}")
            return jsonify({
                'code': 400,
                'msg': '图片数据格式错误'
            })

        # 提取人脸特征
        args = Configs()
        args = args.get_config()
        model_path1 = r'mywork/weights/520.pth'
        model_path2 = r'mywork/weights/520_v3.pth'
        reference_image = r'mywork/reference/me_001.jpg'
        recognizer = DualModelFaceRecognizer(
            args,
            model_path1,
            model_path2,
            reference_image,
            threshold1=0.90,
            threshold2=0.75,
        )
        result = recognizer.predict(image)
        # if not features is None:
        #     return jsonify({
        #         'code': 201,
        #         'msg': '未检测到清晰的人脸，请重新拍照'
        #     })


        if result['final_decision']:
            # 找到匹配的用户，生成token
            user_info = {
                'username': f"Ivan Chan",
                'name': 'Ivan Chan',
                'role': 'Admin'
            }
            token = generate_token(user_info)
            return jsonify({
                'code': 200,
                'msg': f'欢迎回来，Ivan Chan！',
                'token': token,
                'name': 'Ivan Chan'
            })
        else:
            # 未找到匹配的用户
            return jsonify({
                'code': 201,
                'msg': '身份验证失败，未找到匹配的用户信息'
            })

    except Exception as e:
        print(f"人脸识别登录失败: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'code': 500,
            'msg': '服务器处理错误，请稍后重试'
        }), 500


# ==================== 人脸检测相关API ====================

@app.route('/detectFaces', methods=['POST', 'OPTIONS'])
def detect_faces():
    """人脸检测接口"""

    # 处理预检请求
    if request.method == 'OPTIONS':
        print("收到 OPTIONS 预检请求")
        return '', 200

    try:
        print("=" * 60)
        print("收到人脸检测请求")
        print("=" * 60)
        print(f"请求方法: {request.method}")
        print(f"Content-Type: {request.content_type}")
        print(f"Content-Length: {request.content_length}")
        print(f"请求头: {dict(request.headers)}")
        print(f"Files: {list(request.files.keys())}")
        print(f"Form: {list(request.form.keys())}")
        print(f"JSON: {request.is_json}")

        # 检查请求中是否包含文件
        if 'image' not in request.files:
            print("❌ 错误: 请求中没有 'image' 字段")
            print(f"可用的字段: {list(request.files.keys())}")
            return jsonify({
                'code': 1,
                'message': '请上传图片文件，字段名应为 image',
                'data': None
            }), 400

        file = request.files['image']
        print(f"📁 接收到文件:")
        print(f"- 文件名: {file.filename}")
        print(f"- 文件类型: {file.content_type}")
        print(f"- 文件流: {type(file.stream)}")

        # 检查文件是否为空
        if not file.filename or file.filename == '':
            print("❌ 错误: 文件名为空")
            return jsonify({
                'code': 1,
                'message': '未选择文件',
                'data': None
            }), 400

        # 检查文件类型
        if not allowed_file(file.filename):
            print(f"❌ 错误: 不支持的文件格式 - {file.filename}")
            print(f"支持的格式: {ALLOWED_EXTENSIONS}")
            return jsonify({
                'code': 1,
                'message': '不支持的文件格式，请上传PNG、JPG或JPEG格式的图片',
                'data': None
            }), 400

        # 读取图片数据
        print("📖 开始读取图片数据...")
        try:
            # 确保文件指针在开始位置
            file.seek(0)

            # 读取文件内容
            file_content = file.read()
            print(f"- 读取字节数: {len(file_content)}")

            if len(file_content) == 0:
                raise ValueError("文件内容为空")

            # 重置文件指针并用PIL打开
            file.seek(0)
            image = Image.open(file.stream)
            print(f"✅ 图片读取成功:")
            print(f"- 尺寸: {image.size}")
            print(f"- 模式: {image.mode}")
            print(f"- 格式: {image.format}")

        except Exception as e:
            print(f"❌ 图片读取失败: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'code': 1,
                'message': f'图片文件损坏或格式不正确: {str(e)}',
                'data': None
            }), 400

        # 确保图片是RGB模式
        if image.mode != 'RGB':
            print(f"🔄 转换图片模式: {image.mode} -> RGB")
            image = image.convert('RGB')

        # 调用人脸检测算法
        print("🔍 开始人脸检测处理...")
        try:
            processed_image = process_face_detection(image)
            print("✅ 人脸检测处理完成")

            # 验证处理后的图片
            if processed_image is None:
                raise ValueError("人脸检测返回空结果")

            print(f"- 处理后图片尺寸: {processed_image.size}")
            print(f"- 处理后图片模式: {processed_image.mode}")

        except Exception as e:
            print(f"❌ 人脸检测处理失败: {str(e)}")
            traceback.print_exc()
            # 如果检测失败，返回原图
            processed_image = image
            print("⚠️ 使用原图作为结果")

        # 将处理后的图片转换为base64
        print("🔄 转换图片为base64...")
        try:
            # 使用自定义函数转换（返回完整的data URL）
            result_base64_full = image_to_base64(processed_image)
            print(f"- 完整data URL长度: {len(result_base64_full)}")
            print(f"- 前缀: {result_base64_full[:50]}")

            # 提取纯base64数据（去掉data:image/jpeg;base64,前缀）
            if result_base64_full.startswith('data:image/'):
                result_base64 = result_base64_full.split(',')[1]
                print(f"- 纯base64长度: {len(result_base64)}")
            else:
                result_base64 = result_base64_full
                print("⚠️ 未检测到data URL前缀")

            # 验证base64数据
            if len(result_base64) < 100:
                raise ValueError("base64数据太短，可能转换失败")

            print("✅ base64转换完成")

        except Exception as e:
            print(f"❌ base64转换失败: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'code': 1,
                'message': f'图片转换失败: {str(e)}',
                'data': None
            }), 500

        # 构建响应数据
        response_data = {
            'code': 0,
            'message': '人脸检测完成',
            'data': {
                'image': result_base64,  # 返回纯base64数据，前端会添加前缀
                'original_size': f"{image.size[0]}x{image.size[1]}",
                'processed_size': f"{processed_image.size[0]}x{processed_image.size[1]}",
                'image_mode': processed_image.mode,
                'data_length': len(result_base64)
            }
        }

        return jsonify(response_data)

    except Exception as e:
        # 记录详细错误信息
        error_msg = str(e)
        print("=" * 60)
        print(f"❌ 处理过程中发生严重错误: {error_msg}")
        print("错误堆栈:")
        traceback.print_exc()
        print("=" * 60)

        return jsonify({
            'code': 1,
            'message': f'服务器处理失败: {error_msg}',
            'data': None
        }), 500


# ==================== 权限验证相关 ====================

def require_auth(f):
    """权限验证装饰器"""

    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({
                'code': 401,
                'msg': '缺少认证token'
            }), 401

        # 移除Bearer前缀
        if token.startswith('Bearer '):
            token = token[7:]

        # 验证token
        payload = verify_token(token)
        if not payload:
            return jsonify({
                'code': 401,
                'msg': 'token无效或已过期'
            }), 401

        # 将用户信息添加到request对象
        request.current_user = payload
        return f(*args, **kwargs)

    decorated_function.__name__ = f.__name__
    return decorated_function


@app.route('/user/info', methods=['GET'])
@require_auth
def get_user_info():
    """获取当前用户信息"""
    return jsonify({
        'code': 200,
        'msg': '获取成功',
        'data': request.current_user
    })





@app.route('/system/info', methods=['GET'])
def system_info():
    """系统信息接口"""
    return jsonify({
        'code': 0,
        'message': '系统信息获取成功',
        'data': {
            'name': '智能登录系统',
            'version': '1.0.0',
            'supported_login_methods': ['password', 'face_recognition'],
            'default_accounts': [
                {'username': 'admin', 'password': '123456', 'name': '管理员'},
                {'username': 'user1', 'password': 'hello', 'name': '用户1'},
                {'username': 'demo', 'password': 'demo', 'name': '演示用户'}
            ]
        }
    })


# ==================== 错误处理 ====================

@app.errorhandler(413)
def too_large(e):
    """文件过大错误处理"""
    return jsonify({
        'code': 1,
        'message': '文件大小超过10MB限制',
        'data': None
    }), 413


@app.errorhandler(404)
def not_found(e):
    """404错误处理"""
    return jsonify({
        'code': 404,
        'message': '请求的接口不存在',
        'data': None
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """内部服务器错误处理"""
    return jsonify({
        'code': 1,
        'message': '服务器内部错误',
        'data': None
    }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("正在启动智能登录系统后端服务...")
    print("=" * 60)
    print("前端地址: http://localhost:8088")
    print("后端地址: http://127.0.0.1:8089")
    print()
    print("默认账号:")
    print("  管理员 - 用户名: admin, 密码: 123456")
    print("  用户1 - 用户名: user1, 密码: hello")
    print("  演示用户 - 用户名: demo, 密码: demo")


    # 启动服务器
    app.run(
        host='127.0.0.1',
        port=8089,
        debug=True,  # 开发模式，生产环境请设置为False
        threaded=True
    )