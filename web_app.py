"""
Flask Web应用 - MNIST手写数字识别
提供网页画板功能，可以手写数字并实时识别
"""

import os
import io
import base64
import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template_string, request, jsonify

import config
from models.cnn_model import MNIST_CNNModel

app = Flask(__name__)

# 全局变量
device = None
model = None


def load_model():
    """加载训练好的模型"""
    global model, device
    
    # 获取设备
    if config.USE_CUDA and torch.cuda.is_available():
        device = torch.device(config.DEVICE)
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    # 创建模型
    model = MNIST_CNNModel()
    
    # 加载权重
    model_path = os.path.join(config.MODEL_DIR, 'mnist_cnn_best.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型已加载: {model_path}")
    else:
        print("警告: 未找到训练好的模型!")
    
    model = model.to(device)
    model.eval()


def preprocess_image(image):
    """
    预处理图像 - 与训练时保持一致
    
    Args:
        image: PIL图像（用户在白色背景上画的黑色数字）
    
    Returns:
        tensor: 处理后的张量
    """
    # 转换为灰度图
    if image.mode != 'L':
        image = image.convert('L')
    
    # 调整大小为28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 转换为numpy数组
    img_array = np.array(image).astype('float32')
    
    # 反转颜色：白色背景(255)上的黑色数字(0) -> 黑色背景(0)上的白色数字(255)
    # MNIST数据集是黑色背景上的白色数字
    img_array = 255 - img_array
    
    # 归一化到[0,1]
    img_array = img_array / 255.0
    
    # 应用与训练时相同的标准化 (mean=0.1307, std=0.3081)
    img_array = (img_array - 0.1307) / 0.3081
    
    # 转换为张量并添加批次和通道维度 [1, 1, 28, 28]
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor


def predict_digit(image_tensor):
    """
    预测数字
    
    Args:
        image_tensor: 预处理后的图像张量
    
    Returns:
        predicted_digit: 预测的数字
        probabilities: 各类别的概率
        confidence: 置信度
    """
    global model, device
    
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        
        # 计算概率
        probabilities = torch.softmax(output, dim=1)[0]
        
        # 获取预测结果
        confidence, predicted = torch.max(probabilities, dim=0)
        
        predicted_digit = predicted.item()
        confidence = confidence.item()
    
    return predicted_digit, probabilities.cpu().numpy(), confidence


# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MNIST手写数字识别</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 30px;
            max-width: 600px;
            width: 100%;
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 20px;
            font-size: 28px;
        }
        
        .canvas-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }
        
        .drawing-area {
            position: relative;
            width: 280px;
            height: 280px;
            border: 3px solid #667eea;
            border-radius: 10px;
            background: white;
            cursor: crosshair;
            overflow: hidden;
        }
        
        canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }
        
        .btn-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        button {
            padding: 12px 24px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        
        .btn-primary {
            background: #667eea;
            color: white;
        }
        
        .btn-primary:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: #f56565;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #e53e3e;
            transform: translateY(-2px);
        }
        
        .btn-success {
            background: #48bb78;
            color: white;
        }
        
        .btn-success:hover {
            background: #38a169;
            transform: translateY(-2px);
        }
        
        .result-container {
            margin-top: 25px;
            text-align: center;
        }
        
        .prediction {
            font-size: 48px;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }
        
        .confidence {
            font-size: 18px;
            color: #666;
            margin-bottom: 20px;
        }
        
        .probabilities {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 15px;
        }
        
        .prob-item {
            background: #f0f0f0;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
        }
        
        .prob-item.highlight {
            background: #667eea;
            color: white;
        }
        
        .prob-value {
            font-weight: bold;
        }
        
        .tips {
            margin-top: 20px;
            padding: 15px;
            background: #f7fafc;
            border-radius: 8px;
            font-size: 14px;
            color: #666;
        }
        
        .tips h3 {
            color: #333;
            margin-bottom: 8px;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-container {
            animation: fadeIn 0.3s ease-out;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 MNIST手写数字识别</h1>
        
        <div class="canvas-container">
            <div class="drawing-area">
                <canvas id="canvas" width="280" height="280"></canvas>
            </div>
            
            <div class="btn-group">
                <button class="btn-primary" onclick="recognize()">🔍 识别数字</button>
                <button class="btn-secondary" onclick="clearCanvas()">🗑️ 清除</button>
                <button class="btn-success" onclick="useExample()">📝 使用示例</button>
            </div>
        </div>
        
        <div class="result-container" id="result" style="display: none;">
            <div>预测结果：</div>
            <div class="prediction" id="prediction">-</div>
            <div class="confidence">置信度：<span id="confidence">-</span></div>
            <div class="probabilities" id="probabilities"></div>
        </div>
        
        <div class="tips">
            <h3>💡 使用说明：</h3>
            <ul>
                <li>在白色画板区域用鼠标书写数字(0-9)</li>
                <li>点击"识别数字"按钮进行预测</li>
                <li>点击"清除"按钮重新书写</li>
                <li>点击"使用示例"查看MNIST数据集样本</li>
            </ul>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;
        
        // 设置画笔样式
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        // 鼠标事件
        canvas.addEventListener('mousedown', (e) => {
            isDrawing = true;
            [lastX, lastY] = [e.offsetX, e.offsetY];
        });
        
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', () => isDrawing = false);
        canvas.addEventListener('mouseout', () => isDrawing = false);
        
        // 触摸事件（支持移动端）
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            isDrawing = true;
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            lastX = touch.clientX - rect.left;
            lastY = touch.clientY - rect.top;
        });
        
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (!isDrawing) return;
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();
            [lastX, lastY] = [x, y];
        });
        
        canvas.addEventListener('touchend', () => isDrawing = false);
        
        function draw(e) {
            if (!isDrawing) return;
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
            [lastX, lastY] = [e.offsetX, e.offsetY];
        }
        
        function clearCanvas() {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('result').style.display = 'none';
        }
        
        function getImageData() {
            return canvas.toDataURL('image/png');
        }
        
        async function recognize() {
            const imageData = getImageData();
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ image: imageData })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert('识别失败: ' + data.error);
                    return;
                }
                
                // 显示结果
                document.getElementById('result').style.display = 'block';
                document.getElementById('prediction').textContent = data.prediction;
                document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(2) + '%';
                
                // 显示概率分布
                const probContainer = document.getElementById('probabilities');
                probContainer.innerHTML = '';
                data.probabilities.forEach((prob, idx) => {
                    const item = document.createElement('div');
                    item.className = 'prob-item' + (idx === data.prediction ? ' highlight' : '');
                    item.innerHTML = `<span>${idx}</span>: <span class="prob-value">${(prob * 100).toFixed(1)}%</span>`;
                    probContainer.appendChild(item);
                });
                
            } catch (error) {
                alert('请求失败: ' + error);
            }
        }
        
        async function useExample() {
            try {
                const response = await fetch('/random_example');
                const data = await response.json();
                
                if (data.error) {
                    alert('获取示例失败: ' + data.error);
                    return;
                }
                
                // 显示示例图像
                const img = new Image();
                img.onload = function() {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = data.image;
                
            } catch (error) {
                alert('请求失败: ' + error);
            }
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        
        # 从base64获取图像数据
        image_data = image_data.split(',')[1]  # 去除data:image/png;base64,
        image_bytes = base64.b64decode(image_data)
        
        # 转换为PIL图像
        image = Image.open(io.BytesIO(image_bytes))
        
        # 预处理
        img_tensor = preprocess_image(image)
        
        # 预测
        pred, probs, conf = predict_digit(img_tensor)
        
        return jsonify({
            'prediction': int(pred),
            'confidence': float(conf),
            'probabilities': probs.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/random_example', methods=['GET'])
def random_example():
    """获取随机示例"""
    try:
        from torchvision import datasets
        
        # 加载测试集
        test_dataset = datasets.MNIST(
            root=config.MNIST_DATA_PATH,
            train=False,
            download=True
        )
        
        # 随机选择一个样本
        np.random.seed()
        idx = np.random.randint(0, len(test_dataset))
        img, label = test_dataset[idx]
        
        # 转换为base64
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = 'data:image/png;base64,' + base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'image': img_base64,
            'label': int(label)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """主函数"""
    print("=" * 60)
    print(" MNIST手写数字识别 - Web应用")
    print("=" * 60)
    
    # 加载模型
    print("\n加载模型...")
    load_model()
    
    # 启动Flask应用
    print("\n启动Web服务器...")
    print("请在浏览器中打开: http://localhost:5000")
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()