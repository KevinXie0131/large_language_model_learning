// 获取DOM元素
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const scoreElement = document.getElementById('score');
const startBtn = document.getElementById('startBtn');
const restartBtn = document.getElementById('restartBtn');

// 游戏配置
const GRID_SIZE = 20;  // 格子大小
const TILE_COUNT = canvas.width / GRID_SIZE;  // 格子数量

// 游戏状态
let snake = [];
let food = { x: 0, y: 0 };
let direction = { x: 0, y: 0 };
let nextDirection = { x: 0, y: 0 };
let score = 0;
let gameLoop = null;
let isRunning = false;

// 初始化蛇
function initSnake() {
    snake = [
        { x: 10, y: 10 },
        { x: 9, y: 10 },
        { x: 8, y: 10 }
    ];
    direction = { x: 1, y: 0 };
    nextDirection = { x: 1, y: 0 };
}

// 生成食物
function generateFood() {
    let validPosition = false;
    while (!validPosition) {
        food.x = Math.floor(Math.random() * TILE_COUNT);
        food.y = Math.floor(Math.random() * TILE_COUNT);
        
        validPosition = !snake.some(segment => 
            segment.x === food.x && segment.y === food.y
        );
    }
}

// 绘制游戏
function draw() {
    // 清空画布
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 绘制网格线（可选）
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    for (let i = 0; i <= TILE_COUNT; i++) {
        ctx.beginPath();
        ctx.moveTo(i * GRID_SIZE, 0);
        ctx.lineTo(i * GRID_SIZE, canvas.height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i * GRID_SIZE);
        ctx.lineTo(canvas.width, i * GRID_SIZE);
        ctx.stroke();
    }
    
    // 绘制蛇
    snake.forEach((segment, index) => {
        // 蛇头和蛇身不同的颜色
        const gradient = ctx.createRadialGradient(
            segment.x * GRID_SIZE + GRID_SIZE / 2,
            segment.y * GRID_SIZE + GRID_SIZE / 2,
            0,
            segment.x * GRID_SIZE + GRID_SIZE / 2,
            segment.y * GRID_SIZE + GRID_SIZE / 2,
            GRID_SIZE / 2
        );
        
        if (index === 0) {
            gradient.addColorStop(0, '#00ff88');
            gradient.addColorStop(1, '#00cc6a');
        } else {
            gradient.addColorStop(0, '#4ade80');
            gradient.addColorStop(1, '#22c55e');
        }
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.roundRect(
            segment.x * GRID_SIZE + 1,
            segment.y * GRID_SIZE + 1,
            GRID_SIZE - 2,
            GRID_SIZE - 2,
            5
        );
        ctx.fill();
        
        // 给蛇头添加眼睛
        if (index === 0) {
            ctx.fillStyle = '#000';
            const eyeSize = 4;
            const eyeOffset = 5;
            
            // 根据方向调整眼睛位置
            let dx1, dy1, dx2, dy2;
            if (direction.x === 1) {
                dx1 = 10; dy1 = 5; dx2 = 10; dy2 = 15;
            } else if (direction.x === -1) {
                dx1 = 5; dy1 = 5; dx2 = 5; dy2 = 15;
            } else if (direction.y === -1) {
                dx1 = 5; dy1 = 5; dx2 = 15; dy2 = 5;
            } else {
                dx1 = 5; dy1 = 10; dx2 = 15; dy2 = 10;
            }
            
            ctx.beginPath();
            ctx.arc(
                segment.x * GRID_SIZE + dx1,
                segment.y * GRID_SIZE + dy1,
                eyeSize, 0, Math.PI * 2
            );
            ctx.fill();
            ctx.beginPath();
            ctx.arc(
                segment.x * GRID_SIZE + dx2,
                segment.y * GRID_SIZE + dy2,
                eyeSize, 0, Math.PI * 2
            );
            ctx.fill();
        }
    });
    
    // 绘制食物
    const foodGradient = ctx.createRadialGradient(
        food.x * GRID_SIZE + GRID_SIZE / 2,
        food.y * GRID_SIZE + GRID_SIZE / 2,
        0,
        food.x * GRID_SIZE + GRID_SIZE / 2,
        food.y * GRID_SIZE + GRID_SIZE / 2,
        GRID_SIZE / 2
    );
    foodGradient.addColorStop(0, '#ff6b6b');
    foodGradient.addColorStop(1, '#ee5a5a');
    
    ctx.fillStyle = foodGradient;
    ctx.beginPath();
    ctx.arc(
        food.x * GRID_SIZE + GRID_SIZE / 2,
        food.y * GRID_SIZE + GRID_SIZE / 2,
        GRID_SIZE / 2 - 2,
        0, Math.PI * 2
    );
    ctx.fill();
    
    // 食物发光效果
    ctx.shadowColor = '#ff6b6b';
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
}

// 更新游戏状态
function update() {
    direction = { ...nextDirection };
    
    // 计算新蛇头位置
    const head = { 
        x: snake[0].x + direction.x, 
        y: snake[0].y + direction.y 
    };
    
    // 检查碰撞（墙壁）
    if (head.x < 0 || head.x >= TILE_COUNT || 
        head.y < 0 || head.y >= TILE_COUNT) {
        gameOver();
        return;
    }
    
    // 检查碰撞（自身）
    if (snake.some(segment => segment.x === head.x && segment.y === head.y)) {
        gameOver();
        return;
    }
    
    // 添加新蛇头
    snake.unshift(head);
    
    // 检查是否吃到食物
    if (head.x === food.x && head.y === food.y) {
        score += 10;
        scoreElement.textContent = score;
        generateFood();
    } else {
        // 移除蛇尾
        snake.pop();
    }
}

// 游戏结束
function gameOver() {
    isRunning = false;
    clearInterval(gameLoop);
    
    // 显示游戏结束画面
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = '#ff6b6b';
    ctx.font = 'bold 40px Microsoft YaHei';
    ctx.textAlign = 'center';
    ctx.fillText('游戏结束!', canvas.width / 2, canvas.height / 2 - 20);
    
    ctx.fillStyle = '#fff';
    ctx.font = '20px Microsoft YaHei';
    ctx.fillText(`最终分数: ${score}`, canvas.width / 2, canvas.height / 2 + 30);
    
    startBtn.textContent = '开始游戏';
    startBtn.disabled = false;
}

// 开始游戏
function startGame() {
    if (isRunning) return;
    
    initSnake();
    generateFood();
    score = 0;
    scoreElement.textContent = score;
    isRunning = true;
    startBtn.textContent = '游戏中...';
    startBtn.disabled = true;
    
    // 100ms刷新一次
    gameLoop = setInterval(() => {
        update();
        draw();
    }, 100);
}

// 键盘控制
document.addEventListener('keydown', (e) => {
    if (!isRunning) return;
    
    switch (e.key) {
        case 'ArrowUp':
        case 'w':
        case 'W':
            if (direction.y !== 1) {
                nextDirection = { x: 0, y: -1 };
            }
            break;
        case 'ArrowDown':
        case 's':
        case 'S':
            if (direction.y !== -1) {
                nextDirection = { x: 0, y: 1 };
            }
            break;
        case 'ArrowLeft':
        case 'a':
        case 'A':
            if (direction.x !== 1) {
                nextDirection = { x: -1, y: 0 };
            }
            break;
        case 'ArrowRight':
        case 'd':
        case 'D':
            if (direction.x !== -1) {
                nextDirection = { x: 1, y: 0 };
            }
            break;
    }
    
    // 防止方向键滚动页面
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'w', 'a', 's', 'd', 'W', 'A', 'S', 'D'].includes(e.key)) {
        e.preventDefault();
    }
});

// 按钮事件
startBtn.addEventListener('click', startGame);
restartBtn.addEventListener('click', () => {
    clearInterval(gameLoop);
    isRunning = false;
    startBtn.textContent = '开始游戏';
    startBtn.disabled = false;
    initSnake();
    generateFood();
    draw();
    startGame();
});

// 初始化显示
initSnake();
generateFood();
draw();