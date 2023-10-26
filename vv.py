import pygame
import sys
import math
import random

# 初始化Pygame
pygame.init()

# 设置屏幕尺寸和颜色
screen_width, screen_height = 800, 600
bg_color = (255, 255, 255)

# 创建屏幕
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Circle Animation")

# 定义大圆圈和小圆圈的半径
big_circle_radius = 200
small_circle_radius = 20

# 定义小圆圈的颜色
blue_circle_index = random.randint(0, 11)  # 随机选择一个小圆圈作为蓝色
small_circle_colors = [(0, 0, 255) if i == blue_circle_index else (0, 0, 0) for i in range(12)]

# 定义小圆圈的位置和速度向量
x = [random.randint(big_circle_radius, screen_width - big_circle_radius) for _ in range(12)]
y = [random.randint(big_circle_radius, screen_height - big_circle_radius) for _ in range(12)]
speeds = [(0, 0) for _ in range(12)]  # 初始速度为零

# 吸引力参数
attraction_factor = 0.1  # 调整吸引力的强度

# 主循环
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill(bg_color)

    # 绘制大圆圈
    pygame.draw.circle(screen, (0, 0, 0), (screen_width // 2, screen_height // 2), big_circle_radius, 2)

    # 更新小圆圈的位置和速度
    for i in range(12):
        # 计算小圆圈受到的吸引力向量
        dx = (screen_width // 2) - x[i]
        dy = (screen_height // 2) - y[i]
        distance = max(math.hypot(dx, dy), 1)
        acceleration = (dx / distance * attraction_factor, dy / distance * attraction_factor)

        # 更新速度向量
        speeds[i] = (speeds[i][0] + acceleration[0], speeds[i][1] + acceleration[1])

        # 限制速度以避免小圆圈重叠
        speed_magnitude = math.hypot(speeds[i][0], speeds[i][1])
        if speed_magnitude > 3:
            speeds[i] = (speeds[i][0] * 3 / speed_magnitude, speeds[i][1] * 3 / speed_magnitude)

        # 更新小圆圈的位置
        x[i] += speeds[i][0]
        y[i] += speeds[i][1]

        pygame.draw.circle(screen, small_circle_colors[i], (int(x[i]), int(y[i])), small_circle_radius)

    pygame.display.flip()
    clock.tick(60)









