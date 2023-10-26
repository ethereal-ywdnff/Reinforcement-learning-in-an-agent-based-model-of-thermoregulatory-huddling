import pygame
import sys
import math
import random

# 初始化Pygame
pygame.init()

# 设置屏幕尺寸和颜色
screen_width, screen_height = 600, 600
border_color = (0, 0, 0)  # 边框颜色为黑色
bg_color = (255, 255, 255)

# 创建屏幕
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Circle Animation")

# 定义大圆圈和小圆圈的半径
big_circle_radius = 200
small_circle_radius = 20

# 定义小圆圈的颜色
blue_circle_index = random.randint(0, 11)  # 随机选择一个小圆圈作为初始蓝色
small_circle_colors = [(0, 0, 255) if i == blue_circle_index else (192, 192, 192) for i in range(12)]

# 定义小圆圈的位置和速度向量
x = [random.randint(big_circle_radius, screen_width - big_circle_radius) for _ in range(12)]
y = [random.randint(big_circle_radius, screen_height - big_circle_radius) for _ in range(12)]
speeds = [(0, 0) for _ in range(12)]  # 初始速度为零

# 吸引力参数
attraction_factor = 0.1  # 调整吸引力的强度

# 颜色变动参数
color_change_interval = 20  # 每隔多少帧改变颜色
frame_counter = 0

# 距离中心点的最小距离
min_distance_to_center = big_circle_radius - small_circle_radius - 5  # 略小于大圆圈半径

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
        # pygame.draw.circle(screen, border_color, (int(x[i]), int(y[i])), small_circle_radius + 1, 1)

    # 每隔一定帧数改变颜色
    frame_counter += 1
    if frame_counter >= color_change_interval:
        frame_counter = 0
        # 找到当前蓝色小圆圈的索引
        blue_indices = [i for i, color in enumerate(small_circle_colors) if color == (0, 0, 255)]
        if blue_indices:
            for index in blue_indices:
                # 将当前蓝色小圆圈颜色改回黑色
                small_circle_colors[index] = (0, 0, 0)
            # 随机选择一个新的小圆圈作为蓝色
            new_blue_index = random.choice([i for i in range(12) if i not in blue_indices])
            small_circle_colors[new_blue_index] = (0, 0, 255)
        for i in range(12):
            if i != new_blue_index:
                distance_to_center = math.hypot(x[i] - screen_width // 2, y[i] - screen_height // 2)
                if distance_to_center <= min_distance_to_center:
                    # 将小圆圈颜色改为红色
                    small_circle_colors[i] = (255, 0, 0)
                    for t in range(12):
                        if t != i and t != new_blue_index:
                            small_circle_colors[t] = (192, 192, 192)

    pygame.display.flip()
    clock.tick(60)




