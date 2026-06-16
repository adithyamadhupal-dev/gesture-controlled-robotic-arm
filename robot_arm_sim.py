import pygame
import math

pygame.init()

WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robotic Arm Simulator")

clock = pygame.time.Clock()

# Arm settings
base_x = WIDTH // 2
base_y = HEIGHT - 120

joint1_angle = -45
joint2_angle = 60

arm1_length = 180
arm2_length = 140

running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # Temporary controls
    if keys[pygame.K_LEFT]:
        joint1_angle -= 1

    if keys[pygame.K_RIGHT]:
        joint1_angle += 1

    if keys[pygame.K_UP]:
        joint2_angle += 1

    if keys[pygame.K_DOWN]:
        joint2_angle -= 1

    screen.fill((25, 25, 35))

    # First arm segment
    x1 = base_x + arm1_length * math.cos(math.radians(joint1_angle))
    y1 = base_y + arm1_length * math.sin(math.radians(joint1_angle))

    # Second arm segment
    x2 = x1 + arm2_length * math.cos(math.radians(joint1_angle + joint2_angle))
    y2 = y1 + arm2_length * math.sin(math.radians(joint1_angle + joint2_angle))

    # Draw base
    pygame.draw.circle(screen, (220, 220, 220), (base_x, base_y), 20)

    # Draw arm segments
    pygame.draw.line(screen, (0, 180, 255), (base_x, base_y), (x1, y1), 12)
    pygame.draw.line(screen, (255, 180, 0), (x1, y1), (x2, y2), 10)

    # Draw joints
    pygame.draw.circle(screen, (255, 255, 255), (int(x1), int(y1)), 12)
    pygame.draw.circle(screen, (255, 100, 100), (int(x2), int(y2)), 10)

    font = pygame.font.SysFont(None, 36)

    text = font.render(
        f"Joint1: {joint1_angle}  Joint2: {joint2_angle}",
        True,
        (255, 255, 255)
    )

    screen.blit(text, (20, 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()