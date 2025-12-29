import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import pygame
import random
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================== PYGAME SETUP ==================
pygame.init()
WIDTH, HEIGHT = 600, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Gesture Snake Game")

clock = pygame.time.Clock()
snake_block = 20
font = pygame.font.SysFont("arial", 26)

GREEN = (0, 200, 0)
RED = (200, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# ================== MEDIAPIPE TASKS ==================
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ================== GAME VARIABLES ==================
snake = [(300, 300)]
direction = "RIGHT"
food = (
    random.randrange(0, WIDTH, snake_block),
    random.randrange(0, HEIGHT, snake_block)
)

prev_x, prev_y = None, None
smooth_x, smooth_y = None, None

alpha = 0.25        # EMA smoothing (FIXED)
DEAD_ZONE = 20
frame_timestamp_ms = 0

# ================== GAME LOOP ==================
running = True
while running:
    clock.tick(10)
    win.fill(BLACK)

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    frame_timestamp_ms += 100
    result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    # ================= HAND TRACKING =================
    if result.hand_landmarks and result.hand_world_landmarks:
        landmarks = result.hand_landmarks[0]
        index_finger = landmarks[8]

        h, w, _ = frame.shape
        raw_x = int(index_finger.x * w)
        raw_y = int(index_finger.y * h)

        # EMA smoothing
        if smooth_x is None:
            smooth_x, smooth_y = raw_x, raw_y
        else:
            smooth_x = int(alpha * raw_x + (1 - alpha) * smooth_x)
            smooth_y = int(alpha * raw_y + (1 - alpha) * smooth_y)

        cx, cy = smooth_x, smooth_y

        if prev_x is not None and prev_y is not None:
            dx, dy = cx - prev_x, cy - prev_y

            if abs(dx) > abs(dy):
                if dx > DEAD_ZONE and direction != "LEFT":
                    direction = "RIGHT"
                elif dx < -DEAD_ZONE and direction != "RIGHT":
                    direction = "LEFT"
            else:
                if dy > DEAD_ZONE and direction != "UP":
                    direction = "DOWN"
                elif dy < -DEAD_ZONE and direction != "DOWN":
                    direction = "UP"

        prev_x, prev_y = cx, cy
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

    # ================= MOVE SNAKE =================
    x, y = snake[0]

    if direction == "UP":
        y -= snake_block
    elif direction == "DOWN":
        y += snake_block
    elif direction == "LEFT":
        x -= snake_block
    elif direction == "RIGHT":
        x += snake_block

    # -------- BOUNCE OFF WALLS --------
    if x < 0:
        x = 0
        direction = "RIGHT"
    elif x >= WIDTH:
        x = WIDTH - snake_block
        direction = "LEFT"

    if y < 0:
        y = 0
        direction = "DOWN"
    elif y >= HEIGHT:
        y = HEIGHT - snake_block
        direction = "UP"

    new_head = (x, y)
    snake.insert(0, new_head)

    if new_head == food:
        food = (
            random.randrange(0, WIDTH, snake_block),
            random.randrange(0, HEIGHT, snake_block)
        )
    else:
        snake.pop()

    # ================= DRAW =================
    for part in snake:
        pygame.draw.rect(win, GREEN, (*part, snake_block, snake_block))

    pygame.draw.rect(win, RED, (*food, snake_block, snake_block))

    score = (len(snake) - 1) * 10
    win.blit(font.render(f"Score: {score}", True, WHITE), (10, 10))

    pygame.display.update()
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
pygame.quit()
