import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

class SnakeGame:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.snake = [(width//2, height//2)]
        self.food = None
        self.score = 0
        self.game_over = False
        self.place_food()

    def reset(self):
        self.snake = [(self.width//2, self.height//2)]
        self.food = None
        self.score = 0
        self.game_over = False
        self.place_food()

    def update(self, x, y):
        head = (x, y)
        self.snake.insert(0, head)

        if self.food and head == self.food:
            self.food = None
            self.place_food()
            self.score += 1
        else:
            self.snake.pop()

        if self.is_collision():
            self.game_over = True

    def is_collision(self):
        head = self.snake[0]
        if head in self.snake[1:]:
            return True
        if head[0] < 0 or head[0] >= self.width or head[1] < 0 or head[1] >= self.height:
            return True
        return False

    def place_food(self):
        while True:
            x = random.randint(0, (self.width // 20) - 1) * 20
            y = random.randint(0, (self.height // 20) - 1) * 20
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

def draw_snake(img, snake):
    for part in snake:
        cv2.circle(img, part, 10, (0, 255, 0), -1)

def draw_food(img, food):
    if food:
        cv2.circle(img, food, 10, (0, 0, 255), -1)

def display_score(img, score):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'Score: {score}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    game = SnakeGame(width=640, height=480)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_img)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * img.shape[1]), int(index_finger_tip.y * img.shape[0])
                
                if not game.game_over:
                    game.update(x, y)

        draw_snake(img, game.snake)
        draw_food(img, game.food)
        display_score(img, game.score)

        cv2.imshow("Snake Game", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if game.game_over:
            game.reset()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
