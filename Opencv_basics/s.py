import cv2
import mediapipe as mp
import random

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

class SnakeGame:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.snake = [(width//2, height//2), (width//2-20, height//2), (width//2+20, height//2)]
        self.food = None
        self.direction = 'RIGHT'
        self.game_over = False

    def reset(self):
        self.snake = [(self.width//2, self.height//2), (self.width//2-20, self.height//2), (self.width//2+20, self.height//2)]
        self.food = None
        self.direction = 'RIGHT'
        self.game_over = False

    def update(self, x, y):
        head = self.snake[0]
        if self.direction == 'UP' and y!= head[1]:
            self.snake.insert(0, (head[0], head[1]-20))
        elif self.direction == 'DOWN' and y!= head[1]:
            self.snake.insert(0, (head[0], head[1]+20))
        elif self.direction == 'LEFT' and x!= head[0]:
            self.snake.insert(0, (head[0]-20, head[1]))
        elif self.direction == 'RIGHT' and x!= head[0]:
            self.snake.insert(0, (head[0]+20, head[1]))

        if self.food:
            if self.food == self.snake[0]:
                self.food = None
                self.snake.append(self.snake[-1])
        else:
            self.snake.pop()

        if self.is_collision():
            self.game_over = True

    def is_collision(self):
        head = self.snake[0]
        if head in self.snake[1:]:
            return True
        if head[0] < 0 or head[0] > self.width or head[1] < 0 or head[1] > self.height:
            return True
        return False

    def place_food(self):
        while True:
            x = random.randint(0, self.width//20)*20
            y = random.randint(0, self.height//20)*20
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

def draw_snake(snake):
    """Draw the snake on the screen."""
    for part in snake:
        cv2.circle(img, part, 10, (0, 255, 0), -1)

def draw_food(food):
    """Draw the food on the screen."""
    cv2.circle(img, food, 10, (0, 0, 255), -1)

def main():
    cap = cv2.VideoCapture(0)
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    game = SnakeGame(width=640, height=480)

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # Convert BGR image to RGB.
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image and find hand landmarks.
        result = hands.process(rgb_img)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the position of the middle finger tip.
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                x, y = int(middle_finger_tip.x * img.shape[1]), int(middle_finger_tip.y * img.shape[0])

                # Update the game state based on hand position.
                game.update(x, y)

                draw_snake(game.snake)
                draw_food(game.food)

                if game.food:
                    game.place_food()

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
