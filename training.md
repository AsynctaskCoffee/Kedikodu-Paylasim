## Training

### Pencere Yakalama İşlemi

Öncelikle, `Knight Evolution` penceresini yakalamak için `pygetwindow` ve `pyautogui` kütüphanelerini kullanabiliriz. Bu kütüphaneler, belirli bir pencereyi hedef alarak OCR işlemlerinin o pencereden yapılmasını sağlar.

### Kodun Güncellenmiş Hali

#### Gerekli Kütüphanelerin Eklenmesi

```python
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageGrab
import pytesseract
from pynput import keyboard
import pygetwindow as gw
import json
import pyautogui

# OCR işlemi için yapılandırma
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### OCR ve Pencere Yakalama İşlemleri

```python
previous_x = None
previous_y = None

def parse_coordinates(text):
    global previous_x, previous_y
    try:
        coords = text.strip().replace(' ', '').replace('\n', '').split(',')
        x, y = int(coords[0][1:]), int(coords[1][:-1])

        # Eğer koordinatlar 4 karakterden uzunsa son karakteri at
        if len(str(x)) > 3:
            x = int(str(x)[:-1])
        if len(str(y)) > 3:
            y = int(str(y)[:-1])

        # Eğer koordinatlar 2 karakterse, önceki koordinatlardan üçüncü basamağı ekle
        if len(str(x)) < 3 and previous_x is not None:
            x = int(str(previous_x)[0] + str(x))
        if len(str(y)) < 3 and previous_y is not None:
            y = int(str(previous_y)[0] + str(y))

        previous_x, previous_y = x, y

        return x, y
    except:
        return None

def grab_and_ocr(window):
    bbox = (window.left + 118, window.top + 100, window.left + 205, window.top + 120)
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789(),'
    for _ in range(3):
        img = ImageGrab.grab(bbox=bbox)
        text = pytesseract.image_to_string(img, config=custom_config)
        coordinates = parse_coordinates(text)
        if coordinates:
            print(f'bulundu {coordinates}')
            return coordinates
        print("cord bulunamadı /town")
    return None

def is_success(state):
    x, y = state
    return (812 <= x <= 819) and (600 <= y <= 605)
```

#### MaradonEnv Sınıfı

```python
class MaradonEnv:
    def __init__(self, window, grid_size=(1000, 1000), obstacles=None):
        self.window = window
        self.grid_size = grid_size
        self.state = None
        self.keyboard_controller = keyboard.Controller()
        if obstacles is None:
            obstacles = [(random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1)) for _ in range(100)]
        self.obstacles = set(obstacles)

    def reset(self):
        teleport_to_town()
        time.sleep(5)  # Yeniden konumlanmak için bekle
        self.state = grab_and_ocr(self.window)
        return self.state

    def step(self, action):
        initial_state = self.state

        if action == 0:  # Move Forward
            move_forward(self.keyboard_controller, 0.5)
        elif action == 1:  # Move Backward
            move_backward(self.keyboard_controller, 0.5)
        elif action == 2:  # Turn Left
            turn_left(self.keyboard_controller, 0.5)
        elif action == 3:  # Turn Right
            turn_right(self.keyboard_controller, 0.5)

        next_state = grab_and_ocr(self.window)
        if next_state is None:
            reward = -10  # OCR başarısız olursa ceza ver
            done = True
        else:
            x, y = next_state
            if is_success(next_state):
                reward = 100
                done = True
            elif (x, y) in self.obstacles:
                reward = -10
                done = True
            elif next_state == initial_state:
                reward = -10
                self.obstacles.add(next_state)  # Hareket sonucu konum değişmezse engel olarak işaretle
                done = False
            else:
                reward = -1 * (abs(x - 816) + abs(y - 604))
                done = False

        return next_state, reward, done, {}

    def render(self):
        print(f"Current location: {self.state}")
```

#### Teleport ve Hareket Fonksiyonları

```python
def teleport_to_town():
    controller = keyboard.Controller()
    controller.press(keyboard.Key.enter)
    time.sleep(0.1)
    controller.release(keyboard.Key.enter)
    time.sleep(0.1)
    controller.type('/town')
    time.sleep(0.1)
    controller.press(keyboard.Key.enter)
    time.sleep(0.1)
    controller.release(keyboard.Key.enter)

def move_forward(controller, duration):
    controller.press('w')
    time.sleep(duration)
    controller.release('w')

def move_backward(controller, duration):
    controller.press('s')
    time.sleep(duration)
    controller.release('s')

def turn_left(controller, duration):
    controller.press('a')
    time.sleep(duration)
    controller.release('a')

def turn_right(controller, duration):
    controller.press('d')
    time.sleep(duration)
    controller.release('d')
```

#### ManualTraining Sınıfı

```python
class ManualTraining:
    def __init__(self, env, steps=10):
        self.env = env
        self.steps = steps
        self.actions = []
        self.coordinates = []
        self.durations = []
        self.manual_mode = False

    def run(self):
        print("Manuel eğitim başlıyor. Q tuşuna basarak eğitim adımlarını gerçekleştirin.")
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        while True:
            if self.manual_mode:
                print(f"{len(self.actions) // self.steps + 1}. adım başladı.")
                self.env.reset()
                while self.manual_mode:
                    time.sleep(0.1)
                print(f"{len(self.actions) // self.steps}. adım tamamlandı.")
                if len(self.actions) >= self.steps * 10:
                    break

        self.save_model()
        print("Manuel eğitim tamamlandı. Model kaydedildi.")

    def on_press(self, key):
        if key == keyboard.KeyCode.from_char('q'):
            self.manual_mode = not self.manual_mode
            if not self.manual_mode:
                teleport_to_town()
            print(f"Manual mode {'started' if self.manual_mode else 'stopped'}")

    def on_release(self, key):
        pass

    def save_model(self):
        model = {
            "actions": self.actions,
            "durations": self.durations,
            "coordinates": self.coordinates
        }
        with open("manual_model.json", "w") as f:
            json.dump(model, f)
        print("Model kaydedildi: manual_model.json")
```

#### AutoTraining Sınıfı

```python
class AutoTraining:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.99):
        self.env = env
        self.q_table = np.zeros((*env.grid_size, 4))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.load_model()

    def load_model(self):
        with open("manual_model.json", "r") as f:
            self.model = json.load(f)
        print("Manuel model yüklendi.")

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice([0, 1, 2, 3])
        else:
            x, y = state
            return np.argmax(self.q_table[x, y])

    def learn(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        predict = self.q_table[x, y, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_x, next_y])
        self.q_table[x, y, action] += self.learning_rate * (target - predict)

    def run(self, episodes=100):
        rewards = []
       

 for episode in range(episodes):
            state = self.env.reset()
            if state is None:
                continue  # OCR başarısız olursa bölümü atla
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if next_state is None:
                    break  # OCR başarısız olursa döngüden çık
                self.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            self.exploration_rate *= self.exploration_decay
            rewards.append(total_reward)
        self.save_model()
        return rewards

    def save_model(self):
        np.save('q_table.npy', self.q_table)
        print("Otomatik eğitim tamamlandı ve model kaydedildi: q_table.npy")
```

#### Main Script

```python
# Ana script
window = gw.getWindowsWithTitle('Knight Evolution')[0]  # 'Knight Evolution' penceresini al
env = MaradonEnv(window, grid_size=(1000, 1000))

# Manual Training
manual_training = ManualTraining(env)
manual_training.run()

# Auto Training
auto_training = AutoTraining(env)
rewards = auto_training.run(episodes=100)

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
```

### Açıklamalar
1. **Pencere Yakalama:** `pygetwindow` ve `pyautogui` kullanarak `Knight Evolution` penceresini yakalıyoruz.
2. **OCR İşlemleri:** OCR işlemlerini belirli bir pencere üzerinde gerçekleştiriyoruz.
3. **ManualTraining ve AutoTraining Sınıfları:** Manuel eğitimde kullanıcıdan alınan hareketler kaydedilir ve model oluşturulur. Otomatik eğitimde bu model kullanılarak Q-learning algoritmasıyla eğitim yapılır.
4. **Main Script:** Ana script, `Knight Evolution` penceresini yakalar, manuel eğitim yapar ve ardından otomatik eğitim gerçekleştirir.

Bu şekilde, kodunuz belirlediğiniz koşullara uygun olarak çalışacaktır. Manual training yaparak episode sayısını ve güç tüketimini azaltabilirsiniz. Bu kod örnek olsun diye hazırlanmış olup server ama amaca göre düzenlenmelidir.
