import pygame
from typing import List
from neural_network import Perceptron, TEST_DATA
import time

prediction_allow_time = 0

pygame.init()

# Set the dimensions of the window
GRID_SIZE = 28
PIXEL_SIZE = 10
SCREEN_WIDTH = GRID_SIZE * PIXEL_SIZE * 1.5
SCREEN_HEIGHT = GRID_SIZE * PIXEL_SIZE

# Create the window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
mouse_down = False


class Drawer:
    def __init__(self):
        self.surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.data = [0.0 for _ in range(28 * 28)]

    def set_data(self, got_data: List[float]):
        self.data = got_data.copy()

    def load_data(self):
        return self.data.copy()
    
    def _put_pixel(self, col: int, row: int, pressure: float, recursion_level: int):
        if recursion_level < 0:
            return

        if (col < 0 or col >= GRID_SIZE) or (row < 0 or row >= GRID_SIZE):
            return

        index = row * GRID_SIZE + col

        self.data[index] += pressure
        if self.data[index] > 255: self.data[index] = 255

        # Left of
        self._put_pixel(col-1, row, pressure * 0.5, recursion_level=recursion_level-1)
        # Right of
        self._put_pixel(col+1, row, pressure * 0.5, recursion_level=recursion_level-1)
        # Up of
        self._put_pixel(col, row-1, pressure * 0.5, recursion_level=recursion_level-1)
        # Below of
        self._put_pixel(col, row+1, pressure * 0.5, recursion_level=recursion_level-1)

        return True

    def handle_event(self):
        if not mouse_down:
            return False

        m_x, m_y = pygame.mouse.get_pos()

        if m_x > PIXEL_SIZE * GRID_SIZE or m_y > PIXEL_SIZE * GRID_SIZE:
            return False

        col = m_x // PIXEL_SIZE
        row = m_y // PIXEL_SIZE

        self._put_pixel(col, row, 16, recursion_level=2)
        return True

    def build_image(self):
        self.surface.fill((0,0,0))

        pixel = pygame.Surface((PIXEL_SIZE, PIXEL_SIZE))

        for i, d in enumerate(self.data):
            row = i // GRID_SIZE
            col = i % GRID_SIZE

            pixel.fill((int(d), int(d), int(d)))

            self.surface.blit(
                pixel,
                (col * PIXEL_SIZE, row * PIXEL_SIZE)
            )

    def draw(self, surface: pygame.Surface):
        self.build_image()
        surface.blit(self.surface, (0, 0))


class Button:
    def __init__(self, x, y, width, height, on_click, text='', color=(0, 0, 255), font=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.text = text
        self.font = font
        self.on_click = on_click

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        if self.text:
            font_surface = self.font.render(self.text, True, (0,0,0))
            font_rect = font_surface.get_rect(center=self.rect.center)
            surface.blit(font_surface, font_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            # handle button click
            if self.on_click:
                self.on_click()


class Stats:
    def __init__(self, x, y, width, height):
        self.results = []
        self.rect = pygame.Rect(x, y, width, height)
        self.surface = pygame.surface.Surface((width, height))

    def set_results(self, results: List[float]):
        self.results = results.copy()

    def build_image(self):
        p_x = p_y = 0
        font = pygame.font.Font(None, 16)

        scalable_rect_height = 5

        for index, result in enumerate(self.results):
            rect = pygame.Rect(p_x, p_y, GRID_SIZE, GRID_SIZE)
            font_surface = font.render(str(index), False, (255, 255, 255))
            font_rect = font_surface.get_rect(center=rect.center)
            self.surface.blit(font_surface, font_rect)
            p_y += font_rect.height + font_rect.height * 0.3

            scalable_rect = pygame.Rect(
                font_rect.x,
                font_rect.y,
                int((self.rect.width - font_rect.width) * result),
                scalable_rect_height
            )
            scalable_rect.x += GRID_SIZE
            scalable_rect.y += int(scalable_rect_height * 0.5)

            pygame.draw.rect(self.surface, (255,255,255), rect=scalable_rect)

    def draw(self, surface: pygame.Surface):
        self.surface.fill((0,0,0))
        self.build_image()
        surface.blit(self.surface, self.rect.topleft)


nn = Perceptron([1])
print('loading')
nn.load()
print('loaded')

drawer = Drawer()
data_line = []

b_x = PIXEL_SIZE * GRID_SIZE + 10
b_y = 10

file = open(TEST_DATA, 'r')

def button_next_handler():
    data = file.readline()
    data = data.replace("\n", "")
    data = data.split(",")
    data = [float(f) for f in data[1:]]
    drawer.set_data(data)
    results = nn.predict(inputs=data)
    stats.set_results(results)

button_font = pygame.font.Font(None, 36)
button_next = Button(x=b_x, y=b_y, width=120, height=GRID_SIZE, on_click=button_next_handler, text='Next', color=(255,255,255), font=button_font)

def recognize_symbol():
    global prediction_allow_time

    data = drawer.load_data()
    if time.time() < prediction_allow_time:  # Is no time yet to come.
        return

    results = nn.predict(inputs=data)
    stats.set_results(results)
    prediction_allow_time = time.time() + 0.01

def button_clear_handler():
    data = [0.0 for _ in range(GRID_SIZE**2)]
    drawer.set_data(data)
    results = nn.predict(inputs=data)
    stats.set_results(results)

b_y += 10 + GRID_SIZE
button_clear = Button(x=b_x, y=b_y, width=120, height=28, on_click=button_clear_handler, text='Clear', color=(255,255,255), font=button_font)

b_y += 10 + GRID_SIZE

stats = Stats(b_x, b_y, 120, 500)

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True

        if event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False

        if drawer.handle_event():
            recognize_symbol()

        button_next.handle_event(event)
        button_clear.handle_event(event)

    # Fill the background with white
    screen.fill((255, 255, 255))
    drawer.draw(screen)
    button_next.draw(screen)
    button_clear.draw(screen)
    stats.draw(screen)

    # Update the display
    pygame.display.update()

