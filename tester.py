import pygame
from typing import List
from neural_network import Perceptron, TEST_DATA

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

    def handle_event(self):
        if not mouse_down:
            return False
        m_x, m_y = pygame.mouse.get_pos()

        if m_x > PIXEL_SIZE * GRID_SIZE or m_y > PIXEL_SIZE * GRID_SIZE:
            return False

        col = m_x // PIXEL_SIZE
        row = m_y // PIXEL_SIZE

        index = row * GRID_SIZE + col

        self.data[index] += 16
        if self.data[index] > 255: self.data[index] = 255
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
            p_y += font_rect.height + font_rect.height * 0.4

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
    data_line = file.readline()
    data_line = data_line.replace("\n", "")
    data_line = data_line.split(",")
    data_line = [float(f) for f in data_line[1:]]
    drawer.set_data(data_line)
    results = nn.predict(inputs=data_line)
    stats.set_results(results)

button_font = pygame.font.Font(None, 36)
button_next = Button(x=b_x, y=b_y, width=120, height=GRID_SIZE, on_click=button_next_handler, text='Next', color=(255,255,255), font=button_font)

def button_recognize_handler():
    data = drawer.load_data()
    results = nn.predict(inputs=data)
    stats.set_results(results)

b_y += 10 + GRID_SIZE
button_recognize = Button(x=b_x, y=b_y, width=120, height=28, on_click=button_recognize_handler, text='Recognize', color=(255,255,255), font=button_font)

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
            button_recognize_handler()
        button_next.handle_event(event)
        button_recognize.handle_event(event)

    # Fill the background with white
    screen.fill((255, 255, 255))
    drawer.draw(screen)
    button_next.draw(screen)
    button_recognize.draw(screen)
    stats.draw(screen)

    # Update the display
    pygame.display.update()


