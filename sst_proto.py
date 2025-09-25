# sst_mouse_task.py
import pygame
import random
import math
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

# ---------- Settings ----------
SCREEN_WIDTH, SCREEN_HEIGHT = 900, 700
BG_COLOR = (240, 240, 240)
FPS = 60
NUM_ARROWS = 40
MOVE_DURATION = 800  # ms round duration
ROUNDS = 10
PERC_OPTIONS = [0, 10, 25, 50, 80]
CSV_FILE = "results.csv"
MIN_ARROW_DIST = 28
ARROW_LENGTH = 20

# ---------- Colors ----------
BLACK = (0, 0, 0)
BLUE = (0, 0, 220)
GREEN = (0, 180, 0)
RED = (220, 0, 0)
GRAY = (150, 150, 150)

# ---------- Init ----------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Arrow Motion Task")
big_font = pygame.font.SysFont("Arial", 72)
med_font = pygame.font.SysFont("Arial", 36)
small_font = pygame.font.SysFont("Arial", 22)
clock = pygame.time.Clock()

# ---------- UI Geometry ----------
line_y = SCREEN_HEIGHT * 3 // 4
start_button = pygame.Rect(SCREEN_WIDTH//2 - 70, 625, 140, 60)
results_button = pygame.Rect(SCREEN_WIDTH//2 - 70, 625, 140, 60)  # NEW

left_button = pygame.Rect(20, 20, 110, 50)
right_button = pygame.Rect(SCREEN_WIDTH - 130, 20, 110, 50)
ARROW_X_MIN = 120
ARROW_X_MAX = SCREEN_WIDTH - 120
ARROW_Y_MIN = 100
ARROW_Y_MAX = line_y - 40


# ---------- Arrow class ----------
class Arrow:
    def __init__(self, x, y, angle_deg, special=False):
        self.x = float(x)
        self.y = float(y)
        self.angle = angle_deg
        self.special = special
        self.speed = 2.5 if special else 0.0
        self.dx = math.cos(math.radians(self.angle)) * self.speed
        self.dy = math.sin(math.radians(self.angle)) * self.speed

    def update(self, moving):
        if moving and self.special:
            self.x += self.dx
            self.y += self.dy

    def draw(self, surface):
        end_x = self.x + ARROW_LENGTH * math.cos(math.radians(self.angle))
        end_y = self.y + ARROW_LENGTH * math.sin(math.radians(self.angle))
        pygame.draw.line(surface, BLACK, (self.x, self.y), (end_x, end_y), 3)
        ha1 = math.radians(self.angle + 150)
        ha2 = math.radians(self.angle - 150)
        pygame.draw.line(surface, BLACK, (end_x, end_y),
                         (end_x + 10 * math.cos(ha1), end_y + 10 * math.sin(ha1)), 2)
        pygame.draw.line(surface, BLACK, (end_x, end_y),
                         (end_x + 10 * math.cos(ha2), end_y + 10 * math.sin(ha2)), 2)

# ---------- Utility ----------
def non_overlapping_positions(n, xmin, xmax, ymin, ymax, min_dist, max_attempts=5000):
    positions = []
    attempts = 0
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        x = random.randint(xmin, xmax)
        y = random.randint(ymin, ymax)
        ok = True
        for (px, py) in positions:
            if (px - x)**2 + (py - y)**2 < min_dist**2:
                ok = False
                break
        if ok:
            positions.append((x, y))
    if len(positions) < n:
        while len(positions) < n:
            positions.append((random.randint(xmin, xmax), random.randint(ymin, ymax)))
    return positions

def create_arrows(move_percent, global_dir_deg):
    arrows = []
    num_special = NUM_ARROWS * move_percent // 100
    special_indices = set(random.sample(range(NUM_ARROWS), num_special))
    pos_list = non_overlapping_positions(NUM_ARROWS, ARROW_X_MIN, ARROW_X_MAX, ARROW_Y_MIN, ARROW_Y_MAX, MIN_ARROW_DIST)
    for i, (x, y) in enumerate(pos_list):
        if i in special_indices:
            arrows.append(Arrow(x, y, global_dir_deg, special=True))
        else:
            arrows.append(Arrow(x, y, random.randint(0, 359), special=False))
    return arrows

def polygon_area(points):
    if len(points) < 3:
        return 0.0
    a = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0

def compute_mouse_kinematics(mouse_trace, fps=60):
    if len(mouse_trace) < 2:
        return np.array([]), [], np.array([]), []

    velocities = [
        ((mouse_trace[i][1] - mouse_trace[i-1][1]) /
         max((mouse_trace[i][0] - mouse_trace[i-1][0]), 1e-6)) * fps
        for i in range(1, len(mouse_trace))
    ]
    times = np.linspace(0, 1, len(velocities))

    accelerations = [
        (velocities[i] - velocities[i-1]) / max(times[i] - times[i-1], 1e-9)
        for i in range(1, len(velocities))
    ]
    acc_times = times[1:]

    return times, velocities, acc_times, accelerations

# ---------- CSV ----------
def setup_csv():
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Start_Pos", "End_Pos", "Arrow_Direction", "Choice",
                         "Move_Percent", "SpeedZeroFlag", "CrossedLine", "AUC"])

def log_result(row):
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

# ---------- Prepare rounds ----------
move_percents = PERC_OPTIONS * 2
random.shuffle(move_percents)

# ---------- Main ----------
def main():
    setup_csv()
    round_index = 0
    state = "waiting_start"
    arrows = []
    moving = False
    move_start_time = None
    move_percent = 0
    global_dir_deg = 0
    arrow_direction_str = None
    start_pos = None
    end_pos = None
    mouse_path = []
    show_message_until = None
    all_mouse_paths = []

    while True:
        dt = clock.tick(FPS)
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if state == "waiting_start" and start_button.collidepoint(mouse_pos):
                    if round_index >= ROUNDS:
                        continue
                    move_percent = move_percents[round_index]
                    global_dir_deg = random.choice([0, 180])
                    arrow_direction_str = "RIGHT" if global_dir_deg == 0 else "LEFT"
                    start_pos = mouse_pos
                    arrows = create_arrows(move_percent, global_dir_deg)
                    moving = True
                    move_start_time = pygame.time.get_ticks()
                    state = "playing"
                    mouse_path = []

                elif state == "finished" and round_index >= ROUNDS:
                    if results_button.collidepoint(mouse_pos):
                        show_results(all_mouse_paths)

        now = pygame.time.get_ticks()
        if state == "playing":
            if moving and (pygame.time.get_ticks() - move_start_time) > MOVE_DURATION:
                moving = False
                end_pos = None
                choice = "NONE"
                auc, crossed_line = compute_auc_and_cross(start_pos, mouse_path, end_pos, move_percent)
                log_result([round_index + 1, start_pos, end_pos, arrow_direction_str, choice,
                            move_percent, int(move_percent == 0), int(crossed_line), auc])
                all_mouse_paths.extend(mouse_path)
                if move_percent > 0:
                    show_message_until = pygame.time.get_ticks() + 1000
                    state = "message"
                else:
                    round_index += 1
                    state = "finished" if round_index >= ROUNDS else "waiting_start"
                arrows = []
            else:
                mouse_path.append((mouse_pos[0], mouse_pos[1]))

        elif state == "message":
            if show_message_until and pygame.time.get_ticks() > show_message_until:
                round_index += 1
                state = "finished" if round_index >= ROUNDS else "waiting_start"

        if arrows:
            for a in arrows:
                a.update(moving)

        # ---------- Drawing ----------
        screen.fill(BG_COLOR)
        pygame.draw.line(screen, GRAY, (0, line_y), (SCREEN_WIDTH, line_y), 3)

        if state in ("playing", "message"):
            for a in arrows:
                a.draw(screen)

        if state == "waiting_start":
            pygame.draw.rect(screen, BLUE, start_button, 2)
            start_label = med_font.render("START", True, BLACK)
            screen.blit(start_label, start_label.get_rect(center=start_button.center))

        if state in ("playing", "message"):
            pygame.draw.rect(screen, GREEN, left_button, 2)
            left_label = small_font.render("LEFT", True, BLACK)
            screen.blit(left_label, left_label.get_rect(center=left_button.center))

            pygame.draw.rect(screen, GREEN, right_button, 2)
            right_label = small_font.render("RIGHT", True, BLACK)
            screen.blit(right_label, right_label.get_rect(center=right_button.center))

        if state == "message":
            msg = med_font.render("Click faster!", True, RED)
            screen.blit(msg, msg.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)))

        if state == "finished" and round_index >= ROUNDS:
            pygame.draw.rect(screen, BLUE, results_button, 2)
            results_label = med_font.render("RESULTS", True, BLACK)
            screen.blit(results_label, results_label.get_rect(center=results_button.center))

        rc_text = small_font.render(f"Round {min(round_index+1, ROUNDS)}/{ROUNDS}", True, BLACK)
        screen.blit(rc_text, (SCREEN_WIDTH - 140, 10))

        pygame.display.flip()

# ---------- Helper ----------
def compute_auc_and_cross(start_pos, mouse_path, end_pos, move_percent):
    if start_pos is None:
        return 0.0, 0
    sx, sy = start_pos
    if not mouse_path:
        return 0.0, 0
    crossed = 0
    if move_percent == 0:
        for _, my in mouse_path:
            if my < line_y:
                crossed = 1
                break
    if end_pos is not None:
        ex = end_pos[0]
    else:
        ex = mouse_path[-1][0]
    if abs(ex - sx) < 1e-6:
        return 0.0, crossed
    xmin = min(sx, ex)
    xmax = max(sx, ex)
    selected = [(x, y) for (x, y) in mouse_path if xmin <= x <= xmax]
    if not selected:
        selected = [mouse_path[0], mouse_path[-1]]
    if sx > ex:
        selected = sorted(selected, key=lambda p: p[0], reverse=True)
    else:
        selected = sorted(selected, key=lambda p: p[0])
    poly = [(sx, sy)] + selected + [(ex, sy)]
    auc = polygon_area(poly)
    return auc, crossed

def show_results(all_mouse_paths):
    if len(all_mouse_paths) < 2:
        print("Not enough data")
        return

    mouse_vel = [
        ((all_mouse_paths[i][1] - all_mouse_paths[i-1][1]) /
         max((all_mouse_paths[i][0] - all_mouse_paths[i-1][0]), 1e-6)) * 60
        for i in range(1, len(all_mouse_paths))
    ]
    times = np.linspace(0, 1, len(mouse_vel))

    mouse_acc = [
        (mouse_vel[i] - mouse_vel[i-1]) / max((times[i] - times[i-1]), 1e-9)
        for i in range(1, len(mouse_vel))
    ]
    acc_times = times[1:]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    axes[0].plot(times, mouse_vel, marker='o')
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Velocity (px/s)")
    axes[0].set_title("Mouse Velocity Over Time")
    axes[0].grid(True)

    axes[1].plot(acc_times, mouse_acc, marker='o', color='r')
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration (px/s²)")
    axes[1].set_title("Mouse Acceleration Over Time")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
