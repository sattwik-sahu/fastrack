"""4th Oct - CBPattanaik"""

from type_decls import *
import numpy as np
import matplotlib.pyplot as plt
import math

PRINT = 1

class Carrot_Chase:
    def __init__(self, path: Path, location: Point, heading=(1, 0)):
        self.path = path                    
        self.location = Point(location.x, location.y)
        self.heading = np.array(heading, dtype=float)
        self.heading /= np.linalg.norm(self.heading)
        self.velocity = 1.5
        self.max_speed = 4.0
        self.kp = 1.0
        self.look_ahead = 2.0
        self.max_turn_rad = np.radians(30)
        self.current_idx = 0
        self.finished = False
        self.traj_x = [self.location.x]
        self.traj_y = [self.location.y]
        self.current_carrot = None

    def get_lookahead_point(self) -> Point:
        if self.current_idx >= len(self.path) - 1:
            self.current_carrot = self.path[-1]
            return self.path[-1]
        
        pos = np.array([self.location.x, self.location.y], dtype=float)
        for i in range(self.current_idx, len(self.path)):
            p = np.array([self.path[i].x, self.path[i].y])
            if np.linalg.norm(p - pos) >= self.look_ahead:
                self.current_idx = i
                self.current_carrot = self.path[i]
                return self.path[i]
        self.finished = True
        self.current_carrot = self.path[-1]
        return self.path[-1]

    def update(self, dt=0.1) -> Point:
        if self.finished:
            return self.location

        carrot = self.get_lookahead_point()

        desired_vec = np.array([carrot.x - self.location.x, carrot.y - self.location.y], dtype=float)
        desired_vec /= np.linalg.norm(desired_vec)

        current_angle = math.atan2(self.heading[1], self.heading[0])
        desired_angle = math.atan2(desired_vec[1], desired_vec[0])
        error = desired_angle - current_angle
        error = math.atan2(math.sin(error), math.cos(error))  # Wrap [-pi, pi]

        angular_velocity = self.kp * error
        angular_velocity = np.clip(angular_velocity, -self.max_turn_rad, self.max_turn_rad)

        rot_matrix = np.array([[math.cos(angular_velocity), -math.sin(angular_velocity)],
                               [math.sin(angular_velocity),  math.cos(angular_velocity)]])
        self.heading = rot_matrix @ self.heading
        self.heading /= np.linalg.norm(self.heading)

        speed = min(self.velocity, self.max_speed)
        self.location.x += self.heading[0] * speed * dt
        self.location.y += self.heading[1] * speed * dt

        self.traj_x.append(self.location.x)
        self.traj_y.append(self.location.y)

        goal = self.path[-1]
        if math.hypot(self.location.x - goal.x, self.location.y - goal.y) < 0.5:
            self.finished = True

        return self.location

    def draw(self, ax):
        ax.plot(self.traj_x, self.traj_y, color='orange', linewidth=2, label='Carrot Chase Path')
        ax.scatter(self.location.x, self.location.y, color='k', s=30, zorder=5, label='Agent')

        if PRINT:
            if hasattr(self, "current_carrot") and self.current_carrot is not None:
                ax.scatter(self.current_carrot.x, self.current_carrot.y, color='magenta', s=50, zorder=6, label='Lookahead')

        ax.legend()



    
# TESTING
# path = [Point(0, 0), Point(5, 3), Point(10, 7), Point(15, 10), Point(20, 12)]
# carrot = Carrot_Chase(path, location=path[0])

# plt.ion()
# fig, ax = plt.subplots()
# ax.set_xlim(-2, 22)
# ax.set_ylim(-2, 15)
# ax.set_aspect('equal')

# for _ in range(500):
#     carrot.update(dt=0.1)
#     ax.clear()
#     ax.plot([p.x for p in path], [p.y for p in path], 'g--', label='Reference Path')
#     carrot.draw(ax)
#     plt.pause(0.05)
#     if carrot.finished:
#         break

# plt.ioff()
# plt.show()