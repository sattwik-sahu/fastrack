"""4th Oct - CBPattanaik"""


import math, random, time
from type_decls import Point, Environment

PRINT = 0

class RRT:
    def __init__(self, start: Point, step_size=0.5, max_iter=2000):
        self.start = start
        self.step_size = step_size
        self.max_iter = max_iter
        self.nodes = []
        self.parent = {0: None}
        self.path = []

    def is_in_obstacle(self, p: Point, env: Environment) -> bool:
        for obs in env.obstacles:
            cx, cy = obs.center
            r = obs.radius
            if (p.x - cx)**2 + (p.y - cy)**2 <= r*r:
                return True
        return False

    def is_edge_valid(self, p1: Point, p2: Point, env: Environment) -> bool:
        for obs in env.obstacles:
            cx, cy = obs.center
            r = obs.radius
            dx, dy = p2.x - p1.x, p2.y - p1.y
            fx, fy = p1.x - cx, p1.y - cy
            a = dx*dx + dy*dy
            b = 2 * (fx*dx + fy*dy)
            c = (fx*fx + fy*fy) - r*r

            if a == 0:
                continue

            disc = b*b - 4*a*c
            if disc >= 0:
                d = math.sqrt(disc)
                t1 = (-b - d) / (2*a)
                t2 = (-b + d) / (2*a)
                if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                    return False
        return True

    def get_nearest(self, p: Point):
        dists = [math.dist((n.x, n.y), (p.x, p.y)) for n in self.nodes]
        return min(range(len(dists)), key=lambda i: dists[i])

    def steer(self, from_node: Point, to_point: Point) -> Point:
        dx, dy = to_point.x - from_node.x, to_point.y - from_node.y
        dist = math.hypot(dx, dy)
        if dist < self.step_size:
            return to_point
        theta = math.atan2(dy, dx)
        return Point(from_node.x + self.step_size * math.cos(theta),
                     from_node.y + self.step_size * math.sin(theta))

    def is_goal_reached(self, node: Point, goal: Point, tol=1.0) -> bool:
        return math.dist((node.x, node.y), (goal.x, goal.y)) < tol

    def build_rrt(self, env: Environment, ax=None):
        self.nodes = [self.start]
        for _ in range(self.max_iter):
            if random.random() < 0.1:
                rnd = env.target
            else:
                rnd = Point(random.uniform(0, 30), random.uniform(0, 30))

            nearest_idx = self.get_nearest(rnd)
            nearest = self.nodes[nearest_idx]
            new_node = self.steer(nearest, rnd)

            if PRINT:
                if ax is not None:
                    ax.scatter(rnd.x, rnd.y, c="orange", s=8, alpha=0.4, zorder=2)

            if self.is_in_obstacle(new_node, env):
                continue
            if not self.is_edge_valid(nearest, new_node, env):
                continue

            self.nodes.append(new_node)
            self.parent[len(self.nodes)-1] = nearest_idx

            if PRINT:
                if ax is not None:
                    ax.scatter(new_node.x, new_node.y, c="black", s=2, zorder=3)  
                    ax.plot([nearest.x, new_node.x], [nearest.y, new_node.y],
                            c="gray", linewidth=0.8, alpha=0.6, zorder=1)

            if self.is_goal_reached(new_node, env.target):
                self.nodes.append(env.target)
                self.parent[len(self.nodes)-1] = len(self.nodes)-2
                self.extract_path()

                if PRINT:
                    if ax is not None:
                        ax.plot([new_node.x, env.target.x], [new_node.y, env.target.y],
                                c="blue", linewidth=1.5, zorder=4)

                return True

        return False

    def extract_path(self):
        path = []
        idx = len(self.nodes)-1
        while idx is not None:
            path.append(self.nodes[idx])
            idx = self.parent[idx]
        self.path = list(reversed(path))

    def compute_path_length(self) -> float:
        path_length = 0.0
        for i in range(1, len(self.path)):
            dx = self.path[i].x - self.path[i-1].x
            dy = self.path[i].y - self.path[i-1].y
            path_length += math.sqrt(dx*dx + dy*dy)
        return path_length
