"""4th Oct - CBPattanaik"""


import matplotlib.pyplot as plt
import time
from type_decls import Environment, Point, Obstacle
from mod_rrt import RRT
from primary_controller import *

class EnvironmentSim:
    def __init__(self, env: Environment, planner:RRT):
        self.env = env
        self.planner = planner
        self.obstacles = env.obstacles if env.obstacles else []
        self.global_path = []
        self.agent = None

    def planner_func(self, ax) -> List:
        success = self.planner.build_rrt(self.env, ax)
        if success:
            self.global_path = self.planner.path
            return self.planner.path
        return None

    def run(self):
        start_time = time.time()

        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("FASTRACK_TEST1")

        ax.scatter(self.env.start.x, self.env.start.y, color='green', label='Start', zorder=5)
        ax.scatter(self.env.target.x, self.env.target.y, color='red', label='Goal', zorder=5)

        for obs in self.obstacles:
            circle = plt.Circle(obs.center, obs.radius, color=obs.color, alpha=obs.alpha)
            ax.add_patch(circle)

        path = self.planner_func(ax)
        end_time = time.time()

        if path:
            print("Path Found !!")
            ax.plot([p.x for p in path], [p.y for p in path], c='blue', linewidth=2, label='Global Path')
            path_length = self.planner.compute_path_length()
            print(f"Path length: {path_length:.2f}")
            print(f"Computation Time: {end_time - start_time:.2f}s")

            self.agent = Carrot_Chase(path, location=path[0])

            while not self.agent.finished:
                self.agent.update(dt=0.1)
                ax.clear()

                ax.set_xlim(0, 30)
                ax.set_ylim(0, 30)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title("FASTRACK_TEST1")
                ax.scatter(self.env.start.x, self.env.start.y, color='green', zorder=5)
                ax.scatter(self.env.target.x, self.env.target.y, color='red', zorder=5)
                for obs in self.obstacles:
                    circle = plt.Circle(obs.center, obs.radius, color=obs.color, alpha=obs.alpha)
                    ax.add_patch(circle)

                ax.plot([p.x for p in path], [p.y for p in path], c='blue', linewidth=0.5, label='Global Path')

                self.agent.draw(ax)

                plt.pause(0.05)

        else:
            print("Path Not Found !!")

        ax.legend()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    start = Point(1, 1)
    target = Point(20, 20)
    obstacles = [
        Obstacle((4.5, 3.0), 2),
        Obstacle((3.0, 12.0), 2),
        Obstacle((15.0, 15.0), 2)
    ]

    env = Environment(start, target, obstacles)
    planner = RRT(start)
    sim = EnvironmentSim(env, planner)
    sim.run()