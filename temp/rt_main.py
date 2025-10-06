"""4th Oct"""

import time
import matplotlib.pyplot as plt
from type_decls import *
from primary_controller import *
from mod_rrt import *

class EnvironmentSim:
    def __init__(self, env: Environment, planner:RRT):
        self.env = env
        self.planner = planner
        self.obstacles = env.obstacles if env.obstacles else []
        self.global_path = []
        self.agent = None
        self.replan_interval = 2.0  
        self.lookahead_replan_dist = 2.0  

    def planner_func(self, start_point: Point, ax):
        self.planner.start = start_point
        success = self.planner.build_rrt(self.env, ax)
        if success:
            return self.planner.path
        return None

    def run(self):
        start_time = time.time()
        last_replan_time = start_time

        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("FASTRACK_RT_TEST1")

        ax.scatter(self.env.start.x, self.env.start.y, color='green', label='Start', zorder=5)
        ax.scatter(self.env.target.x, self.env.target.y, color='red', label='Goal', zorder=5)
        for obs in self.obstacles:
            circle = plt.Circle(obs.center, obs.radius, color=obs.color, alpha=obs.alpha)
            ax.add_patch(circle)

        path = self.planner_func(self.env.start, ax)
        if not path:
            print("Initial path not found !!")
            return
        self.global_path = path
        self.agent = Carrot_Chase(self.global_path, location=path[0])

        while not self.agent.finished:
            current_time = time.time()

            if current_time - last_replan_time >= self.replan_interval:
                last_replan_time = current_time
                new_start = Point(self.agent.location.x, self.agent.location.y)
                new_path = self.planner_func(new_start, ax)
                
                if new_path:
                    cleared_idx = self.agent.current_idx
                    dist_accum = 0.0
                    prev_point = Point(self.agent.location.x, self.agent.location.y)
                    while cleared_idx < len(self.global_path):
                        p = self.global_path[cleared_idx]
                        d = ((p.x - prev_point.x)**2 + (p.y - prev_point.y)**2)**0.5
                        dist_accum += d
                        prev_point = p
                        cleared_idx += 1
                        if dist_accum >= self.lookahead_replan_dist:
                            break
                    
                    lookahead_buffer = 2
                    self.global_path = new_path[lookahead_buffer:] 
                    self.agent.path = self.global_path
                    self.agent.current_idx = 0
                    print("Replanning successful from current location!")



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

            ax.plot([p.x for p in self.global_path], [p.y for p in self.global_path], c='blue', linewidth=2, label='Global Path')
            
            self.agent.draw(ax)

            plt.pause(0.05)

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
