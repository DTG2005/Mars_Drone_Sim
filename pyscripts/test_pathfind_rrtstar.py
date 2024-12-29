import numpy as np
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, bounds, obstacles, step_size=1.0, search_radius=5.0, max_iter=500):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.bounds = bounds
        self.obstacles = obstacles
        self.step_size = step_size
        self.search_radius = search_radius
        self.max_iter = max_iter
        self.nodes = [self.start]

    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

    def is_collision_free(self, new_node, nearest_node):
        for obstacle in self.obstacles:
            x_min, y_min, x_max, y_max = obstacle
            step = 0.01
            for i in np.arange(0, 1 + step, step):
                x = nearest_node.x + i * (new_node.x - nearest_node.x)
                y = nearest_node.y + i * (new_node.y - nearest_node.y)
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return False
        return True

    def get_nearest_node(self, node):
        nearest_node = self.start
        min_dist = float('inf')
        for existing_node in self.nodes:
            dist = self.distance(node, existing_node)
            if dist < min_dist:
                nearest_node = existing_node
                min_dist = dist
        return nearest_node

    def rewire(self, new_node):
        for other_node in self.nodes:
            if self.distance(new_node, other_node) < self.search_radius and self.is_collision_free(new_node, other_node):
                cost = new_node.cost + self.distance(new_node, other_node)
                if cost < other_node.cost:
                    other_node.parent = new_node
                    other_node.cost = cost

    def generate_random_node(self):
        x = random.uniform(self.bounds[0], self.bounds[2])
        y = random.uniform(self.bounds[1], self.bounds[3])
        return Node(x, y)

    def extract_path(self):
        path = []
        node = self.goal
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent
        return path[::-1]

    def plan(self):
        for _ in range(self.max_iter):
            random_node = self.generate_random_node()
            nearest_node = self.get_nearest_node(random_node)

            # Move in the direction of the random node by step_size
            theta = np.arctan2(random_node.y - nearest_node.y, random_node.x - nearest_node.x)
            new_x = nearest_node.x + self.step_size * np.cos(theta)
            new_y = nearest_node.y + self.step_size * np.sin(theta)
            new_node = Node(new_x, new_y)

            # Check for collisions
            if self.is_collision_free(new_node, nearest_node):
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + self.distance(new_node, nearest_node)
                self.nodes.append(new_node)
                self.rewire(new_node)

                # Check if the goal is within step_size
                if self.distance(new_node, self.goal) <= self.step_size:
                    self.goal.parent = new_node
                    self.goal.cost = new_node.cost
                    self.nodes.append(self.goal)
                    break
        return self.extract_path()

    def draw(self, path=None):
        fig, ax = plt.subplots()
        ax.set_xlim(self.bounds[0], self.bounds[2])
        ax.set_ylim(self.bounds[1], self.bounds[3])

        # Draw obstacles
        for obstacle in self.obstacles:
            x_min, y_min, x_max, y_max = obstacle
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, color='gray')
            ax.add_patch(rect)

        # Draw nodes
        for node in self.nodes:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], 'g--', linewidth=0.5)

        # Draw path
        if path:
            for i in range(len(path) - 1):
                plt.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], 'r-', linewidth=2)

        plt.plot(self.start.x, self.start.y, 'bo', label='Start')
        plt.plot(self.goal.x, self.goal.y, 'ro', label='Goal')
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    start = (0, 0)
    goal = (10, 10)
    bounds = (0, 0, 12, 12)
    obstacles = [(3, 3, 5, 5), (7, 8, 9, 10), (6, 2, 8, 4)]

    rrt_star = RRTStar(start, goal, bounds, obstacles)
    path = rrt_star.plan()
    rrt_star.draw(path)
