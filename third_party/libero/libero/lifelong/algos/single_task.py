import copy

from libero.lifelong.algos.base import Sequential


class SingleTask(Sequential):
    """
    序贯 BC 基线。
    """

    def __init__(self, n_tasks, cfg):
        super().__init__(n_tasks, cfg)
        self.init_pi = copy.deepcopy(self.policy)

    def start_task(self, task):
        # 为每个新任务重新初始化
        self.policy = copy.deepcopy(self.init_pi)
        super().start_task(task)
