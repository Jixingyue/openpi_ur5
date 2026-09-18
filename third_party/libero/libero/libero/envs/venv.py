import cloudpickle
import ctypes
import gym
import numpy as np
import numpy as np
import warnings
import time

from abc import ABC, abstractmethod
from collections import OrderedDict
from multiprocessing import Array, Pipe, connection
from multiprocessing.context import Process
from typing import Any, Callable, List, Optional, Tuple, Union


gym_old_venv_step_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
warnings.simplefilter("once", DeprecationWarning)
_NP_TO_CT = {
    np.bool_: ctypes.c_bool,
    np.uint8: ctypes.c_uint8,
    np.uint16: ctypes.c_uint16,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
    np.int8: ctypes.c_int8,
    np.int16: ctypes.c_int16,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
}


def deprecation(msg: str) -> None:
    """弃用警告包装器。"""
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


class CloudpickleWrapper(object):
    """在SubprocVectorEnv中使用的cloudpickle包装器。"""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> str:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: str) -> None:
        self.data = cloudpickle.loads(data)


GYM_RESERVED_KEYS = [
    "metadata",
    "reward_range",
    "spec",
    "action_space",
    "observation_space",
]


################################################################################
#
# 工作器
#
################################################################################


class EnvWorker(ABC):
    """环境的抽象工作器。"""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self._env_fn = env_fn
        self.is_closed = False
        self.result: Union[
            gym_old_venv_step_type,
            gym_new_venv_step_type,
            Tuple[np.ndarray, dict],
            np.ndarray,
        ]
        # self.action_space = self.get_env_attr("action_space")  # noqa: B009
        self.is_reset = False

    @abstractmethod
    def get_env_attr(self, key: str) -> Any:
        pass

    @abstractmethod
    def set_env_attr(self, key: str, value: Any) -> None:
        pass

    def send(self, action: Optional[np.ndarray]) -> None:
        """向低层工作器发送动作信号。

        当action为None时，表示发送“reset”信号；否则
        表示“step”信号。“recv”函数的配对返回值
        由这种不同的信号决定。
        """
        if hasattr(self, "send_action"):
            deprecation(
                "send_action will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            if action is None:
                self.is_reset = True
                self.result = self.reset()
            else:
                self.is_reset = False
                self.send_action(action)

    def recv(
        self,
    ) -> Union[
        gym_old_venv_step_type,
        gym_new_venv_step_type,
        Tuple[np.ndarray, dict],
        np.ndarray,
    ]:  # noqa:E125
        """从低层工作器接收结果。

        如果上次“send”函数发送了NULL动作，它只返回一个
        单独的观测；否则它返回(obs, rew, done,
        info)或(obs, rew, terminated, truncated, info)的元组，基于
        环境是使用旧的step API还是新的。
        """
        if hasattr(self, "get_result"):
            deprecation(
                "get_result will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            if not self.is_reset:
                self.result = self.get_result()
        return self.result

    @abstractmethod
    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        pass

    def step(
        self, action: np.ndarray
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """执行环境动态的一个时间步。

        “send”和“recv”在同步仿真中是耦合的，因此用户只调用
        “step”函数。但它们可以在异步仿真中分开调用，
        即某人先调用“send”，然后稍后调用“recv”。
        """
        self.send(action)
        return self.recv()  # type: ignore

    @staticmethod
    def wait(
        workers: List["EnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["EnvWorker"]:
        """给定一组工作器，返回已就绪的工作器。"""
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        # return self.action_space.seed(seed)  # issue 299
        pass

    @abstractmethod
    def render(self, **kwargs: Any) -> Any:
        """渲染环境。"""
        pass

    @abstractmethod
    def close_env(self) -> None:
        pass

    def close(self) -> None:
        if self.is_closed:
            return None
        self.is_closed = True
        self.close_env()


class ShArray:
    """多进程Array的包装器。"""

    def __init__(self, dtype: np.generic, shape: Tuple[int]) -> None:
        self.arr = Array(_NP_TO_CT[dtype.type], int(np.prod(shape)))  # type: ignore
        self.dtype = dtype
        self.shape = shape

    def save(self, ndarray: np.ndarray) -> None:
        assert isinstance(ndarray, np.ndarray)
        dst = self.arr.get_obj()
        dst_np = np.frombuffer(dst, dtype=self.dtype).reshape(
            self.shape
        )  # type: ignore
        np.copyto(dst_np, ndarray)

    def get(self) -> np.ndarray:
        obj = self.arr.get_obj()
        return np.frombuffer(obj, dtype=self.dtype).reshape(self.shape)  # type: ignore


def _setup_buf(space: gym.Space) -> Union[dict, tuple, ShArray]:
    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict)
        return {k: _setup_buf(v) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(space.spaces, tuple)
        return tuple([_setup_buf(t) for t in space.spaces])
    else:
        return ShArray(space.dtype, space.shape)  # type: ignore


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                env_return = env.step(data)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                p.send(env_return)
            elif cmd == "reset":
                retval = env.reset(**data)
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs = retval
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                if reset_returns_info:
                    p.send((obs, info))
                else:
                    p.send(obs)
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
            elif cmd == "check_success":
                p.send(env.check_success())
            elif cmd == "get_segmentation_of_interest":
                p.send(env.get_segmentation_of_interest(data))
            elif cmd == "get_sim_state":
                p.send(env.get_sim_state())
            elif cmd == "set_init_state":
                obs = env.set_init_state(data)
                p.send(obs)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class DummyEnvWorker(EnvWorker):
    """在顺序向量环境中使用的虚拟工作器。"""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self.env = env_fn()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        return getattr(self.env, key)

    def set_env_attr(self, key: str, value: Any) -> None:
        setattr(self.env.unwrapped, key, value)

    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        return self.env.reset(**kwargs)

    @staticmethod
    def wait(  # type: ignore
        workers: List["DummyEnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["DummyEnvWorker"]:
        # 顺序EnvWorker对象始终就绪
        return workers

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        if action is None:
            self.result = self.env.reset(**kwargs)
        else:
            self.result = self.env.step(action)  # type: ignore

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        super().seed(seed)
        try:
            return self.env.seed(seed)  # type: ignore
        except (AttributeError, NotImplementedError):
            self.env.reset(seed=seed)
            return [seed]  # type: ignore

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        self.env.close()

    def check_success(self):
        return self.env.check_success()

    def get_segmentation_of_interest(self, segmentation_image):
        return self.env.get_segmentation_of_interest(segmentation_image)

    def get_sim_state(self):
        return self.env.get_sim_state()

    def set_init_state(self, init_state):
        return self.env.set_init_state(init_state)


class SubprocEnvWorker(EnvWorker):
    """在SubprocVectorEnv和ShmemVectorEnv中使用的子进程工作器。"""

    def __init__(
        self, env_fn: Callable[[], gym.Env], share_memory: bool = False
    ) -> None:
        self.parent_remote, self.child_remote = Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        self.parent_remote.send(["getattr", key])
        return self.parent_remote.recv()

    def set_env_attr(self, key: str, value: Any) -> None:
        self.parent_remote.send(["setattr", {"key": key, "value": value}])

    def _decode_obs(self) -> Union[dict, tuple, np.ndarray]:
        def decode_obs(
            buffer: Optional[Union[dict, tuple, ShArray]]
        ) -> Union[dict, tuple, np.ndarray]:
            if isinstance(buffer, ShArray):
                return buffer.get()
            elif isinstance(buffer, tuple):
                return tuple([decode_obs(b) for b in buffer])
            elif isinstance(buffer, dict):
                return {k: decode_obs(v) for k, v in buffer.items()}
            else:
                raise NotImplementedError

        return decode_obs(self.buffer)

    @staticmethod
    def wait(  # type: ignore
        workers: List["SubprocEnvWorker"],
        wait_num: int,
        timeout: Optional[float] = None,
    ) -> List["SubprocEnvWorker"]:
        remain_conns = conns = [x.parent_remote for x in workers]
        ready_conns: List[connection.Connection] = []
        remain_time, t1 = timeout, time.time()
        while len(remain_conns) > 0 and len(ready_conns) < wait_num:
            if timeout:
                remain_time = timeout - (time.time() - t1)
                if remain_time <= 0:
                    break
            # 如果列表为空，connection.wait会挂起
            new_ready_conns = connection.wait(remain_conns, timeout=remain_time)
            ready_conns.extend(new_ready_conns)  # type: ignore
            remain_conns = [conn for conn in remain_conns if conn not in ready_conns]
        return [workers[conns.index(con)] for con in ready_conns]

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        if action is None:
            if "seed" in kwargs:
                super().seed(kwargs["seed"])
            self.parent_remote.send(["reset", kwargs])
        else:
            self.parent_remote.send(["step", action])

    def recv(
        self,
    ) -> Union[
        gym_old_venv_step_type,
        gym_new_venv_step_type,
        Tuple[np.ndarray, dict],
        np.ndarray,
    ]:  # noqa:E125
        result = self.parent_remote.recv()
        if isinstance(result, tuple):
            if len(result) == 2:
                obs, info = result
                if self.share_memory:
                    obs = self._decode_obs()
                return obs, info
            obs = result[0]
            if self.share_memory:
                obs = self._decode_obs()
            return (obs, *result[1:])  # type: ignore
        else:
            obs = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs

    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        self.parent_remote.send(["reset", kwargs])

        result = self.parent_remote.recv()
        if isinstance(result, tuple):
            obs, info = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs, info
        else:
            obs = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        super().seed(seed)
        self.parent_remote.send(["seed", seed])
        ret = self.parent_remote.recv()
        return ret

    def render(self, **kwargs: Any) -> Any:
        self.parent_remote.send(["render", kwargs])
        return self.parent_remote.recv()

    def close_env(self) -> None:
        try:
            self.parent_remote.send(["close", None])
            # mp可能已被删除，因此可能抛出AttributeError
            self.parent_remote.recv()
            self.process.join()
        except (BrokenPipeError, EOFError, AttributeError):
            pass
        # 确保子进程已终止
        self.process.terminate()

    def check_success(self):
        self.parent_remote.send(["check_success", None])
        return self.parent_remote.recv()

    def get_segmentation_of_interest(self, segmentation_image):
        self.parent_remote.send(["get_segmentation_of_interest", segmentation_image])
        return self.parent_remote.recv()

    def get_sim_state(self):
        self.parent_remote.send(["get_sim_state", None])
        return self.parent_remote.recv()

    def set_init_state(self, init_state):
        self.parent_remote.send(["set_init_state", init_state])
        obs = self.parent_remote.recv()
        if self.share_memory:
            obs = self._decode_obs()
        return obs


################################################################################
#
# VecEnvs
#
################################################################################


class BaseVectorEnv(object):
    """向量化环境的基类。

    用法：
    ::

        env_num = 8
        envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(env_num)])
        assert len(envs) == env_num

    它接受一个环境生成器列表。换句话说，某个特定任务的
    环境生成器 ``efn`` 意味着 ``efn()`` 返回该给定任务的
    环境，例如 ``gym.make(task)``。

    所有的 VectorEnv 都必须继承 :class:`~tianshou.env.BaseVectorEnv`。
    以下是一些其他用法：
    ::

        envs.seed(2)  # 等价于下一行
        envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # 为每个环境设置特定的种子
        obs = envs.reset()  # 重置所有环境
        obs = envs.reset([0, 5, 7])  # 重置 3 个特定环境
        obs, rew, done, info = envs.step([1] * 8)  # 同步步进
        envs.render()  # 渲染所有环境
        envs.close()  # 关闭所有环境

    .. warning::

        如果你使用自己的环境，请确保 ``seed`` 方法
        已正确设置，例如：
        ::

            def seed(self, seed):
                np.random.seed(seed)

        否则，这些环境的输出可能彼此相同。

    :param env_fns: 一个可调用环境的列表，``env_fns[i]()`` 生成第 i 个环境。
    :param worker_fn: 一个可调用工作器，``worker_fn(env_fns[i])`` 生成一个
        包含第 i 个环境的工作器。
    :param int wait_num: 用于异步仿真，当 ``env.step`` 的时间开销
        随时间变化，且同步等待所有环境完成一个步进
        很浪费时间时。在这种情况下，我们可以在 ``wait_num`` 个环境
        完成一个步进时就返回，并在这些环境中继续
        仿真。如果为 ``None``，则禁用异步仿真；
        否则 ``1 <= wait_num <= env_num``。
    :param float timeout: 同上述用于异步仿真，在每个向量化
        步进中只处理那些耗时在 ``timeout`` 秒以内的
        环境。
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        worker_fn: Callable[[Callable[[], gym.Env]], EnvWorker],
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._env_fns = env_fns
        # 一个 VectorEnv 包含一个 EnvWorker 池，它们负责
        # 与给定的环境交互（一个工作器 <-> 一个环境）。
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.worker_class = type(self.workers[0])
        assert issubclass(self.worker_class, EnvWorker)
        assert all([isinstance(w, self.worker_class) for w in self.workers])

        self.env_num = len(env_fns)
        self.wait_num = wait_num or len(env_fns)
        assert (
            1 <= self.wait_num <= len(env_fns)
        ), f"wait_num should be in [1, {len(env_fns)}], but got {wait_num}"
        self.timeout = timeout
        assert (
            self.timeout is None or self.timeout > 0
        ), f"timeout is {timeout}, it should be positive if provided!"
        self.is_async = self.wait_num != len(env_fns) or timeout is not None
        self.waiting_conn: List[EnvWorker] = []
        # self.ready_id 中的环境是真正就绪的
        # 而 self.waiting_id 中的环境在检查时只是处于等待状态，
        # 它们现在可能已经就绪，但在我们在 step() 函数中
        # 检查之前这是未知的
        self.waiting_id: List[int] = []
        # 一开始所有环境都是就绪的
        self.ready_id = list(range(self.env_num))
        self.is_closed = False

    def _assert_is_not_closed(self) -> None:
        assert (
            not self.is_closed
        ), f"Methods of {self.__class__.__name__} cannot be called after close."

    def __len__(self) -> int:
        """返回 len(self)，即环境的数量。"""
        return self.env_num

    def __getattribute__(self, key: str) -> Any:
        """根据 key 切换属性获取方式。

        任何继承 ``gym.Env`` 的类都会继承一些属性，例如
        ``action_space``。然而，我们希望属性查找直接进入
        工作器（实际上，这个向量环境的 action_space 总是 None）。
        """
        if key in GYM_RESERVED_KEYS:  # reserved keys in gym.Env
            return self.get_env_attr(key)
        else:
            return super().__getattribute__(key)

    def get_env_attr(
        self,
        key: str,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> List[Any]:
        """从底层环境获取一个属性。

        如果 id 是一个整数，则从索引为 id 的工作器底层环境中
        检索由 key 表示的属性。结果以包含单个元素的
        列表形式返回。否则，检索索引为 id 的所有工作器的该属性，
        并返回一个与 id 顺序对应的列表。

        :param str key: 所需属性的键。
        :param id: 所需工作器的索引。默认为 None，表示所有 env_id。

        :return list: 环境属性的列表。
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        return [self.workers[j].get_env_attr(key) for j in id]

    def set_env_attr(
        self,
        key: str,
        value: Any,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> None:
        """在底层环境中设置一个属性。

        如果 id 是一个整数，则将索引为 id 的工作器底层环境中
        由 key 表示的属性设置为 value。
        否则，为索引为 id 的所有工作器设置该属性。

        :param str key: 所需属性的键。
        :param Any value: 该属性的新值。
        :param id: 所需工作器的索引。默认为 None，表示所有 env_id。
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        for j in id:
            self.workers[j].set_env_attr(key, value)

    def _wrap_id(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Union[List[int], np.ndarray]:
        if id is None:
            return list(range(self.env_num))
        return [id] if np.isscalar(id) else id  # type: ignore

    def _assert_id(self, id: Union[List[int], np.ndarray]) -> None:
        for i in id:
            assert (
                i not in self.waiting_id
            ), f"Cannot interact with environment {i} which is stepping now."
            assert (
                i in self.ready_id
            ), f"Can only interact with ready environments {self.ready_id}."

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """重置某些环境的状态并返回初始观测值。

        如果 id 为 None，则重置所有环境的状态并返回
        初始观测值；否则重置具有给定 id（可以是整数或列表）的
        特定环境。
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # 在工作器中 send(None) == reset()
        for i in id:
            self.workers[i].send(None, **kwargs)
        ret_list = [self.workers[i].recv() for i in id]

        reset_returns_info = (
            isinstance(ret_list[0], (tuple, list))
            and len(ret_list[0]) == 2
            and isinstance(ret_list[0][1], dict)
        )
        if reset_returns_info:
            obs_list = [r[0] for r in ret_list]
        else:
            obs_list = ret_list

        if isinstance(obs_list[0], tuple):
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )
        try:
            obs = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs = np.array(obs_list, dtype=object)

        if reset_returns_info:
            infos = [r[1] for r in ret_list]
            return obs, infos  # type: ignore
        else:
            return obs

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """运行某些环境动力学的一个时间步。

        如果 id 为 None，则运行所有环境动力学的一个时间步；
        否则为具有给定 id（可以是整数或列表）的某些环境运行
        一个时间步。当到达一个回合的末尾时，你有责任
        调用 reset(id) 来重置该环境的状态。

        接受一批动作，并以 numpy 格式返回一个元组
        (batch_obs, batch_rew, batch_done, batch_info)。

        :param numpy.ndarray action: 由智能体提供的一批动作。

        :return: 一个由以下内容组成的元组：

            * ``obs`` 一个 numpy.ndarray，智能体对当前环境的观测值
            * ``rew`` 一个 numpy.ndarray，上一个动作之后返回的奖励量
            * ``done`` 一个 numpy.ndarray，这些回合是否已结束，\
                若已结束，则进一步的 step() 调用将返回未定义的结果
            * ``info`` 一个 numpy.ndarray，包含辅助诊断信息\
                （有助于调试，有时也有助于学习）

            或：

            * ``obs`` 一个 numpy.ndarray，智能体对当前环境的观测值
            * ``rew`` 一个 numpy.ndarray，上一个动作之后返回的奖励量
            * ``terminated`` 一个 numpy.ndarray，这些回合是否已被\
                终止
            * ``truncated`` 一个 numpy.ndarray，这些回合是否已被截断
            * ``info`` 一个 numpy.ndarray，包含辅助诊断信息\
                （有助于调试，有时也有助于学习）

            情况区分基于底层环境使用的是旧的 step API
            （第一种情况）还是新的 step API（第二种情况）。

        对于异步仿真：

        向环境提供给定的动作。动作序列应与 ``id`` 参数
        对应，且 ``id`` 参数应是上一次返回的 ``info`` 中
        ``env_id`` 的子集（初始时它们是所有环境的 env_id）。
        如果 action 为 None，则改为获取未完成的 step() 调用。
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if not self.is_async:
            assert len(action) == len(id)
            for i, j in enumerate(id):
                self.workers[j].send(action[i])
            result = []
            for j in id:
                env_return = self.workers[j].recv()
                env_return[-1]["env_id"] = j
                result.append(env_return)
        else:
            if action is not None:
                self._assert_id(id)
                assert len(action) == len(id)
                for act, env_id in zip(action, id):
                    self.workers[env_id].send(act)
                    self.waiting_conn.append(self.workers[env_id])
                    self.waiting_id.append(env_id)
                self.ready_id = [x for x in self.ready_id if x not in id]
            ready_conns: List[EnvWorker] = []
            while not ready_conns:
                ready_conns = self.worker_class.wait(
                    self.waiting_conn, self.wait_num, self.timeout
                )
            result = []
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                # env_return 可以是 (obs, reward, done, info) 或
                # (obs, reward, terminated, truncated, info)
                env_return = conn.recv()
                env_return[-1]["env_id"] = env_id  # 将 `env_id` 添加到 info 中
                result.append(env_return)
                self.ready_id.append(env_id)
        return_lists = tuple(zip(*result))
        obs_list = return_lists[0]
        try:
            obs_stack = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs_stack = np.array(obs_list, dtype=object)
        other_stacks = map(np.stack, return_lists[1:])
        return (obs_stack, *other_stacks)  # type: ignore

    def seed(
        self,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> List[Optional[List[int]]]:
        """为所有环境设置种子。

        接受 ``None``、一个整数（它会将 ``i`` 扩展为
        ``[i, i + 1, i + 2, ...]``）或一个列表。

        :return: 本环境随机数生成器中使用的种子列表。
            列表中的第一个值应该是"主"种子，即复现者
            传给 "seed" 的值。
        """
        self._assert_is_not_closed()
        seed_list: Union[List[None], List[int]]
        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        return [w.seed(s) for w, s in zip(self.workers, seed_list)]

    def render(self, **kwargs: Any) -> List[Any]:
        """渲染所有环境。"""
        self._assert_is_not_closed()
        if self.is_async and len(self.waiting_id) > 0:
            raise RuntimeError(
                f"Environments {self.waiting_id} are still stepping, cannot "
                "render them now."
            )
        return [w.render(**kwargs) for w in self.workers]

    def close(self) -> None:
        """关闭所有环境。

        该函数只会被调用一次（如果未调用，它将在垃圾回收
        时被调用）。这样，就可以保证所有工作器的 ``close`` 都被执行。
        """
        self._assert_is_not_closed()
        for w in self.workers:
            w.close()
        self.is_closed = True


class DummyVectorEnv(BaseVectorEnv):
    """虚拟向量化环境包装器，以 for 循环实现。

    .. seealso::

        关于其他 API 的用法，请参阅 :class:`~tianshou.env.BaseVectorEnv`。
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        super().__init__(env_fns, DummyEnvWorker, **kwargs)

    def check_success(self):
        return [w.check_success() for w in self.workers]

    def get_segmentation_of_interest(self, segmentation_images):
        return [
            w.get_segmentation_of_interest(img)
            for w, img in zip(self.workers, segmentation_images)
        ]

    def get_sim_state(self):
        return [w.get_sim_state() for w in self.workers]

    def set_init_state(
        self,
        init_state: Optional[Union[int, List[int], np.ndarray]] = None,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """重置某些环境的状态并返回初始观测值。
        如果 id 为 None，则重置所有环境的状态并返回
        初始观测值；否则重置具有给定 id（可以是整数或列表）的
        特定环境。
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # 在工作器中 send(None) == reset()
        obs_list = []
        for j, i in enumerate(id):
            obs = self.workers[i].set_init_state(init_state[j])
            obs_list.append(obs)
        obs = np.stack(obs_list)
        return obs


class SubprocVectorEnv(BaseVectorEnv):
    """基于子进程的向量化环境包装器。

    .. seealso::

        关于其他 API 的用法，请参阅 :class:`~tianshou.env.BaseVectorEnv`。
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            return SubprocEnvWorker(fn, share_memory=False)

        super().__init__(env_fns, worker_fn, **kwargs)

    def check_success(self):
        return [w.check_success() for w in self.workers]

    def get_segmentation_of_interest(self, segmentation_images):
        return [
            w.get_segmentation_of_interest(img)
            for w, img in zip(self.workers, segmentation_images)
        ]

    def get_sim_state(self):
        return [w.get_sim_state() for w in self.workers]

    def set_init_state(
        self,
        init_state: Optional[Union[int, List[int], np.ndarray]] = None,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """重置某些环境的状态并返回初始观测值。
        如果 id 为 None，则重置所有环境的状态并返回
        初始观测值；否则重置具有给定 id（可以是整数或列表）的
        特定环境。
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # 在工作器中 send(None) == reset()
        obs_list = []
        for j, i in enumerate(id):
            obs = self.workers[i].set_init_state(init_state[j])
            obs_list.append(obs)
        obs = np.stack(obs_list)
        return obs
