import numpy as np
import os
import robosuite.utils.transform_utils as T

from copy import deepcopy
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial
import robosuite.macros as macros

import mujoco

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs.robots import *
from libero.libero.envs.utils import *
from libero.libero.envs.object_states import *
from libero.libero.envs.objects import *
from libero.libero.envs.regions import *
from libero.libero.envs.arenas import *


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

TASK_MAPPING = {}


def register_problem(target_class):
    """我们将映射设计为不区分大小写的。"""
    TASK_MAPPING[target_class.__name__.lower()] = target_class


import time


class BDDLBaseDomain(SingleArmEnv):
    """
    用于解析bddl文件的基础域。
    """

    def __init__(
        self,
        bddl_file_name,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_latch=False,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        object_property_initializers=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        table_full_size=(1.0, 1.0, 0.05),
        workspace_offset=(0.0, 0.0, 0.0),
        arena_type="table",
        scene_xml="scenes/libero_base_style.xml",
        scene_properties={},
        **kwargs,
    ):
        t0 = time.time()
        # 桌面设置（硬编码，因为它不是环境的核心部分）
        self.workspace_offset = workspace_offset
        # 奖励配置
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # 是否使用真实物体状态
        self.use_object_obs = use_object_obs

        # 物体放置初始化器
        self.placement_initializer = placement_initializer
        self.conditional_placement_initializer = None
        self.conditional_placement_on_objects_initializer = None

        # 物体属性初始化器

        if object_property_initializers is not None:
            self.object_property_initializers = object_property_initializers
        else:
            self.object_property_initializers = list()

        # 跟踪任务中的可移动物体
        self.objects_dict = {}
        # 跟踪任务中的固定物体
        self.fixtures_dict = {}
        # 跟踪任务中的站点对象。站点对象
        # （SiteObject的实例）
        self.object_sites_dict = {}
        # 这是一个存储所有物体状态
        # 接口的字典
        self.object_states_dict = {}

        # 对于那些需要视觉特征变化的物体，每个时间步都更新状态以避免遗漏状态变化。我们跟踪这类物体以使谓词检查更高效。
        self.tracking_object_states_change = []

        self.object_sites_dict = {}

        self.objects = []
        self.fixtures = []
        # self.custom_material_dict = {}

        self.custom_asset_dir = os.path.abspath(os.path.join(DIR_PATH, "../assets"))

        self.bddl_file_name = bddl_file_name
        self.parsed_problem = BDDLUtils.robosuite_parse_problem(self.bddl_file_name)

        self.obj_of_interest = self.parsed_problem["obj_of_interest"]

        self._assert_problem_name()

        self._arena_type = arena_type
        self._arena_xml = os.path.join(self.custom_asset_dir, scene_xml)
        self._arena_properties = scene_properties

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            **kwargs,
        )

    def seed(self, seed):
        np.random.seed(seed)

    def reward(self, action=None):
        """
        任务的奖励函数。

        稀疏未归一化奖励：

            - 如果任务成功，提供1.0的离散奖励。

        参数:
            action (np.array): [未使用]

        返回:
            float: 奖励值
        """
        reward = 0.0

        # 稀疏完成奖励
        if self._check_success():
            reward = 1.0

        # 如果请求，缩放奖励
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _assert_problem_name(self):
        """实现此方法以确保加载的bddl文件具有正确的问题名称规范。"""
        assert (
            self.parsed_problem["problem_name"] == self.__class__.__name__.lower()
        ), "Problem name mismatched"

    def _load_fixtures_in_arena(self, mujoco_arena):
        """
        根据bddl文件描述加载固定装置。请在自定义问题文件中重写此方法。
        """
        raise NotImplementedError

    def _load_objects_in_arena(self, mujoco_arena):
        """
        根据bddl文件描述加载可移动物体
        """
        raise NotImplementedError

    def _load_sites_in_arena(self, mujoco_arena):
        """
        从每个物体加载站点信息，以便跟踪它们进行谓词检查
        """
        raise NotImplementedError

    def _generate_object_state_wrapper(
        self, skip_object_names=["main_table", "floor", "countertop", "coffee_table"]
    ):
        object_states_dict = {}
        tracking_object_states_changes = []
        for object_name in self.objects_dict.keys():
            if object_name in skip_object_names:
                continue
            object_states_dict[object_name] = ObjectState(self, object_name)
            if (
                self.objects_dict[object_name].category_name
                in VISUAL_CHANGE_OBJECTS_DICT
            ):
                tracking_object_states_changes.append(object_states_dict[object_name])

        for object_name in self.fixtures_dict.keys():
            if object_name in skip_object_names:
                continue
            object_states_dict[object_name] = ObjectState(
                self, object_name, is_fixture=True
            )
            if (
                self.fixtures_dict[object_name].category_name
                in VISUAL_CHANGE_OBJECTS_DICT
            ):
                tracking_object_states_changes.append(object_states_dict[object_name])

        for object_name in self.object_sites_dict.keys():
            if object_name in skip_object_names:
                continue
            object_states_dict[object_name] = SiteObjectState(
                self,
                object_name,
                parent_name=self.object_sites_dict[object_name].parent_name,
            )
        self.object_states_dict = object_states_dict
        self.tracking_object_states_change = tracking_object_states_changes

    def _load_distracting_objects(self, mujoco_arena):
        raise NotImplementedError

    def _load_custom_material(self):
        """
        定义所有纹理
        """
        # self.custom_material_dict = dict()

        # tex_attrib = {
        #     "type": "cube"
        # }

        # self.custom_material_dict["bread"] = CustomMaterial(
        #     texture="Bread",
        #     tex_name="bread",
        #     mat_name="MatBread",
        #     tex_attrib=tex_attrib,
        #     mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        # )

    def _setup_camera(self, mujoco_arena):
        # 修改默认agentview相机
        mujoco_arena.set_camera(
            camera_name="canonical_agentview",
            pos=[0.5386131746834771, 0.0, 1.4903500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5886131746834771, 0.0, 1.4903500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

    def _load_model(self):
        """
        加载xml模型，将其放入self.model中
        """
        super()._load_model()
        # 相应地调整基座位姿

        if self._arena_type == "table":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](
                self.table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = TableArena(
                table_full_size=self.table_full_size,
                table_offset=self.workspace_offset,
                table_friction=(0.6, 0.005, 0.0001),
                xml=self._arena_xml,
                **self._arena_properties,
            )
        elif self._arena_type == "kitchen":
            xpos = self.robots[0].robot_model.base_xpos_offset["kitchen_table"](
                self.kitchen_table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = KitchenTableArena(
                table_full_size=self.kitchen_table_full_size,
                table_offset=self.workspace_offset,
                xml=self._arena_xml,
                **self._arena_properties,
            )

        elif self._arena_type == "floor":
            xpos = self.robots[0].robot_model.base_xpos_offset["empty"]
            self.robots[0].robot_model.set_base_xpos(xpos)

            mujoco_arena = EmptyArena(
                xml=self._arena_xml,
                **self._arena_properties,
            )
        elif self._arena_type == "coffee_table":
            xpos = self.robots[0].robot_model.base_xpos_offset["coffee_table"](
                self.coffee_table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = CoffeeTableArena(
                xml=self._arena_xml,
                **self._arena_properties,
            )

        elif self._arena_type == "living_room":
            xpos = self.robots[0].robot_model.base_xpos_offset["living_room_table"](
                self.living_room_table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = LivingRoomTableArena(
                xml=self._arena_xml,
                **self._arena_properties,
            )

        elif self._arena_type == "study":
            xpos = self.robots[0].robot_model.base_xpos_offset["study_table"](
                self.study_table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = StudyTableArena(
                xml=self._arena_xml,
                **self._arena_properties,
            )

        # 场景始终设置为零原点
        mujoco_arena.set_origin([0, 0, 0])

        self._setup_camera(mujoco_arena)

        self._load_custom_material()

        self._load_fixtures_in_arena(mujoco_arena)

        self._load_objects_in_arena(mujoco_arena)

        self._load_sites_in_arena(mujoco_arena)

        self._generate_object_state_wrapper()

        self._setup_placement_initializer(mujoco_arena)

        self.objects = list(self.objects_dict.values())
        self.fixtures = list(self.fixtures_dict.values())

        # 任务包括场景、机器人和感兴趣的物体
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects + self.fixtures,
        )

        for fixture in self.fixtures:
            self.model.merge_assets(fixture)

    def _setup_placement_initializer(self, mujoco_arena):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.conditional_placement_initializer = SiteSequentialCompositeSampler(
            name="ConditionalSiteSampler"
        )
        self.conditional_placement_on_objects_initializer = SequentialCompositeSampler(
            name="ConditionalObjectSampler"
        )
        self._add_placement_initializer()

    def _setup_references(self):
        """
        设置对重要组件的引用。引用通常是一个索引或索引列表，
        指向展平数组中的相应元素，这是MuJoCo存储物理仿真数据的方式。
        """
        super()._setup_references()

        # 此环境的额外物体引用
        self.obj_body_id = dict()

        for (object_name, object_body) in self.objects_dict.items():
            self.obj_body_id[object_name] = self.sim.model.body_name2id(
                object_body.root_body
            )

        for (fixture_name, fixture_body) in self.fixtures_dict.items():
            self.obj_body_id[fixture_name] = self.sim.model.body_name2id(
                fixture_body.root_body
            )

    def _setup_observables(self):
        """
        设置用于此环境的可观测量。如果启用，创建基于物体的可观测量

        返回:
            OrderedDict: 将可观测量名称映射到其对应Observable对象的字典
        """
        observables = super()._setup_observables()

        observables["robot0_joint_pos"]._active = True

        # 低层物体信息
        if self.use_object_obs:
            # 获取机器人前缀并定义可观测量模态
            pf = self.robots[0].robot_model.naming_prefix
            sensors = []
            names = [s.__name__ for s in sensors]

            # 如果使用带有可旋转把手的锁定抽屉版本，还要追加把手qpos

            # 创建可观测量
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality="object")
        def world_pose_in_gripper(obs_cache):
            return (
                T.pose_inv(
                    T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))
                )
                if f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache
                else np.eye(4)
            )

        sensors.append(world_pose_in_gripper)
        names.append("world_pose_in_gripper")

        for (i, obj) in enumerate(self.objects):
            obj_sensors, obj_sensor_names = self._create_obj_sensors(
                obj_name=obj.name, modality="object"
            )

            sensors += obj_sensors
            names += obj_sensor_names

        for name, s in zip(names, sensors):
            if name == "world_pose_in_gripper":
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=True,
                    active=False,
                )
            else:
                observables[name] = Observable(
                    name=name, sensor=s, sampling_rate=self.control_freq
                )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        为给定物体创建传感器的辅助函数。这被抽象为单独的函数调用，
        以避免在_setup_observables()调用期间出现局部函数命名冲突。

        参数:
            obj_name (str): 要创建传感器的物体名称
            modality (str): 分配给所有传感器的模态

        返回:
            2-tuple:
                sensors (list): 给定物体的传感器数组
                names (list): 对应的可观测量名称数组
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(
                self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw"
            )

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # 如果缓存为空，立即返回默认值
            if any(
                [
                    name not in obs_cache
                    for name in [
                        f"{obj_name}_pos",
                        f"{obj_name}_quat",
                        "world_pose_in_gripper",
                    ]
                ]
            ):
                return np.zeros(3)
            obj_pose = T.pose2mat(
                (obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"])
            )
            rel_pose = T.pose_in_A_to_pose_in_B(
                obj_pose, obs_cache["world_pose_in_gripper"]
            )
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return (
                obs_cache[f"{obj_name}_to_{pf}eef_quat"]
                if f"{obj_name}_to_{pf}eef_quat" in obs_cache
                else np.zeros(4)
            )

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [
            f"{obj_name}_pos",
            f"{obj_name}_quat",
            f"{obj_name}_to_{pf}eef_pos",
            f"{obj_name}_to_{pf}eef_quat",
        ]

        return sensors, names

    def _add_placement_initializer(self):

        mapping_inv = {}
        for k, values in self.parsed_problem["fixtures"].items():
            for v in values:
                mapping_inv[v] = k
        for k, values in self.parsed_problem["objects"].items():
            for v in values:
                mapping_inv[v] = k

        regions = self.parsed_problem["regions"]
        initial_state = self.parsed_problem["initial_state"]
        problem_name = self.parsed_problem["problem_name"]

        conditioned_initial_place_state_on_sites = []
        conditioned_initial_place_state_on_objects = []
        conditioned_initial_place_state_in_objects = []

        for state in initial_state:
            if state[0] == "on" and state[2] in self.objects_dict:
                conditioned_initial_place_state_on_objects.append(state)
                continue

            # (Yifeng) 鉴于物体需要具有某个“包含”区域才能保持“In”关系，我们假设用户需要已经指定物体的包含区域。
            if state[0] == "in" and state[2] in regions:
                conditioned_initial_place_state_in_objects.append(state)
                continue
            # 检查谓词是否为On(object, region)的形式
            if state[0] == "on" and state[2] in regions:
                object_name = state[1]
                region_name = state[2]
                target_name = regions[region_name]["target"]
                x_ranges, y_ranges = rectangle2xyrange(regions[region_name]["ranges"])
                yaw_rotation = regions[region_name]["yaw_rotation"]
                if (
                    target_name in self.objects_dict
                    or target_name in self.fixtures_dict
                ):
                    conditioned_initial_place_state_on_sites.append(state)
                    continue
                if self.is_fixture(object_name):
                    # 这是用于放置环境固定装置的。
                    fixture_sampler = MultiRegionRandomSampler(
                        f"{object_name}_sampler",
                        mujoco_objects=self.fixtures_dict[object_name],
                        x_ranges=x_ranges,
                        y_ranges=y_ranges,
                        rotation=yaw_rotation,
                        rotation_axis="z",
                        z_offset=self.z_offset,  # -self.table_full_size[2],
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=False,
                        reference_pos=self.workspace_offset,
                    )
                    self.placement_initializer.append_sampler(fixture_sampler)
                else:
                    # 这是用于放置可移动物体的。
                    region_sampler = get_region_samplers(
                        problem_name, mapping_inv[target_name]
                    )(
                        object_name,
                        self.objects_dict[object_name],
                        x_ranges=x_ranges,
                        y_ranges=y_ranges,
                        rotation=self.objects_dict[object_name].rotation,
                        rotation_axis=self.objects_dict[object_name].rotation_axis,
                        reference_pos=self.workspace_offset,
                    )
                    self.placement_initializer.append_sampler(region_sampler)
            if state[0] in ["open", "close"]:
                # 如果实现了“open”，我们假设也实现了“close”
                if state[1] in self.object_states_dict and hasattr(
                    self.object_states_dict[state[1]], "set_joint"
                ):
                    obj = self.get_object(state[1])
                    if state[0] == "open":
                        joint_ranges = obj.object_properties["articulation"][
                            "default_open_ranges"
                        ]
                    else:
                        joint_ranges = obj.object_properties["articulation"][
                            "default_close_ranges"
                        ]

                    property_initializer = OpenCloseSampler(
                        name=obj.name,
                        state_type=state[0],
                        joint_ranges=joint_ranges,
                    )
                    self.object_property_initializers.append(property_initializer)
            elif state[0] in ["turnon", "turnoff"]:
                # 如果实现了“turnon”，我们假设也实现了“turnoff”。
                if state[1] in self.object_states_dict and hasattr(
                    self.object_states_dict[state[1]], "set_joint"
                ):
                    obj = self.get_object(state[1])
                    if state[0] == "turnon":
                        joint_ranges = obj.object_properties["articulation"][
                            "default_turnon_ranges"
                        ]
                    else:
                        joint_ranges = obj.object_properties["articulation"][
                            "default_turnoff_ranges"
                        ]

                    property_initializer = TurnOnOffSampler(
                        name=obj.name,
                        state_type=state[0],
                        joint_ranges=joint_ranges,
                    )
                    self.object_property_initializers.append(property_initializer)

        # 放置在站点上的物体
        for state in conditioned_initial_place_state_on_sites:
            object_name = state[1]
            region_name = state[2]
            target_name = regions[region_name]["target"]
            site_xy_size = self.object_sites_dict[region_name].size[:2]
            sampler = SiteRegionRandomSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                x_ranges=[[-site_xy_size[0] / 2, site_xy_size[0] / 2]],
                y_ranges=[[-site_xy_size[1] / 2, site_xy_size[1] / 2]],
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_initializer.append_sampler(
                sampler, {"reference": target_name, "site_name": region_name}
            )
        # 放置在其他物体上的物体
        for state in conditioned_initial_place_state_on_objects:
            object_name = state[1]
            other_object_name = state[2]
            sampler = ObjectBasedSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                x_ranges=[[0.0, 0.0]],
                y_ranges=[[0.0, 0.0]],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_on_objects_initializer.append_sampler(
                sampler, {"reference": other_object_name}
            )
        # 放置在一些包含区域内部的物体
        for state in conditioned_initial_place_state_in_objects:
            object_name = state[1]
            region_name = state[2]
            target_name = regions[region_name]["target"]

            site_xy_size = self.object_sites_dict[region_name].size[:2]
            sampler = InSiteRegionRandomSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                # x_ranges=[[-site_xy_size[0] / 2, site_xy_size[0] / 2]],
                # y_ranges=[[-site_xy_size[1] / 2, site_xy_size[1] / 2]],
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_initializer.append_sampler(
                sampler, {"reference": target_name, "site_name": region_name}
            )

    def _reset_internal(self):
        """
        重置仿真内部配置。
        """
        super()._reset_internal()

        # 如果不是直接从xml加载，使用初始化采样器重置所有物体位置
        if not self.deterministic_reset:

            # 从所有物体的放置初始化器中采样
            for object_property_initializer in self.object_property_initializers:
                if isinstance(object_property_initializer, OpenCloseSampler):
                    joint_pos = object_property_initializer.sample()
                    self.object_states_dict[object_property_initializer.name].set_joint(
                        joint_pos
                    )
                elif isinstance(object_property_initializer, TurnOnOffSampler):
                    joint_pos = object_property_initializer.sample()
                    self.object_states_dict[object_property_initializer.name].set_joint(
                        joint_pos
                    )
                else:
                    print("Warning!!! This sampler doesn't seem to be used")
            # robosuite没有提供此步进的api。我们手动执行此步进以提高重置仿真的速度。
            mujoco.mj_step1(self.sim.model._model, self.sim.data._data)

            object_placements = self.placement_initializer.sample()
            object_placements = self.conditional_placement_initializer.sample(
                self.sim, object_placements
            )
            object_placements = (
                self.conditional_placement_on_objects_initializer.sample(
                    object_placements
                )
            )
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.name not in list(self.fixtures_dict.keys()):
                    # 这是用于可移动物体重置的
                    self.sim.data.set_joint_qpos(
                        obj.joints[-1],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )
                else:
                    # 这是用于固定装置重置的
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = obj_pos
                    self.sim.model.body_quat[body_id] = obj_quat

    def _check_success(self):
        """
        这需要与bddl文件中的目标描述匹配

        返回:
            bool: 如果抽屉已打开则为True
        """
        return False

    def visualize(self, vis_settings):
        """
        除了父类调用外，还可视化与抽屉把手距离成比例的夹爪站点。

        参数:
            vis_settings (dict): 可视化关键字映射到T/F，确定是否应可视化该特定
                组件。应包含“grippers”关键字以及任何其他指定的相关选项。
        """
        # 先运行父类方法
        super().visualize(vis_settings=vis_settings)

    def step(self, action):
        if self.action_dim == 4 and len(action) > 4:
            # 转换OSC_POSITION动作
            action = np.array(action)
            action = np.concatenate((action[:3], action[-1:]), axis=-1)

        obs, reward, done, info = super().step(action)
        done = self._check_success()

        return obs, reward, done, info

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        self._post_process()

        return reward, done, info

    def _post_process(self):
        # 更新一些物体状态，例如灯光开关等
        for object_state in self.tracking_object_states_change:
            object_state.update_state()

    def get_robot_state_vector(self, obs):
        return np.concatenate(
            [obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
        )

    def is_fixture(self, object_name):
        """
        检查物体是否在任务中被定义为固定装置

        参数:
            object_name (str): 查询物体的名称字符串
        """
        return object_name in list(self.fixtures_dict.keys())

    @property
    def language_instruction(self):
        return self.parsed_problem["language"]

    def get_object(self, object_name):
        for query_dict in [
            self.fixtures_dict,
            self.objects_dict,
            self.object_sites_dict,
        ]:
            if object_name in query_dict:
                return query_dict[object_name]
