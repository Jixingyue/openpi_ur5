import time
import sys
import IPython
e = IPython.embed

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from constants import MASTER2PUPPET_JOINT_FN, DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE
from robot_utils import torque_on, torque_off, move_arms, move_grippers, get_arm_gripper_positions

def prep_robots(master_bot, puppet_bot):
    # 重启夹爪电机，并为所有电机设置工作模式
    puppet_bot.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit
    torque_on(puppet_bot)
    torque_on(master_bot)

    # 将机械臂移动到起始位置
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot, puppet_bot], [start_arm_qpos] * 2, move_time=1)
    # 将夹爪移动到起始位置
    move_grippers([master_bot, puppet_bot], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE], move_time=0.5)


def press_to_start(master_bot):
    # 捏合夹爪以开始数据采集
    # 仅关闭主机器人夹爪关节的力矩，以允许用户移动
    master_bot.dxl.robot_torque_enable("single", "gripper", False)
    print(f'Close the gripper to start')
    close_thresh = -0.3
    pressed = False
    while not pressed:
        gripper_pos = get_arm_gripper_positions(master_bot)
        if gripper_pos < close_thresh:
            pressed = True
        time.sleep(DT/10)
    torque_off(master_bot)
    print(f'Started!')


def teleop(robot_side):
    """ 一个用于实验遥操作的独立函数。不进行数据记录。 """
    puppet_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_{robot_side}', init_node=True)
    master_bot = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name=f'master_{robot_side}', init_node=False)

    prep_robots(master_bot, puppet_bot)
    press_to_start(master_bot)

    ### 遥操作循环
    gripper_command = JointSingleCommand(name="gripper")
    while True:
        # 同步关节位置
        master_state_joints = master_bot.dxl.joint_states.position[:6]
        puppet_bot.arm.set_joint_positions(master_state_joints, blocking=False)
        # 同步夹爪位置
        master_gripper_joint = master_bot.dxl.joint_states.position[6]
        puppet_gripper_joint_target = MASTER2PUPPET_JOINT_FN(master_gripper_joint)
        gripper_command.cmd = puppet_gripper_joint_target
        puppet_bot.gripper.core.pub_single.publish(gripper_command)
        # 睡眠 DT
        time.sleep(DT)


if __name__=='__main__':
    side = sys.argv[1]
    teleop(side)
