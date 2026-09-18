import os
import pickle
import select
import sys
import termios
import tty
from image_server.camera import CameraCapture
from image_server.wrist_camera import RealSenseCamera
import time
import cv2
from utils.utils import resize_image
from ur10_arm import UR10ArmSubscriber
from datetime import datetime
from controller.robotiq_2f85 import Robotiq2F85


def poll_keyboard():
    pressed_keys = []
    # 使用 os.read 直接读取底层描述符，避开 sys.stdin 的缓冲区问题，避免漏键或阻塞
    while select.select([sys.stdin], [], [], 0)[0]:
        try:
            ch = os.read(sys.stdin.fileno(), 1).decode("utf-8", errors="ignore")
            pressed_keys.append(ch)
        except:
            break
    return pressed_keys


if __name__ == "__main__":
    # ============ 配置参数 ============
    GLOBAL_CAMERA_INDEX = 4
    USE_WRIST_CAMERA = True

    ARM_IP = "192.168.0.77"
    ARM_PORT = 30004

    USE_GRIPPER = True
    GRIPPER_PORT = "/dev/ttyUSB0"
    GRIPPER_BAUDRATE = 115200
    GRIPPER_SLAVE_ID = 9
    GRIPPER_SPEED = 100      # 降低速度，原为255
    GRIPPER_FORCE = 25      # 降低力度，避免捏扁物品，原为120
    
    GRIPPER_OPEN_POS = 0    # 张开目标位置
    GRIPPER_CLOSE_POS = 170 # 闭合目标位置，不去死压到底(255)，避免损坏物品

    # 全自动夹爪控制（已关闭，现在使用键盘 o/c 手动控制）
    AUTO_GRIPPER = False
    AUTO_GRIPPER_OPEN_INTERVAL = 1.0   # 保持张开的时间（秒），可以根据需要修改
    AUTO_GRIPPER_CLOSE_INTERVAL = 1.0  # 保持闭合的时间（秒），可以根据需要修改

    COLLECTION_INTERVAL = 1 / 30  # 30Hz

    # ============ 初始化设备 ============
    print("正在初始化设备...")

    global_camera = CameraCapture(camera_index=GLOBAL_CAMERA_INDEX)
    global_camera.start()
    print("✓ 全局相机已启动")

    wrist_camera = None
    if USE_WRIST_CAMERA:
        try:
            wrist_camera = RealSenseCamera()
            print("✓ 腕部相机已启动")
        except Exception as e:
            print(f"⚠ 腕部相机启动失败: {e}")
            USE_WRIST_CAMERA = False

    arm = UR10ArmSubscriber(robot_ip=ARM_IP, port=ARM_PORT)
    arm.start()
    print("✓ UR3机械臂连接已启动")
    print(f"✓ 机械臂状态: {arm.get_status_summary()}")
    if arm.is_simulation_mode():
        print("⚠ 当前处于模拟模式，本次采集不会保存，避免混入假关节角数据")

    gripper = None
    if USE_GRIPPER:
        try:
            gripper = Robotiq2F85(
                port=GRIPPER_PORT,
                baudrate=GRIPPER_BAUDRATE,
                slave_id=GRIPPER_SLAVE_ID,
            )
            activated = gripper.activate(wait=True, timeout=5.0)
            if activated:
                print("✓ Robotiq 2F-85 已连接并激活")
            else:
                print("⚠ Robotiq 2F-85 已连接，但激活超时")
        except Exception as e:
            print(f"⚠ Robotiq 2F-85 启动失败: {e}")
            gripper = None

    terminal_settings = None
    keyboard_enabled = False
    current_gripper_target = 0
    last_gripper_command = None

    if gripper is not None:
        try:
            s = gripper.get_status()
            print(f"  夹爪初始状态: pos={s.get('position')} req={s.get('requested_position')}")
        except Exception:
            pass

        if sys.stdin.isatty():
            terminal_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            keyboard_enabled = True
            print("✓ 已启用夹爪键盘控制")
            print("  - o: 张开夹爪")
            print("  - c: 闭合夹爪")

    print("正在预热设备...")
    time.sleep(2)

    # ============ 主程序 ============
    all_data = []

    pre_angles = arm.get_angles()
    if pre_angles is None:
        pre_angles = {}
    pre_gripper_state = None
    last_dout0_state = 0  # 记录示教器数字输出信号的先前状态

    print("\n开始数据采集...")
    print("按 Ctrl+C 停止采集")
    if AUTO_GRIPPER:
        print(f"提示: 【全自动模式】夹爪张开保持 {AUTO_GRIPPER_OPEN_INTERVAL}s，闭合保持 {AUTO_GRIPPER_CLOSE_INTERVAL}s！")
    else:
        print("提示: 【手动控制模式】请按终端键盘 'o' 张开夹爪，按 'c' 闭合夹爪。也可以通过示教器 I/O 控制。")
    print("\n")

    last_auto_toggle = time.time()
    auto_gripper_status = "open"

    try:
        frame_count = 0
        while True:
            loop_start_time = time.time()
            gripper_event = None
            action = None
            if gripper is not None:
                # 1. 自动循环时间控制
                if AUTO_GRIPPER:
                    current_interval = AUTO_GRIPPER_OPEN_INTERVAL if auto_gripper_status == "open" else AUTO_GRIPPER_CLOSE_INTERVAL
                    if time.time() - last_auto_toggle >= current_interval:
                        last_auto_toggle = time.time()
                        if auto_gripper_status == "open":
                            current_gripper_target = GRIPPER_CLOSE_POS
                            auto_gripper_status = "close"
                            action = "close"
                        else:
                            current_gripper_target = GRIPPER_OPEN_POS
                            auto_gripper_status = "open"
                            action = "open"

                # 2. 键盘控制（优先级高，会覆盖自动产生的值）
                if keyboard_enabled:
                    pressed_keys = poll_keyboard()
                    for key in pressed_keys:
                        if key.lower() == "o":
                            current_gripper_target = GRIPPER_OPEN_POS
                            action = "open"
                        elif key.lower() == "c":
                            current_gripper_target = GRIPPER_CLOSE_POS
                            action = "close"

                # 监听示教器状态释放双手
                dout = arm.get_digital_out()
                dout0_state = dout & 1
                if dout0_state != last_dout0_state:
                    last_dout0_state = dout0_state
                    action = "close" if dout0_state == 1 else "open"
                    current_gripper_target = GRIPPER_CLOSE_POS if dout0_state == 1 else GRIPPER_OPEN_POS

                if action is not None:
                    try:
                        gripper.move(
                            position=current_gripper_target,
                            speed=GRIPPER_SPEED,
                            force=GRIPPER_FORCE,
                        )
                        gripper_event = {
                            "timestamp": time.time(),
                            "action": action,
                            "target_position": current_gripper_target,
                        }
                        last_gripper_command = gripper_event.copy()
                        print(f"夹爪命令 | {action} → target={current_gripper_target}")
                    except Exception as e:
                        print(f"⚠ 发送夹爪命令失败: {e}")

            frame = global_camera.get_frame()
            if frame is not None:
                frame = resize_image(frame)
            else:
                print("⚠ 无法获取全局相机图像")
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, COLLECTION_INTERVAL - elapsed)
                time.sleep(sleep_time)
                continue

            rgb_frame = None
            if USE_WRIST_CAMERA and wrist_camera is not None:
                rgb_frame = wrist_camera.get_rgb_frame()
                if rgb_frame is not None:
                    rgb_frame = resize_image(rgb_frame)

            angles = arm.get_angles()
            if angles is None:
                angles = {}

            gripper_state = None
            gripper_pos_norm = 0.0
            if gripper is not None:
                try:
                    gripper_state = gripper.get_status()
                    # 夹爪位置映射到 0~1 的区间写入数据
                    pos = gripper_state.get("position", 0)
                    gripper_pos_norm = min(1.0, max(0.0, pos / 255.0))
                except Exception as e:
                    if frame_count % 30 == 0:
                        print(f"⚠ 读取夹爪状态失败: {e}")
                    gripper_state = None

            frame_data = {
                "timestamp": time.time(),
                "frame": frame,
                "rgb_frame": rgb_frame,
                "angles": angles,
                "state": pre_angles,
                
                "gripper_pos_norm": gripper_pos_norm,  # 映射到 [0, 1] 区间的值
                "gripper": gripper_state,
                "gripper_state": pre_gripper_state,
                "gripper_command": last_gripper_command.copy() if isinstance(last_gripper_command, dict) else None,
                "gripper_event": gripper_event.copy() if isinstance(gripper_event, dict) else None,
            }

            pre_angles = angles.copy() if angles else {}
            pre_gripper_state = gripper_state.copy() if isinstance(gripper_state, dict) else None

            all_data.append(frame_data)
            frame_count += 1

            if frame_count % 30 == 0:
                if isinstance(gripper_state, dict):
                    g_pos = gripper_state.get("position", "?")
                    g_req = gripper_state.get("requested_position", "?")
                    g_obj = gripper_state.get("object_status", "?")
                    gripper_info = f"夹爪 pos={g_pos} req={g_req} obj={g_obj}"
                else:
                    gripper_info = "夹爪: N/A"
                print(f"已采集 {frame_count} 帧 | 关节数: {len(angles)} | {gripper_info}")

            if angles and frame_count % 90 == 0:
                print(f"机械臂角度: {angles}")
            elif not angles and frame_count % 10 == 0:
                print("等待机械臂数据...")

            elapsed = time.time() - loop_start_time
            sleep_time = max(0, COLLECTION_INTERVAL - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n正在停止采集...")
    finally:
        if arm.is_simulation_mode():
            print("\n⚠ 检测到模拟模式，已跳过数据保存")
            print(f"⚠ 本次内存中累计了 {len(all_data)} 帧，但未写入磁盘")
        else:
            print("\n正在保存数据...")

            os.makedirs("saved_data/all_data", exist_ok=True)

            timestep = datetime.now().strftime("%Y%m%d%H%M%S")
            save_path = f"saved_data/all_data/recorded_data_ur3_{timestep}.pkl"

            with open(save_path, "wb") as f:
                pickle.dump(all_data, f)

            print(f"✓ 数据已保存到: {os.path.abspath(save_path)}")
            print(f"✓ 总共采集了 {len(all_data)} 帧数据")

        print("\n正在释放资源...")

        global_camera.stop()
        global_camera.release()
        print("✓ 全局相机已释放")

        if wrist_camera is not None:
            wrist_camera.stop()
            print("✓ 腕部相机已释放")

        arm.stop()
        print("✓ 机械臂连接已断开")

        if gripper is not None:
            gripper.close_port()
            print("✓ 夹爪串口已关闭")

        if terminal_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal_settings)
            print("✓ 终端键盘模式已恢复")

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("\n清理完成。再见 👋")
