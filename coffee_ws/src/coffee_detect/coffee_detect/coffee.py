#!/home/wzb/miniconda3/envs/sam/bin/python
import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from scipy.spatial.transform import Rotation as R_scipy
import open3d as o3d
from sklearn.decomposition import PCA
import warnings
import time
warnings.filterwarnings("ignore")


class CoffeeDetectNode(Node):
    def __init__(self):
        super().__init__('coffee_detect')

        # 声明参数（无默认值！）
        self.declare_parameter('sam_checkpoint')
        self.declare_parameter('device')
        self.declare_parameter('camera_fx')
        self.declare_parameter('camera_fy')
        self.declare_parameter('camera_cx')
        self.declare_parameter('camera_cy')
        self.declare_parameter('min_area')
        self.declare_parameter('handeye_pos')
        self.declare_parameter('handeye_quat')

        # 从参数服务器获取值
        self.sam_checkpoint = self.get_parameter('sam_checkpoint').value
        self.device = self.get_parameter('device').value
        self.fx = self.get_parameter('camera_fx').value
        self.fy = self.get_parameter('camera_fy').value
        self.cx = self.get_parameter('camera_cx').value
        self.cy = self.get_parameter('camera_cy').value
        self.min_area = self.get_parameter('min_area').value
        self.handeye_pos = self.get_parameter('handeye_pos').value
        self.handeye_quat = self.get_parameter('handeye_quat').value

        # 验证路径
        if not Path(self.sam_checkpoint).exists():
            self.get_logger().fatal(f"SAM checkpoint not found: {self.sam_checkpoint}")
            raise FileNotFoundError(f"SAM checkpoint not found: {self.sam_checkpoint}")

        # 加载 SAM
        self.get_logger().info("Loading SAM model...")
        sam = sam_model_registry["vit_h"](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        self.get_logger().info(f"SAM loaded on {self.device}.")

        # 手眼标定矩阵
        self.T_e2c = self.pose_to_matrix(self.handeye_pos, self.handeye_quat)

        # ROS 接口
        self.bridge = CvBridge()
        self.grasp_pub = self.create_publisher(PoseStamped, '/my_pose_cmd', 10)
        self.gripper_pub = self.create_publisher(Float64, '/my_gripper_cmd', 10)

        rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')
        pose_sub = Subscriber(self, PoseStamped, '/end_pose_stamped')

        self.ts = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, pose_sub],
            queue_size=5,
            slop=0.1
        )
        self.ts.registerCallback(self.callback)

        # 缓存最新同步帧（不在回调中做重处理！）
        self.latest_frame = None

        self.get_logger().info("Coffee detect node ready (waiting for synchronized frames).")

    def pose_to_matrix(self, pos, quat):
        T = np.eye(4)
        T[:3, :3] = R_scipy.from_quat(quat).as_matrix()
        T[:3, 3] = pos
        return T

    def matrix_to_pose(self, T):
        pos = T[:3, 3]
        quat = R_scipy.from_matrix(T[:3, :3]).as_quat()
        return pos, quat

    def move_pose_along_axis(self, position, quaternion, distance, axis):
        ''' 
        position沿指定axis轴负方向平移distance(cm)
        '''
        position = np.asarray(position, dtype=float)
        quaternion = np.asarray(quaternion, dtype=float)
        rot = R_scipy.from_quat(quaternion)
        ax_dict = {'x': 0, 'y': 1, 'z': 2}
        if axis not in ax_dict:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        axis_vector = rot.as_matrix()[:, ax_dict[axis]]
        new_position = position - distance * axis_vector
        return new_position, quaternion

    def callback(self, rgb_msg, depth_msg, end_pose_msg):
        """仅做轻量级转换并缓存，不阻塞！"""
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            # 缓存最新帧（覆盖旧帧，避免堆积）
            self.latest_frame = (rgb, depth, end_pose_msg)
        except Exception as e:
            self.get_logger().error(f"Frame conversion failed: {e}")

    def process_frame(self, rgb, depth, end_pose_msg):
        """执行完整处理逻辑（SAM、点云、发布等）"""
        try:
            self.get_logger().info("Processing synchronized frame...")
            rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # --- Color mask ---
            hsv = cv2.cvtColor(rgb_rgb, cv2.COLOR_RGB2HSV)
            lower = np.array([0, 30, 20])
            upper = np.array([40, 255, 200])
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            mask_clean = np.zeros_like(mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > self.min_area:
                    mask_clean[labels == i] = 255

            # --- Find vertical bag ---
            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_bag = None
            max_area = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if h < w:
                    continue
                if area > max_area:
                    max_area = area
                    best_bag = (x, y, x + w, y + h)

            if best_bag is None:
                self.get_logger().warn("No valid bag detected.")
                return

            # --- SAM refinement ---
            self.sam_predictor.set_image(rgb_rgb)
            input_box = np.array(best_bag)
            masks, _, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=True,
            )
            final_mask = masks[2].astype(np.uint8)

            # --- Point cloud ---
            ys, xs = np.where(final_mask == 1)
            if len(xs) == 0:
                self.get_logger().warn("Empty SAM mask.")
                return

            depths = depth[ys, xs].astype(np.float32)
            if depth.dtype == np.uint16:
                depths /= 1000.0  # mm → m
            valid = depths > 0
            xs, ys, depths = xs[valid], ys[valid], depths[valid]
            if len(xs) == 0:
                self.get_logger().warn("No valid depth.")
                return

            zs = depths
            xs_3d = (xs - self.cx) * zs / self.fx
            ys_3d = (ys - self.cy) * zs / self.fy
            points_3d = np.stack([xs_3d, ys_3d, zs], axis=-1)

            # --- RANSAC + PCA ---
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=5, num_iterations=1000)
            if len(inliers) < 100:
                self.get_logger().warn("RANSAC failed.")
                return

            front_points = np.asarray(pcd.select_by_index(inliers).points)
            normal = np.array(plane_model[:3])
            z_axis = normal / np.linalg.norm(normal)

            min_b = np.min(front_points, axis=0)
            max_b = np.max(front_points, axis=0)
            origin = (min_b + max_b) / 2

            points_centered = front_points - origin
            proj = points_centered - np.outer(np.dot(points_centered, z_axis), z_axis)
            pca = PCA(n_components=2)
            pca.fit(proj)
            long_dir = pca.components_[0]
            short_dir = pca.components_[1]

            y_axis = long_dir / np.linalg.norm(long_dir)
            x_axis = short_dir / np.linalg.norm(short_dir)

            if np.dot(np.cross(y_axis, z_axis), x_axis) < 0:
                y_axis = -y_axis

            length_long = np.max(proj @ y_axis) - np.min(proj @ y_axis)
            length_short = np.max(proj @ x_axis) - np.min(proj @ x_axis)
            center_point = origin - y_axis * (length_long / 2)

            self.get_logger().info(f"Bag dimensions - Length: {length_long:.3f} m, Width: {length_short:.3f} m")

            # if not (0.17 < length_short < 0.25 and 0.25 < length_long < 0.32):
            #     self.get_logger().warn("Detected bag size out of expected range.")
            #     return

            R_cam = np.column_stack((x_axis, y_axis, z_axis))
            R_adjust = R_scipy.from_euler('z', 90, degrees=True).as_matrix()
            R_grasp = R_cam @ R_adjust
            quat_cam = R_scipy.from_matrix(R_grasp).as_quat()

            # --- Transform to base ---
            T_obj_in_cam = self.pose_to_matrix(center_point, quat_cam)
            T_ee_in_base = self.pose_to_matrix(
                [end_pose_msg.pose.position.x, end_pose_msg.pose.position.y, end_pose_msg.pose.position.z],
                [end_pose_msg.pose.orientation.x, end_pose_msg.pose.orientation.y,
                 end_pose_msg.pose.orientation.z, end_pose_msg.pose.orientation.w]
            )
            T_obj_in_base = T_ee_in_base @ self.T_e2c @ T_obj_in_cam
            pos_base, quat_base = self.matrix_to_pose(T_obj_in_base)

            # --- Retreat along local -Z ---
            pos_base, quat_base = self.move_pose_along_axis(pos_base, quat_base, distance=0.08, axis='z')

            self.get_logger().info(
                f"Pos = [{pos_base[0]:.3f}, {pos_base[1]:.3f}, {pos_base[2]:.3f}], "
                f"Quat = [{quat_base[0]:.3f}, {quat_base[1]:.3f}, {quat_base[2]:.3f}, {quat_base[3]:.3f}]"
            )

        except Exception as e:
            self.get_logger().error(f"Processing failed: {e}")

        return pos_base, quat_base

    def publish_pos(self, position, quaternion):
            # --- Publish ---
            grasp_msg = PoseStamped()
            grasp_msg.header.stamp = self.get_clock().now().to_msg()
            grasp_msg.header.frame_id = "base_link"
            grasp_msg.pose.position.x = float(position[0])
            grasp_msg.pose.position.y = float(position[1])
            grasp_msg.pose.position.z = float(position[2])
            grasp_msg.pose.orientation.x = float(quaternion[0])
            grasp_msg.pose.orientation.y = float(quaternion[1])
            grasp_msg.pose.orientation.z = float(quaternion[2])
            grasp_msg.pose.orientation.w = float(quaternion[3])

            self.grasp_pub.publish(grasp_msg)
            self.get_logger().info(f"Pose published: "
                                   f"xyz = {position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}, "
                                   f"xyzw = {quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}")
    
    def publish_gripper(self, gap):
        '''
        Docstring for publish_gripper
        发送夹爪控制命令
        :param gap: 实际夹爪开合距离为gap*2，单位米
        '''
        gripper_msg = Float64()
        gripper_msg.data = gap
        self.gripper_pub.publish(gripper_msg)
        self.get_logger().info(f"Gripper command published: gap = {gap*2:.4f} m")

def main(args=None):
    rclpy.init(args=args)
    node = CoffeeDetectNode()

    try:
        time.sleep(3.0)  # 等待初始化完成
        # 观测姿态 方便拍摄
        node.publish_pos(
            position=np.array([-0.065623, 0.001, 0.400744]),
            quaternion=np.array([0.004185950432734016, 0.7850548335728314, 0.01612455744139523, 0.6186224168049711])
        )
        time.sleep(3.0)  # 确保动作完成

        while rclpy.ok():

            # 非阻塞地处理一次回调（接收新消息）
            rclpy.spin_once(node, timeout_sec=0.5)

            # 如果有新帧，处理它
            if node.latest_frame is not None:
                rgb, depth, end_pose_msg = node.latest_frame
                node.latest_frame = None  # 清空，防止重复处理

                # 执行完整处理流程
                p, q = node.process_frame(rgb, depth, end_pose_msg)

                # 提手位姿
                p, q = node.move_pose_along_axis(p, q, distance=-0.04, axis='y')

                # 抓取前位姿
                p0, q0 = node.move_pose_along_axis(p, q, distance=0.1, axis='z')

                node.publish_pos(p0, q0)
                time.sleep(3.0)  # 确保动作完成

                # 打开夹爪
                node.publish_gripper(gap=0.035)

                # 移动到抓取点
                node.publish_pos(p, q)
                time.sleep(3.0)

                # 关闭夹爪
                node.publish_gripper(gap=0.0)
                time.sleep(3.0)

                # 提起物体
                pe = np.array([0.046216, 0.002615, 0.478671])
                qe = np.array([-0.0030402966631021544, 0.6838194050006842, 0.02950326107270329, 0.7290482395059921])
                node.publish_pos(pe, qe)

                break  # 处理完一帧后退出（可根据需要修改）

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()