#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch  
import time
import os
import numpy as np
from typing import Optional

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        
        self.get_logger().info("Initializing ImageProcessor node...")

   
        self.declare_parameter('queue_size', 10000)
        self.declare_parameter('crop_coords', [1032, 1544, 850, 1362])  # y1, y2, x1, x2
        self.declare_parameter('normalization_mean', [114., 121., 134.])
        self.declare_parameter('display_images', False)
        self.declare_parameter('camera_topic_prefix', '/vimbax_camera_')
        self.declare_parameter('camera_topic_suffix', '/image_raw')

        
        queue_size = self.get_parameter('queue_size').value
        prefix = self.get_parameter('camera_topic_prefix').value
        suffix = self.get_parameter('camera_topic_suffix').value

       
        self.bridge = CvBridge()

       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        try:
            self.model = self.load_model()
            self.get_logger().info("Model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {str(e)}")
            raise

        
        input_topic = self.find_image_topic(prefix, suffix)
        if not input_topic:
            self.get_logger().error("No matching camera image topic found.")
            raise RuntimeError("Image topic not found")

        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            queue_size
        )
        self.get_logger().info(f"Subscribed to: {input_topic}")

       
        self.output_dirs = {
            'images': os.path.join(os.getcwd(), "images"),
            'masks': os.path.join(os.getcwd(), "masks")
        }
        for dir_name, dir_path in self.output_dirs.items():
            os.makedirs(dir_path, mode=0o777, exist_ok=True)

        
        self.last_processed_time = time.time()
        self.frame_count = 0
        self.processing_times = []

        self.get_logger().info("ImageProcessor initialized successfully")

    def find_image_topic(self, prefix: str, suffix: str) -> Optional[str]:
        """Search for the first matching image topic"""
        all_topics = self.get_topic_names_and_types()
        for topic, types in all_topics:
            if topic.startswith(prefix) and topic.endswith(suffix):
                if 'sensor_msgs/msg/Image' in types:
                    return topic
        return None

    def load_model(self) -> torch.nn.Module:
        model_path = "epoch_99.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = torch.load(model_path, map_location=self.device)
        model.to(self.device)
        model.eval()
        return model

    def image_callback(self, msg: Image) -> None:
        start_time = time.time()
        self.frame_count += 1
        try:
            cv_image_ = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            y1, y2, x1, x2 = self.get_parameter('crop_coords').value
            cv_image = cv_image_[y1:y2, x1:x2]

            timestamp = int(time.time())
            image_path = os.path.join(self.output_dirs['images'], f"frame_{timestamp}_{self.frame_count}.jpg")
            cv2.imwrite(image_path, cv_image)

            processed_image = self.run_model(cv_image)

            mask_path = os.path.join(self.output_dirs['masks'], f"frame_{timestamp}_{self.frame_count}.png")
            cv2.imwrite(mask_path, processed_image)

            if self.get_parameter('display_images').value:
                cv2.imshow("Original Image", cv_image)
                cv2.imshow("Processed Mask", processed_image)
                cv2.waitKey(1)

            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            if self.frame_count % 10 == 0:
                avg_time = np.mean(self.processing_times[-10:])
                self.get_logger().info(
                    f"Processed {self.frame_count} frames | "
                    f"Avg processing time: {avg_time:.4f}s | "
                    f"Current: {processing_time:.4f}s"
                )

        except Exception as e:
            self.get_logger().error(f"Error processing frame {self.frame_count}: {str(e)}")

    def np2Tensor(self, array: np.ndarray) -> torch.Tensor:
        if len(array.shape) == 4 and array.shape[-1] == 1:
            array = array.squeeze(axis=-1)
        if len(array.shape) == 2:
            tensor = torch.FloatTensor(array[np.newaxis, :, :].astype(float))
        elif len(array.shape) == 3:
            tensor = torch.FloatTensor(array.transpose(2, 0, 1).astype(float))
        else:
            raise ValueError(f"Unexpected array shape: {array.shape}")
        return tensor

    def run_model(self, image: np.ndarray) -> np.ndarray:
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            norm_mean = self.get_parameter('normalization_mean').value
            image = (image.astype(np.float32) - norm_mean) / 255.0
            input_tensor = self.np2Tensor(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out, _ = self.model(input_tensor)
                output = torch.sigmoid(out).squeeze(0)
            output_image = output.detach().cpu().numpy().squeeze(0) * 255
            return output_image.astype('uint8')
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    try:
        node = ImageProcessor()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down due to keyboard interrupt")
    except Exception as e:
        node.get_logger().error(f"Node crashed: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
