import numpy as np
import cv2
import sys
from pylibfreenect2 import (
    Freenect2, SyncMultiFrameListener, FrameType, Registration, Frame,
    createConsoleLogger, setGlobalLogger, LoggerLevel, CpuPacketPipeline
)

pipeline = CpuPacketPipeline()

# Create and set logger
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

listener = SyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)

device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

registration = Registration(device.getIrCameraParams(), device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

# Optional: Enable these if you need to use them
need_bigdepth = False
need_color_depth_map = False

bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
color_depth_map = np.zeros((424, 512), np.int32).ravel() if need_color_depth_map else None

frame_count = 0

try:
    while True:
        frames = listener.waitForNewFrame()

        color = frames["color"]
        ir = frames["ir"]
        depth = frames["depth"]

        registration.apply(color, depth, undistorted, registered,
                            bigdepth=bigdepth,
                            color_depth_map=color_depth_map)

        # Instead of showing images, you can process/save them here
        ir_np = ir.asarray() / 65535.
        depth_np = depth.asarray() / 4500.
        color_np = color.asarray()

        # Example: Print basic info
        print(f"Frame {frame_count}: color shape {color_np.shape}, ir shape {ir_np.shape}, depth shape {depth_np.shape}")

        frame_count += 1

        listener.release(frames)

except KeyboardInterrupt:
    print("Interrupted by user, shutting down...")

finally:
    device.stop()
    device.close()
    sys.exit(0)
