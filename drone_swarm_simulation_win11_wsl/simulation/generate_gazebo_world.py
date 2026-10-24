#!/usr/bin/env python3
"""
Tự động tạo file world SDF cho Gazebo Harmonic dựa trên số lượng drone (3 hoặc 5).
Giúp Gazebo chỉ spawn đúng số drone mong muốn, tránh drone thừa nằm bất động ở bãi đỗ.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DRONE_PROPERTIES = {
    1: {"name": "Red",     "rgba": "1 0.1 0.1 1",      "y": 0.0},
    2: {"name": "Green",   "rgba": "0.1 0.95 0.2 1",   "y": 1.6},
    3: {"name": "Blue",    "rgba": "0.15 0.5 1 1",     "y": 3.2},
    4: {"name": "Yellow",  "rgba": "1 0.9 0.1 1",      "y": 4.8},
    5: {"name": "Magenta", "rgba": "0.85 0.15 0.85 1", "y": 6.4},
}

def generate_world_sdf(num_drones: int = 3, output_path: str = None) -> str:
    if output_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(current_dir, "gazebo_swarm_world.sdf")

    xml_header = """<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="default">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Scene lighting (Tắt cast_shadows để tăng FPS cực mạnh trên WSLg) -->
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 20 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.2 0.22 0.25 1</ambient>
            <diffuse>0.3 0.33 0.38 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""

    drone_models = []
    for i in range(1, num_drones + 1):
        props = DRONE_PROPERTIES.get(i, {
            "name": f"Drone {i}",
            "rgba": "0.8 0.8 0.8 1",
            "y": float(i - 1) * 1.6
        })
        model_xml = f"""
    <!-- DRONE {i}: {props['name']} -->
    <model name="drone{i}">
      <static>true</static>
      <pose>0.0 {props['y']:.1f} 0.05 0 0 0</pose>
      <link name="base_link">
        <visual name="mesh_visual">
          <geometry>
            <mesh>
              <uri>model://kamikaze_drone/kamikaze_drone_ros.dae</uri>
              <scale>1 1 1</scale>
            </mesh>
          </geometry>
        </visual>
        <visual name="led_indicator">
          <pose>0 0 0.08 0 0 0</pose>
          <geometry>
            <sphere><radius>0.04</radius></sphere>
          </geometry>
          <material>
            <ambient>{props['rgba']}</ambient>
            <diffuse>{props['rgba']}</diffuse>
            <emissive>{props['rgba']}</emissive>
          </material>
        </visual>
      </link>
    </model>"""
        drone_models.append(model_xml)

    xml_footer = """
  </world>
</sdf>
"""

    full_sdf = xml_header + "".join(drone_models) + xml_footer

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_sdf)

    print(f"✅ Đã tạo file {output_path} cho {num_drones} Drones thành công.")
    return output_path


if __name__ == "__main__":
    count = 3
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        count = int(sys.argv[1])
    generate_world_sdf(count)
