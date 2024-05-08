import yaml

class LaunchFileGenerator:
    def __init__(self, yaml_file):
        with open(yaml_file) as file:
            self.start_pos = yaml.safe_load(file)

    def generate_launch_file(self, output_file):
        num_robots = len(self.start_pos)  # Count the number of states (robots)

        with open(output_file, 'w') as file:
            file.write('<?xml version="1.0"?>\n')
            file.write('<launch>\n')
            file.write('  <arg name="use_sim_time" default="true" />\n')
            file.write('  <arg name="gui" default="true" />\n')
            file.write('  <arg name="headless" default="false" />\n\n')
            file.write(f'  <arg name="num_robots" default="{num_robots}" />\n\n')
            file.write('  <include file="$(find gazebo_ros)/launch/empty_world.launch">\n')
            file.write('    <arg name="debug" value="0" />\n')
            file.write('    <arg name="gui" value="$(arg gui)" />\n')
            file.write('    <arg name="use_sim_time" value="$(arg use_sim_time)" />\n')
            file.write('    <arg name="headless" value="$(arg headless)" />\n')
            file.write('    <arg name="paused" value="false"/>\n')
            file.write('  </include>\n\n')

            for i, (key, value) in enumerate(self.start_pos.items(), start=1):
                file.write(f'  <group ns="nexus{i}">\n')
                file.write('    <!-- Load robot description -->\n')
                file.write('    <include file="$(find sml_nexus_description)/launch/sml_nexus_description.launch" />\n')
                file.write(f'    <!-- Spawn the robot -->\n')
                file.write(f'    <node name="urdf_spawner" pkg="gazebo_ros" type="spawn_model" args="-urdf -model nexus{i} -param robot_description -x {value[0]} -y {value[1]} -z 0.5" />\n')
                file.write('    <!-- Controller node -->\n')
                file.write(f'    <node name="controller" pkg="sml_nexus_tutorials" type="controller.py" output="screen">\n')
                file.write(f'      <param name="robot_name" value="nexus{i}"/>\n')
                file.write('      <param name="num_robots" value="$(arg num_robots)"/>\n')
                file.write('    </node>\n')
                file.write('  </group>\n\n')

            file.write('  <!-- Load manager -->\n')
            file.write('  <node name="manager" pkg="sml_nexus_tutorials" type="manager_node.py" output="screen" />\n\n')
            file.write('  <!-- Motion capture system simulation -->\n')
            file.write('  <include file="$(find mocap_simulator)/launch/qualisys_simulator.launch" />\n')
            file.write('</launch>\n')

# Example usage
generator = LaunchFileGenerator("/home/benjamin/catkin_ws/src/sml_nexus_tutorials/sml_nexus_tutorials/yaml_files/start_pos.yaml")
generator.generate_launch_file("/home/benjamin/catkin_ws/src/sml_nexus_tutorials/sml_nexus_tutorials/launch/generated_launch_file.launch")
