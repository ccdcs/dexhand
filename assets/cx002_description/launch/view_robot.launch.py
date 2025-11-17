from launch import LaunchDescription
import launch_ros.descriptions
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            description="Start RViz2 automatically with this launch file.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_name",
            default_value="cx002.urdf",
            description="robot_name of this view with this launch file.",
        )
    )

    # Initialize Arguments
    gui = LaunchConfiguration("gui")
    robot_name = LaunchConfiguration("robot_name")

    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("cx002_description"),
                    "urdf",
                    robot_name,
                ]
            ),
        ]
    )
    robot_description = {"robot_description": launch_ros.descriptions.ParameterValue(
        robot_description_content)}
    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare("cx002_description"), "rviz", "cx002.rviz"])

    joint_state_publisher_gui_node = Node(
        condition=IfCondition(gui),
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui')

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=['-d', rviz_config_path],
        condition=IfCondition(gui),
    )

    nodes = [
        joint_state_publisher_gui_node,
        robot_state_pub_node,
        rviz_node,
    ]

    return LaunchDescription(declared_arguments + nodes)
