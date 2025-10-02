#!/usr/bin/env python3
"""
XACRO to URDF 转换工具
使用标准的 xacro 库将 .xacro 文件转换为 .urdf 文件
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import xacro
except ImportError:
    print("错误: 需要安装 xacro 库")
    print("请运行: pip install xacro")
    sys.exit(1)


def convert_xacro_to_urdf(xacro_file_path, urdf_file_path, mappings=None):
    """
    将 xacro 文件转换为 URDF 文件
    
    Args:
        xacro_file_path (str): 输入的 .xacro 文件路径
        urdf_file_path (str): 输出的 .urdf 文件路径  
        mappings (dict): 参数映射字典，例如 {'robot_name': 'my_robot'}
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(xacro_file_path):
            raise FileNotFoundError(f"输入文件不存在: {xacro_file_path}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(urdf_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 设置参数映射
        if mappings is None:
            mappings = {
                'ns_header': 'fanuc',
                'base_position_x': '0.0',
                'base_position_y': '0.0',
                'base_position_z': '0.0',
                'base_euler_r': '0.0',
                'base_euler_p': '0.0',
                'base_euler_y': '0.0'
            }
        
        # 构建 xacro 参数列表
        xacro_args = []
        for key, value in mappings.items():
            xacro_args.extend([f'{key}:={value}'])
        
        print(f"转换文件: {xacro_file_path}")
        print(f"输出到: {urdf_file_path}")
        if xacro_args:
            print(f"使用参数: {xacro_args}")
        
        # 处理 .xacro 文件并生成 URDF 字符串
        doc = xacro.process_file(xacro_file_path, mappings=mappings)
        urdf_string = doc.toprettyxml(indent="  ")
        
        # 保存转换后的 .urdf 文件
        with open(urdf_file_path, 'w', encoding='utf-8') as f:
            f.write(urdf_string)
        
        print(f"✅ 成功转换: {xacro_file_path} -> {urdf_file_path}")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='将 XACRO 文件转换为 URDF 文件')
    parser.add_argument('input', help='输入的 .xacro 文件路径')
    parser.add_argument('-o', '--output', help='输出的 .urdf 文件路径')
    parser.add_argument('--ns_header', default='fanuc', help='命名空间前缀 (默认: fanuc)')
    parser.add_argument('--base_x', default='0.0', help='基座 X 坐标 (默认: 0.0)')
    parser.add_argument('--base_y', default='0.0', help='基座 Y 坐标 (默认: 0.0)')
    parser.add_argument('--base_z', default='0.0', help='基座 Z 坐标 (默认: 0.0)')
    parser.add_argument('--base_roll', default='0.0', help='基座翻滚角 (默认: 0.0)')
    parser.add_argument('--base_pitch', default='0.0', help='基座俯仰角 (默认: 0.0)')
    parser.add_argument('--base_yaw', default='0.0', help='基座偏航角 (默认: 0.0)')
    
    args = parser.parse_args()
    
    # 如果没有指定输出文件，自动生成
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.with_suffix('.urdf'))
    
    # 构建参数映射
    mappings = {
        'ns_header': args.ns_header,
        'base_position_x': args.base_x,
        'base_position_y': args.base_y,
        'base_position_z': args.base_z,
        'base_euler_r': args.base_roll,
        'base_euler_p': args.base_pitch,
        'base_euler_y': args.base_yaw
    }
    
    # 执行转换
    success = convert_xacro_to_urdf(args.input, args.output, mappings)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    # 如果直接运行脚本，使用默认配置
    if len(sys.argv) == 1:
        # 默认配置
        xacro_file = '/home/baai/Projects/C/Dexhand/Dexhand_single/FANUC_with_gripper_XACRO/lrmate_model.xacro'
        urdf_file = '/home/baai/Projects/C/Dexhand/Dexhand_single/FANUC_with_gripper_XACRO/test/lrmate_model.urdf'
        
        convert_xacro_to_urdf(xacro_file, urdf_file)
    else:
        main()