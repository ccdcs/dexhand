# 软管形变力矩计算 - 使用说明

## 概述

本模块实现了基于论文的液压柔顺性建模，用于计算软管形变产生的力矩。根据当前l0值和目标l值，可以计算出由于软管形变而产生的对接触面的力矩。

## 核心功能

### 1. 液压柔顺性建模 (`HydraulicComplianceModel`)

基于论文中的液压系统柔顺性分析，包括：

- **体积公式**（论文公式6）：`V = πh/3 * (D²/4 + d²/4 + Dd/4)`
- **变形直径计算**（论文公式9）：考虑内部压力对直径的影响
- **柔顺性计算**：计算有效线性柔顺性Cl

### 2. 柔顺性扭矩计算 (`ComplianceTorqueCalculator`)

实现论文中的雅可比矩阵变换和扭矩计算：

- **雅可比矩阵计算**（论文公式4）
- **柔顺性扭矩计算**（论文公式10）：`τ = (1/Cl) * J^T * Δl`
- **关节空间刚度矩阵**（论文公式11）：`Kq = (1/Cl) * J^T * J`

## 使用方法

### 基本使用

```python
from compliance_torque_calculator import calculate_compliance_torque_from_l_values

# 材料参数
material_params = {
    'd0': 0.005,    # 5mm 内径
    'D0': 0.008,    # 8mm 外径
    'h0': 0.03,     # 30mm 初始长度
    't': 0.0015,    # 1.5mm 壁厚
    'E': 2e6        # 2MPa 弹性模量
}

# 几何参数
geometry_params = {
    'R': 15,
    'D1': 10.5,
    'D2': 5,
    'D3': 3.58,
    'D4': 3.5,
    'L': 27,
    'S': 3.5
}

# 输入数据
l_target = [0.03, 0.03, 0.03]  # 目标连杆长度
l0 = [0.025, 0.025, 0.025]     # 初始连杆长度
joint_positions = [0.1, 0.1, 0.1]  # 关节位置

# 计算柔顺性扭矩
torque, force, compliance = calculate_compliance_torque_from_l_values(
    l_target, l0, material_params, geometry_params, joint_positions
)

if torque is not None:
    print(f"柔顺性扭矩: {torque}")
    print(f"液压力: {force:.2f} N")
    print(f"柔顺性值: {compliance:.6f}")
```

### 在仿真中使用

```python
# 在仿真循环中集成
def run_simulator_with_compliance(sim, scene):
    # 初始化柔顺性计算器
    calculator = ComplianceTorqueCalculator(material_params, geometry_params)
    
    while simulation_app.is_running():
        # 获取当前l值和目标l值
        l_target = [l1_target, l2_target, l3_target]
        l0 = [l1_0, l2_0, l3_0]
        joint_positions = [theta1, theta2, theta3]
        
        # 计算柔顺性扭矩
        result = calculator.compute_compliance_torque(l_target, l0, joint_positions)
        
        if result is not None:
            compliance_torque, hydraulic_force, compliance_value = result
            
            # 将柔顺性扭矩应用到关节控制
            target_action = scene["Dofbot"].data.default_joint_pos.clone()
            target_action[:, 0] += compliance_torque[0]  # L1_joint
            target_action[:, 2] += compliance_torque[1]  # L2_pre_joint
            target_action[:, 4] += compliance_torque[2]  # L3_pre_joint
            
            scene["Dofbot"].set_joint_position_target(target_action)
```

## 参数说明

### 材料参数 (`material_params`)

- `d0`: 初始内径 (m)
- `D0`: 初始外径 (m)
- `h0`: 初始长度 (m)
- `t`: 壁厚 (m)
- `E`: 弹性模量 (Pa)

### 几何参数 (`geometry_params`)

- `R`: 径向距离
- `D1, D2, D3, D4`: 偏移量
- `L`: 执行器端点间距离
- `S`: 偏移参数

## 输出说明

### 柔顺性扭矩 (`compliance_torque`)

返回一个3维向量，对应三个关节的柔顺性扭矩：
- `compliance_torque[0]`: L1_joint的柔顺性扭矩
- `compliance_torque[1]`: L2_pre_joint的柔顺性扭矩
- `compliance_torque[2]`: L3_pre_joint的柔顺性扭矩

### 液压力 (`hydraulic_force`)

计算得到的液压力，单位为牛顿(N)。

### 柔顺性值 (`compliance_value`)

有效线性柔顺性，单位为m/(N·m²)。

## 测试和验证

运行测试脚本：

```bash
python scripts/test_compliance_calculation.py
```

这将：
1. 从CSV文件读取实际数据
2. 计算柔顺性扭矩
3. 进行统计分析
4. 保存结果到文件

## 注意事项

1. **参数调整**：材料参数和几何参数需要根据实际的软管和机械结构进行调整
2. **数值稳定性**：在某些极端情况下，数值求解可能失败，需要检查输入参数
3. **单位一致性**：确保所有输入参数使用相同的单位系统（推荐使用SI单位）

## 理论基础

本实现基于论文中的以下关键公式：

1. **体积约束**（公式6）：`V = V₀ = constant`
2. **变形直径**（公式9）：考虑压力对直径的影响
3. **柔顺性扭矩**（公式10）：`Δτ = (1/Cl) * J^T * Δl`
4. **刚度矩阵**（公式11）：`Kq = (1/Cl) * J^T * J`

这些公式描述了软管在液压作用下的变形行为以及由此产生的柔顺性效应。 