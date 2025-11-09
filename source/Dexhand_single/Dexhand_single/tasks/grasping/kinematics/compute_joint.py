import numpy as np
from scipy.optimize import fsolve


def compute_jacobian(joint_positions, R, D1, D2, L, S):
    """
    计算雅可比矩阵（根据论文公式）
    
    参数:
    joint_positions: 关节位置 [theta1, theta2, theta3, ...]
    R, D1, D2, L, S: 几何参数
    
    返回:
    jacobian: 雅可比矩阵 J = [∂θ1/∂l1  ∂θ1/∂l2]
                              [∂θ2/∂l1  ∂θ2/∂l2]
    """
    theta1, theta2 = joint_positions
    
    # 计算中间变量
    A = R * (1 - np.cos(theta2)) + D2 * np.sin(theta1)
    H = D1 + R * np.sin(theta2) + D2 * np.cos(theta2)
    V = L / 2
    
    # 计算 l1 和 l2 的表达式（从论文公式(1)）
    l1_squared = A**2 + (H * np.cos(theta1) - V * np.sin(theta1) - S)**2 + (H * np.sin(theta1) + V * np.cos(theta1) - V)**2
    l2_squared = A**2 + (H * np.cos(theta1) + V * np.sin(theta1) - S)**2 + (H * np.sin(theta1) - V * np.cos(theta1) + V)**2
    
    l1 = np.sqrt(l1_squared)
    l2 = np.sqrt(l2_squared)
    
    # ∂l1/∂θ1
    dl1_dtheta1 = (-H * np.sin(theta1) - V * np.cos(theta1)) * (H * np.cos(theta1) - V * np.sin(theta1) - S) / l1 + \
                  (H * np.cos(theta1) - V * np.sin(theta1)) * (H * np.sin(theta1) + V * np.cos(theta1) - V) / l1
    
    # ∂l1/∂θ2  
    dl1_dtheta2 = (R * np.sin(theta2) - D2 * np.sin(theta2)) * A / l1 + \
                  (R * np.cos(theta2) - D2 * np.sin(theta2)) * (H * np.cos(theta1) - V * np.sin(theta1) - S) / l1 + \
                  (R * np.cos(theta2) - D2 * np.cos(theta2)) * (H * np.sin(theta1) + V * np.cos(theta1) - V) / l1
    
    # ∂l2/∂θ1
    dl2_dtheta1 = (-H * np.sin(theta1) + V * np.cos(theta1)) * (H * np.cos(theta1) + V * np.sin(theta1) - S) / l2 + \
                  (H * np.cos(theta1) + V * np.sin(theta1)) * (H * np.sin(theta1) - V * np.cos(theta1) + V) / l2
    
    # ∂l2/∂θ2
    dl2_dtheta2 = (R * np.sin(theta2) - D2 * np.sin(theta2)) * A / l2 + \
                  (R * np.cos(theta2) - D2 * np.sin(theta2)) * (H * np.cos(theta1) + V * np.sin(theta1) - S) / l2 + \
                  (R * np.cos(theta2) - D2 * np.cos(theta2)) * (H * np.sin(theta1) - V * np.cos(theta1) + V) / l2
    
    J_forward = np.array([[dl1_dtheta1, dl1_dtheta2],
                          [dl2_dtheta1, dl2_dtheta2]])
    
    # 返回前向雅可比矩阵（用于IK求解）
    return J_forward



def inverse_kinematics_with_jacobian(l1_target, l2_target, l3_target, R, D1, D2, D3, D4, L, S, current_theta1=0.0, current_theta2=0.0, max_iterations=50, tolerance=1e-3):
    """
    使用雅可比矩阵迭代求解逆运动学（θ1,θ2用雅可比矩阵，θ3单独求解）
    
    参数:
    l1_target, l2_target, l3_target: 目标连杆长度
    R, D1, D2, D3, D4, L, S: 几何参数
    max_iterations: 最大迭代次数
    tolerance: 收敛容差
    
    返回:
    theta1, theta2, theta3: 关节角度（弧度）
    """
    # 首先单独求解θ3（根据论文公式）
    def solve_theta3(l3_target, R, D3, D4):
        """根据l3求解θ3"""
        # 使用数值方法求解：l3² = H3² + V3²
        def equation_theta3(theta3):
            H3 = D3 + R * np.sin(theta3) + D4 * np.cos(theta3)
            V3 = R * (1 - np.cos(theta3)) + D4 * np.sin(theta3)
            return l3_target**2 - (H3**2 + V3**2)
        
        # 尝试多个初始值
        for guess in [0.0, np.pi/6, -np.pi/6]:
            try:
                theta3 = fsolve(equation_theta3, guess)[0]
                if -np.pi/9 <= theta3 <= np.pi/9:
                    return theta3
            except:
                continue
        return None
    
    # 求解θ3
    theta3 = solve_theta3(l3_target, R, D3, D4)
    if theta3 is None:
        print("警告：无法求解θ3")
        return None, None, None
    
    # 使用雅可比矩阵迭代求解θ1, θ2
    # 使用当前角度作为初始值，提高收敛性
    theta = np.array([current_theta1, current_theta2])  # [theta1, theta2]
    l_target = np.array([l1_target, l2_target])
    
    # 添加自适应步长控制参数
    step_size = 0.5  # 增大初始步长因子，提高收敛速度
    prev_error_norm = float('inf')
    best_theta = theta.copy()  # 记录最佳解
    best_error = float('inf')
    
    for iteration in range(max_iterations):
        # 计算当前连杆长度
        theta1, theta2 = theta
        
        # 计算l1, l2（根据论文公式）
        A = R * (1 - np.cos(theta2)) + D2 * np.sin(theta1)
        H = D1 + R * np.sin(theta2) + D2 * np.cos(theta2)
        V = L / 2
        
        l1_current = np.sqrt(A**2 + (H * np.cos(theta1) - V * np.sin(theta1) - S)**2 + 
                            (H * np.sin(theta1) + V * np.cos(theta1) - V)**2)
        l2_current = np.sqrt(A**2 + (H * np.cos(theta1) + V * np.sin(theta1) - S)**2 + 
                            (H * np.sin(theta1) - V * np.cos(theta1) + V)**2)
        
        l_current = np.array([l1_current, l2_current])
        
        # 计算长度误差
        l_error = l_target - l_current
        current_error_norm = np.linalg.norm(l_error)
        
        # 记录最佳解
        if current_error_norm < best_error:
            best_error = current_error_norm
            best_theta = theta.copy()
        
        # 自适应步长控制
        if current_error_norm > prev_error_norm:
            step_size *= 0.7  # 如果误差增大，更大幅度减小步长
        else:
            step_size = min(step_size * 1.2, 1.0)  # 如果误差减小，适当增大步长
        
        prev_error_norm = current_error_norm
    
        current_joint_pos = np.array([theta1, theta2])  
        jacobian = compute_jacobian(current_joint_pos, R, D1, D2, L, S)
        
        # 使用雅可比矩阵更新关节角度
        try:
            # 使用伪逆避免奇异性
            jacobian_inv = np.linalg.pinv(jacobian)
            delta_theta = jacobian_inv @ l_error
            
            # 使用步长控制更新关节角度
            theta += step_size * delta_theta
            
            # 限制角度范围
            theta = np.clip(theta, -np.pi/9, np.pi/9)
            
        except np.linalg.LinAlgError:
            print(f"警告：第{iteration}次迭代中雅可比矩阵奇异")
            break
        # 检查收敛 - 使用更宽松的条件
        if current_error_norm < tolerance:
            # 检查解是否在合理范围内
            if (-np.pi/9 <= theta1 <= np.pi/9 and -np.pi/9 <= theta2 <= np.pi/9):
                return theta1, theta2, theta3
        
        # 如果步长太小，说明可能陷入局部最小值，尝试重新初始化
        if step_size < 0.01:
            theta = np.array([current_theta1, current_theta2])  # 重新初始化到当前角度
            step_size = 0.5
            prev_error_norm = float('inf')
    
    print(f"警告：迭代{max_iterations}次后未收敛，最终误差: {best_error:.6f}")
    # 返回最佳解而不是当前解
    if (-np.pi/9 <= best_theta[0] <= np.pi/9 and -np.pi/9 <= best_theta[1] <= np.pi/9):
        return best_theta[0], best_theta[1], theta3
    return None, None, None