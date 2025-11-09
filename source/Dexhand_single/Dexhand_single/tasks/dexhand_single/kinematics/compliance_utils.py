import numpy as np
from scipy.optimize import fsolve

def compliance_cl(params, Fh):
    """
    输入:
        params: dict, 包含如下键：
            'd0': 初始内径 (m)
            'D0': 初始外径 (m)
            'h0': 初始长度 (m)
            't': 壁厚 (m)
            'E': 弹性模量 (Pa)
        Fh: 液压力 (N)
    返回:
        h: 变形后长度 (m)
        Ch: 柔顺性 (m/N)
        Cl: 有效线性柔顺性 (m/(N·m^2))
    """
    d0, D0, h0, t, E = params['d0'], params['D0'], params['h0'], params['t'], params['E']
    def get_deformed_diameter(Fh):
        P = -4 * Fh / (np.pi * d0**2)
        d = d0 + (P * d0**2) / (2 * E * t)
        D = D0 + (P * D0**2) / (2 * E * t)
        return d, D
    def volume_formula(h, d, D):
        return (np.pi * h / 3) * ((D/2)**2 + (d/2)**2 + (D*d)/4)
    V0 = volume_formula(h0, d0, D0)
    def equation(h):
        d, D = get_deformed_diameter(Fh)
        V = volume_formula(h, d, D)
        return V - V0
    h = fsolve(equation, h0)[0]
    # 有限差分法求Ch
    delta = 1e-4
    h1 = h
    d, D = get_deformed_diameter(Fh + delta)
    h2 = fsolve(lambda h_: volume_formula(h_, d, D) - V0, h1)[0]
    Ch = (h2 - h1) / delta
    # 有效截面积
    A = np.pi * d0**2 / 4
    Cl = Ch / A
    return h, Ch, Cl 