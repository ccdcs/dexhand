# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

# Note: agents can be added later when needed
# from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-CX002-Teleop-Direct-v0",
    entry_point=f"{__name__}.cx002_teleop_env:Cx002TeleopEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cx002_teleop_env_cfg:Cx002TeleopEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

