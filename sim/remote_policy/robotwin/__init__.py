"""RoboTwin adapter that runs the policy on the Jetson through the remote link.

Link this folder into RoboTwin as policy/remote (RoboTwin 13c3c47 layout):

    ln -sfn <repo>/sim/remote_policy/robotwin <RoboTwin>/policy/remote
    python script/eval_policy.py --config policy/remote/deploy_policy.yml \
        --overrides --task_name beat_block_hammer --task_config demo_clean \
        --ckpt_setting remote --instruction_type unseen --seed 10000 --test_num 10 \
        --policy_name policy.remote
"""

from .deploy_policy import encode_obs, eval, get_model, reset_model  # noqa: A004

__all__ = ["encode_obs", "get_model", "eval", "reset_model"]
