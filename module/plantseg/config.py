"""Experiment configuration functions
"""
from pathlib import Path

import git


def get_expid(tensorboard_dir):
    """Find a new incremental experiment ID
    """
    tb_dir = Path(tensorboard_dir)

    runs = [dir for dir in tb_dir.glob('*')
            if dir.is_dir()]

    exp_nums = set()
    for run_dir in runs:
        name = run_dir.name.split('-')
        try:
            exp_nums.add(int(name[0]))
        except:
            continue

    if exp_nums:
        my_exp = max(exp_nums) + 1
    else:
        my_exp = 0

    assert my_exp not in exp_nums

    repo = git.Repo(search_parent_directories=True)
    gitsha = repo.head.object.hexsha[:5]
    return '-'.join(["{:03d}".format(my_exp), gitsha])
