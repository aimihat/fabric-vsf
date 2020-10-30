"""
Script for running VisMPC and analytic policies in sim.
"""
import argparse
import datetime
import logging
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from os.path import join

import numpy as np
import pkg_resources
import yaml

from analysis.visualize_demos import visualize_last_demo
from gym_cloth.envs import ClothEnv
from vismpc.cost_functions import L2, SSIM, coverage
from vismpc.mpc import VISMPC
from vismpc.visualize import Viz

np.set_printoptions(edgeitems=10, linewidth=180, suppress=True)

# Adi: Now adding the 'oracle_reveal' demonstrator policy which in reveals occluded corners.
POLICIES = ["oracle", "harris", "wrinkle", "highest", "random", "oracle_reveal"]
RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi / 180.0
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


class Policy(object):
    def __init__(self):
        pass

    def get_action(self, obs, t):
        raise NotImplementedError()

    def set_env_cfg(self, env, cfg):
        self.env = env
        self.cfg = cfg

    def _data_delta(self, pt, targx, targy, shrink=True):
        """Given pt and target locations, return info needed for action.

        Assumes DELTA actions. Returns x, y of the current point (which should
        be the target) but also the cx, and cy, which should be used if we are
        'clipping' it into [-1,1], but for the 80th time, this really means
        _expanding_ the x,y.
        """
        x, y = pt.x, pt.y
        cx = (x - 0.5) * 2.0
        cy = (y - 0.5) * 2.0
        dx = targx - x
        dy = targy - y
        dist = np.sqrt((x - targx) ** 2 + (y - targy) ** 2)
        # ----------------------------------------------------------------------
        # Sometimes we grab the top, and can 'over-pull' toward a background
        # corner. Thus we might as well try and reduce it a bit. Experiment!  I
        # did 0.95 for true corners, but if we're pulling one corner 'inwards'
        # then we might want to try a smaller value, like 0.9.
        # ----------------------------------------------------------------------
        if shrink:
            dx *= 0.90
            dy *= 0.90
        return (x, y, cx, cy, dx, dy, dist)


class RandomPolicy(Policy):
    def __init__(self):
        """Two possible types of random policies, pick one.

        Should work for all the cloth tiers.
        """
        super().__init__()
        self.type = "over_xy_plane"

    def get_action(self, obs, t):
        # edge_bias=True biases pick points toward edges of fabric
        return self.env.get_random_action(edge_bias=False, atype=self.type)


class RandomVideoPolicy(Policy):
    def __init__(self):
        """Two possible types of random policies, pick one.

        Should work for all the cloth tiers.
        """
        super().__init__()
        self.type = "over_xy_plane"

    def get_action(self, obs, t):
        # edge_bias=True biases pick points toward edges of fabric
        return self.env.get_random_action(edge_bias=True, atype=self.type)


def run(args, policy, model_name, cost_fn="L2"):
    """Run an analytic policy, using similar setups as baselines-fork.

    If we have a random seed in the args, we use that instead of the config
    file. That way we can run several instances of the policy in parallel for
    faster data collection.

    model_name and cost_fn only have semantic meaning for vismpc
    """
    with open(args.cfg_file, "r") as fh:
        cfg = yaml.safe_load(fh)
        
        if args.seed is not None:
            seed = args.seed
            cfg["seed"] = seed  # Actually I don't think it's needed but doesn't hurt?
        else:
            seed = cfg["seed"]
        
        if seed == 1500 or seed == 1600:
            print("Ideally, avoid using these two seeds.")
            sys.exit()
        
        if args.policy != "vismpc":
            model_name = "NA"
            cost_fn = "NA"
        
        stuff = "-seed-{}-{}-model-{}-cost-{}_epis_{}".format(
            seed,
            cfg["init"]["type"],
            model_name.replace("/", "_"),
            cost_fn,
            args.max_episodes,
        )
        
        result_path = args.result_path.replace(".pkl", "{}.pkl".format(stuff))
    
    np.random.seed(seed)

    # Should seed env this way, following gym conventions.  NOTE: we pass in
    # args.cfg_file here, but then it's immediately loaded by ClothEnv. When
    # env.reset() is called, it uses the ALREADY loaded parameters, and does
    # NOT re-query the file again for parameters (that'd be bad!).
    env = ClothEnv(args.cfg_file)
    env.seed(seed)
    env.render(filepath=args.render_path)
    if args.policy == "vismpc":
        policy.set_env_cfg(env, cfg, model_name, cost_fn)
    else:
        policy.set_env_cfg(env, cfg)

    # Book-keeping.
    num_episodes = 0
    stats_all = []
    coverage = []
    variance_inv = []
    nb_steps = []

    for ep in range(args.max_episodes):
        obs = env.reset()
        # Go through one episode and put information in `stats_ep`.
        # Don't forget the first obs, since we need t _and_ t+1.
        stats_ep = defaultdict(list)
        stats_ep["obs"].append(obs)
        done = False
        num_steps = 0
        
        while not done:
            action = policy.get_action(obs, t=num_steps)
            if cfg["env"]["intermediary_frames"]:
                interm_obs, obs, rew, done, info = env.step(action)
                stats_ep["interm_obs"].append(interm_obs)
            else:
                obs, rew, done, info = env.step(action)
            stats_ep["obs"].append(obs)
            stats_ep["rew"].append(rew)
            stats_ep["act"].append(action)
            stats_ep["done"].append(done)
            stats_ep["info"].append(info)
            num_steps += 1
        num_episodes += 1
        coverage.append(info["actual_coverage"])
        variance_inv.append(info["variance_inv"])
        nb_steps.append(num_steps)
        stats_all.append(stats_ep)
        print("\nInfo for most recent episode: {}".format(info))
        print("Finished {} episodes.".format(num_episodes))
        print(
            "  {:.3f} +/- {:.3f} (coverage)".format(np.mean(coverage), np.std(coverage))
        )
        print(
            "  {:.2f} +/- {:.1f} ((inv)variance)".format(
                np.mean(variance_inv), np.std(variance_inv)
            )
        )
        print(
            "  {:.2f} +/- {:.2f} (steps per episode)".format(
                np.mean(nb_steps), np.std(nb_steps)
            )
        )

        # Just dump here to keep saving and overwriting.
        with open(result_path, "wb") as fh:
            pickle.dump(stats_all, fh)

    assert len(stats_all) == args.max_episodes, len(stats_all)
    if env.render_proc is not None:
        env.render_proc.terminate()
        env.cloth.stop_render()

    input('Press a key to play demo')
    visualize_last_demo(result_path)



if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument("policy", type=str, help="name of the policy to use")
    pp.add_argument("--max_episodes", type=int, default=10)
    pp.add_argument("--seed", type=int)
    pp.add_argument(
        "--model_path",
        type=str,
        default="/data/pure_random",
        help="[for vismpc policy] SV2P model path, which should be parent of \
        sv2p_model_cloth and sv2p_data_cloth",
    )
    args = pp.parse_args()
    args.policy = (args.policy).lower()
    if args.policy == "random":
        policy = RandomPolicy()
    elif args.policy == "video_random":
        policy = RandomVideoPolicy()
    else:
        raise ValueError(args.policy)

    # Use this to store results. For example, these can be used to save the
    # demonstrations that we later load to augment DeepRL training. We can
    # augment the file name later in `run()`. Add policy name so we know the
    # source. Fortunately, different trials can be combined in a larger lists.
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    result_pkl = "demos-{}-pol-{}.pkl".format(date, args.policy)

    # Each time we use the environment, we need to pass in some configuration.
    args.file_path = fp = os.path.dirname(os.path.realpath(__file__))
    args.cfg_file = join(fp, "../cfg/video_dataset.yaml")
    args.render_path = join(fp, "../render/build")  # Must be compiled!
    args.result_path = join(fp, "../logs/{}".format(result_pkl))

    run(args, policy, args.model_path)
