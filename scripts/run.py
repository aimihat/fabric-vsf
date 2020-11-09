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


class OracleCornerPolicy(Policy):

    def __init__(self):
        """Oracle corner based policy, cheating as we know the position of points.
        Note the targets, expressed as (x,y):
          upper right: (1,1)
          lower right: (1,0)
          lower left:  (0,0)
          upper left:  (0,1)
        The order in which we pull is important, though!  Choose the method to
        be rotation or distance-based. The latter seems to be more reasonable:
        pick the corner that is furthest from its target.
        Use `np.arctan2(deltay,deltax)` for angle in [-pi,pi] if we use angles.
        Be careful about the action parameterization and if we clip or not.  If
        clipping, we have to convert the x and y to each be in [0,1].
        For tier2 we may have different corner targets for a given point index.
        """
        super().__init__()
        #self._method = 'rotation'
        self._method = 'distance'

    def get_action(self, obs, t):
        """Analytic oracle corner policy.
        """
        if self.cfg['env']['delta_actions']:
            return self._corners_delta(t)
        else:
            return self._corners_nodelta(t)

    def _corners_delta(self, t):
        """Corner-based policy, assuming delta actions.
        """
        pts = self.env.cloth.pts
        assert len(pts) == 625, len(pts)
        cloth = self.env.cloth
        if self.cfg['init']['type'] == 'tier2' and (not cloth.init_side):
            self._ll = 576  # actual corner: 600
            self._ul = 598  # actual corner: 624
            self._lr = 26   # actual corner: 0
            self._ur = 48   # actual corner: 24
            print('NOTE! Flip the corner indices due to init side, tier 2')
            print(self._ll, self._ul, self._lr, self._ur)
        else:
            self._ll = 26   # actual corner: 0
            self._ul = 48   # actual corner: 24
            self._lr = 576  # actual corner: 600
            self._ur = 598  # actual corner: 624
            print('Corners are at the usual indices.')
            print(self._ll, self._ul, self._lr, self._ur)
        x0, y0, cx0, cy0, dx0, dy0, dist0 = self._data_delta(pts[self._ur], targx=1, targy=1)
        x1, y1, cx1, cy1, dx1, dy1, dist1 = self._data_delta(pts[self._lr], targx=1, targy=0)
        x2, y2, cx2, cy2, dx2, dy2, dist2 = self._data_delta(pts[self._ll], targx=0, targy=0)
        x3, y3, cx3, cy3, dx3, dy3, dist3 = self._data_delta(pts[self._ul], targx=0, targy=1)
        maxdist = max([dist0, dist1, dist2, dist3])

        if self._method == 'rotation':
            # Rotate through the corners.
            if t % 4 == 0:
                x, y, cx, cy, dx, dy = x0, y0, cx0, cy0, dx0, dy0
            elif t % 4 == 1:
                x, y, cx, cy, dx, dy = x1, y1, cx1, cy1, dx1, dy1
            elif t % 4 == 2:
                x, y, cx, cy, dx, dy = x2, y2, cx2, cy2, dx2, dy2
            elif t % 4 == 3:
                x, y, cx, cy, dx, dy = x3, y3, cx3, cy3, dx3, dy3
        elif self._method == 'distance':
            # Pick cloth corner furthest from the target.
            if dist0 == maxdist:
                x, y, cx, cy, dx, dy = x0, y0, cx0, cy0, dx0, dy0
            elif dist1 == maxdist:
                x, y, cx, cy, dx, dy = x1, y1, cx1, cy1, dx1, dy1
            elif dist2 == maxdist:
                x, y, cx, cy, dx, dy = x2, y2, cx2, cy2, dx2, dy2
            elif dist3 == maxdist:
                x, y, cx, cy, dx, dy = x3, y3, cx3, cy3, dx3, dy3
        else:
            raise ValueError(self._method)

        if self.cfg['env']['clip_act_space']:
            action = (cx, cy, dx, dy)
        else:
            action = (x, y, dx, dy)
        return action

    def _corners_nodelta(self, t):
        print('Warning! Are you sure you want the no-delta actions?')
        print('We normally do not use this due to pi and -pi angles')

        def _get_data(pt, targx, targy):
            x, y = pt.x, pt.y
            cx = (x - 0.5) * 2.0
            cy = (y - 0.5) * 2.0
            a = np.arctan2(targy-y, targx-x)
            l = np.sqrt( (x-targx)**2 + (y-targy)**2 )
            return (x, y, cx, cy, l, a)

        pts = self.env.cloth.pts
        x0, y0, cx0, cy0, l0, a0 = _get_data(pts[-1], targx=1, targy=1)
        x1, y1, cx1, cy1, l1, a1 = _get_data(pts[-25], targx=1, targy=0)
        x2, y2, cx2, cy2, l2, a2 = _get_data(pts[0], targx=0, targy=0)
        x3, y3, cx3, cy3, l3, a3 = _get_data(pts[24], targx=0, targy=1)
        maxdist = max([l0, l1, l2, l3])

        if self._method == 'rotation':
            # Rotate through the corners.
            if t % 4 == 0:
                x, y, cx, cy, l, a = x0, y0, cx0, cy0, l0, a0
            elif t % 4 == 1:
                x, y, cx, cy, l, a = x1, y1, cx1, cy1, l1, a1
            elif t % 4 == 2:
                x, y, cx, cy, l, a = x2, y2, cx2, cy2, l2, a2
            elif t % 4 == 3:
                x, y, cx, cy, l, a = x3, y3, cx3, cy3, l3, a3
        elif self._method == 'distance':
            # Pick cloth corner furthest from the target.
            if dist0 == maxdist:
                x, y, cx, cy, dx, dy = x0, y0, cx0, cy0, dx0, dy0
            elif dist1 == maxdist:
                x, y, cx, cy, dx, dy = x1, y1, cx1, cy1, dx1, dy1
            elif dist2 == maxdist:
                x, y, cx, cy, dx, dy = x2, y2, cx2, cy2, dx2, dy2
            elif dist3 == maxdist:
                x, y, cx, cy, dx, dy = x3, y3, cx3, cy3, dx3, dy3
        else:
            raise ValueError(self._method)

        # Apply scaling factor to length if needed, since for non-delta actions,
        # length is just the fraction of the maximum number of pulls, which is
        # itself a tuned quantity. Not the same reasoning as the scaling I use
        # for delta actions, but has same effect of reducing pull length.
        l = l * 1.0

        action = (x, y, l, a)
        if self.cfg['env']['clip_act_space']:
            action = (cx, cy, (l-0.5)*2, a/np.pi)
        else:
            action = (x, y, l, a)
        return action


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
    elif args.policy == 'oracle':
        policy = OracleCornerPolicy()
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
