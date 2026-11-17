"""CLI entry point: python -m gradient_td_rl.train --algorithm tdc"""
from __future__ import annotations

import argparse

from gradient_td_rl.agents.gtd2 import GTD2
from gradient_td_rl.agents.semi_gradient_td import SemiGradientTD
from gradient_td_rl.agents.tdc import TDC
from gradient_td_rl.envs.bairds_star import BairdsStar, CLASSIC_W_INIT
from gradient_td_rl.metrics import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["td", "gtd2", "tdc"], default="tdc")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.algorithm == "td":
        agent = SemiGradientTD(8, alpha=args.alpha, w_init=CLASSIC_W_INIT)
    elif args.algorithm == "gtd2":
        agent = GTD2(8, alpha=args.alpha, beta=args.beta, w_init=CLASSIC_W_INIT)
    else:
        agent = TDC(8, alpha=args.alpha, beta=args.beta, w_init=CLASSIC_W_INIT)

    env = BairdsStar(seed=args.seed)
    weight_norms, mspbe_values = run_training(agent, env, n_steps=args.steps, track_every=args.steps // 20)

    print(f"||w||: start={weight_norms[0]:.2f}, end={weight_norms[-1]:.3g}")
    print(f"MSPBE: start={mspbe_values[0]:.3g}, end={mspbe_values[-1]:.3g}")


if __name__ == "__main__":
    main()
