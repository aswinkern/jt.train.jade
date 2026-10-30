import os
import sys
import yaml
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import train as tr
from tools import analyse as an
from tools import export as ex


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JADE_v3 — DINOv2 Anomaly Detection")
    parser.add_argument("-c", "--config", type=str, default="./config.yaml",
                        help="Path to config yaml file")
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    if config_dict["do_training"]:
        print("\n========== TRAINING ==========")
        tr.train(config_dict)

    if config_dict["do_analysis"]:
        print("\n========== ANALYSIS ==========")
        an.analyse(config_dict)

    if config_dict["do_export"]:
        print("\n========== EXPORT ==========")
        ex.export(config_dict)
