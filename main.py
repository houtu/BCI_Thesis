"""
Main script
"""

import argparse
from neuro.stats import STAT


config = {
    "survey": {
            "path_save": "results/survey.png", 
        },
    "within_group": {
        "method": "SparseFBCSP+LDA",
        "path_save": "results/withingroup.png",
    },
    "between_group": {
        "method": "Cov+Tangent+LR",
        "path_save": "results/betweengroup1.png",
    }
}


######################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="between_group", 
                            help='name of analysis (survey / within_group / between_group)')
    args = parser.parse_args()
    s = STAT()
    s.get(args.mode, config[args.mode])