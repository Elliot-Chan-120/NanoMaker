# Entry point for nanomaker's functionality
from src.nano_maker.paths import *
from src.nano_maker.nanomaker import NanoMaker
from src.nano_maker.pocketwatcher import PocketWatcher

import argparse
import sys

# 1. generate
# -> input: smiles and outputfilename
# -> output: nanopkt file targeting that smiles, named after filename

# 2. visualize
# -> input: access filename, visualization setting, output filename (optional)
# checks: valid filename and valid vis. setting
# -> output: plotly visualization of the pocket color coded to the setting

# 3. report
# -> input: access filename, output filename (optional)
# checks: valid filename
# -> output: report function of visualization

def generate(args):
    pass


def visualize(args):
    pass


def report(args):
    pass



def main():
    parser = argparse.ArgumentParser(
        prog="NanoMaker",
        description="Dual transformer protein binding pocket designer system",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog=
        """
        NanoMaker command list:
        
        note: every saved item goes in src/output_container
        
            [1] generate
                - input: chemical smiles string and an output_filename (o)
                - output: output_filename.nanopkt targeting the given smiles
            
            [2] visualize
                - input: access_filename, visualization mode and --output_filename (optional) 
                - output: opens + visualizes access_filename.nanopkt and optionally saves it as output_filename.html
                
            [3] report
                - input: access_filename and --output_filename (optional)
                - output: opens + generates report for access_filename.nanopkt and optionally saves it as output_filename.txt
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # [1] generate
    gen_nnpkt = subparsers.add_parser(
        'generate',
    )

    gen_nnpkt.add_argument(
        'smiles',
        type=str,
    )

    gen_nnpkt.add_argument(
        'o',
        type=str
    )

    # [2] visualize
    vis_nnpkt = subparsers.add_parser(
        'visualize',
    )

    vis_nnpkt.add_argument(
        'access',
        type=str,
    )

    vis_nnpkt.add_argument(
        'mode',
        type=str,
        choices = ['skeleton', 'color_code', 'polar_character', 'hydrophobicity', 'flexibility', 'steric_accessibility']

    )

    vis_nnpkt.add_argument(
        '--o',
        type=str
    )

    # [3] report
    rep_nnpkt = subparsers.add_parser(
        'report',
        type=str,
    )

    rep_nnpkt.add_argument(
        'access',
        type=str
    )

    rep_nnpkt.add_argument(
        '--o',
        type=str,
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == 'generate':
        generate(args)
    elif args.command == 'visualize':
        visualize(args)
    elif args.command == 'report':
        report(args)


if __name__ == "__main__":
    main()