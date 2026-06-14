# Entry point for nanomaker's functionality
from src.nano_maker.paths import *
from src.nano_maker.container.configs import *
from src.nano_maker.nanomaker import NanoMaker
from src.nano_maker.pocketwatcher import PocketWatcher
from src.nano_maker.utility import *

import argparse
import sys
import re

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

# initialize NanoMaker system
nnmkr = NanoMaker(
    version_code=version_code,
    skeleton_weight_filename=skeleton_weight_filename,
    naanobot_weight_filename=naanobot_weight_filename,
    skeleton_cfg=skeleton_config,
    naanobot_cfg=naanobot_config,
    radial_cfg=radial_config
)

pkt_wtchr = PocketWatcher(
    naanoeng_config=naanoeng_config,
    pocket_config=pocketwatcher_config
)

def generate(args):
    smiles_str = args.smiles
    sampling_temp = args.temp if args.temp is not None else 0.3
    if not (0.01 <= sampling_temp <= 1):
        print('temp must be a float between 0.01 and 1')
        sys.exit()
    output_filename = sanitize_filename(args.o) if args.o is not None else None


    nnmkr.ingest_chemical(smiles_str)
    pocket_data = nnmkr.generate_nanopkt_data(sampling_temp=sampling_temp)
    save_nano_pocket(pocket_data, output_filename)
    return True


def sanitize_filename(string):
    string = string.strip()
    string = re.sub(r"[^\w.\-]", "_", string)
    string = re.sub(r"_+", "_", string)
    return string


def visualize(args):
    a_file = args.access
    if not file_exists(a_file):
        print(f"{a_file} not found in output container")
        return False
    o_file = args.access if args.save is True else None   # file will have the same name if save turned on
    identifier = args.mode
    pkt_wtchr.ingest_file(a_file)
    pkt_wtchr.visualize_pocket(identifier, o_file)
    return True


def report(args):
    a_file = args.access
    if not file_exists(a_file):
        print(f"{a_file} not found in output container")
        return False
    o_file = args.access if args.save is True else None

    pkt_wtchr.ingest_file(a_file)
    report_contents = pkt_wtchr.pocket_report(o_file)
    print(report_contents)
    return True


def file_exists(filename):
    filepath = POCKET_DATA / f"{filename}.nanopkt"
    if filepath.is_file():
        return True
    else:
        return False


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
        '--temp',
        type=float,
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
        '--save',
        action='store_true'
    )

    # [3] report
    rep_nnpkt = subparsers.add_parser(
        'report',
    )

    rep_nnpkt.add_argument(
        'access',
        type=str
    )

    rep_nnpkt.add_argument(
        '--save',
        action='store_true',
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