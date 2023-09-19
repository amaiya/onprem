import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter

try:
    # streamlit>=1.12
    from streamlit.web import cli as stcli
except ImportError:
    # streamlit<1.12
    from streamlit import cli as stcli

from onprem import __version__


def cli():
    # if "--version" in sys.argv[1:]:
    # print(__version__)
    # exit(0)
    parser = ArgumentParser(
        description=(
            "Start the streamlit app \n"
            "Example: onprem --port 8000 \n\n"
        ),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8501,
        help=("Port to use; default is 8501"),
    )
    parser.add_argument(
        "-a",
        "--address",
        type=str,
        default="0.0.0.0",
        help=("Address to bind; default is 0.0.0.0"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help=("Print app version"),
    )
    args = parser.parse_args()

    if args.version:
        print(f"OnPrem.LLM v{__version__}")
        exit(0)

    appfile = os.path.join(os.path.dirname(__file__), "ui.py")
    sys.argv = [
        "streamlit",
        "run",
        "--server.port",
        str(args.port),
        "--server.address",
        str(args.address),
        appfile,
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    cli()
