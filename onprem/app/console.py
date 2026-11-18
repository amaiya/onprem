import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter

try:
    from streamlit.web import cli as stcli
except ImportError:
    raise ImportError(
        "Please install the latest version of streamlit: pip install streamlit"
    )

from onprem import __version__


def cli():
    parser = ArgumentParser(
        description=(
            "Start the OnPrem.LLM web app \n" "Example: onprem --port 8000 \n\n"
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
        default="localhost",
        help=("Address to bind; default is localhost; Set to 0.0.0.0 if hosting on server."),
    )
    parser.add_argument(
        "--enable-xsrf-protection",
        action="store_true",
        help=(
            "Keep Streamlit's XSRF/CORS protection enabled. By default, OnPrem disables it "
            "when binding to a non-localhost address so that file uploads work when the app "
            "is accessed from other machines on the network (otherwise uploads fail with an "
            "'AxiosError: Network Error'). Only pass this if the app is served on localhost or "
            "behind a proxy that handles XSRF."
        ),
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

    appfile = os.path.join(os.path.dirname(__file__), "OnPrem.py")
    sys.argv = [
        "streamlit",
        "run",
        "--server.port",
        str(args.port),
        "--server.address",
        str(args.address),
    ]

    # When the app is reachable from other machines (non-localhost bind),
    # Streamlit's XSRF protection blocks the separate file-upload request,
    # surfacing as "AxiosError: Network Error" in the browser even though the
    # main app loads fine. Disable XSRF (and CORS, which Streamlit forces on
    # when XSRF is enabled) so uploads work across the network. Users can opt
    # back into protection with --enable-xsrf-protection.
    is_localhost = str(args.address) in ("localhost", "127.0.0.1", "::1")
    if not args.enable_xsrf_protection and not is_localhost:
        sys.argv += [
            "--server.enableXsrfProtection",
            "false",
            "--server.enableCORS",
            "false",
        ]

    sys.argv.append(appfile)
    sys.exit(stcli.main())


if __name__ == "__main__":
    cli()
