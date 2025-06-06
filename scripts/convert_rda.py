import argparse
import json
import pickle
from pathlib import Path

from rpy2 import robjects
from rpy2.robjects import default_converter
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


def rda_to_python(path: Path):
    """Load a single object from an R ``.rda`` file and return it as a
    Python object using :mod:`rpy2` conversions."""
    env = robjects.Environment()
    robjects.r["load"](str(path), env)
    if len(env) != 1:
        raise ValueError("Expected a single object in the .rda file")
    obj = next(env.values())
    with localconverter(default_converter + pandas2ri.converter):
        return robjects.conversion.rpy2py(obj)


def main():
    parser = argparse.ArgumentParser(
        description="Convert an R .rda file to JSON or pickle for use in Python"
    )
    parser.add_argument("rda_path", help="path to .rda file")
    parser.add_argument(
        "output", help="output filename (.json or .pkl/.pickle)")
    args = parser.parse_args()

    data = rda_to_python(Path(args.rda_path))
    out_path = Path(args.output)
    if out_path.suffix == ".json":
        with open(out_path, "w") as fh:
            json.dump(data, fh)
    elif out_path.suffix in {".pkl", ".pickle"}:
        with open(out_path, "wb") as fh:
            pickle.dump(data, fh)
    else:
        raise ValueError("Output file must end with .json, .pkl or .pickle")


if __name__ == "__main__":
    main()
