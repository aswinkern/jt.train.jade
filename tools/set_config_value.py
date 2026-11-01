"""
Sets one line under model: of a config.yaml, changing nothing else in the file (the line keeps the file's own indentation).
The line is replaced when it is there, else it is put just above model.pixelwise_threshold.

  python set_config_value.py <config.yaml> heatmap_scaling_factor 0.058983

Used by training/train.py (the run's own config) and by run.sh (the config.yaml next to it) when a training has finished.
"""
import os
import re
import sys


def set_model_value(path, key, value):
    lines = open(path).read().split("\n")
    for i, l in enumerate(lines):
        if re.match(r"^\s+%s:" % re.escape(key), l):
            lines[i] = re.match(r"^\s+", l).group(0) + "%s: %s" % (key, value)
            break
    else:
        for i, l in enumerate(lines):
            m = re.match(r"^(\s+)pixelwise_threshold:", l)                  # the new line gets the indentation of this line
            if m:
                lines.insert(i, "%s%s: %s" % (m.group(1), key, value))
                break
        else:
            raise ValueError("%s has no model.pixelwise_threshold line to put %s above" % (path, key))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
    os.chmod(tmp, os.stat(path).st_mode)
    os.replace(tmp, path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: set_config_value.py <config.yaml> <key> <value>")
    set_model_value(sys.argv[1], sys.argv[2], sys.argv[3])
