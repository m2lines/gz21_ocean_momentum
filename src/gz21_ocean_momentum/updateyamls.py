# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:07:41 2020

@author: Arthur
This script modifies the .yaml files so that the uri for the artifacts
location is correctly modified.
"""

import os
import sys
from os.path import basename, isdir, join

if len(sys.argv) > 1:
    experiment = sys.argv[1]
else:
    experiment = "0"

# Check that the folder mlruns exits
if not isdir("mlruns"):
    raise Exception("Folder mlruns not found.")

exp_rel_path = join("mlruns", experiment)
# Check that the experiment exists
if not isdir(exp_rel_path):
    raise Exception("Experiment {} not found.".format(experiment))

# We go through the different runs and modify all the .yaml files
cwd = os.getcwd()
for root, dirs, files in os.walk(exp_rel_path):
    if os.path.basename(root) == experiment:
        for run in dirs:
            print(run + "...")
            new_content = ""
            with open(join(exp_rel_path, run, "meta.yaml"), "r") as f:
                for i_line, line in enumerate(f):
                    if i_line == 0:
                        artifacts_location = join(cwd, root, run, "artifacts")
                        line = "artifact_uri: file:///" + artifacts_location
                        line = line + "\n"
                    new_content += line
            with open(join(exp_rel_path, run, "meta.yaml"), "w") as f:
                f.writelines(new_content)
