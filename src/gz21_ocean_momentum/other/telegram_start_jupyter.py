#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 02:51:59 2020

@author: arthur
In this script we read the updates for a telegram bot and let the user
start a program by text.
"""

import requests
import json
import sys
from telegram import send_message
import subprocess
import time
from os.path import join
import hashlib
import logging

with open("/home/ag7531/.bot_token") as f:
    token = f.readline().strip()


def get_updates():
    r = requests.get("https://api.telegram.org/bot" + token + "/getupdates")
    return r


def start_jupyter():
    cmd_text = "/opt/slurm/bin/sbatch run-jupyter.sbatch"
    r = subprocess.run(
        cmd_text, shell=True, capture_output=True, cwd="/home/ag7531/myjupyter/"
    )
    return r


def get_output_file(job_id: int, chat_id: str, verbose: str = True):
    file_name = "".join(("slurm-", str(job_id), ".out"))
    file_path = join("/home/ag7531/myjupyter", file_name)
    n = 0
    while True:
        n += 1
        if n >= 2:
            return (
                1,
                [
                    "Output file not found...",
                    "This might be due to the job being stuck on the queue.",
                    "We will send you a text if the job does start at some " "point.",
                ],
            )
        try:
            if verbose:
                send_message("Looking for output file " + file_path, chat_id)
            with open(file_path) as f:
                if verbose:
                    send_message("Found the file!", chat_id)
                lines = f.readlines()
                for line in lines:
                    if "Serving notebooks from" in line:
                        return (0, lines)
        except FileNotFoundError:
            pass
        time.sleep(20)


def check_user(user_id: int):
    try:
        with open(".telegram_users") as f:
            return user_id in [int(s.strip()) for s in f.readlines()]
    except FileNotFoundError:
        return False


def register_new_user(user_id):
    with open(".telegram_users", "a") as f:
        f.write(str(user_id) + "\n")


# We read updates, if we find the expected message we start the jupyter script
r = get_updates()
updates = json.loads(r.text)

if not updates["ok"]:
    sys.exit(1)

# Read the last update from file
try:
    with open(".last_update_id", "r") as f:
        last_update_id = f.readline()
        if last_update_id == "":
            last_update_id = 0
        else:
            last_update_id = int(last_update_id)
except FileNotFoundError:
    last_update_id = 0

updates = updates["result"]
for update in updates:
    update_id = update["update_id"]
    if update_id > last_update_id:
        # Update last_update_id in file
        with open(".last_update_id", "w") as f:
            f.write(str(update_id))
        # Check user
        user_id = int(update["message"]["from"]["id"])
        chat_id = update["message"]["chat"]["id"]
        if check_user(user_id):
            if update["message"]["text"] == "start jupyter":
                send_message("Trying to start jupyter...", chat_id)
                r = start_jupyter()
                s = r.stdout.decode()
                send_message(s, chat_id)
                job_id = s.split()[-1]
                logging.log(logging.DEBUG, f"job id: {job_id}.")
                exit_code, output_file = get_output_file(int(job_id), chat_id)
                if exit_code == 0:
                    print(f"output file content: {output_file}")
                    send_message("You can now connect via ssh", chat_id)
                else:
                    print("Output file not found")
                    for line in output_file:
                        send_message(line, chat_id)
                    with open(".jobs_on_queue", "w") as f:
                        f.writelines(
                            [
                                job_id + ":" + str(chat_id),
                            ]
                        )
            else:
                send_message("Did not understand your request, sorry.", chat_id)
        else:
            message = update["message"]["text"]
            hash_m = hashlib.sha256(message.encode()).hexdigest()
            if (
                hash_m
                == "141398e3d78065d224cc535a984d7aa000a0429b1ead2687f16a81e05c8f5f41"
            ):
                register_new_user(user_id)
                send_message("Thanks, you are now registered.", chat_id)
                send_message(
                    'To start a new session, please type "start ' 'jupyter"', chat_id
                )
                send_message(
                    "Note that the server may take a minute or two " "to reply.",
                    chat_id,
                )
            else:
                send_message("You are not registered as a user yet.", chat_id)
                send_message("Please reply with the password", chat_id)

# Check jobs on the queue
with open(".jobs_on_queue", "r") as f:
    lines = f.readlines()
    new_lines = []
    for line in lines:
        job_id, chat_id = line.split(":")
        exit_code, output_file = get_output_file(int(job_id), chat_id)
        if exit_code == 0:
            print(f"output file content: {output_file}")
            for line in output_file:
                send_message(line, chat_id)
            send_message("Done!", chat_id)
        else:
            new_lines.append(line)
with open(".jobs_on_queue", "w") as f:
    f.writelines(new_lines)
