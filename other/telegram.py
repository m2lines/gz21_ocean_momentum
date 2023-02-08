#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 03:01:49 2020

@author: arthur
"""

import requests

with open('/home/ag7531/.bot_token') as f:
    token = f.readline().strip()

def send_message(text: str, chat_id: str):
    parameters = {'chat_id': chat_id, 'text': text}
    r = requests.get('https://api.telegram.org/bot' + token + '/sendMessage',
                     params=parameters)