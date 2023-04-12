# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/3/26 7:39 PM
# @File: __init__.py
# @Email: mlshenkai@163.com
import json
import json

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        buffer = ''
        for line in f:
            buffer += line.strip()
            while buffer:
                try:
                    result, index = json.JSONDecoder().raw_decode(buffer)
                    yield result
                    buffer = buffer[index:]
                except ValueError:
                    # Not enough data to decode, read more
                    break

# 使用迭代器逐个解析JSON对象
for json_obj in read_jsonl('./WAIYUN-COSCO-2210-2303.jsonl'):
    print(json_obj)
