import os
from time import sleep
from typing import List


def scale_workers(worker_type: str, num_workers: int):
    os.system(f"PYTHONPATH='' LD_LIBRARY_PATH='' /apollo/bin/env -e HoverboardScaleCLI scale {worker_type}:{num_workers}")


def scale_down_workers(worker_type: str, wait_before_scale_down=300):
    if wait_before_scale_down:
        sleep(wait_before_scale_down)
    scale_workers(worker_type=worker_type, num_workers=0)


def get_worker_ips():
    return list(
        (
            x[1:-1]
            for x in
            os.popen(
                "/usr/bin/curl -s https://host-discovery.hoverboard | jq -c '.[] | .privateIpAddress'"
            ).read().split()
            if x != 'null'
        )
    )