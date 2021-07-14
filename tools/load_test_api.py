import requests
import time
import multiprocessing as mp
import json
import os
import matplotlib.pyplot as plt

from functools import partial


def load_json_file(path):
    with open(path, 'r') as file:
        return json.loads(file.read())


def get_response_time(_, endpoint, payload):
    start_time = time.time()
    requests.post(endpoint, json=payload)
    return time.time() - start_time


def plot_response_times(times_list):
    plt.plot(times_list)
    plt.xlabel('requests')
    plt.ylabel('response_time')
    plt.savefig('load_testing_results.png')


def main(endpoint, parallel_processes, total_requests):
    payload = load_json_file(os.path.join('data', 'sample_payloads', 'sample_payload1.json'))
    _get_response_time = partial(get_response_time, endpoint=endpoint, payload=payload)
    pool = mp.Pool(processes=parallel_processes)
    result = pool.map(_get_response_time, range(1, total_requests))
    plot_response_times(result)


if __name__ == "__main__":
    main('http://127.0.0.1:5000/predict', 4, 50)
