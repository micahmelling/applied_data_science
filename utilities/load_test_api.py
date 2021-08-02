import requests
import time
import multiprocessing as mp
import json
import os
import matplotlib.pyplot as plt

from functools import partial


def load_json_file(path):
    """
    Loads a json file.

    :param path: path to json file
    :returns: json file (dictionary)
    """
    with open(path, 'r') as file:
        return json.loads(file.read())


def get_response_time(_, endpoint, payload):
    """
    Gets the response time of htting an endpoint.

    :param endpoint: HTTP endpoint to hit
    :param payload: payload to send to endpoint
    """
    start_time = time.time()
    requests.post(endpoint, json=payload)
    return time.time() - start_time


def plot_response_times(times_list):
    """
    Plots the load testing results.

    :param times_list: list of response times
    """
    plt.plot(times_list)
    plt.xlabel('requests')
    plt.ylabel('response_time')
    plt.savefig('load_testing_results.png')
    plt.clf()


def main(endpoint, parallel_processes, total_requests):
    """
    Executes load testing of API.

    :param endpoint: endpoint to test
    :param parallel_processes: number of processes to run in parallel
    :param total_requests: total number of requests to send
    """
    payload = load_json_file(os.path.join('data', 'sample_payloads', 'sample_payload1.json'))
    _get_response_time = partial(get_response_time, endpoint=endpoint, payload=payload)
    with mp.Pool(processes=parallel_processes) as pool:
        result = pool.map(_get_response_time, range(1, total_requests))
    plot_response_times(result)


if __name__ == "__main__":
    main(
        endpoint='http://127.0.0.1:5000/predict',
        parallel_processes=4,
        total_requests=50
    )
