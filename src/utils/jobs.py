from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar, cast

from tqdm import tqdm

TJob = TypeVar("TJob")
TResult = TypeVar("TResult")


def run_jobs(
    jobs: list[TJob],
    worker: Callable[[TJob], TResult],
    *,
    workers: int,
    desc: str,
) -> list[TResult]:
    if not jobs:
        return []

    results: list[TResult | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_index = {
            pool.submit(worker, job): index for index, job in enumerate(jobs)
        }
        with tqdm(total=len(jobs), desc=desc) as progress:
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()
                progress.update(1)
    return cast(list[TResult], results)
