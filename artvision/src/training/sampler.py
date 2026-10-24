import math
import random
from typing import Dict, Iterator, List
from torch.utils.data import Sampler


class BalancedMultiTaskBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that guarantees a minimum number of labeled samples
    per task in each batch during multi-task training.
    """
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        min_samples_per_task: Dict[str, int] = None,
        drop_last: bool = False,
        seed: int = 42
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = random.Random(seed)
        
        self.min_samples = min_samples_per_task or {}
        
        # Build task-specific pools
        self.task_pools = {}
        for task, min_count in self.min_samples.items():
            if min_count > 0:
                indices = dataset.get_task_indices(task)
                if len(indices) > 0:
                    self.task_pools[task] = (indices, min_count)
                    
        self.all_indices = list(range(len(dataset)))
        self.num_batches = len(dataset) // batch_size if drop_last else math.ceil(len(dataset) / batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        # Clone index pools for this epoch
        shuffled_all = list(self.all_indices)
        self.rng.shuffle(shuffled_all)
        
        task_shuffled = {}
        task_ptrs = {}
        for task, (indices, min_count) in self.task_pools.items():
            pool = list(indices)
            self.rng.shuffle(pool)
            task_shuffled[task] = pool
            task_ptrs[task] = 0

        all_ptr = 0

        for _ in range(self.num_batches):
            batch = set()

            # 1. Guarantee minimums from task pools
            for task, (pool, min_count) in self.task_pools.items():
                shuffled = task_shuffled[task]
                ptr = task_ptrs[task]
                needed = min_count
                
                added = 0
                while added < needed:
                    if ptr >= len(shuffled):
                        self.rng.shuffle(shuffled)
                        ptr = 0
                    idx = shuffled[ptr]
                    ptr += 1
                    if idx not in batch:
                        batch.add(idx)
                        added += 1
                task_ptrs[task] = ptr

            # 2. Fill the remainder of the batch from general pool
            while len(batch) < self.batch_size:
                if all_ptr >= len(shuffled_all):
                    self.rng.shuffle(shuffled_all)
                    all_ptr = 0
                idx = shuffled_all[all_ptr]
                all_ptr += 1
                batch.add(idx)

            batch_list = list(batch)
            self.rng.shuffle(batch_list)
            yield batch_list

    def __len__(self) -> int:
        return self.num_batches
