import math  # For exp() and other math operations used in Gaussian scoring
import random  # For random sampling of buckets and dataset indices
from typing import Dict, List, Optional  # Type hints for cleaner code
from torch.utils.data import Sampler  # Base PyTorch sampler class


def gaussian_schedule(
    t: int,  # Current curriculum step (microstep index)
    T: int,  # Total number of curriculum steps across the whole run
    num_tasks: int,  # Number of difficulty buckets, e.g. 4 for n=2,3,4,5
    curriculum_beta: float = 0.5,  # Controls how fast the center moves early vs late
    sigma: float = 0.5,  # Controls how wide the Gaussian spread is
    min_prob: float = 0.0,  # Optional floor probability for every task
) -> List[float]:
    """
    Fixed Gaussian curriculum schedule.

    Curriculum center:
        x_t = (t / (T - 1))^curriculum_beta * (K - 1)

    Bucket probabilities:
        p_t(k) ∝ exp(- (k - x_t)^2 / (2 sigma^2))

    We divide by (T - 1), not T, so that if t runs from 0 to T-1,
    the final step can exactly reach the last bucket.
    """

    # Safety check: total curriculum length must be positive
    if T <= 0:
        raise ValueError("T must be > 0")

    # Safety check: we must have at least one task bucket
    if num_tasks <= 0:
        raise ValueError("num_tasks must be > 0")

    # Safety check: sigma must be positive, otherwise Gaussian is invalid
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    # Safety check: minimum probability cannot be negative
    if min_prob < 0:
        raise ValueError("min_prob must be >= 0")

    # Safety check: floor probability across all tasks cannot exceed 1
    if num_tasks * min_prob >= 1.0:
        raise ValueError("num_tasks * min_prob must be < 1")

    # Clamp t into the valid range [0, T-1]
    t = min(max(t, 0), T - 1)

    # Use T-1 in the denominator so the last valid step reaches the final bucket
    denom = max(T - 1, 1)

    # Compute the moving curriculum center x_t
    x_t = (t / denom) ** curriculum_beta * (num_tasks - 1)

    # Compute unnormalized Gaussian scores for each bucket id k = 0,1,...,K-1
    raw_scores = [
        math.exp(-((k - x_t) ** 2) / (2 * sigma ** 2))
        for k in range(num_tasks)
    ]

    # Normalize the scores into probabilities that sum to 1
    total = sum(raw_scores)
    probs = [score / total for score in raw_scores]

    # Optional probability floor so that no bucket becomes exactly impossible
    if min_prob > 0:
        probs = [min_prob + (1 - num_tasks * min_prob) * p for p in probs]
        z = sum(probs)
        probs = [p / z for p in probs]

    # Return the probability vector over internal bucket ids
    return probs


class TaskSampler(Sampler):
    """
    Curriculum-aware sampler for Countdown training.

    High-level behavior:
    1. Group dataset indices by difficulty_n
    2. At each curriculum step, compute probabilities over difficulty buckets
    3. Sample a bucket according to those probabilities
    4. Sample one example index from that bucket
    5. Yield indices to the DataLoader

    Debug mode can:
    - print sampled difficulties every N steps
    - save them to a TSV file
    """

    def __init__(
        self,
        dataset,  # Hugging Face dataset containing difficulty_n per row
        total_iterations: int,  # Total number of curriculum steps across the run
        batch_size: int,  # Per-device batch size used by the DataLoader
        data_schedule: str = "balanced",  # Either "balanced" or "gaussian"
        scheduler_params: Optional[Dict] = None,  # Params like curriculum_beta, sigma, min_prob
        seed: int = 42,  # Seed for reproducible sampling
        debug: bool = False,  # Whether to print/save sampler debug info
        debug_every: int = 5,  # Log every N curriculum steps
        debug_file: Optional[str] = None,  # Path to save debug TSV file
    ):
        # Store dataset reference
        self.dataset = dataset

        # Store total number of curriculum steps
        self.total_iterations = total_iterations

        # Store per-device batch size
        self.batch_size = batch_size

        # Store which schedule to use
        self.data_schedule = data_schedule

        # Store scheduler parameters
        self.scheduler_params = scheduler_params or {}

        # Save seed for reproducibility
        self.seed = seed

        # Create a dedicated random generator
        self.rng = random.Random(seed)

        # Debug controls
        self.debug = debug
        self.debug_every = debug_every
        self.debug_file = debug_file

        # Dictionary mapping actual difficulty -> list of dataset indices
        # Example: {2: [0, 5, 9], 3: [1, 7], 4: [2, 3], 5: [4, 6, 8]}
        self.bucket_to_indices: Dict[int, List[int]] = {}

        # Scan dataset and group indices by difficulty_n
        for idx in range(len(dataset)):
            row = dataset[idx]
            difficulty = int(row["difficulty_n"])

            if difficulty not in self.bucket_to_indices:
                self.bucket_to_indices[difficulty] = []

            self.bucket_to_indices[difficulty].append(idx)

        # Ensure we found at least one difficulty bucket
        if len(self.bucket_to_indices) == 0:
            raise ValueError("No difficulty buckets found in dataset.")

        # Sort real difficulty values, e.g. [2, 3, 4, 5]
        self.sorted_difficulties = sorted(self.bucket_to_indices.keys())

        # Map internal bucket ids 0,1,2,... to real difficulty values 2,3,4,5
        self.bucket_id_to_difficulty = {
            i: difficulty for i, difficulty in enumerate(self.sorted_difficulties)
        }

        # Number of difficulty buckets
        self.num_tasks = len(self.sorted_difficulties)

        # If debug logging to file is enabled, create/overwrite the file header once
        if self.debug and self.debug_file is not None:
            with open(self.debug_file, "w") as f:
                f.write("step\tprobs\tsampled_difficulties\n")

    def _get_probs_for_step(self, step: int) -> List[float]:
        """
        Compute the probability of sampling each internal bucket id
        for the current curriculum step.

        Example return:
            [0.80, 0.18, 0.02, 0.00]
        """
        # Balanced schedule means uniform sampling across all buckets
        if self.data_schedule == "balanced":
            return [1.0 / self.num_tasks] * self.num_tasks

        # Gaussian schedule means easy-to-hard curriculum sampling
        if self.data_schedule == "gaussian":
            return gaussian_schedule(
                t=step,
                T=self.total_iterations,
                num_tasks=self.num_tasks,
                curriculum_beta=self.scheduler_params.get("curriculum_beta", 0.5),
                sigma=self.scheduler_params.get("sigma", 0.5),
                min_prob=self.scheduler_params.get("min_prob", 0.0),
            )

        # Fail loudly if the schedule name is unknown
        raise ValueError(f"Unknown data_schedule: {self.data_schedule}")

    def __iter__(self):
        """
        Yield dataset indices one by one.

        For each curriculum step:
        - compute bucket probabilities
        - sample batch_size many difficulties/examples
        - optionally log the sampled difficulties
        - yield the indices
        """
        # Loop over all curriculum steps
        for step in range(self.total_iterations):
            # Get the probability of each difficulty bucket at this step
            probs = self._get_probs_for_step(step)

            # Temporary storage for this step's sampled indices
            sampled_indices = []

            # Temporary storage for this step's sampled difficulties
            sampled_difficulties = []

            # Sample one batch worth of indices for this curriculum step
            for _ in range(self.batch_size):
                # Sample internal bucket id using the Gaussian curriculum probabilities
                bucket_id = self.rng.choices(
                    population=list(range(self.num_tasks)),
                    weights=probs,
                    k=1,
                )[0]

                # Convert internal bucket id to the real dataset difficulty
                difficulty = self.bucket_id_to_difficulty[bucket_id]

                # Randomly choose one dataset example from that difficulty bucket
                chosen_idx = self.rng.choice(self.bucket_to_indices[difficulty])

                # Save index and difficulty for later logging/yielding
                sampled_indices.append(chosen_idx)
                sampled_difficulties.append(difficulty)

            # Debug print/save every N steps
            if self.debug and (step % self.debug_every == 0):
                # Round probabilities to make logs easier to read
                rounded_probs = [round(p, 4) for p in probs]

                # Create a readable debug message
                msg = (
                    f"[TaskSampler Debug] step={step} "
                    f"probs={rounded_probs} "
                    f"sampled_difficulties={sampled_difficulties}"
                )

                # Print to console
                print(msg)

                # Append to TSV file if requested
                if self.debug_file is not None:
                    with open(self.debug_file, "a") as f:
                        f.write(f"{step}\t{rounded_probs}\t{sampled_difficulties}\n")

            # Yield the sampled indices one by one to the DataLoader
            for idx in sampled_indices:
                yield idx

    def __len__(self):
        """
        Return the total number of dataset indices yielded by this sampler.

        Since each curriculum step emits 'batch_size' indices:
            total length = total_iterations * batch_size
        """
        return self.total_iterations * self.batch_size