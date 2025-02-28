import os
from typing import Callable, Iterable, List, Optional, Tuple

from trlx.data.configs import TRLConfig
from trlx.utils import set_seed
from trlx.utils.loading import get_model, get_orchestrator, get_pipeline


def train(
    model_path: Optional[str] = None,
    reward_fn: Optional[Callable] = None,
    dataset: Optional[Iterable[Tuple[str, float]]] = None,
    prompts: Optional[List[str]] = None,
    eval_prompts: Optional[List[str]] = None,
    metric_fn: Optional[Callable] = None,
    config: Optional[TRLConfig] = None,
    split_token: Optional[str] = None,
    logit_mask: Optional[List[List[bool]]] = None,
):
    """
    Dispatches online or offline reinforcement training depending on whether a reward function or a list of samples & rewards is given

    Args:
        model_path (Optional[str]): Path to either huggingface checkpoint or a local directory
        reward_fn (List[str] -> List[float]): Function to rate batches of generated samples
        dataset (List[str], List[float]): Lists of samples and rewards
        prompts (List[str]): Prompts to sample off from during online training
        eval_prompts (List[str]): Prompts to periodically validate training on
        metric_fn (Optional[Callable[List[str], List[float]]]): Function to compute statistics on validation samples
        config (Optional[TRLConfig]): TRL configuration object to override default settings
        split_token (Optional[str]): Split samples in the dataset on prompts and continuations
        logit_mask (Optional[List]): Bigram masking matrix
    """
    if reward_fn is not None:
        if config is None:
            config = TRLConfig.load_yaml("configs/ppo_config.yml")
        set_seed(config.train.seed)

        if model_path:
            config.model.model_path = model_path

        model = get_model(config.model.model_type)(config)

        batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
        
        prompts = []
        response_gt = []
        with open("/home/fuxi-common/jiangdanyang/trlx-t5/trlx-t5/samples.tsv") as f:
            for line in f:
                line = line.strip()
                one_p = line.split("\t")[0]
                one_r = line.split("\t")[1]
                prompts.append(one_p)
                response_gt.append(one_r)

        if eval_prompts is None:
            eval_prompts = prompts[:batch_size]

        pipeline = get_pipeline(config.train.pipeline)(prompts, response_gt, model.tokenizer)
        orch = get_orchestrator(config.train.orchestrator)(
            model, pipeline, reward_fn=reward_fn, chunk_size=config.method.chunk_size
        )
        orch.make_experience(config.method.num_rollouts)
        eval_pipeline = get_pipeline(config.train.pipeline)(
            eval_prompts, response_gt, model.tokenizer
        )
        model.add_eval_pipeline(eval_pipeline)

    elif dataset is not None:
        samples, rewards = dataset

        if len(samples) != len(rewards):
            raise ValueError(
                f"Number of samples {len(samples)} should match the number of rewards {len(rewards)}"
            )

        if config is None:
            config = TRLConfig.load_yaml("configs/ilql_config.yml")
        set_seed(config.train.seed)

        if model_path:
            config.model.model_path = model_path

        model = get_model(config.model.model_type)(
            config=config,
            logit_mask=logit_mask,
            metric_fn=metric_fn,
        )

        batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
        if eval_prompts is None:
            eval_prompts = [model.tokenizer.bos_token] * batch_size
        eval_pipeline = get_pipeline(config.train.pipeline)(
            eval_prompts, model.tokenizer
        )

        orch = get_orchestrator(config.train.orchestrator)(
            model, split_token=split_token
        )
        orch.make_experience(samples, rewards)
        model.add_eval_pipeline(eval_pipeline)

    else:
        raise ValueError(f"Either dataset or reward_fn should be given")

    model.learn()
    return model
