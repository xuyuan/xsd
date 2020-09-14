import os
import warnings
import threading
import time
import queue
from functools import partial
from collections.abc import MutableMapping
from pathlib import Path
from datetime import datetime
import numpy as np

import socket
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter

# patch pymysql to handle numpy.float64
try:
    import pymysql
    pymysql.converters.encoders[np.float64] = pymysql.converters.escape_float
    pymysql.converters.conversions = pymysql.converters.encoders.copy()
    pymysql.converters.conversions.update(pymysql.converters.decoders)
except:
    pass


def default_log_dir():
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + '_' + socket.gethostname())
    return os.getenv("TRAIN_LOG_DIR", log_dir)


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class MLFlowClientThread(threading.Thread):
    def __init__(self, mlflow_tracking_uri, run_name):
        super().__init__(daemon=True)
        self.queue = queue.Queue()

        self.mlflow = None
        if mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", None):
            try:
                import mlflow
                self.mlflow = mlflow
            except:
                # error only when mlflow_tracking_uri is specified
                # MLFLOW_TRACKING_URI used only when MLFlow is installed
                if mlflow_tracking_uri:
                    raise RuntimeError("MLFlow not installed!")

        if self.mlflow:
            if mlflow_tracking_uri:
                self.mlflow.set_tracking_uri(mlflow_tracking_uri)
            MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", None)
            if MLFLOW_EXPERIMENT_NAME is not None:
                self.mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            self.mlflow.start_run(run_name=run_name)

            # set tags with gitlab CI variables
            GITLAB_CI = os.getenv("GITLAB_CI", None)
            if GITLAB_CI:
                tag_env = {"mlflow.user": "GITLAB_USER_NAME",
                           "mlflow.source.name": "CI_JOB_URL",
                           "mlflow.source.git.branch": "CI_COMMIT_REF_NAME",
                           "mlflow.source.git.repoURL": "CI_PROJECT_URL",
                           "mlflow.note.content": "CI_COMMIT_MESSAGE",
                           "CI_RUNNER_DESCRIPTION": "CI_RUNNER_DESCRIPTION",
                           "CI_RUNNER_TAGS": "CI_RUNNER_TAGS"}
                tags = {"mlflow.source.type": "JOB"}
                for k, e in tag_env.items():
                    v = os.getenv(e, None)
                    if v:
                        tags[k] = v
                self.mlflow.set_tags(tags)

            self.start()

    def log_metric(self, *args, **kwargs):
        if self.mlflow:
            func = partial(self.mlflow.log_metric, *args, **kwargs)
            self.queue.put(func)

    def log_metrics(self, *args, **kwargs):
        if self.mlflow:
            func = partial(self.mlflow.log_metrics, *args, **kwargs)
            self.queue.put(func)

    def log_params(self, *args, **kwargs):
        if self.mlflow:
            func = partial(self.mlflow.log_params, *args, **kwargs)
            self.queue.put(func)

    def end_run(self):
        if self.mlflow:
            self.queue.join()
            self.mlflow.end_run()
            self.mlflow = None

    def run(self):
        while self.mlflow:
            func = self.queue.get()

            while self.mlflow:
                # keep trying
                try:
                    func()
                    break
                except Exception as e:
                    warnings.warn(str(e))
                    time.sleep(30)

            self.queue.task_done()


class Logger(object):
    def __init__(self, log_dir, mlflow_tracking_uri):
        self.tb_writer = SummaryWriter(str(log_dir))
        print("logging into {}".format(log_dir))

        self.mlflow = MLFlowClientThread(mlflow_tracking_uri, Path(log_dir).name)

    def add_graph(self, model, input_to_model=None, verbose=False, profile_with_cuda=False, **kwargs):
        self.tb_writer.add_graph(model=model, input_to_model=input_to_model, verbose=verbose, profile_with_cuda=profile_with_cuda, **kwargs)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        self.tb_writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step, walltime=walltime)
        if self.mlflow:
            self.mlflow.log_metric(tag, scalar_value, step=global_step)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        self.tb_writer.add_image(tag=tag, img_tensor=img_tensor, global_step=global_step, walltime=walltime, dataformats=dataformats)

    def add_hparams(self, hparam_dict=None, metric_dict=None, name=None, global_step=None):
        if metric_dict:
            metric_dict = flatten_dict(metric_dict)

        try:
            self.tb_writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict, name=name, global_step=global_step)
        except Exception as e:
            warnings.warn(f'failed to write tb hparams: {e}')
        if self.mlflow:
            if hparam_dict:
                self.mlflow.log_params(hparam_dict)
            if metric_dict:
                self.mlflow.log_metrics(metric_dict, step=global_step)

    def close(self):
        self.tb_writer.close()
        if self.mlflow:
            self.mlflow.end_run()

