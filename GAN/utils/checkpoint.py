# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import logging

import torch

from GAN.utils.model_serialization import load_state_dict


class Checkpointer(object):
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            save_dir="",
            save_to_disk=None,
            logger=None,
    ):
        """
        initial function

        :param model: nn.Module object
        :param optimizer: torch.optim
        :param scheduler: torch.optim.lr_scheduler
        :param save_dir: save directory
        :param save_to_disk: save the file only if rank==0
        :param logger: logger
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        """
        Save the entire training-related state.

        :param name: file name (does not contain "pth")
        :param kwargs: other state variables (epoch, etc)
        :return: None
        """
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, use_latest=True):
        """
        load the file specified by f.
        data related to the model, optimizer, ect. will be directly stored in the
        self variables after loading, and other data will be returned in the form
        of a dictionary.

        :param f: a str which denotes file name and path
        :param use_latest: whether to load the last checkpoint
        :return: a dict which denotes the other data
        """
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        return checkpoint

    def has_checkpoint(self):
        """
        check if the last checkpoint exists

        :return: True of False
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        """
        get the last checkpoint's name and path from the file ”last_checkpoint“

        :return: a str which denotes the last checkpoint's name and path
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        """
        save the last checkpoint's name and path to the file "last_checkpoint"

        :param last_filename: a str which denotes the last checkpoint's name and path
        :return: None
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        """
        use the torch.load method to load the file specified by f

        :param f: a str which denotes file name and path
        :return: a dict
        """
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


class GAN_Checkpointer(object):
    def __init__(
            self,
            model1,
            model2,
            optimizer1=None,
            optimizer2=None,
            scheduler=None,
            save_dir="",
            save_to_disk=None,
            logger=None,
    ):
        self.model1 = model1
        self.model2 = model2
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model1"] = self.model1.state_dict()
        data["model2"] = self.model2.state_dict()
        if self.optimizer1 is not None:
            data["optimizer1"] = self.optimizer1.state_dict()
        if self.optimizer2 is not None:
            data["optimizer2"] = self.optimizer2.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, use_latest=True):

        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer1" in checkpoint and self.optimizer1:
            self.logger.info("Loading optimizer1 from {}".format(f))
            self.optimizer1.load_state_dict(checkpoint.pop("optimizer1"))
        if "optimizer2" in checkpoint and self.optimizer2:
            self.logger.info("Loading optimizer2 from {}".format(f))
            self.optimizer2.load_state_dict(checkpoint.pop("optimizer2"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model1, checkpoint.pop("model1"))
        load_state_dict(self.model2, checkpoint.pop("model2"))
