import random
import os
import pickle
import time
from collections import deque
import numpy as np

from RoboticsSuite.utils import postprocess_model_xml
from RoboticsSuite.utils.mjcf_utils import xml_path_completion


class DemoSampler:
    def __init__(
        self, file_path, demo_config, need_xml=False, preload=False, number=-1
    ):
        self.demo_config = demo_config

        """ Load the demo file...
            If it's a .pkl file with a list of objects, then proceed as usual.
            If it's a .pkl file with a list of numbers, open the corresponding file with all of them;
                if we want to preload the samples, then read the big file onto a list, and
                    then pretend that we just read a normal .pkl file
                if we want to lazy load from the filesystem, simply keep self.demo_big_file pointing
                    to the bigfile and self.demo_data as a list of numbers
        """
        with open(xml_path_completion(file_path), "rb") as f:
            self.demo_data = pickle.load(f)
            is_bkl = self.demo_data[0] == 0
            if number > 0:
                random.seed(3141)
                self.demo_data = random.sample(self.demo_data, number)
                random.seed(time.time() * 1000 * ord(os.urandom(1)))
                print("Subsampled:", self.demo_data)
            if is_bkl:
                self.demo_big_file = open(
                    xml_path_completion(file_path).replace(".pkl", ".bkl"), "rb"
                )
                if preload:
                    print("Preloading...")
                    data = []
                    for ofs in self.demo_data:
                        self.demo_big_file.seek(ofs)
                        data.append(pickle.load(self.demo_big_file))
                    self.demo_data = data
                    self.demo_big_file.close()
                    self.demo_big_file = None
                    print("Preloaded!")
            else:
                print("Not a bkl!")
                self.demo_big_file = None

        self.need_xml = need_xml
        self.demo_sampled = 0

        self.sample_method_dict = {
            "random": "_random_sample",
            "uniform": "_uniform_sample",
            "forward": "_forward_sample_adaptive"
            if self.demo_config.adaptive
            else "_forward_sample_openloop",
            "reverse": "_reverse_sample_adaptive"
            if self.demo_config.adaptive
            else "_reverse_sample_openloop",
        }

        self.mixing = demo_config.mixing
        self.mixing_ratio = np.asarray(demo_config.mixing_ratio)
        self.ratio_step = np.asarray(demo_config.ratio_step)

        assert len(self.mixing) == len(self.mixing_ratio) and len(
            self.mixing_ratio
        ) == len(self.ratio_step)
        assert sum(self.mixing_ratio) == 1.0 and sum(self.ratio_step) == 0.0

        if demo_config.adaptive:
            self.curriculum_length = demo_config.curriculum_length
            self.history_length = demo_config.history_length
            self.prev_curriculum_score = None
            self.curr_curriculum_scores = deque(maxlen=demo_config.history_length)
            self.current_curriculum = 0

    def log_score(self, score):
        if self.demo_config.adaptive:
            self.curr_curriculum_scores.append(score)
            # checking whether to update curriculum
            if (
                len(self.curr_curriculum_scores) == self.history_length
                and self._performance_improved()
            ):

                self.current_curriculum += 1
                self.prev_curriculum_score = sum(self.curr_curriculum_scores) / len(
                    self.curr_curriculum_scores
                )
                self.curr_curriculum_scores.clear()

    def sample(self):
        seed = random.uniform(0, 1)
        ratio = np.cumsum(self.mixing_ratio)
        ratio = ratio > seed
        for i, v in enumerate(ratio):
            if v:
                break

        sample_method = getattr(self, self.sample_method_dict[self.mixing[i]])
        return sample_method()

    def _performance_improved(self):
        new_avg = sum(self.curr_curriculum_scores) / len(self.curr_curriculum_scores)

        if self.prev_curriculum_score is None:
            self.prev_curriculum_score = new_avg
            return False

        improve = new_avg / max(self.prev_curriculum_score, 1e-4) - 1
        return improve >= self.demo_config.improve_threshold

    def _random_sample(self):
        return None

    def _uniform_sample(self):
        episode = random.choice(self.demo_data)
        if self.demo_big_file is not None:
            self.demo_big_file.seek(episode)
            episode = pickle.load(self.demo_big_file)
        state = random.choice(episode["states"])

        if self.need_xml:
            xml = postprocess_model_xml(episode["model.xml"])
            return state, xml
        return state

    def _reverse_sample_openloop(self):
        episode = random.choice(self.demo_data)
        if self.demo_big_file is not None:
            self.demo_big_file.seek(episode)
            episode = pickle.load(self.demo_big_file)
        eps_len = len(episode["states"])
        index = np.random.randint(
            max(eps_len - self.demo_config.sample_window_width, 0), eps_len
        )
        state = episode["states"][index]

        self.demo_sampled += 1
        if self.demo_sampled >= self.demo_config.increment_frequency:
            if self.demo_config.sample_window_width < eps_len:
                self.demo_config.sample_window_width += self.demo_config.increment
            self.demo_sampled = 0

        if self.need_xml:
            xml = postprocess_model_xml(episode["model.xml"])
            return state, xml

        return state

    def _forward_sample_openloop(self):
        episode = random.choice(self.demo_data)
        if self.demo_big_file is not None:
            self.demo_big_file.seek(episode)
            episode = pickle.load(self.demo_big_file)
        eps_len = len(episode["states"])
        index = np.random.randint(0, min(self.demo_config.sample_window_width, eps_len))
        state = episode["states"][index]

        self.demo_sampled += 1
        if self.demo_sampled >= self.demo_config.increment_frequency:
            if self.demo_config.sample_window_width < eps_len:
                self.demo_config.sample_window_width += self.demo_config.increment
            self.demo_sampled = 0

        if self.need_xml:
            xml = postprocess_model_xml(episode["model.xml"])
            return state, xml

        return state

    def _reverse_sample_adaptive(self):
        episode = random.choice(self.demo_data)
        if self.demo_big_file is not None:
            self.demo_big_file.seek(episode)
            episode = pickle.load(self.demo_big_file)
        eps_len = len(episode["states"])

        index_start = max(
            eps_len - (self.current_curriculum + 1) * self.curriculum_length, 0
        )
        index = np.random.randint(index_start, index_start + self.curriculum_length)
        state = episode["states"][index]

        if self.need_xml:
            xml = postprocess_model_xml(episode["model.xml"])
            return state, xml

        return state

    def _forward_sample_adaptive(self):
        episode = random.choice(self.demo_data)
        if self.demo_big_file is not None:
            self.demo_big_file.seek(episode)
            episode = pickle.load(self.demo_big_file)
        eps_len = len(episode["states"])

        index_start = min(
            self.current_curriculum * self.curriculum_length,
            eps_len - self.curriculum_length,
        )
        index = np.random.randint(index_start, index_start + self.curriculum_length)
        state = episode["states"][index]

        if self.need_xml:
            xml = postprocess_model_xml(episode["model.xml"])
            return state, xml

        return state
