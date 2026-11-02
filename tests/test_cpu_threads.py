"""Проверки выбора числа потоков CPU под режим сети."""

import unittest

import torch

from device_runtime import MAX_CPU_THREADS, apply_cpu_threads, cpu_threads


class CpuThreadsTests(unittest.TestCase):
    def test_matched_uses_single_thread(self):
        self.assertEqual(cpu_threads("matched"), 1)

    def test_raw_stays_within_limit(self):
        self.assertGreaterEqual(cpu_threads("raw"), 1)
        self.assertLessEqual(cpu_threads("raw"), MAX_CPU_THREADS)

    def test_apply_changes_pytorch_setting(self):
        saved = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, saved)

        self.assertEqual(apply_cpu_threads("matched"), 1)
        self.assertEqual(torch.get_num_threads(), 1)


if __name__ == "__main__":
    unittest.main()
