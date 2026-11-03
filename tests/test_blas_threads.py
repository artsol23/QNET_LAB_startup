"""Проверки ограничения числа потоков BLAS."""

import os
import unittest

from blas_threads import (
    BLAS_THREADS_VAR,
    MAX_BLAS_THREADS,
    blas_threads,
    limit_blas_threads,
)


class BlasThreadsTests(unittest.TestCase):
    def setUp(self):
        saved = os.environ.get(BLAS_THREADS_VAR)

        def restore():
            if saved is None:
                os.environ.pop(BLAS_THREADS_VAR, None)
            else:
                os.environ[BLAS_THREADS_VAR] = saved

        self.addCleanup(restore)

    def test_threads_within_bounds(self):
        self.assertGreaterEqual(blas_threads(), 1)
        self.assertLessEqual(blas_threads(), MAX_BLAS_THREADS)
        self.assertLessEqual(blas_threads(), os.cpu_count() or 1)

    def test_limit_sets_variable(self):
        os.environ.pop(BLAS_THREADS_VAR, None)

        limit_blas_threads()

        self.assertEqual(os.environ[BLAS_THREADS_VAR], str(blas_threads()))

    def test_limit_keeps_user_value(self):
        os.environ[BLAS_THREADS_VAR] = "3"

        limit_blas_threads()

        self.assertEqual(os.environ[BLAS_THREADS_VAR], "3")


if __name__ == "__main__":
    unittest.main()
