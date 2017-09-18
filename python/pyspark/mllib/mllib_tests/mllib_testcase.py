import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from time import time, sleep



if sys.version_info[:2] <= (2, 6):
    try:
        import unittest2 as unittest
    except ImportError:
        sys.stderr.write('Please install unittest2 to test with Python 2.6 or earlier')
        sys.exit(1)
else:
    import unittest


class MLlibTestCase(unittest.TestCase):
    def setUp(self):
        self.sc = SparkContext('local[4]', "MLlib mllib_tests")
        self.spark = SparkSession(self.sc)

    def tearDown(self):
        self.spark.stop()


class MLLibStreamingTestCase(unittest.TestCase):
    def setUp(self):
        self.sc = SparkContext('local[4]', "MLlib mllib_tests")
        self.ssc = StreamingContext(self.sc, 1.0)

    def tearDown(self):
        self.ssc.stop(False)
        self.sc.stop()

    @staticmethod
    def _eventually(condition, timeout=30.0, catch_assertions=False):
        """
        Wait a given amount of time for a condition to pass, else fail with an error.
        This is a helper utility for streaming ML mllib_tests.
        :param condition: Function that checks for termination conditions.
                          condition() can return:
                           - True: Conditions met. Return without error.
                           - other value: Conditions not met yet. Continue. Upon timeout,
                                          include last such value in error message.
                          Note that this method may be called at any time during
                          streaming execution (e.g., even before any results
                          have been created).
        :param timeout: Number of seconds to wait.  Default 30 seconds.
        :param catch_assertions: If False (default), do not catch AssertionErrors.
                                 If True, catch AssertionErrors; continue, but save
                                 error to throw upon timeout.
        """
        start_time = time()
        lastValue = None
        while time() - start_time < timeout:
            if catch_assertions:
                try:
                    lastValue = condition()
                except AssertionError as e:
                    lastValue = e
            else:
                lastValue = condition()
            if lastValue is True:
                return
            sleep(0.01)
        if isinstance(lastValue, AssertionError):
            raise lastValue
        else:
            raise AssertionError(
                "Test failed due to timeout after %g sec, with last condition returning: %s"
                % (timeout, lastValue))