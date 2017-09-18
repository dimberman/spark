from tests import *
_have_scipy = False

if __name__ == "__main__":
    from pyspark.mllib.mllib_tests import *
    if not _have_scipy:
        print("NOTE: Skipping SciPy mllib_tests as it does not seem to be installed")
    if xmlrunner:
        unittest.main(testRunner=xmlrunner.XMLTestRunner(output='target/test-reports'))
    else:
        unittest.main()
    if not _have_scipy:
        print("NOTE: SciPy mllib_tests were skipped as it does not seem to be installed")
    sc.stop()