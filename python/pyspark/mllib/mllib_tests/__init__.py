from tests import *
_have_scipy = False

if __name__ == "__main__":
    from pyspark.mllib.tests import *
    if not _have_scipy:
        print("NOTE: Skipping SciPy tests as it does not seem to be installed")
    if xmlrunner:
        unittest.main(testRunner=xmlrunner.XMLTestRunner(output='target/test-reports'))
    else:
        unittest.main()
    if not _have_scipy:
        print("NOTE: SciPy tests were skipped as it does not seem to be installed")
    sc.stop()