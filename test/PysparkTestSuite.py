import unittest.loader
import sys
import logging
import traceback

from pyspark.sql import SparkSession
from testresources.fixtures import loadFixtures

"""
Globals for all tests
"""

def get_or_create_global_spark():
    session = (SparkSession
               .builder
               .master("local[*]")
               .appName("Unit-tests")
               .enableHiveSupport()
               .getOrCreate())
    return session


def load_global_fixtures(spark):
    return loadFixtures(spark)


def get_logger():
    logger = logging.getLogger("Py4j")
    return logger


globalSpark = get_or_create_global_spark()
globalFixtures = load_global_fixtures(globalSpark)
globalLogger = get_logger()

"""
TestCaseExtension is the parent class used for each testing
subclass
"""


class TestCaseWithSparkSession(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        global globalSpark
        global globalFixtures
        global globalLogger
        self.spark = globalSpark
        self.fixtures = globalFixtures
        self.logger = globalLogger


"""
Here are auxiliary methods to the testing framework
"""


def select_test(test_case):
    test_loader = unittest.TestLoader()
    if test_case == "all":
        return test_loader.discover(".", "*test.py")
    if ".py" in test_case:
        return test_loader.discover(".", test_case)
    if "." not in test_case:
        return test_loader.discover(".", test_case + "*")
    test_case_list = test_case.split('.')
    if len(test_case_list) >= 2:
        try:
            # when test case points to a method
            module_name = ".".join(test_case_list[:-1])
            __import__(module_name)
            test_class = getattr(sys.modules[module_name], test_case_list[-2])
            inst = test_class(test_case_list[-1])
            return unittest.TestSuite([inst])
        except Exception as e:
            logging.error(traceback.format_exc())
    else:
        try:
            return test_loader.discover(".", test_case_list[-1] + "*")
        except Exception as e2:
            logging.error(traceback.format_exc())
            ValueError('Please refer to the docker-compose.yml documentation,' +
                     ' the TEST_CASE argument passed must folow a specific' +
                     ' pattern to run properly, your input was: ', test_case)


"""
Testing Suite main
"""

if __name__ == '__main__':
    # Command line argument
    test_case = sys.argv[1]
    print("== running " + test_case + " ==")
    test_suite = select_test(test_case)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)

    globalSpark.stop()
