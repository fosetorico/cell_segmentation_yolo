from cellSegmentation.logger import logging
from cellSegmentation.exception import AppException
import sys

logging.info("welcome!!!")

try:
    a = 4/0
except Exception as e:
    raise AppException(e, sys)