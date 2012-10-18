import magnum.magneto as magneto
import logging
import magnum.console as console

# I. Custom log message formater (with colors!)

_color_map = {
  logging.DEBUG: 2,
  logging.INFO: 2,
  logging.WARNING: 12,
  logging.ERROR: 1,
  logging.CRITICAL: 1
}

class MyFormatter(logging.Formatter):
  def format(self, record):
    return console.color(_color_map[record.levelno]) + logging.Formatter.format(self, record) + console.nocolor()

# II. Create logger
ch = logging.StreamHandler()
ch.setFormatter(MyFormatter("[%(levelname)5s] - %(message)s", "%Y-%m-%d %H:%M:%S"))
logger = logging.getLogger("MagNum")
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
del ch

# III. Set debug callback (called from C++ code to communicate with Python logger)
def callback(level, msg):
  if level == 0:
    logger.debug(msg)
  elif level == 1:
    logger.info(msg)
  elif level == 2:
    logger.warning(msg)
  elif level == 3:
    logger.error(msg)
  elif level == 4:
    logger.critical(msg)
magneto.setDebugCallback(callback)
del callback
