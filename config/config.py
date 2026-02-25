"""
Configuration file.
"""

def customConfig(filename="detector.log", save_output=True):
	"""
	Custom config for logger.
	"""
	logger.setLevel(logging.INFO)
	if save_output:
		handler = logging.FileHandler(filename=filename, mode="w")
	else:
		handler = logging.StreamHandler()

	formatter = jsonlogger.JsonFormatter(
		"%(asctime)s %(name)s %(levelname)s %(message)s"
		)

	handler.setFormatter(formatter)
	logger.addHandler(handler)