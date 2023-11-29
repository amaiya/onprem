import importlib
from sys import modules

importlib.import_module('pysqlite3')
modules['sqlite3'] = modules.pop('pysqlite3')
