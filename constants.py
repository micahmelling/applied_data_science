import pathlib
import os

path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'data', 'rds-ca-2019-root.pem')

print(path)
