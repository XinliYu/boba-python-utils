import uuid

from pyspark.sql.functions import udf


@udf
def uuid4():
    return str(uuid.uuid4())
