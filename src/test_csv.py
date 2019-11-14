if __name__ == '__main__':


    import pandas as pd
    import numpy as np
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.sql.functions import udf

    spark = SparkSession\
        .builder\
        .appName("OneHotEncoderExample")\
        .getOrCreate()

    a = pd.DataFrame(np.random.randint(0,2,(1000,11)),columns=list('abcdefghijk'))
    b = spark.createDataFrame(a)
    assembler = VectorAssembler(
                    inputCols=list('abcdefghij'),
                    outputCol='feature')
    c = assembler.transform(b)
    c = c.drop(*list('abcdefghij'))

    rf = RandomForestClassifier(featuresCol='feature', labelCol='k', predictionCol='Prediction')
    train, test = c.randomSplit([.7, .3])
    model = rf.fit(train)
    predictions = model.transform(test)
    split1_udf = udf(lambda value: value[0].item(), FloatType())
    split2_udf = udf(lambda value: value[1].item(), FloatType())
    predictions2 = predictions.select('Prediction', split1_udf('probability').alias('prob0'), split2_udf('probability').alias('prob1'))
    predictions2.coalesce(1).write.csv('bob/test', mode='overwrite')