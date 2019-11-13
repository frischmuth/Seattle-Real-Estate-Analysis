import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, Normalizer, StandardScaler

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder



def run_model(numFolds = 10, 
            gis_filepath='data/Parcels_for_King_County_with_Address_with_Property_Information__parcel_address_area.csv',
            parcels_filepath='data/EXTR_Parcel.csv',
            permit_filepath='data/Building_permits.csv'
            ):
    

    
    data = create_full_dataframe(gis_filepath, parcels_filepath, permit_filepath, numFolds)
    
    rf = RandomForestClassifier(featuresCol='all_features', labelCol='TARGET', predictionCol='Prediction')
    
    # pipeline = Pipeline(stages=[lr])
    # paramGrid = (ParamGridBuilder().build())

    # crossval = CrossValidator(
    #             estimator=lr,
    #             # estimatorParamMaps=paramGrid,
    #             evaluator=evaluator,
    #             numFolds=numFolds)
    
    # cvModel = crossval.fit(data)
    # prediction = cvModel.transform(data)

    for i in range(0,numFolds):
        train = data.filter(data.fold != i)
        test = data.filter(data.fold == i)
        model = rf.fit(train)
        if i == 0:
            predictions = model.transform(test)
        else:
            predictions = predictions.union(model.transform(test))

    evaluator = BinaryClassificationEvaluator(rawPredictionCol='Prediction', labelCol='TARGET')
    accuracy = evaluator.evaluate(predictions)
    # print("Test Error = %g" % (1.0 - accuracy))
    return accuracy, predictions

def run_first_model():
    
    data = create_full_dataframe()
   
    
    lr = LogisticRegression(featuresCol='all_features', labelCol='TARGET', predictionCol='Prediction')
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='Prediction', labelCol='TARGET')
    
    # pipeline = Pipeline(stages=[lr])
    # paramGrid = (ParamGridBuilder().build())

    # crossval = CrossValidator(
    #             estimator=lr,
    #             # estimatorParamMaps=paramGrid,
    #             evaluator=evaluator,
    #             numFolds=numFolds)
    
    # cvModel = crossval.fit(data)
    # prediction = cvModel.transform(data)

    train, test = data.randomSplit([.7, .3])
    # # rf = RandomForestClassifier(featuresCol='all_features', labelCol='TARGET', predictionCol='Prediction')
    # # model = rf.fit(train)
    # # predictions = model.transform(test)

    
    model = lr.fit(train)
    predictions = model.transform(test)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol='Prediction', labelCol='TARGET')
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))
    return predictions

def create_full_dataframe(gis_filepath, parcels_filepath, permit_filepath, numFolds):

    # Get the X vector
    gis = gis_data_to_spark(numFolds, gis_filepath)
    parcel = get_parcels_to_spark(parcels_filepath)

    # Join into one dataframe
    all_data = gis.alias('g').join(parcel.alias('p'), gis.PIN==parcel.PIN).select('g.PIN', 'g.fold', 'g.MAJOR', 'g.MINOR', 'g.ADDR_FULL', 'g.gis_features', 'p.parcel_features', 'g.TARGET')

    input_columns = ['gis_features', 'parcel_features']
    assembler = VectorAssembler(
    inputCols= input_columns,
    outputCol='all_features')
    all_data = assembler.transform(all_data)
    all_data = all_data.drop(*input_columns)

    # USE SMALL DATASET FOR PRACTICE. REMOVE!!!!!
    small, large = all_data.randomSplit([.01,.99])

    return all_data

def gis_data_to_spark(numFolds, gis_filepath='data/Parcels_for_King_County_with_Address_with_Property_Information__parcel_address_area.csv'):
    # Comment out to only use initial SparkSession
    # spark = SparkSession\
    # .builder\
    # .master('Local[4]')\
    # .appName("Get_GIS_Data")\
    # .config("spark.master", "local")\
    # .getOrCreate()
    
    # Initially read in pre-cleaned Pandas DataFrame into Spark DataFrame
    gis_pd = get_gis_data(gis_filepath)
    
    gis_pd['fold'] = np.random.randint(0,numFolds,gis_pd.shape[0])
    gis = spark.createDataFrame(gis_pd)
    
    # Normalize numerical data
    numerical_cols = ['LAT','LON','LOTSQFT','APPRLNDVAL','APPR_IMPR','TAX_LNDVAL','TAX_IMPR','Shape_Length','Shape_Area']
    numerical_assembler = VectorAssembler(
    inputCols=numerical_cols,
    outputCol='num_features')
    gis = numerical_assembler.transform(gis)

    gis = StandardScaler(inputCol='num_features', outputCol='num_features_std').fit(gis).transform(gis)

    # Create index and dummy_vector column names of categorical colums, eventually dropping categorical and index columns
    cat_cols = ['KCTP_STATE', 'SITETYPE', 'LEVYCODE', 'NEW_CONSTR', 'TAXVAL_RSN', 'QTS', 'SEC', 'TWP', 'RNG', 'KCA_ZONING', 'PROPTYPE', 'PREUSE_DESC']
    cat_index = []
    dummies = []
    for col in cat_cols:
        cat_index.append(col+'_index')
        dummies.append(col+'_dummy_vector')
    
    # Create and populate categorical index columns
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(gis) for column in cat_cols]
    cat_pipeline = Pipeline(stages=indexers)
    gis = cat_pipeline.fit(gis).transform(gis)

    # Encode dummy_vector columns from categorical indeces
    encoder = OneHotEncoderEstimator(inputCols=cat_index,outputCols=dummies)
    model = encoder.fit(gis)
    gis = model.transform(gis)   
    
    # Drop categorical and index columns
    gis = gis.drop(*cat_cols)
    gis = gis.drop(*cat_index)
    gis = gis.drop(*numerical_cols)
    
    # Combine all features into single vector 
    ignore = ['PIN', 'MAJOR', 'MINOR', 'ADDR_FULL', 'TARGET', 'fold']
    assembler = VectorAssembler(
    inputCols=[col for col in gis.columns if col not in ignore],
    outputCol='gis_features')
    gis = assembler.transform(gis)

    # Drop all columns that are now in the features column
    ignore.append('gis_features')
    gis = gis.drop(*[col for col in gis.columns if col not in ignore])
    
    # Write to parquet - not sure if I will eventually open from this, but that's the idea
    # gis.write.parquet('data/gis_parquet',mode='overwrite')
    
    
    return gis

def get_parcels_to_spark(parcels_filepath='data/EXTR_Parcel.csv'):
    # Comment out to only use initial SparkSession
    # spark = SparkSession\
    # .builder\
    # .master('Local[4]')\
    # .appName("Get_Parcel_Data")\
    # .config("spark.master", "local")\
    # .getOrCreate()    

    # Initially read in pre-cleaned Pandas DataFrame into Spark DataFrame
    parcel_pd = get_parcels(parcels_filepath)
    parcel = spark.createDataFrame(parcel_pd)

    # Normalize numerical data
    numerical_cols = ['PcntUnusable', 'WfntFootage', ]
    numerical_assembler = VectorAssembler(
    inputCols=numerical_cols,
    outputCol='num_features')
    parcel = numerical_assembler.transform(parcel)

    parcel = StandardScaler(inputCol='num_features', outputCol='num_features_std').fit(parcel).transform(parcel)

    # Create index and dummy_vector column names of categorical colums, eventually dropping categorical and index columns
    cat_cols = ['Range', 'Township', 'Section', 'QuarterSection', 'Area',
       'SubArea', 'LevyCode',
       'CurrentZoning', 'PresentUse',
       'SqFtLot', 'WaterSystem', 'SewerSystem', 'Access', 'Topography',
       'StreetSurface', 'InadequateParking', 'MtRainier', 'Olympics', 'Cascades',
       'Territorial', 'SeattleSkyline', 'PugetSound', 'LakeWashington', 'SmallLakeRiverCreek', 'OtherView', 'WfntLocation',
        'WfntBank', 'WfntPoorQuality', 'WfntRestrictedAccess',
       'WfntAccessRights',  'TidelandShoreland',
       'LotDepthFactor', 'TrafficNoise', 'NbrBldgSites', 'Contamination', ]

    cat_index = []
    dummies = []
    for col in cat_cols:
        cat_index.append(col+'_index')
        dummies.append(col+'_dummy_vector')
    
    # Create and populate categorical index columns
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(parcel) for column in cat_cols]
    cat_pipeline = Pipeline(stages=indexers)
    parcel = cat_pipeline.fit(parcel).transform(parcel)

    # Encode dummy_vector columns from categorical indeces
    encoder = OneHotEncoderEstimator(inputCols=cat_index,outputCols=dummies)
    model = encoder.fit(parcel)
    parcel = model.transform(parcel)   
    
    # Drop categorical and index columns
    parcel = parcel.drop(*cat_cols)
    parcel = parcel.drop(*cat_index)
    parcel = parcel.drop(*numerical_cols)
    
    # Combine all features into single vector 
    ignore = ['PIN']
    assembler = VectorAssembler(
    inputCols=[col for col in parcel.columns if col not in ignore],
    outputCol='parcel_features')
    parcel = assembler.transform(parcel)

    # Drop all columns that are now in the features column
    ignore.append('parcel_features')
    parcel = parcel.drop(*[col for col in parcel.columns if col not in ignore])
    
    # # Write to parquet - not sure if I will eventually open from this, but that's the idea
    # # gis.write.parquet('data/gis_parquet',mode='overwrite')
    
    
    return parcel


def get_pending_demo_permits(permit_filepath='data/Building_permits.csv'):
    building_permits = pd.read_csv(permit_filepath ,parse_dates=[10,11,12,13], infer_datetime_format=True)
    res_bldg_prmts = building_permits[building_permits['PermitClassMapped']=='Residential'].copy()
    demo = res_bldg_prmts[((res_bldg_prmts['PermitTypeDesc']=='Demolition') | (res_bldg_prmts['PermitTypeMapped']=='Demolition'))].copy()
    pending_demo = demo[(demo['StatusCurrent']!='Closed')
                        &(demo['StatusCurrent']!='Completed')
                        &(demo['OriginalAddress1'].notnull())]
    pending_demo = pending_demo[[ 'OriginalAddress1', 'OriginalCity', 'OriginalState', 'OriginalZip', 'Latitude', 'Longitude']].copy()
    return pending_demo

def get_gis_data(gis_filepath='data/Parcels_for_King_County_with_Address_with_Property_Information__parcel_address_area.csv'):
    gis = pd.read_csv(gis_filepath, low_memory=False)
    
    # Limit to Seattle Residential
    gis = gis[gis['CTYNAME']=='SEATTLE']
    gis = gis[gis['QTS']!='  ']
    gis = gis[gis['QTS'].notnull()]
    gis['PREUSE_DESC'].fillna('NaN', inplace=True)
    # gis = gis[gis['PROPTYPE']=='R']
    
    # Clear out NaNs
    # This first one might not be great, as it includes lots of vacant properties, but it also doesn't include add, lat, or lon
    gis = gis[gis['SITETYPE'].notnull()]
    gis = gis[gis['KCTP_STATE'].notnull()]
    
    # drop ACCNT_NUMBER - cannot tie to Seattle data
    gis = gis[['PIN', 'MAJOR', 'MINOR',  'SITETYPE', 'ADDR_FULL', 'LAT', 'LON', 
                'KCTP_STATE', 'LOTSQFT', 'LEVYCODE', 'NEW_CONSTR', 'TAXVAL_RSN', 'APPRLNDVAL', 
                'APPR_IMPR', 'TAX_LNDVAL', 'TAX_IMPR', 'QTS', 'SEC', 'TWP', 'RNG', 
                'Shape_Length', 'Shape_Area', 'PROPTYPE', 'KCA_ZONING', 
                'KCA_ACRES', 'PREUSE_DESC']]

    # Create Dummy Columns
    # dummy_cols = ['SITETYPE', 'LEVYCODE', 'NEW_CONSTR', 'TAXVAL_RSN', 'QTS', 'SEC', 'TWP', 'RNG', 
    #             'PROPTYPE', 'KCA_ZONING', 'PREUSE_DESC']
    # for col in dummy_cols:
    #     gis = pd.concat([gis,pd.get_dummies(gis[col], prefix=col,dummy_na=True)],axis=1).drop([col],axis=1)
    
    # Get the y column
    demo = get_pending_demo_permits()
    gis['TARGET'] = gis['ADDR_FULL'].apply(lambda x: x.lower()).isin(demo['OriginalAddress1'].apply(lambda x: x.lower()))
    gis['TARGET'] = gis['TARGET']*1

    
    
    # matches = []
    # non_matches = []
    # add_set = set(gis['ADDR_FULL'].apply(lambda x: x.lower()))
    # for address in demo['OriginalAddress1']:
    #     if address.lower() not in add_set:
    #         gis['Demo'] = 0
    #     else:
    #         matches.append(address)
    
    
    
    
    return gis
    

def get_parcels(parcels_filepath='data/EXTR_Parcel.csv'):
    parcels = pd.read_csv(parcels_filepath, encoding="Latin1")
    # parcels = parcels[(parcels.PropType=='R')]
    parcels['PIN'] = parcels['Major'].map(str).apply(lambda x: x.zfill(6)) + parcels['Minor'].map(str).apply(lambda x: x.zfill(4))
    parcels = parcels[parcels['DistrictName']=='SEATTLE']
    # parcels = parcels.set_index('PIN')

    parcels = parcels.drop(columns=['Major', 'Minor','PropName','PlatName','PlatLot','PlatBlock','PropType','SpecArea','SpecSubArea',
            'HBUAsIfVacant','HBUAsImproved','RestrictiveSzShape','DNRLease','TranspConcurrency', 'DistrictName', 'LandfillBuffer'])

    parcels['Unbuildable'] = parcels['Unbuildable'].apply(lambda x: 1 if x==True else 0)
    
    binary_cols = ['AdjacentGolfFairway', 'AdjacentGreenbelt', 'HistoricSite',
        'CurrentUseDesignation', 'NativeGrowthProtEsmt', 'Easements',
        'OtherDesignation', 'DeedRestrictions', 'DevelopmentRightsPurch',
        'CoalMineHazard', 'CriticalDrainage', 'ErosionHazard',
        'HundredYrFloodPlain', 'SeismicHazard', 'LandslideHazard',
        'SteepSlopeHazard', 'Stream', 'Wetland', 'SpeciesOfConcern',
        'SensitiveAreaTract', 'WaterProblems',
        'OtherProblems','WfntProximityInfluence','PowerLines', 
        'OtherNuisances', 'OtherProblems', 'WfntProximityInfluence']

    for col in binary_cols:
        parcels[col] = parcels[col].apply(lambda x: 1 if x=='Y' else 0)

    return parcels


def get_units(filepath='data/EXTR_UnitBreakdown.csv'):
    units = pd.read_csv(filepath)


    return units



def get_res_bld(filepath):
    res_build = pd.read_csv('data/EXTR_ResBldg.csv', low_memory=False)


    return res_build





# CSV_PATH = "data/mllib/2004_10000_small.csv"
# APP_NAME = "Random Forest Example"
# SPARK_URL = "local[*]"
# RANDOM_SEED = 13579
# TRAINING_DATA_RATIO = 0.7
# RF_NUM_TREES = 10
# RF_MAX_DEPTH = 30
# RF_MAX_BINS = 2048
# LABEL = "DepDelay15Min"
# CATEGORICAL_FEATURES = ["UniqueCarrier", "Origin", "Dest"]

# from pyspark import SparkContext
# from pyspark.ml.feature import StringIndexer
# from pyspark.ml import Pipeline
# from pyspark.mllib.linalg import Vectors
# from pyspark.mllib.tree import RandomForest
# from pyspark.mllib.regression import LabeledPoint
# from pyspark.sql import SparkSession
# from time import *

# # Creates Spark Session
# spark = SparkSession.builder.appName(APP_NAME).master(SPARK_URL).getOrCreate()

# # Reads in CSV file as DataFrame
# # header: The first line of files are used to name columns and are not included in data. All types are assumed to be string.
# # inferSchema: Automatically infer column types. It requires one extra pass over the data.
# df = spark.read.options(header = "true", inferschema = "true").csv(CSV_PATH)

# # Transforms all strings into indexed numbers
# indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in CATEGORICAL_FEATURES]
# pipeline = Pipeline(stages=indexers)
# df = pipeline.fit(df).transform(df)

# # Removes old string columns
# df = df.drop(*CATEGORICAL_FEATURES)

# # Moves the label to the last column
# df = StringIndexer(inputCol=LABEL, outputCol=LABEL+"_label").fit(df).transform(df)
# df = df.drop(LABEL)

# # Converts the DataFrame into a LabeledPoint Dataset with the last column being the label and the rest the features.
# transformed_df = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))

# # Splits the dataset into a training and testing set according to the defined ratio using the defined random seed.
# splits = [TRAINING_DATA_RATIO, 1.0 - TRAINING_DATA_RATIO]
# training_data, test_data = transformed_df.randomSplit(splits, RANDOM_SEED)

# print("Number of training set rows: %d" % training_data.count())
# print("Number of test set rows: %d" % test_data.count())

# # Run algorithm and measure runtime
# start_time = time()

# model = RandomForest.trainClassifier(training_data, numClasses=2, categoricalFeaturesInfo={}, numTrees=RF_NUM_TREES, featureSubsetStrategy="auto", impurity="gini", maxDepth=RF_MAX_DEPTH, maxBins=RF_MAX_BINS, seed=RANDOM_SEED)

# end_time = time()
# elapsed_time = end_time - start_time
# print("Time to train model: %.3f seconds" % elapsed_time)

# # Make predictions and compute accuracy
# predictions = model.predict(test_data.map(lambda x: x.features))
# labels_and_predictions = test_data.map(lambda x: x.label).zip(predictions)
# acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())
# print("Model accuracy: %.3f%%" % (acc * 100))

if __name__ == '__main__':
    spark = SparkSession\
    .builder\
    .master('Local[4]')\
    .appName("Seattle_Real_Estate")\
    .getOrCreate()     
    
    accuracy, prediction = run_model()
    prediction.coalesce(1).write.csv('predictions.csv')
    print(accuracy)