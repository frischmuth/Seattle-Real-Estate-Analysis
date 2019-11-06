import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, Normalizer, StandardScaler

def gis_data_to_spark(filepath='data/Parcels_for_King_County_with_Address_with_Property_Information__parcel_address_area.csv'):
    spark = SparkSession\
    .builder\
    .master('Local[4]')\
    .appName("Get_GIS_Data")\
    .getOrCreate()
    
    # Initially read in pre-cleaned Pandas DataFrame into Spark DataFrame
    gis_pd = get_gis_data(filepath)
    gis = spark.createDataFrame(gis_pd)
    
    # # Normalize numerical data
    numerical_cols = ['LAT','LON','LOTSQFT','APPRLNDVAL','APPR_IMPR','TAX_LNDVAL','TAX_IMPR','Shape_Length','Shape_Area']
    numerical_assembler = VectorAssembler(
    inputCols=numerical_cols,
    outputCol='num_features')
    gis = numerical_assembler.transform(gis)

    gis = StandardScaler(inputCol='num_features', outputCol='num_features_std').fit(gis).transform(gis)

    # Create index and dummy_vector column names of categorical colums, eventually dropping categorical and index columns
    cat_cols = ['KCTP_STATE', 'SITETYPE', 'LEVYCODE', 'NEW_CONSTR', 'TAXVAL_RSN', 'QTS', 'SEC', 'TWP', 'RNG', 'KCA_ZONING', 'PREUSE_DESC', 'PROPTYPE']
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
    # gis = gis.drop(*numerical_cols)
    
    # Combine all features into single vector 
    ignore = ['PIN', 'MAJOR', 'MINOR', 'ADDR_FULL']
    assembler = VectorAssembler(
    inputCols=[col for col in gis.columns if col not in ignore],
    outputCol='features')
    gis = assembler.transform(gis)

    # Drop all columns that are now in the features column
    ignore.append('features')
    gis = gis.drop(*[col for col in gis.columns if col not in ignore])
    
    # Write to parquet - not sure if I will eventually open from this, but that's the idea
    gis.write.parquet('data/gis_parquet',mode='overwrite')
    
    
    return gis










def get_pending_demo_permits(filepath='data/Building_permits.csv'):
    building_permits = pd.read_csv(filepath ,parse_dates=[10,11,12,13], infer_datetime_format=True)
    res_bldg_prmts = building_permits[building_permits['PermitClassMapped']=='Residential'].copy()
    demo = res_bldg_prmts[res_bldg_prmts['PermitTypeDesc']=='Demolition']
    pending_demo = demo[(demo['StatusCurrent']!='Closed')
                        &(demo['StatusCurrent']!='Completed')
                        &(demo['OriginalAddress1'].notnull())]
    pending_demo = pending_demo[[ 'OriginalAddress1', 'OriginalCity', 'OriginalState', 'OriginalZip', 'Latitude', 'Longitude']].copy()
    return pending_demo

def get_gis_data(filepath='data/Parcels_for_King_County_with_Address_with_Property_Information__parcel_address_area.csv'):
    gis = pd.read_csv(filepath, low_memory=False)
    
    # Limit to Seattle Residential
    gis = gis[gis['CTYNAME']=='SEATTLE']
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
    return gis.head(1000)
    

def get_parcels(filepath='data/EXTR_Parcel.csv'):
    parcels = pd.read_csv(filepath, encoding="Latin1")
    parcels = parcels[(parcels.PropType=='R')]
    parcels['PIN'] = parcels['Major'].map(str).apply(lambda x: x.zfill(6)) + parcels['Minor'].map(str).apply(lambda x: x.zfill(4))
    parcels = parcels[parcels['DistrictName']=='SEATTLE']
    parcels = parcels.set_index('PIN')

    parcels = parcels.drop(columns=['PropName','PlatName','PlatLot','PlatBlock','PropType','SpecArea','SpecSubArea',
            'HBUAsIfVacant','HBUAsImproved','RestrictiveSzShape','DNRLease','TranspConcurrency'])

    dummy_cols = ['Range', 'Township', 'Section', 'QuarterSection', 'Area',
       'SubArea','DistrictName', 'LevyCode',
       'CurrentZoning', 'PresentUse',
       'SqFtLot', 'WaterSystem', 'SewerSystem', 'Access', 'Topography',
       'StreetSurface', 'InadequateParking', 
       'Unbuildable', 'MtRainier', 'Olympics', 'Cascades',
       'Territorial', 'SeattleSkyline', 'PugetSound', 'LakeWashington',
       'LakeSammamish', 'SmallLakeRiverCreek', 'OtherView', 'WfntLocation',
        'WfntBank', 'WfntPoorQuality', 'WfntRestrictedAccess',
       'WfntAccessRights',  'TidelandShoreland',
       'LotDepthFactor', 'TrafficNoise', 'NbrBldgSites', 'Contamination']
    
    # for col in dummy_cols:
    #     parcels = pd.concat([parcels,pd.get_dummies(parcels[col], prefix=col,dummy_na=True)],axis=1).drop([col],axis=1)
    
    # Create Binary Columns
    binary_cols = ['AdjacentGolfFairway', 'AdjacentGreenbelt', 'HistoricSite',
        'CurrentUseDesignation', 'NativeGrowthProtEsmt', 'Easements',
        'OtherDesignation', 'DeedRestrictions', 'DevelopmentRightsPurch',
        'CoalMineHazard', 'CriticalDrainage', 'ErosionHazard', 'LangisillBuffer',
        'HundredYrFloodPlain', 'SeismicHazard', 'LandslideHazard',
        'SteepSlopeHazard', 'Stream', 'Wetland', 'SpeciesOfConcern',
        'SensitiveAreaTract', 'WaterProblems', 'TranspConcurrency',
        'OtherProblems''WfntProximityInfluence','PowerLines', 
       'OtherNuisances']
    # for col in binary_cols:
        # parcels[col] = parcels[col].map(lambda x: True if x=='Y' else False)
    
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