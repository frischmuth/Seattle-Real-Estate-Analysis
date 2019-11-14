import numpy as np
import pandas as pd
import time

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, Normalizer, StandardScaler

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Filepaths for running on AWS EMR
gis_filepath='/home/hadoop/Seattle-Real-Estate-Analysis/data/Parcels_for_King_County_with_Address_with_Property_Information__parcel_address_area.csv'
parcels_filepath='/home/hadoop/Seattle-Real-Estate-Analysis/data/EXTR_Parcel.csv'
permit_filepath='/home/hadoop/Seattle-Real-Estate-Analysis/data/Building_Permits.csv'
numFolds = 10

def run_forest_model(gis_filepath, parcels_filepath, permit_filepath, numFolds=10):
    

    # Individual functions below read in the csv, then convert from pandas to spark, then combine
    data = create_full_dataframe(gis_filepath, parcels_filepath, permit_filepath, numFolds)
    
    rf = RandomForestClassifier(featuresCol='all_features', labelCol='TARGET', predictionCol='Prediction')
    
    # Train and test ten random forests, each predicting 1/numFolds of the dataset
    for i in range(0,numFolds):
        train = data.filter(data.fold != i)
        test = data.filter(data.fold == i)
        model = rf.fit(train)
        if i == 0:
            predictions = model.transform(test)
        else:
            predictions = predictions.union(model.transform(test))

    # Split the prediction vector into two columns to allow for writing to CSV
    split1_udf = udf(lambda value: value[0].item(), FloatType())
    split2_udf = udf(lambda value: value[1].item(), FloatType())
    predictions = predictions.select('PIN', 'MAJOR','MINOR','ADDR_FULL','TARGET','Prediction', split1_udf('probability').alias('prob0'), split2_udf('probability').alias('prob1'))
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    prediction.toPandas().to_csv('predictions_'+timestr+'.csv').toPandas().to_csv('prediction.csv')

    return predictions

def run_regression_model():
    
    data = create_full_dataframe(gis_filepath, parcels_filepath, permit_filepath, numFolds)   
    
    lr = LogisticRegression(featuresCol='all_features', labelCol='TARGET', predictionCol='Prediction')
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='Prediction', labelCol='TARGET')
    
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

    # Create single feature column vector and drop originals
    input_columns = ['gis_features', 'parcel_features']
    assembler = VectorAssembler(
    inputCols= input_columns,
    outputCol='all_features')
    all_data = assembler.transform(all_data)
    all_data = all_data.drop(*input_columns)

    # For testing, return small, for production, return all_data
    # small, large = all_data.randomSplit([.005,.995])

    return all_data

def gis_data_to_spark(numFolds, gis_filepath='data/Parcels_for_King_County_with_Address_with_Property_Information__parcel_address_area.csv'):
    
    # Initially read in pre-cleaned Pandas DataFrame into Spark DataFrame
    gis_pd = get_gis_data(gis_filepath)
    
    gis_pd['fold'] = np.random.randint(0,numFolds,gis_pd.shape[0])
    gis = spark.createDataFrame(gis_pd)
    
    # Create new feature columns
    gis['value_per_area'] = gis['APPRLNDVAL']/gis['Shape_Area']
    gis['improvement_over_land'] = gis['APPR_IMPR']/gis['APPRLNDVAL']


    # Normalize numerical data
    numerical_cols = ['LAT','LON','LOTSQFT','APPRLNDVAL','APPR_IMPR','TAX_LNDVAL','TAX_IMPR',
                        'Shape_Length','Shape_Area', 'value_per_area', 'improvement_over_land']
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


def get_pending_demo_permits(permit_filepath='data/Building_Permits.csv'):
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


if __name__ == '__main__':
    spark = SparkSession\
    .builder\
    .appName("Seattle_Real_Estate")\
    .getOrCreate()     
    
    prediction = run_forest_model(gis_filepath=gis_filepath, parcels_filepath=parcels_filepath, permit_filepath=permit_filepath, numFolds=10)
