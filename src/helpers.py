import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler

def gis_data_to_spark(filepath='data/Parcels_for_King_County_with_Address_with_Property_Information__parcel_address_area.csv'):
    spark = SparkSession\
    .builder\
    .master('Local[4]')\
    .appName("GetGISData")\
    .getOrCreate()
    
    gis_pd = get_gis_data()
    gis = spark.createDataFrame(gis_pd)
    
    cat_cols = ['SITETYPE', 'LEVYCODE', 'NEW_CONSTR', 'TAXVAL_RSN', 'QTS', 'SEC', 'TWP', 'RNG', 'KCA_ZONING', 'PREUSE_DESC']
    cat_index = []
    dummies = []
    for col in cat_cols:
        cat_index.append(col+'_index')
        dummies.append(col+'_dummy_vector')
    
    # Transforms all strings into indexed numbers
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(gis) for column in cat_cols]
    pipeline = Pipeline(stages=indexers)
    gis = pipeline.fit(gis).transform(gis)

    gis = gis.drop(*cat_cols)
    encoder = OneHotEncoderEstimator(inputCols=cat_index,outputCols=dummies)
    model = encoder.fit(gis)
    gis = model.transform(gis)   
    
    
    
    
    
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
    gis = gis[gis['PROPTYPE']=='R']
    
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
    return gis
    

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