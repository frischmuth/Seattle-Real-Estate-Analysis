import numpy as np
import pandas as pd

def get_pending_demo_permits(filepath):
    building_permits = pd.read_csv(filepath ,parse_dates=[10,11,12,13], infer_datetime_format=True)
    res_bldg_prmts = building_permits[building_permits['PermitClassMapped']=='Residential'].copy()
    demo = res_bldg_prmts[res_bldg_prmts['PermitTypeDesc']=='Demolition']
    pending_demo = demo[(demo['StatusCurrent']!='Closed')
                        &(demo['StatusCurrent']!='Completed')
                        &(demo['OriginalAddress1'].notnull())]
    pending_demo = pending_demo[[ 'OriginalAddress1', 'OriginalCity', 'OriginalState', 'OriginalZip', 'Latitude', 'Longitude']].copy()
    return pending_demo

def get_gis_data(filepath='data/Parcels_for_King_County_with_Address_with_Property_Information__parcel_address_area.csv'):
    gis = pd.read_csv(filepath, index_col='PIN', low_memory=False)
    gis = gis[gis['CTYNAME']=='SEATTLE']
    gis = gis[gis['PROPTYPE']=='R']
    gis = gis[['MAJOR', 'MINOR',  'SITETYPE', 'ADDR_FULL', 'LAT', 'LON', 
                'KCTP_STATE', 'LOTSQFT', 'LEVYCODE', 'LEVY_JURIS', 'NEW_CONSTR', 'TAXVAL_RSN', 'APPRLNDVAL', 
                'APPR_IMPR', 'TAX_LNDVAL', 'TAX_IMPR', 'ACCNT_NUM', 'KCTP_TAXYR', 'QTS', 'SEC', 'TWP', 'RNG', 'PRIMARY_ADDR', 
                'Shape_Length', 'Shape_Area', 'PROPTYPE', 'KCA_ZONING', 
                'KCA_ACRES', 'PREUSE_CODE', 'PREUSE_DESC']]
    return gis
    

def get_parcels(filepath='data/EXTR_Parcel.csv'):
    parcels = pd.read_csv(filepath, encoding="Latin1")
    parcels = parcels[(parcels.PropType=='R')]
    parcels['PIN'] = parcels['Major'].map(str).apply(lambda x: x.zfill(6)) + parcels['Minor'].map(str).apply(lambda x: x.zfill(4))
    parcels = parcels[parcels['DistrictName']=='SEATTLE']
    parcels = parcels.set_index('PIN')

def get_units(filepath='data/EXTR_UnitBreakdown.csv'):
    units = pd.read_csv(filepath)