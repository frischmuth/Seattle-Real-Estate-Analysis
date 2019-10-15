import numpy as np
import pandas as pd

def get_pending_demo_permits(file):
    building_permits = pd.read_csv(file ,parse_dates=[10,11,12,13], infer_datetime_format=True)
    res_bldg_prmts = building_permits[building_permits['PermitClassMapped']=='Residential'].copy()
    demo = res_bldg_prmts[res_bldg_prmts['PermitTypeDesc']=='Demolition']
    pending_demo = demo[(demo['StatusCurrent']!='Closed')&(demo['StatusCurrent']!='Completed')&(demo['OriginalAddress1'].notnull())][[ 'OriginalAddress1', 'OriginalCity', 'OriginalState', 'OriginalZip', 'Latitude', 'Longitude']].copy()

    return pending_demo