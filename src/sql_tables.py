import os
import pandas as pd
from sqlalchemy import create_engine
import psycopg2
conn = psycopg2.connect(dbname='seattle_housing', user = 'postgres', host='localhost')

alchemyEngine = create_engine('postgresql+psycopg2://postgres:postgres@localhost/seattle_housing', pool_recycle=3600)
postgreSQLConnection = alchemyEngine.connect()

for filename in os.listdir('/Users/ross/Galvanize/Seattle-Real-Estate-Analysis/data/csv_to_sql/'):
    if filename[:5]=='EXTR_':
        clean_filename = filename[5:]
    postgreSQLTable = clean_filename.split('.')[0].lower()
    df = pd.read_csv('/Users/ross/Galvanize/Seattle-Real-Estate-Analysis/data/csv_to_sql/'+ filename,
                    low_memory=False, encoding='latin_1').head(0)
    try:
        frame = df.to_sql(postgreSQLTable, postgreSQLConnection, if_exists='fail', index=False,)
        # curr=conn.cursor()
        # query = 'copy (table) from (file) CSV Header;',postgreSQLTable, 
        # curr.execute(query, {'table':postgreSQLTable, 'file':filename})
        # curr.close()
    except ValueError as vx:
        print(vx)
    except Exception as ex:  
        print(ex)
    else:
        print(f"PostgreSQL Table {postgreSQLTable} has been created successfully.")
    
conn.commit()
conn.close()