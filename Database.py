# -*- coding: utf-8 -*-
import sqlalchemy
import pandas as pd

class Database(object):
    def __init__(self):
        host = '10.5.22.73'
        user = 'u645287'
        password = 'P%40ssw0rd'
        database = 'SMT'
        self.engine = sqlalchemy.create_engine('postgresql://'+ user +':'+ password + '@' + host + ':5432/' + database)
    def __del__(self):
        self.engine.dispose()
    def get_input(self):
        return pd.read_sql('select * from ordc_edreg_input order by year, month',self.engine)
def postgres_upsert(table, conn, keys, data_iter):
    from sqlalchemy.dialects.postgresql import insert

    data = [dict(zip(keys, row)) for row in data_iter]

    insert_statement = insert(table.table).values(data)
    upsert_statement = insert_statement.on_conflict_do_update(
        constraint=f"{table.table.name}_pkey",
        set_={c.key: c for c in insert_statement.excluded},
    )
    conn.execute(upsert_statement)

