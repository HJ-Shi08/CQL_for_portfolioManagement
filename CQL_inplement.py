import d3rlpy
from dataProcessing import Dateset

MDPdataset=Dateset('AAPL.csv')
mdpData=MDPdataset.MDPDateset()


cql=d3rlpy.algos.CQL()
cql.fit(mdpData, n_steps_per_epoch=100)
