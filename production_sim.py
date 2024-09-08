import scheduling_lib as slib
# import pandas as pd

ptime_file = 't_500_20_mon.csv'
job_seq = [4,2,3,5,1]

# ptimes = pd.read_csv(ptime_file, index_col='JobID', nrows=len(job_seq))
# print(ptimes)
# ptimes = ptimes.sort_values(by=['M1'], ascending=True)
# print(ptimes)
# job_seq = ptimes.index.values
# print(job_seq)


makespan = slib.schedule(ptime_file, job_seq)
print(makespan)