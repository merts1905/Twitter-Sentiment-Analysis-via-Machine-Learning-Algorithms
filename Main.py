

from shuffle import SHUFFLE
import csv
shuffle=SHUFFLE()


documents=[]
labell=[]
negative=[]
positive=[]
file="dataset_twitter.csv"
dataset_names=['A','B','C','D','E','F']
Size=[500000,450000,350000,250000,150000,100000]
with open(file) as Data:
    reader=csv.reader(Data)
    records=list(reader)
    negative=shuffle.shuffle(records[1:800000])
    positive=shuffle.shuffle(records[800001:1048575])
    Data.close()
    
for x in range(len(Size)):    
    Data=shuffle.shuffle(negative[0:int(Size[x]*0.6)]+positive[0:int(Size[x]*0.4)])
    with open(dataset_names[x]+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(records[0])
        for row in Data:
            writer.writerow(row)
