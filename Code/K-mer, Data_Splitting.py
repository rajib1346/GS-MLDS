#Import library
import numpy as np
import pandas as pd

#Load dataset

#df = pd.read_csv('E:/R3(DNA)/Material/C.elegans/C_elegans.csv')
#df = pd.read_csv('E:/R3(DNA)/Material/D.melanogaster/D_melanogaster.csv')
#df = pd.read_csv('E:/R3(DNA)/Material/A.thaliana/A_thaliana.csv')
#df = pd.read_csv('E:/R3(DNA)/Material/E.coli/E_coli.csv')
#df = pd.read_csv('E:/R3(DNA)/Material/G.subterraneus/G_subterraneus.csv')
#df = pd.read_csv('E:/R3(DNA)/Material/G.pickeringi/G_pickeringi.csv')
#df = pd.read_csv('E:/R3(DNA)/Material/F.vesca/F_vesca.csv')
df = pd.read_csv('E:/R3(DNA)/Material/R.chinensis/R_chinensis.csv')

df.head()

#Function to convert sequence strings into k-mer words

def getKmers(sequence, size=3):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

df['Gene_Sequence'] = df.apply(lambda x: getKmers(x['Gene Sequence']), axis=1)
df = df.drop('Gene Sequence', axis=1)

dna_texts = list(df['Gene_Sequence'])
for item in range(len(dna_texts)):
    dna_texts[item] = ' '.join(dna_texts[item]) 

y_data = df.iloc[:, 0].values  
dna_kmer = pd.DataFrame(dna_texts)
dna_kmer['Target']=y_data
dna_kmer

#Save dataset after k-mer

#dna_kmer.to_csv('E:/R3(DNA)/Material/C.elegans/C_elegans_Dnakmer.csv')
#dna_kmer.to_csv('E:/R3(DNA)/Material/D.melanogaster/D_melanogaster_Dnakmer.csv')
#dna_kmer.to_csv('E:/R3(DNA)/Material/A.thaliana/A_thaliana_Dnakmer.csv')
#dna_kmer.to_csv('E:/R3(DNA)/Material/E.coli/E_coli_Dnakmer.csv')
#dna_kmer.to_csv('E:/R3(DNA)/Material/G.subterraneus/G_subterraneus_Dnakmer.csv') 
#dna_kmer.to_csv('E:/R3(DNA)/Material/G.pickeringi/G_pickeringi_Dnakmer.csv')
#dna_kmer.to_csv('E:/R3(DNA)/Material/F.vesca/F_vesca_Dnakmer.csv') 
dna_kmer.to_csv('E:/R3(DNA)/Material/R.chinensis/R_chinensis_Dnakmer.csv') 

#Dataset splitting into train and test

#df2 = pd.read_csv('E:/R3(DNA)/Material/C.elegans/C_elegans_Dnakmer.csv')
#df2 = pd.read_csv('E:/R3(DNA)/Material/D.melanogaster/D_melanogaster_Dnakmer.csv')
#df2 = pd.read_csv('E:/R3(DNA)/Material/A.thaliana/A_thaliana_Dnakmer.csv')
#df2 = pd.read_csv('E:/R3(DNA)/Material/E.coli/E_coli_Dnakmer.csv')
#df2 = pd.read_csv('E:/R3(DNA)/Material/G.subterraneus/G_subterraneus_Dnakmer.csv') 
#df2 = pd.read_csv('E:/R3(DNA)/Material/G.pickeringi/G_pickeringi_Dnakmer.csv')
#df2 = pd.read_csv('E:/R3(DNA)/Material/F.vesca/F_vesca_Dnakmer.csv') 
df2 = pd.read_csv('E:/R3(DNA)/Material/R.chinensis/R_chinensis_Dnakmer.csv')
X= df2.drop(['Target'], axis=1)
y= df2['Target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
X_train.shape, X_test.shape

#Save train and dataset into drive
X_train['Target']=y_train
X_test['Target']=y_test

#X_train.to_csv('E:/R3(DNA)/Material/C.elegans/50_50/Train_(50_50).csv')
#X_test.to_csv('E:/R3(DNA)/Material/C.elegans/50_50/Test(50_50).csv')
#X_train.to_csv('E:/R3(DNA)/Material/D.melanogaster/50_50/Train_(50_50).csv')
#X_test.to_csv('E:/R3(DNA)/Material/D.melanogaster/50_50/Test(50_50).csv')
#X_train.to_csv('E:/R3(DNA)/Material/A.thaliana/50_50/Train_(50_50).csv')
#X_test.to_csv('E:/R3(DNA)/Material/A.thaliana/50_50/Test(50_50).csv')
#X_train.to_csv('E:/R3(DNA)/Material/E.coli/50_50/Train_(50_50).csv')
#X_test.to_csv('E:/R3(DNA)/Material/E.coli/50_50/Test(50_50).csv')
#X_train.to_csv('E:/R3(DNA)/Material/G.subterraneus/50_50/Train_(50_50).csv')
#X_test.to_csv('E:/R3(DNA)/Material/G.subterraneus/50_50/Test(50_50).csv')
#X_train.to_csv('E:/R3(DNA)/Material/G.pickeringi/50_50/Train_(50_50).csv')
#X_test.to_csv('E:/R3(DNA)/Material/G.pickeringi/50_50/Test(50_50).csv')
#X_train.to_csv('E:/R3(DNA)/Material/F.vesca/50_50/Train_(50_50).csv')
#X_test.to_csv('E:/R3(DNA)/Material/F.vesca/50_50/Test(50_50).csv')
X_train.to_csv('E:/R3(DNA)/Material/R.chinensis/50_50/Train_(50_50).csv')
X_test.to_csv('E:/R3(DNA)/Material/R.chinensis/50_50/Test(50_50).csv')
