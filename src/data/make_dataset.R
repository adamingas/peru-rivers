#!/usr/bin/Rscript

# This script will open the full data set as provided by the bioinformaticians
# and save only the relevant count data. 
# The input is taxonomy_only_otu.csv, full_otu.csv, and WWF_Samples.txt and they
# can be found in /data/raw
# The output is -fulldf, the full sample-otu matrix 
#               -riverdf, the sample-river_otu which includes only
#               otus associated with the river. This association was determined by the 
#               bioinformatitians using the assigned taxonomies, thus unassigned otus
#               are not included in this dataset
#               -fulldf100s, where samples and otus with under 10000 and 100 total 
#               reads respectively are discarded
#               -riverdf100s same as above for riverdf
#               -taxadf just the taxonomy 
#               -wwfdf metadata
# The output folder is /data/processed

# Go to the directory of the data
setwd("data/raw")
otudata = read.table(file = "taxonomy_only_otu.csv",sep = ",",stringsAsFactors = FALSE)
otutable = t(otudata[-c(1,2,3),-seq(1,9)])
colnames(otutable) <- (t(otudata)[2,-c(1,2,3)])
otudf  <- as.data.frame(x=otutable,stringsAsFactors=FALSE,row.names = otudata[1,-seq(1,9)])
otudf[,]<- sapply(otudf[,], as.numeric)


# Creating an index of otus found in the river using the appropriate
# column in the otudata
index_of_riverotus = names(otudf) %in% otudata[otudata[,1] == "yes","V2"]

# creating the river dataframe
riverdf <- otudf[,index_of_riverotus]

# Getting taxonomy from otudata
taxadf<-data.frame(otudata[-seq(1,3),seq(4,8)],row.names = otudata[-seq(1,3),2])
colnames(taxadf)<- otudata[3,seq(4,8)]

# Creating the fulldataset
fulldata = read.table(header = FALSE,file = "full_otu.csv",sep = ",",stringsAsFactors = F)

fulltable = t(fulldata[-c(1,2,3),-seq(1,8)])
colnames(fulltable) <- fulldata[-c(1,2,3),1]
fulldf =as.data.frame(x = fulltable,row.names =  fulldata[1,-seq(1,8)],stringsAsFactors = FALSE)
fulldf[,] <- sapply(fulldf[,],as.numeric) 

# Preparing metadata
wwf =read.table(file = "WWF_Samples.txt",header =TRUE,sep = "\t",stringsAsFactors = FALSE)
wwfdf <- as.data.frame(x=wwf,row.names=wwf$ID)
wwfdf[,"Area_group"] <- sapply(wwfdf[,"Area_group"],as.factor)
wwfdf[,"Area_group_name"] <- sapply(wwfdf[,"Area_group_name"],as.factor)
wwfdf[,"Water"] <- sapply(wwfdf[,"Water"],as.factor)

wwfdf$ID_nosamples <- gsub(pattern = "[a-zA-Z-]","",wwfdf$ID)
wwfdf[,"ID_nosamples"]<- sapply(wwfdf[,"ID_nosamples"],as.factor)



# Producing smaller datasets with only otus with total read counts >100
# and samples with total read counts >10000
riverdf100 =riverdf[,-(colSums(riverdf)<100)]
fulldf100 = fulldf[,-(colSums(fulldf)<100)]
#Removing samples with less than 10000 read counts
# 8 samples with less than 10000 reads in riverdf100
index_of_riverotus_samples =(rowSums(riverdf) <10000)
riverdf100s = riverdf100[!index_of_riverotus_samples,]
# 3 samples with less than 10000 in fulldf
index_of_full_samples =(rowSums(fulldf) <10000)
fulldf100s = fulldf100[!index_of_full_samples,]


# Saving dataframes as csv
setwd("../processed/")
write.csv(x = riverdf,file = "riverdf")
write.csv(x = riverdf100s,file = "riverdf100s")

write.csv(x = fulldf,file = "fulldf")
write.csv(x = fulldf100s,file = "fulldf100s")

write.csv(x = wwfdf,file = "wwfdf",fileEncoding = "UTF-8")

write.csv(taxadf,"taxadf")
