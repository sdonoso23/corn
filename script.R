library(lubridate)
library(tidyverse)
library(caret)
#library(doParallel)
#registerDoParallel(4)

setwd("C:/Users/Administrador.000/Desktop/Sebastian/Repos/cornprice")

####LOAD DATA####

#load data, transform date to date format, ClassNextday as factor
data<-read_csv2("data/dataSet.csv")
data$DataTimestamp<-dmy(data$DataTimestamp)
data$ClassNextDay<-factor(data$ClassNextDay,levels=c(-1,1),
                             labels=c("Down","Up"))

#verify if there's missing values, count how many 0 are in each variable
colSums(is.na(data))
colSums(data==0)


#separate training and test data
trainidx<-round(nrow(data)*0.85)
testidx<-round((nrow(data)-trainidx))

trainingdata<-data[1:trainidx,]
testdata<-data[(trainidx+1):nrow(data),]

####VARIABLE ANALYSIS####

#correlation matrix
View(cor(trainingdata[2:73]))

#select only columns related to corn
data.corn<- trainingdata %>% 
    select(starts_with("Corn"))

#correlation of only columns related to corn
View(cor(data.corn))

#select only columns related to corn
data.returns<- trainingdata %>%
    select(contains("Returns"))

#correlation of only columns related to corn
View(cor(data.returns))

#select data related with closing prices
data.close<-trainingdata %>%
    select(DataTimestamp,ends_with("Close")) 

##plot of closing prices
data.close  %>%
    gather(CornClose:BrentClose,key=var,value=value) %>% 
    filter(var=="WheatClose" | var=="EURUSDClose") %>%
    ggplot()+geom_line(aes(x=DataTimestamp,y=value,color=var))  

ggplot()+
    geom_density(data=trainingdata,aes(x=BrentReturnsT2,color=ClassNextDay))

####NORMALIZE DATA####

###TRAINING

training.norm<-trainingdata %>%
    select(-1,-74)

minimo<-map_dbl(training.norm,min)
maximo<-map_dbl(training.norm,max)

training.norm<-as.data.frame(scale(training.norm,
                                   center=minimo,scale=maximo-minimo)) %>%
    add_column(ClassNextDay=trainingdata$ClassNextDay,.after = 72) %>%
    add_column(DataTimestamp=trainingdata$DataTimestamp,.before = 1)

###TEST
test.norm<-testdata %>%
    select(-1,-74)

test.norm<-as.data.frame(scale(test.norm,
                                   center=minimo,scale=maximo-minimo)) %>%
    add_column(ClassNextDay=testdata$ClassNextDay,.after = 72) %>%
    add_column(DataTimestamp=testdata$DataTimestamp,.before = 1)


training.norm  %>%
    gather(CornClose:EURUSDMexp,key=var,value=value) %>%
    ggplot()+geom_boxplot(aes(x=var,y=value))+coord_flip()
    


#####VARIABLE SELECTION####
#ELIMINATED EURUSD BY HIGH CORRELATION WITH BRENT
#HIGH,LOW,OPEN,TP BY HIGH CORRELATION WITH CLOSING PRICES
#VRR BY HIGH CORRELATION WITH VOLUME
#Mexp by HIGH CORRELATION WITH CLOSING PRICES

training.filter<-training.norm %>%
    select(-starts_with("EURUSD"),-ends_with("High"),
           -ends_with("Low"),-ends_with("Open"),-ends_with("TP"),
           -ends_with("VRR"),-ends_with("Mexp")) %>%
    filter(ClassNextDay=="Up" | ClassNextDay=="Down")

test.filter<-test.norm %>%
    select(-starts_with("EURUSD"),-ends_with("High"),
           -ends_with("Low"),-ends_with("Open"),-ends_with("TP"),
           -ends_with("VRR"),-ends_with("Mexp")) %>%
    filter(ClassNextDay=="Up" | ClassNextDay=="Down")

boxplot<-training.filter  %>%
    gather(CornClose:BrentRSI,key=var,value=value) %>%
    ggplot()+geom_boxplot(aes(x=var,y=value))+coord_flip()

ggsave("plots/botxplot.jpg",boxplot)

####DATA PARTITION AND CV PARAMETERS####

timeslices<-createTimeSlices(1:nrow(training.filter),
                             initialWindow = 1000,horizon =406, 
                             fixedWindow = FALSE)


fitcontrol<-trainControl(method = "cv",
                         index = timeslices$train,
                         indexOut= timeslices$test,
                         allowParallel = TRUE,
                         classProbs = TRUE,
                         savePredictions = TRUE,
                         returnResamp = "all",
                         verboseIter = TRUE)

####SINGLE LAYER NEURAL NETWORK WITH BFGS ALGORITHM####

###WITH ALL SELECTED VARIABLES

nngrid<-expand.grid(list(size=c(1:5),
                         decay=c(0.001,0.01,0.1,1)))
set.seed(1234)
neural1<-caret::train(ClassNextDay~.-DataTimestamp,data=training.filter,method="nnet",
               trControl=fitcontrol,tuneGrid=nngrid)



###USING STEPWISE TO SELECT VARIABLES

model1<-glm(ClassNextDay~.-DataTimestamp,data=training.filter,
            family="binomial"(link="logit"))

summary(model1)

stepwise1<-step(model1)
summary(stepwise1)

set.seed(1234)
neural2<-caret::train(ClassNextDay~SoybeanClose+SoybeanVolume+CornReturnsT+
                   SoybeanReturnsT+WheatReturnsT+SP500RSI+
                   BrentReturnsT1
               ,data=training.filter,method="nnet",
               trControl=fitcontrol,tuneGrid=nngrid)

####RSNNS: BACKPROPAGATED NEURAL NETWORKS WITH MULTIPLE LAYERS####

rsnnsgrid1<-expand.grid(layer1=c(1:5),layer2=c(0),layer3=c(0))
rsnnsgrid2<-expand.grid(layer1=c(1:5),layer2=c(1:5),layer3=c(0))
rsnnsgrid3<-expand.grid(layer1=c(1:5),layer2=c(1:5),layer3=c(1:5))
rsnnsgrid11<-expand.grid(layer1=c(1:5),layer2=c(0),layer3=c(0),decay=c(0.001,0.005,0.01,0.05,0.1))
rsnnsgrid22<-expand.grid(layer1=c(1:5),layer2=c(1:5),layer3=c(0),decay=c(0.001,0.005,0.01,0.05,0.1))
rsnnsgrid33<-expand.grid(layer1=c(1:5),layer2=c(1:5),layer3=c(1:5),decay=c(0.001,0.005,0.01,0.05,0.1))

set.seed(1234)
rsnns1<-caret::train(ClassNextDay~SoybeanClose+SoybeanVolume+CornReturnsT+
                         SoybeanReturnsT+WheatReturnsT+SP500RSI+
                         BrentReturnsT1
               ,data=training.filter,method="mlpML",
               trControl=fitcontrol,tuneGrid=rsnnsgrid1,
               learnFunc="Std_Backpropagation")

set.seed(1234)
rsnns2<-caret::train(ClassNextDay~SoybeanClose+SoybeanVolume+CornReturnsT+
                         SoybeanReturnsT+WheatReturnsT+SP500RSI+
                         BrentReturnsT1
                     ,data=training.filter,method="mlpML",
                     trControl=fitcontrol,tuneGrid=rsnnsgrid2,
                     learnFunc="Std_Backpropagation")
set.seed(1234)
rsnns3<-caret::train(ClassNextDay~SoybeanClose+SoybeanVolume+CornReturnsT+
                         SoybeanReturnsT+WheatReturnsT+SP500RSI+
                         BrentReturnsT1
                     ,data=training.filter,method="mlpML",
                     trControl=fitcontrol,tuneGrid=rsnnsgrid3,
                     learnFunc="Std_Backpropagation")

set.seed(1234)
rsnns11<-caret::train(ClassNextDay~SoybeanClose+SoybeanVolume+CornReturnsT+
                          SoybeanReturnsT+WheatReturnsT+SP500RSI+
                          BrentReturnsT1
                      ,data=training.filter,method="mlpWeightDecayML",
                      trControl=fitcontrol,tuneGrid=rsnnsgrid11,
                      learnFunc="Std_Backpropagation")

set.seed(1234)
rsnns22<-caret::train(ClassNextDay~SoybeanClose+SoybeanVolume+CornReturnsT+
                          SoybeanReturnsT+WheatReturnsT+SP500RSI+
                          BrentReturnsT1
                      ,data=training.filter,method="mlpWeightDecayML",
                      trControl=fitcontrol,tuneGrid=rsnnsgrid22,
                      learnFunc="Std_Backpropagation")

set.seed(1234)
rsnns33<-caret::train(ClassNextDay~SoybeanClose+SoybeanVolume+CornReturnsT+
                          SoybeanReturnsT+WheatReturnsT+SP500RSI+
                          BrentReturnsT1
                     ,data=training.filter,method="mlpWeightDecayML",
                     trControl=fitcontrol,tuneGrid=rsnnsgrid33,
                     learnFunc="Std_Backpropagation")





####SUPPORT VECTOR MACHINES####

svmradial<-expand.grid(C=c(0.001,0.1,0.25,0.5,1),sigma=c(0.001,0.002,0.005,0.01))

set.seed(1234)
svm<-caret::train(ClassNextDay~.-DataTimestamp,data=training.filter,method="svmRadial",
            trControl=fitcontrol,tuneGrid=svmradial)

set.seed(1234)
svm1<-caret::train(ClassNextDay~SoybeanClose+SoybeanVolume+CornReturnsT+
                       SoybeanReturnsT+WheatReturnsT+SP500RSI+
                       BrentReturnsT1
            ,data=training.filter,method="svmRadialSigma",
            trControl=fitcontrol)

set.seed(1234)
svm2<-caret::train(ClassNextDay~SoybeanClose+SoybeanVolume+CornReturnsT+
                       SoybeanReturnsT+WheatReturnsT+SP500RSI+
                       BrentReturnsT1
            ,data=training.filter,method="svmRadialSigma",
            trControl=fitcontrol,tuneGrid=svmradial)

svmlinear<-expand.grid(C=c(0.01,0.1,0.5,1,2))

set.seed(1234)
svm3<-caret::train(ClassNextDay~SoybeanClose+SoybeanVolume+CornReturnsT+
                       SoybeanReturnsT+WheatReturnsT+SP500RSI+
                       BrentReturnsT1
            ,data=training.filter,method="svmLinear",
            trControl=fitcontrol,tuneGrid=svmlinear)

set.seed(1234)
svm4<-caret::train(ClassNextDay~SoybeanClose+SoybeanVolume+CornReturnsT+
                       SoybeanReturnsT+WheatReturnsT+SP500RSI+
                       BrentReturnsT1
            ,data=training.filter,method="svmPoly",
            trControl=fitcontrol)

####QUICK SUMMARY####

models<-list(BFGSNN=neural2,NN1Layer=rsnns1,NN2Layer=rsnns2,
             NN3Layer=rsnns3,NN1LayerDecay=rsnns11,NN2LayerDecay=rsnns22,
             NN3LayerDecay=rsnns33,SVMRad=svm1,SVMRadGrid=svm2,
             SVMLinear=svm3,SVMPoly=svm4)



results<-resamples(models)

summary(results)

svm1.pred<-predict(object = svm1,newdata = test.filter)

caret::confusionMatrix(svm25.pred,test.filter$ClassNextDay,positive="Up")




####EXPORT MODEL RESULTS####

for (i in 1:length(models)){
      df<-as.data.frame(models[[i]]$resample)
      write_csv(df,paste("output/resamples",names(models[i]),".csv",sep=""))
      }

for (i in 1:length(models)){
    df<-as.data.frame(models[[i]]$pred)
    write_csv(df,paste("output/pred",names(models[i]),".csv",sep=""))
}

read_csv("output/resamplesSVMLinear.csv") %>%
    group_by(C) %>%
    summarize(Acc=mean(Accuracy)) %>%
    arrange(desc(Acc))

read_csv("output/resamplesSVMPoly.csv") %>%
    group_by(C,degree,scale) %>%
    summarize(Acc=mean(Accuracy)) %>%
    arrange(desc(Acc))

read_csv("output/resamplesSVMRad.csv") %>%
    group_by(C,sigma) %>%
    summarize(Acc=mean(Accuracy)) %>%
    arrange(desc(Acc))

read_csv("output/resamplesSVMRadGrid.csv") %>%
    group_by(C,sigma) %>%
    summarize(Acc=mean(Accuracy)) %>%
    arrange(desc(Acc))

read_csv("output/resamplesNN1Layer.csv") %>%
    group_by(layer1,layer2,layer3) %>%
    summarize(Acc=mean(Accuracy)) %>%
    arrange(desc(Acc))

read_csv("output/resamplesNN2Layer.csv") %>%
    group_by(layer1,layer2,layer3) %>%
    summarize(Acc=mean(Accuracy)) %>%
    arrange(desc(Acc))

read_csv("output/resamplesNN3Layer.csv") %>%
    group_by(layer1,layer2,layer3) %>%
    summarize(Acc=mean(Accuracy)) %>%
    arrange(desc(Acc))

read_csv("output/resamplesNN1LayerDecay.csv") %>%
    group_by(layer1,layer2,layer3,decay) %>%
    summarize(Acc=mean(Accuracy)) %>%
    arrange(desc(Acc))

read_csv("output/resamplesNN2LayerDecay.csv") %>%
    group_by(layer1,layer2,layer3,decay) %>%
    summarize(Acc=mean(Accuracy)) %>%
    arrange(desc(Acc))

read_csv("output/resamplesNN3LayerDecay.csv") %>%
    group_by(layer1,layer2,layer3,decay) %>%
    summarize(Acc=mean(Accuracy)) %>%
    arrange(desc(Acc))
