# Load the data
survey_data = read.csv("Simulated_WTP_Survey_group.csv",header=T)
# Row count
N=nrow(survey_data)
# Print top few rows of data
head(survey_data)

# The maximium WTP in data, we can use this as the upper bound for our price search.
# No need to consider a price if noone can afford it.
# Column 2 and 3 contain WTP of non peak and peak

#for (i in 1:N){
#    survey_data$maxWTP[i]=max(survey_data[i,3:12])
#}


# Only using Group for the first time
#maxprice=max(survey_data$maxWTP)

maxprice=max(survey_data[3:12])


# Price for all time slots (i.e., nonpeak) except for peak
basePrice=20

# Creating empty lists
demandNonPeak<-rep(0,maxprice)
demandPeak<-rep(0,maxprice)
revenue<-rep(0,maxprice)


# STEP 1:
# For each client we will obtain their maximum WTP and
# maximum Surplus among the Nonpeak time slot
# Columns 2 is the NonPeak time slot.

maxWTPNonPeak<-rep(0,N)
maxsurplusNonPeak<-rep(0,N)

for (i in 1:N){
    maxWTPNonPeak[i]=max(survey_data[i,c(5:10)])
    maxsurplusNonPeak[i]=max(survey_data[i,c(5:10)]-basePrice)
    
    # We can also generate new column(s) and add this information to our daya nyhc:
    survey_data$maxWTPNonPeak[i]=max(survey_data[i,c(5:10)])
    survey_data$maxsurplusNonPeak[i]=max(survey_data[i,c(5:10)]-basePrice)
}

# Viewing the first ten rows of data
survey_data[1:10,]

surplusPeak<-matrix(0,N,maxprice)

for (p in 1:maxprice) {
    for (i in 1:N) {
        surplusPeak[i, p] <- max(survey_data[i, c(3, 4, 11, 12)]) - p
    }
}

# Viewing a part of data in surplusPeak
# the first ten client's surpluses for Peak time slot if pPeak Price p=50, 51, 52, ..., 60.
colnames(surplusPeak)=paste0("p=",1:maxprice)
surplusPeak[1:10,]


for (p in 1:maxprice){
  demandNonPeak[p]=(sum((maxsurplusNonPeak>surplusPeak[,p])*(maxsurplusNonPeak>=0))) 
  demandPeak[p]=(sum((surplusPeak[,p]>=maxsurplusNonPeak)*(surplusPeak[,p]>=0))) 
  revenue[p]=(basePrice*demandNonPeak[p]+p*demandPeak[p])
}

# Plotting NonPeak Demand vs Peak Period Price
xaxis=1:maxprice
plot(xaxis,demandNonPeak,pch = 16, type="s",col="blue", las=1, xaxt="n",
     xlab="Price for Peak Period",ylab="Non-Peak Period Demand")
xticks <- seq(0, maxprice, by=1)
axis(side = 1, at = xticks)

# Plotting Peak Demand vs Peak Period Price
xaxis=1:maxprice
plot(xaxis,demandPeak,pch = 16, type="s",col="blue", las=1, xaxt="n",
     xlab="Price for Peak Period",ylab="Peak Period Demand")
xticks <- seq(0, maxprice, by=1)
axis(side = 1, at = xticks)

# Plotting Revenue vs Peak Period Price
# Set priceBest (Peak Period Price) to first entry on list
xaxis=1:maxprice
plot(xaxis,revenue,pch = 16, type="s",col="blue", las=1, xaxt="n",
     xlab="Price for Peak Period",ylab="Total Revenue")
xticks <- seq(0, maxprice, by=1)
axis(side = 1, at = xticks)
revenueBest=max(revenue[basePrice:maxprice])
priceBest=which(revenue == revenueBest)
axis(side = 1, at = priceBest) 
lines(c(priceBest[1],priceBest[1]),c(0, revenueBest),lty=2)
axis(side = 2, at = round(revenueBest,3),las=1)
lines(c(0,priceBest[1]),c(revenueBest, revenueBest),lty=2)


