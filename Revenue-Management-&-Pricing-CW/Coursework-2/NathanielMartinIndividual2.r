library(lattice)
library(stargazer)
library(nloptr)

congestion <- read.csv("CongestionPricing.csv")
N <- nrow(congestion)

congestion$maxWTP = apply(congestion[,2:3],1,max)

### 1a

maxprice <- max(c(congestion$Peak_WTP, congestion$Nonpeak_WTP))
demand <- rep(NA, maxprice)
revenue <- rep(NA, maxprice)

for (p in 1:maxprice) {
  demand[p] <- sum(congestion$maxWTP >= p)
  revenue[p] <- p * demand[p]
}

#Calculate best revenue and price
revenueBest <- max(revenue) * (192000 / 345)
priceBest <- which.max(revenue)

print(paste("The optimal single price is:", priceBest))
print(paste("The maximum revenue is:", revenueBest))

#Emissions for single price
num_cars <- demand[priceBest] * (192000 / 345)
avg_speed <- 30 - 0.0625 * (num_cars / 1000)

emissions_per_car <- if (avg_speed < 25) {
  617.5 - 16.7 * avg_speed
} else {
  235.0 - 1.4 * avg_speed
}

#Calculate total emissions
total_em <- emissions_per_car * num_cars

print(paste("The emission level:", round(total_em, 2), "g/km"))

### 1b

maxprice=max(congestion[2:3])

basePrice=7

demandNonPeak<-rep(0,maxprice)
demandPeak<-rep(0,maxprice)
revenue_B<-rep(0,maxprice)

#Calculate surplus
maxsurplusNonPeak<-rep(0,N)

for (i in 1:N){
    maxsurplusNonPeak[i]=max(congestion[i,c(3)]-basePrice)
    congestion$maxsurplusNonPeak[i]=max(congestion[i,c(3)]-basePrice)
}

surplusPeak<-matrix(0,N,maxprice)

for (p in 1:maxprice){
    for (i in 1:N){
        surplusPeak[i,p]=congestion[i,2]-p
    }
}
colnames(surplusPeak)=paste0("p=",1:maxprice)
surplusPeak[1:10,c(1,2,3,4,5,6,7,8,9,10)]

for (p in 1:maxprice){
  demandNonPeak[p]=sum((maxsurplusNonPeak>surplusPeak[,p])*(maxsurplusNonPeak>=0))
  demandPeak[p]=sum((surplusPeak[,p]>=maxsurplusNonPeak)*(surplusPeak[,p]>=0))
  revenue_B[p]=basePrice*demandNonPeak[p]+p*demandPeak[p]
}

peak_revenue_B <- max(revenue_B) * (192000 / 345)
peak_price_B <- which.max(revenue_B)

print(paste("The optimal peak price is:", peak_price_B))
print(paste("The maximum revenue with peak pricing is:", round(peak_revenue_B, 2)))

#Emissions for peak price model
num_cars_peak <- demandPeak[peak_price_B] * (192000 / 345)
num_cars_non_peak <- demandNonPeak[peak_price_B] * (192000 / 345)
total_cars <- num_cars_peak + num_cars_non_peak

avg_speed <- 30 - 0.0625 * (total_cars / 1000)
emissions_per_car <- if (avg_speed < 25) {
  617.5 - 16.7 * avg_speed
} else {
  235.0 - 1.4 * avg_speed
}

#Calculate emissions for peak price
total_emissions_peak <- emissions_per_car * total_cars

print(paste("The total emissions with peak pricing are:", round(total_emissions_peak, 2), "g/km"))


### 1c

#Calculate total emissions for all peak prices
total_emissions_peak <- numeric(maxprice)

for (p in 1:maxprice) {
  total_cars <- demandNonPeak[p] * (192000 / 345) + demandPeak[p] * (192000 / 345)

  avg_speed <- 30 - 0.0625 * (total_cars / 1000)

  emissions_per_car <- if (avg_speed < 25) {
    617.5 - 16.7 * avg_speed
  } else {
    235.0 - 1.4 * avg_speed
  }

  total_emissions_peak[p] <- emissions_per_car * total_cars
}

#Create a data frame
Qc_df <- data.frame(
  peak_price_B = 1:maxprice,
  demandNonPeak = demandNonPeak,
  demandPeak = demandPeak,
  revenue_B = revenue_B,
  total_emissions_peak = total_emissions_peak
)

#Filter data for revenue > 1,100,000
Qc_df_filtered <- subset(Qc_df, revenue_B > 1100000)

minimum_emissions_row <- Qc_df_filtered[which.min(Qc_df_filtered$total_emissions_peak), ]

peak_price_minimum_emission <- minimum_emissions_row$peak_price_B
revenue_minimum_emission <- minimum_emissions_row$revenue_B
minimum_emissions <- minimum_emissions_row$total_emissions_peak

print(paste("The recommended peak price to minimize emissions is:", peak_price_minimum_emission))
print(paste("The revenue at the recommended price is:", revenue_minimum_emission))
print(paste("The emissions at the recommended price are:", round(minimum_emissions, 2), "g/km"))
