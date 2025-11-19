# Inputs
mL <- c(28, 23, 15, 8, 8, 10, 10, 15, 23, 28)  # Mean demand for low-fare (single bookings)
mH <- c(7, 6, 3, 2, 2, 2, 2, 4, 6, 7)         # Mean demand for high-fare (group bookings)
pL <- c(10, 10, 7, 7, 7, 7, 7, 7, 10, 10)               # Price for low-fare
pH <- c(26, 26, 20, 20, 20, 20, 20, 20, 26, 26)         # Price for high-fare
capacity <- 20                                         # Capacity per time slot

# Initialize storage
num_slots <- length(mL)
OptimalProtectionLevels <- rep(0, num_slots)
ExpectedRevenuePerSlot <- rep(0, num_slots)

# Iterate over time slots
for (t in 1:num_slots) {
  ExpRevenue <- rep(0, capacity + 1)
  
  for (i in 1:(capacity + 1)) {
    protect <- i - 1
    availforLowFare <- capacity - protect
    ExpRevenue[i] <- 0
    
    for (dL in 0:200) {  # Low-fare demand loop
      soldLowFare <- min(availforLowFare, dL)
      remainforHighFare <- capacity - soldLowFare
      
      for (dH in 0:200) {  # High-fare demand loop
        soldHighFare <- min(remainforHighFare, dH)
        RevenueThisIter <- pL[t] * soldLowFare + pH[t] * soldHighFare
        ExpRevenue[i] <- ExpRevenue[i] +
          RevenueThisIter * dpois(dL, mL[t]) * dpois(dH, mH[t])
      }
    }
  }
  
  # Optimal protection level for this time slot
  Protectindexbest <- which(ExpRevenue == max(ExpRevenue))
  ProtectBest <- Protectindexbest - 1
  OptimalProtectionLevels[t] <- ProtectBest
  
  # Expected revenue for this time slot
  ExpectedRevenuePerSlot[t] <- max(ExpRevenue)
}

# Total expected revenue for the day
TotalExpectedRevenue <- sum(ExpectedRevenuePerSlot)

# Output results
cat("Optimal Protection Levels for High-Fare Demand (Group Bookings) per Time Slot:\n")
print(OptimalProtectionLevels)
cat("\nExpected Revenue per Time Slot:\n")
print(ExpectedRevenuePerSlot)
cat("\nTotal Expected Revenue for the Day:\n")
print(TotalExpectedRevenue)
