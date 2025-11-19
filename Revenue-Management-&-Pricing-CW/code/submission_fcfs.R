# Inputs
mL <- c(28, 23, 15, 8, 8, 10, 10, 15, 23, 28)               # Mean demand for low-fare (single bookings)
mH <- c(7, 6, 3, 2, 2, 2, 2, 4, 6, 7)                       # Mean demand for high-fare (group bookings)
pL <- c(7, 7, 7, 7, 7, 7, 7, 7, 7, 7)                   # Price for low-fare
pH <- c(20, 20, 20, 20, 20, 20, 20, 20, 20, 20)             # Price for high-fare
capacity <- 20                                             # Capacity per time slot

# Initialize storage
num_slots <- length(mL)                                    # Number of time slots
LowerBoundExpectedRevenue <- rep(0, num_slots)              # Lower Bound Expected Revenue (FCFS) per time slot

# Iterate over time slots
for (t in 1:num_slots) {
  ExpRevenue <- rep(0, capacity + 1)   # Storage for expected revenue given protection levels
  
  # Loop over protection levels (from 0 to capacity)
  for (i in 1:(capacity + 1)) {
    protect <- i - 1                   # Protection level for this iteration
    availforLowFare <- capacity - protect  # Available capacity for low-fare
    
    ExpRevenue[i] <- 0  # Initialize expected revenue for this protection level
    
    # Loop over possible low-fare demand
    for (dL in 0:200) {
      soldLowFare <- min(availforLowFare, dL)  # Sold low-fare tickets
      remainforHighFare <- capacity - soldLowFare  # Remaining capacity for high-fare
      
      # Loop over possible high-fare demand
      for (dH in 0:200) {
        soldHighFare <- min(remainforHighFare, dH)  # Sold high-fare tickets
        RevenueThisIter <- pL[t] * soldLowFare + pH[t] * soldHighFare  # Revenue for this iteration
        
        # Add expected revenue for this demand scenario
        ExpRevenue[i] <- ExpRevenue[i] + RevenueThisIter * dpois(dL, mL[t]) * dpois(dH, mH[t])  # Using Poisson distributions for demand
      }
    }
  }
  
  # Lower bound for expected revenue (FCFS) for this time slot
  LowerBoundExpectedRevenue[t] <- ExpRevenue[1]
}

# Output results
cat("Lower Bound for Expected Revenue (FCFS) per Time Slot:\n")
print(LowerBoundExpectedRevenue)
print(sum(LowerBoundExpectedRevenue))
