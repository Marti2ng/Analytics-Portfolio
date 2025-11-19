m_early <- 32               # Mean demand for Early Res (Poisson)
m_regular <- 18             # Mean demand for Regular Res (Poisson)
price_early <- 195 * .8     # Price for Early Res w/ 20% discount
price_reg <- 195            # Price for Regular Res
capacity <- 23              # Capacity
exp_revenue <- rep(0, capacity + 1)

# First Come First Serve Revenue
for (i in 1:1) {
  protect <- i - 1
  avail_early <- capacity - protect
  exp_revenue[i] <- 0

  # Loop for possible early demand (dE)
  for (dE in 0:200){
    sold_early <- min(avail_early, dE) # Rooms sold to Early Customers
    remaining_for_reg <- capacity - sold_early

    # Loop for possible regular demand (dR)
    for (dR in 0:200){
      sold_reg <- min(remaining_for_reg, dR)
      rev_this_iter <- price_early * sold_early + price_reg * sold_reg

      exp_revenue[i] <- exp_revenue[i] + rev_this_iter * dpois(dE, m_early) * dpois(dR, m_regular)
    }
  }
}

# (A) FCFS Expected Revenue
rev_FCFS <- exp_revenue[1]
print(paste("Expected Daily Revenue (FCFS): £", round(rev_FCFS, 1)))

# Maximizing Expected Revenue by Reserving Rooms for Regular Arrivals

max_revenue <- -Inf         # Initialize max revenue to a very low number

best_protection <- 0        # Variable to track the best number of reserved rooms

# Loop over all possible protection levels (reserving rooms for regular customers)
for (protect in 0:capacity) {
  avail_early <- capacity - protect
  exp_revenue[protect + 1] <- 0

  # Loop for possible early demand (dE)
  for (dE in 0:200) {
    sold_early <- min(avail_early, dE)  # Rooms sold to Early Res
    remaining_for_reg <- capacity - sold_early

    # Loop for possible regular demand (dR)
    for (dR in 0:200) {
      sold_reg <- min(remaining_for_reg, dR)  # Rooms sold to Regular Res
      rev_this_iter <- price_early * sold_early + price_reg * sold_reg

      # Update expected revenue for this protection level
      exp_revenue[protect + 1] <- exp_revenue[protect + 1] + rev_this_iter * dpois(dE, m_early) * dpois(dR, m_regular)
    }
  }

  # Check if this protection level yields higher revenue
  if (exp_revenue[protect + 1] > max_revenue) {
    max_revenue <- exp_revenue[protect + 1]
    best_protection <- protect
  }
}

# (b) Optimal Reservation Policy (Maximizing Expected Revenue)
Protectindexbest <- which(exp_revenue == max(exp_revenue))  # Find the best protection index
ProtectBest <- Protectindexbest - 1                         # The optimal number of rooms to reserve
OptimalExpRevenue <- max(exp_revenue)                       # The maximum expected revenue

# Calculate the percentage improvement over FCFS
PercentImprovement <- ((OptimalExpRevenue - rev_FCFS) / rev_FCFS) * 100

# Output Results
print(paste("The Optimal Number of Rooms to Reserve for Regular Arrivals:", ProtectBest))
print(paste("Expected Daily Revenue with Optimal Protection: £", round(OptimalExpRevenue, 1)))
print(paste("Percent Improvement over FCFS:", round(PercentImprovement, 2), "%"))