### IMDB Movie Clustering using PCA and Unsupervised Learning
## Loading Libraries

library(tidyverse)  # Includes ggplot2, dplyr, and more for data manipulation & visualization
library(dplyr)      # For data manipulation
library(ggplot2)    # For visualization
library(caret)      # For data preprocessing (e.g., scaling)
library(cluster)    # For clustering and silhouette analysis
library(factoextra) # For clustering visualization and PCA tools
library(corrplot) 
library(dbscan)

## Loading the Data
df <- read.csv("IMDB-Movies.csv") %>%
  drop_na()  # Remove rows with missing values

str(df)      # Show structure: variable types and sample data
summary(df)  # Get basic summary stats (mean, min, max, etc.)

# Select only numeric columns, removing 'Year'
# Remove 'X' column as it's just an observation ID with no analytical value
pca_data <- df %>% 
  select_if(is.numeric) %>%
  select(-Year, -X)

# Visualize the correlation matrix
cor_matrix <- cor(pca_data)  # Compute correlation matrix
corrplot(cor_matrix, method = "color", type = "upper",
         tl.cex = 0.8, tl.col = "black",
         col = colorRampPalette(c("blue", "white", "red"))(200),
         addCoef.col = "black", number.cex = 0.8,
         title = "Correlation between Variables", mar = c(0, 0, 2, 0))

## Histograms
df_long <- pca_data %>%
  pivot_longer(cols = where(is.numeric), names_to = "variable", values_to = "value")

ggplot(df_long, aes(x = value)) +
  geom_histogram(fill = "blue", bins = 30, alpha = 0.7) +
  facet_wrap(~variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of Numeric Variables")

# Standardize (center & scale) the numeric data
preprocess_params <- preProcess(pca_data, method = c("center", "scale"))
pca_scaled <- predict(preprocess_params, pca_data)

# Run PCA
# PCA is used to reduce dimensionality while preserving variance. It also aids in visualising the dataset and improves clustering stability by removing collinearity.

pca_model <- prcomp(pca_scaled, center = TRUE, scale. = TRUE)
summary(pca_model)      # View explained variance by each PC
pca_model$rotation       # View loadings (contribution of variables)

# Create a dataframe with PCA results and add movie titles
pca_df <- data.frame(
  PC1 = pca_model$x[, 1],
  PC2 = pca_model$x[, 2],
  Title = df$Title[as.numeric(rownames(pca_scaled))]
)

# Scatter plot of PCA results, with movie titles as labels
ggplot(pca_df, aes(x = PC1, y = PC2, label = Title)) +
  geom_point(alpha = 0.7, size = 3, color = "steelblue") +
  geom_text(size = 2, vjust = 1.5, check_overlap = TRUE) +
  labs(title = "PCA of IMDB Movies Data",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal()

# Clustering: K-Means
# K-Means is efficient and works well with PCA-transformed (Euclidean) data. We determine optimal k using Elbow and Silhouette methods.

# Elbow Method 
fviz_nbclust(pca_df[, 1:2], kmeans, method = "wss", k.max = 10) +
  labs(title = "Elbow Method for K-Means Clustering")


# Silhoutte Score
sil_scores <- c()       # Empty vector to store scores
k_values <- 2:6         # Range of k values to try

for (k in k_values) {
  set.seed(42)
  km <- kmeans(pca_df[, 1:2], centers = k, nstart = 25)
  sil <- silhouette(km$cluster, dist(pca_df[, 1:2]))
  sil_scores <- c(sil_scores, mean(sil[, 3]))  # Store avg silhouette width
}

sil_df <- data.frame(k = k_values, Silhouette = sil_scores)

# --- Silhouette Scores for K-Means --- #
cat("Silhouette Scores for each k:\n")
print(sil_df)

# Identify the k with highest silhouette
best_k <- sil_df$k[which.max(sil_df$Silhouette)]
best_score <- max(sil_df$Silhouette)

cat(paste("Optimal number of clusters based on silhouette score is k =", best_k,
          "with an average silhouette of", round(best_score, 3), "\n"))


ggplot(sil_df, aes(x = k, y = Silhouette)) +
  geom_line() +
  geom_point(size = 3) +
  labs(title = "Average Silhouette Score for Different k",
       x = "Number of Clusters (k)",
       y = "Average Silhouette Width") +
  theme_minimal()

# Clustering K mean (k = 5)
kmeans_result <- kmeans(pca_df[, 1:2], centers = 5, nstart = 25) ## Run K-Means 25 times, each with a different random starting point. Keep the best result
pca_df$KMeans_Cluster <- as.factor(kmeans_result$cluster)

## Method 2: Gives k=4, less control with this method------------------
## fviz_nbclust(pca_df[, 1:2], kmeans, method='silhouette') 
### -------------------------------------------------------------------

## Total Within Cluster Sum of Square
cat("Total Within-Cluster Sum of Square:",kmeans_result$tot.withinss)

# Silhouette plot
fviz_silhouette(silhouette(kmeans_result$cluster, dist(pca_df[, 1:2]))) +
  labs(title = "Silhouette Plot for K-Means")

# Clustered scatter plot
ggplot(pca_df, aes(x = PC1, y = PC2, color = KMeans_Cluster, label = Title)) +
  geom_point(size = 3) +
  geom_text(size = 2, vjust = 1.5, check_overlap = TRUE) +
  labs(title = "K-Means Clustering (k = 5)",
       color = "Cluster") +
  theme_minimal()

# Clustering : Hierarchical Clustering ---------------------------------------------
# Elbow Method
fviz_nbclust(pca_df[, 1:2], FUN = hcut, method = "wss", k.max = 10) +
  labs(title = "Elbow Method for Hierarchical Clustering")

# Silhoutte Scores
sil_scores_hc <- c()
k_values <- 2:8

for (k in k_values) {
  hc <- hclust(dist(pca_df[, 1:2]), method = "complete")
  clusters <- cutree(hc, k)
  sil <- silhouette(clusters, dist(pca_df[, 1:2]))
  sil_scores_hc <- c(sil_scores_hc, mean(sil[, 3]))
}

sil_df_hc <- data.frame(k = k_values, Silhouette = sil_scores_hc)

ggplot(sil_df_hc, aes(x = k, y = Silhouette)) +
  geom_line() +
  geom_point(size = 3) +
  labs(title = "Silhouette Score for Complete Hierarchical Clustering",
       x = "Number of Clusters (k)",
       y = "Average Silhouette Width") +
  theme_minimal()

sil_scores_hc2 <- c()
k_values2 <- 2:8

for (k in k_values2) {
  hc2 <- hclust(dist(pca_df[, 1:2]), method = "average")
  clusters2 <- cutree(hc2, k)
  sil <- silhouette(clusters2, dist(pca_df[, 1:2]))  # <- fixed here
  sil_scores_hc2 <- c(sil_scores_hc2, mean(sil[, 3]))
}

sil_df_hc2 <- data.frame(k = k_values2, Silhouette = sil_scores_hc2)

ggplot(sil_df_hc2, aes(x = k, y = Silhouette)) +
  geom_line() +
  geom_point(size = 3) +
  labs(title = "Silhouette Score for Average Hierarchical Clustering",
       x = "Number of Clusters (k)",
       y = "Average Silhouette Width") +
  theme_minimal()

d <- dist(pca_df[, 1:2])

hc_complete <- hclust(d, method = "complete")
hc_average <- hclust(d, method = "average")

fviz_dend(hc_complete,
          rect = TRUE,
          main = "Dendrogram - Complete Linkage",
          show_labels = TRUE,
          cex = 0.5)

fviz_dend(hc_average,
          rect = TRUE,
          main = "Dendrogram - Average Linkage",
          show_labels = TRUE,
          cex = 0.5)

# Hierarchical Clustering (k=2) 
## For average and complete method of hierarchical clustering the silhoutte score suggests, k=2
# discarding the single method as it isolates single observations and doesn't provide much insights for group clustering
# # Hierarchical clustering offers an alternative perspective with no need to pre-specify k, although cut height is needed. Compared across linkage methods.

## Shouldn't just look at Silhoutte Scores alone, including the WCSS_Score
# We are now looking for the best value of k and model that provides the best balance between clustering metrics

total_wcss <- function(data, clusters) {
  total <- 0
  
  for (cluster in unique(clusters)) {
    points <- data[clusters == cluster, , drop = FALSE]
    center <- colMeans(points)
    total <- total + sum(rowSums((points - center)^2))
  }

  return(total)
}

evaluate_hierarchical_clustering <- function(data, methods = c("complete", "average"), k_range = 2:8) {
  d <- dist(data)
  all_results <- data.frame()
  
  for (m in methods) {
    sil_scores <- c()
    wcss_scores <- c()
    
    for (k in k_range) {
      hc <- hclust(d, method = m)
      clusters <- cutree(hc, k = k)
      sil <- silhouette(clusters, d)
      sil_scores <- c(sil_scores, mean(sil[, 3]))
      wcss_scores <- c(wcss_scores, total_wcss(data, clusters))
    }
    
    temp_df <- data.frame(
      Method = m,
      k = k_range,
      Silhouette = round(sil_scores, 3),
      WCSS = round(wcss_scores, 3)
    )
    
    all_results <- rbind(all_results, temp_df)
  }
  
  return(all_results)
}

# Run the evaluation
results <- evaluate_hierarchical_clustering(pca_df[, 1:2])

# Print results
print(results)

# Find the best combination
best_result <- results[which.max(results$Silhouette - scale(results$WCSS)), ]
cat("\nBest Hierarchical Clustering Configuration:\n")
print(best_result)

# Plot silhouette scores for best method
best_method <- best_result$Method
best_data <- subset(results, Method == best_method)

rownames(pca_df) <- pca_df$Title
hc <- hclust(dist(pca_df[, 1:2]), method = "complete") 
hc_clusters <- cutree(hc, k = 6)
pca_df$Hier_Cluster <- as.factor(hc_clusters)

# Dendrogram
fviz_dend(hc,
          k = 6,
          cex = 0.7,
          show_labels = TRUE,
          labels_track_height = 3,
          main = "Hierarchical Clustering of IMDB Movies")

# PCA Plot with Different Clustering Group
fviz_silhouette(silhouette(hc_clusters, dist(pca_df[, 1:2]))) +
  labs(title = "Silhouette Plot for Hierarchical Clustering")

# Plot clusters
ggplot(pca_df, aes(x = PC1, y = PC2, color = Hier_Cluster, label = Title)) +
  geom_point(size = 3) +
  geom_text(size = 2, vjust = 1.5, check_overlap = TRUE) +
  labs(title = "Hierarchical Clustering (k = 6)",
       color = "Cluster") +
  theme_minimal()

# Metadata / Summary Stats
# Understand what distinguishes the clusters beyond numeric similarity (e.g., genre, rating, popularity)

# K-means
combined_df <- df %>%
  mutate(Title = as.character(Title)) %>%
  inner_join(pca_df[, c("Title", "KMeans_Cluster")], by = "Title")

cluster_summary <- combined_df %>%
  group_by(KMeans_Cluster) %>%
  summarise(
    Count = n(),
    Avg_Rating = mean(Rating, na.rm = TRUE),
    Avg_Votes = mean(Votes, na.rm = TRUE),
    Avg_Revenue = mean(Revenue..Millions., na.rm = TRUE),
    Avg_Runtime = mean(Runtime..Minutes., na.rm = TRUE)
  )

print(cluster_summary)  # Export this to LaTeX table in report

# Movies in Each Cluster
split_movies <- split(combined_df$Title, combined_df$KMeans_Cluster)

for (i in names(split_movies)) {
  cat(paste("\nCluster", i, "Movies:\n"))
  print(split_movies[[i]])
}

# Hierarchical Clustering
combined_hc_df <- df %>%
  mutate(Title = as.character(Title)) %>%
  inner_join(pca_df[, c("Title", "Hier_Cluster")], by = "Title")

hc_summary <- combined_hc_df %>%
  group_by(Hier_Cluster) %>%
  summarise(
    Count = n(),
    Avg_Rating = mean(Rating, na.rm = TRUE),
    Avg_Votes = mean(Votes, na.rm = TRUE),
    Avg_Revenue = mean(Revenue..Millions., na.rm = TRUE),
    Avg_Runtime = mean(Runtime..Minutes., na.rm = TRUE)
  )

print(hc_summary)  # Also useful in LaTeX report

# Movie Titles per Hierarchical Cluster
split_hc_movies <- split(combined_hc_df$Title, combined_hc_df$Hier_Cluster)

for (i in names(split_hc_movies)) {
  cat(paste("\nHierarchical Cluster", i, "Movies:\n"))
  print(split_hc_movies[[i]])
}

## Graphs for K Clustering
# Boxplots of Ratings by Cluster
ggplot(combined_df, aes(x = as.factor(KMeans_Cluster), y = Rating, fill = as.factor(KMeans_Cluster))) +
  geom_boxplot() +
  labs(title = "Rating Distribution by K-Means Cluster",
       x = "Cluster", y = "Rating") +
  theme_minimal() +
  theme(legend.position = "none")

# Average Votes by Cluster
combined_df %>%
  group_by(KMeans_Cluster) %>%
  summarise(Avg_Votes = mean(Votes, na.rm = TRUE)) %>%
  ggplot(aes(x = as.factor(KMeans_Cluster), y = Avg_Votes, fill = as.factor(KMeans_Cluster))) +
  geom_col() +
  labs(title = "Average Popularity by K-Means Cluster",
       x = "Cluster", y = "Average Rating") +
  theme_minimal() +
  theme(legend.position = "none")

# Average Revenue by Cluster
combined_df %>%
  group_by(KMeans_Cluster) %>%
  summarise(Avg_Revenue = mean(Revenue..Millions., na.rm = TRUE)) %>%
  ggplot(aes(x = as.factor(KMeans_Cluster), y = Avg_Revenue, fill = as.factor(KMeans_Cluster))) +
  geom_col() +
  labs(title = "Average Revenue by K-Means Cluster",
       x = "Cluster", y = "Average Revenue (Millions)") +
  theme_minimal() +
  theme(legend.position = "none")


### -------- Appendix
# Determine optimal eps visually
kNNdistplot(pca_df[, 1:2], minPts = round(log(nrow(pca_df))))  
eps_val <- 1.2
abline(h = eps_val, lty = 2) 

minPts <- round(log(nrow(pca_df)))
# Run DBSCAN with selected parameters
dbscan_result <- dbscan(pca_df[, 1:2], eps = eps_val, minPts = minPts)

# Add DBSCAN cluster labels
dbscan_labels <- dbscan_result$cluster
pca_df$DBSCAN_Cluster <- as.factor(dbscan_labels)

# Plot DBSCAN clusters
ggplot(pca_df, aes(x = PC1, y = PC2, color = DBSCAN_Cluster, label = Title)) +
  geom_point(size = 3, alpha = 0.8) +
  geom_text(size = 2, vjust = 1.5, check_overlap = TRUE) +
  labs(title = "DBSCAN Clustering on PCA",
       color = "Cluster") +
  theme_minimal()

## Code from ChatGPT
# Load the dataset
imdb <- read.csv("IMDB-Movies.csv", stringsAsFactors = FALSE)

# Clean the data: remove non-numeric columns and NAs
imdb_clean <- imdb[, sapply(imdb, is.numeric)]
imdb_clean <- imdb_clean[ , !(names(imdb_clean) %in% c("X", "Year"))]
imdb_clean <- na.omit(imdb_clean)

# Perform PCA with scaling
imdb_pca <- prcomp(imdb_clean, scale. = TRUE)

# Print explained variance ratios
explained_variance <- summary(imdb_pca)$importance[2,]
print("Explained Variance Ratios:")
print(explained_variance)

# Print loadings for PC1 and PC2
loadings <- imdb_pca$rotation[, 1:2]
print("Loadings for PC1 and PC2:")
print(loadings)

# Scree plot
fviz_eig(imdb_pca, addlabels = TRUE, ylim = c(0, 50))

# Biplot
fviz_pca_biplot(imdb_pca, repel = TRUE, 
                col.var = "blue", col.ind = "gray30")

# Already done earlier
imdb_clean <- imdb[, sapply(imdb, is.numeric)]
imdb_clean <- imdb_clean[ , !(names(imdb_clean) %in% c("X", "Year"))]
imdb_clean <- na.omit(imdb_clean)

# Standardize the data (important for clustering)
imdb_scaled <- scale(imdb_clean)

# Elbow method
fviz_nbclust(imdb_scaled, kmeans, method = "wss") + 
  labs(title = "Elbow Method")

# Silhouette method
fviz_nbclust(imdb_scaled, kmeans, method = "silhouette") + 
  labs(title = "Silhouette Method")

# Set seed for reproducibility
set.seed(123)

# Apply K-means with k = 7
km_res <- kmeans(imdb_scaled, centers = 7, nstart = 25)

# Add cluster assignment to original data
imdb_clustered <- cbind(imdb_clean, Cluster = as.factor(km_res$cluster))

# Visualize clusters using PCA dimensions
fviz_cluster(km_res, data = imdb_scaled, geom = "point", ellipse.type = "norm", 
             main = "K-Means Clustering (k = 7)", palette = "jco")

# View mean values by cluster
aggregate(imdb_clean, by = list(Cluster = km_res$cluster), FUN = mean)

# Silhouette Plot
# Compute distance matrix
d <- dist(imdb_scaled)

# Generate silhouette object
sil <- silhouette(km_res$cluster, d)

# Plot silhouette
fviz_silhouette(sil) +
  labs(title = "Silhouette Plot for K-Means Clustering (k = 7)")
