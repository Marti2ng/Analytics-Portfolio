library(shiny)
library(rpart)
library(caret)
library(smotefamily)
library(mlbench)
library(rpart.plot)
library(rattle)
library(randomForest)
library(ggplot2)
library(themis)
library(pROC)

# Load dataset
data(PimaIndiansDiabetes)
set.seed(123)  # Ensures reproducibility
pima<- PimaIndiansDiabetes[sample(nrow(PimaIndiansDiabetes), 400, replace = FALSE), ]
pima <- na.omit(pima)  # Remove missing values
pima$diabetes <- as.factor(pima$diabetes)  # Ensure factor type


# Split into training (85%) & testing (15%) sets
set.seed(345)
train.index <- createDataPartition(pima$diabetes, p = 0.85, list = FALSE)
train <- pima[train.index, ]
test <- pima[-train.index, ]

set.seed(345)  # Ensure reproducibility

# Create a SMOTE preprocessing - for equal class representation
recipe_smote <- recipe(diabetes ~ ., data = train) %>%
  step_smote(diabetes, over_ratio = 1) %>%  # Ensures equal balance
  prep() %>%
  juice()  # Extract the balanced dataset

# Convert back to factor
recipe_smote$diabetes <- as.factor(recipe_smote$diabetes)

# Check the new class distribution
table(recipe_smote$diabetes)
train_smote <- recipe_smote

# Check the new class distribution
table(train_smote$diabetes)

### Provided over representation of the minority class 
# Apply SMOTE to balance classes
##set.seed(345)
##smote_data <- SMOTE(train[, -ncol(train)], train$diabetes, K = 2, dup_size = 1)
##train_smote <- smote_data$data
##colnames(train_smote)[ncol(train_smote)] <- "diabetes"
##train_smote$diabetes <- as.factor(train_smote$diabetes)  

# Store variable names for user selection
available_vars <- colnames(train_smote[, -ncol(train_smote)])

test$diabetes <- factor(test$diabetes, levels = c("neg", "pos"))
train_smote$diabetes <- factor(train_smote$diabetes, levels = c("neg", "pos"))

# Define UI for Shiny App

ui <- fluidPage(
  titlePanel("Interactive Decision Tree & Random Forest Model for Diabetes Prediction"),
  
  tabsetPanel(
    # Decision Tree Tab
    tabPanel("Decision Tree",
             sidebarLayout(
               sidebarPanel(
                 sliderInput("maxdepth", "Max Depth:", min = 1, max = 10, value = 1),
                 sliderInput("cp", "Complexity Parameter (cp):", min = 0.0001, max = 0.05, value = 0.0001, step = 0.0005),
                 htmlOutput("metric_explanation")
               ),
               mainPanel(
                 plotOutput("decisionTreePlot"),
                 verbatimTextOutput("performance_metrics"),  
                 htmlOutput("dt_explanation")
               )
             )
    ),
    
    # Random Forest Tab
    tabPanel("Random Forest",
             sidebarLayout(
               sidebarPanel(
                 sliderInput("mtry", "Number of Variables at Each Split (mtry):", min = 1, max = 8, value = 1),
                 sliderInput("nodesize", "Minimum Node Size:", min = 1, max = 10, value = 1),
                 sliderInput("ntree", "Number of Trees:", min = 100, max = 1000, value = 100, step = 100)
               ),
               mainPanel(
                 plotOutput("feature_importance"),  
                 verbatimTextOutput("rf_test_accuracy"),
                 htmlOutput("rf_explanation")
               )
             )
    ),
    
    # Classification Boundary Tab
    tabPanel("Classification Boundary",
             sidebarLayout(
               sidebarPanel(
                 selectInput("model_type", "Select Model:", choices = c("Decision Tree", "Random Forest")),
                 selectInput("var1", "Choose First Variable:", choices = available_vars, selected = "glucose"),
                 selectInput("var2", "Choose Second Variable:", choices = available_vars, selected = "age"),
                 
                 conditionalPanel(
                   condition = "input.model_type == 'Decision Tree'",
                   sliderInput("cp", "Complexity Parameter (cp):", min = 0.0001, max = 0.05, value = 0.0001, step = 0.005),
                   sliderInput("maxdepth", "Max Depth:", min = 1, max = 10, value = 1)
                 ),
                 
                 conditionalPanel(
                   condition = "input.model_type == 'Random Forest'",
                   sliderInput("nodesize", "Minimum Node Size:", min = 1, max = 10, value = 1),
                   sliderInput("ntree", "Number of Trees:", min = 100, max = 1000, value = 100, step = 100),
                   sliderInput("mtry", "Number of Variables at Each Split (mtry):", min = 1, max = 8, value = 1)
                 ),
                 
                 actionButton("update_model", "Update Classification Boundary")
               ),
               mainPanel(
                 plotOutput("classificationBoundary")
               )
             )
    )
  )
)


# Define Server logic

server<- function(input, output, session){
  
  #Metric
  output$metric_explanation <- renderUI({
    HTML(paste0(
      "<h4 style='margin-top: 0;'>Metric Definitions:</h4>",
      
      "<b>Test Accuracy:</b> Measures overall correctness. <br><br>",
      
      "<b>Precision:</b> Measures how many of the positive predictions were actually correct.<br><br>",
      
      "<b>Recall (Sensitivity):</b> Measures how well the model captures actual positives. <br><br>",
      
      "<b>F1-score:</b> Balance between Precision & Recall, signifying better overall performance. <br><br>",
      
      "<b>AUC score:</b> It measures the entire 2D area underneath the ROC curve.<br><br>",
      
      "</div>"
    ))
  })
  
  ## DECISION TREE
  # Train the Decision Tree Model 
  dt_model <- reactive({
    req(input$maxdepth, input$cp)  
    
    fit_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
    
    pima.rpart <- train(
      diabetes ~ ., 
      data = train_smote, 
      method = "rpart", 
      tuneGrid = data.frame(cp = input$cp),
      trControl = fit_control,
      control = rpart.control(minsplit = 2, minbucket = 1, maxdepth = input$maxdepth)
    )
    
    return(pima.rpart)  
  })
  
  # Plot Decision Tree Reactively
  output$decisionTreePlot <- renderPlot({
    dt <- dt_model()$finalModel  
    if (inherits(dt, "rpart")) {
      fancyRpartPlot(dt, main = "Tuned Decision Tree", sub="10-fold Cross Validation")  
    } else {
      plot.new()
      text(0.5, 0.5, "Tree model is not valid.", cex = 1.5)
    }
  })
  
  #Decision Tree Metrics
  output$performance_metrics <- renderText({
    req(dt_model())  
    model <- dt_model()$finalModel  
    pred_prob <- predict(model, test[, -ncol(test)], type = "prob")[, 2]  # Get probability for class "positive"
    roc_obj <- roc(test$diabetes, pred_prob, levels = c("neg", "pos"), direction = "<")
    auc_val <- auc(roc_obj)  # Compute AUC
    
    pred <- predict(model, test[, -ncol(test)], type = "class")
    test_acc <- mean(pred == test$diabetes)
    conf_matrix <- confusionMatrix(pred, test$diabetes)
    precision <- conf_matrix$byClass["Precision"]
    recall <- conf_matrix$byClass["Recall"]
    f1 <- conf_matrix$byClass["F1"]
    
    paste("Test Accuracy:", round(test_acc * 100, 2), "%\n",
          "Precision:", round(precision*100, 2), "%\n",
          "Recall:", round(recall*100, 2), "%\n",
          "F1-score:", round(f1*100, 2), "%\n",
          "AUC:", round(auc_val, 2), "\n")
  })
  
  # Decision Tree Explanation of Parameters
  output$dt_explanation <- renderUI({
    HTML(paste0(
      "<h4>Explanation of Decision Tree Parameters:</h4>",
      "<b>Max Depth </b>: Controls the depth of the decision tree. ",
      "Deeper trees capture more details but risk overfitting.<br><br>",
      
      "<b>Complexity Parameter CP</b>: ",
      "Used for pruning trees. Lower values allow more splits, leading to complex trees, ",
      "while higher values prevent unnecessary complexity and reduce overfitting."
    ))
  })
  
  ## RANDOM FOREST
  # Train Random Forest Model
  rf_model <- reactive({
    req(input$ntree, input$mtry, input$nodesize)  
    
    set.seed(123) 
    
    rf <- randomForest(
      diabetes ~ ., 
      data = train_smote, 
      ntree = input$ntree, 
      mtry = input$mtry, 
      nodesize = input$nodesize,
      importance = TRUE  
    )
    
    return(rf)  
  })
  
  # Display Feature Importance Plot
  output$feature_importance <- renderPlot({
    req(rf_model())  
    
    model <- rf_model() 
    
    # Extract Importance Metrics
    importance_df <- as.data.frame(model$importance)

    # Plot Importance
    varImpPlot(model, main = "Feature Importance in Random Forest", n.var = 8)
  })
  
  # Random Forest Test Accuracy
  output$rf_test_accuracy <- renderText({
    req(rf_model())  
    model <- rf_model() 
    pred_prob <- predict(model, test[, -ncol(test)], type = "prob")[, 2]  # Get probability for class "positive"
    roc_obj <- roc(test$diabetes, pred_prob, levels = c("neg", "pos"), direction = "<")
    auc_val <- auc(roc_obj)  # Compute AUC
    
    pred <- predict(model, test[, -ncol(test)], type = "class")
    test_acc <- mean(pred == test$diabetes)
    conf_matrix <- confusionMatrix(pred, test$diabetes)
    precision <- conf_matrix$byClass["Precision"]
    recall <- conf_matrix$byClass["Recall"]
    f1 <- conf_matrix$byClass["F1"]
    
    paste("Test Accuracy:", round(test_acc * 100, 2), "%\n",
          "Precision:", round(precision*100, 2), "%\n",
          "Recall:", round(recall*100, 2), "%\n",
          "F1-score:", round(f1*100, 2), "%\n",
          "AUC:", round(auc_val, 2), "\n")
  
  })
  
  # Explanation of Random Forest Hyperparameters
  output$rf_explanation <- renderUI({
    HTML(paste0(
      "<h4>Explanation of Random Forest Parameters:</h4>",
      "<b>Number of Variables at Each Split (mtry)</b>: ",
      "Determines how many variables are randomly selected at each tree split. ",
      "Higher values increase model complexity but can lead to overfitting.<br><br>",
      
      "<b>Minimum Node Size (nodesize)</b>: ",
      "Specifies the minimum number of samples required in a leaf node. ",
      "Larger values create simpler trees, while smaller values create more detailed trees.<br><br>",
      
      "<b>Number of Trees (ntree)</b>: ",
      "Defines the total number of trees in the Random Forest. ",
      "A higher number of trees generally improves accuracy but increases computation time. ",
      "Lower values might lead to less stable predictions."
    ))
  })
  
  ## Classification Boundary
  # Ensure that selected variables exist in the dataset
  observe({
    updateSelectInput(session, "var1", choices = available_vars, selected = input$var1)
    updateSelectInput(session, "var2", choices = available_vars, selected = input$var2)
  })
  
  # Train Model Reactively Based on User Selection
  trained_model <- eventReactive(input$update_model, {
    req(input$model_type, input$var1, input$var2)
    
    formula <- as.formula(paste("diabetes ~", input$var1, "+", input$var2))
    
    if (input$model_type == "Decision Tree") {
      model <- rpart(
        formula,
        data = train_smote,
        control = rpart.control(cp = input$cp, maxdepth = input$maxdepth)
      )
    } else {
      model <- randomForest(
        formula,
        data = train_smote,
        ntree = input$ntree,
        mtry = input$mtry,
        nodesize = input$nodesize
      )
    }
    
    return(model)
  })
  
  # Generate Classification Boundary Dynamically
  output$classificationBoundary <- renderPlot({
    model <- trained_model()
    req(model)
    if (!(input$var1 %in% names(train_smote)) | !(input$var2 %in% names(train_smote))) {
      plot.new()
      text(0.5, 0.5, "Error: Selected variables do not exist in dataset", cex = 1.5)
      return(NULL)
    }
    
    # Create a grid of values for selected variables
    x_range <- seq(min(train_smote[[input$var1]], na.rm = TRUE), max(train_smote[[input$var1]], na.rm = TRUE), length.out = 200)
    y_range <- seq(min(train_smote[[input$var2]], na.rm = TRUE), max(train_smote[[input$var2]], na.rm = TRUE), length.out = 200)
    grid_data <- expand.grid(x = x_range, y = y_range)
    colnames(grid_data) <- c(input$var1, input$var2)
    
    if (input$model_type == "Decision Tree") {
      pred_probs <- predict(model, grid_data, type = "class") 
    } else {
      pred_probs <- predict(model, grid_data, type = "response")  
    }
    
    grid_data$Prediction <- as.factor(pred_probs)
    
    # Plot Classification Boundary
    ggplot() +
      geom_tile(data = grid_data, aes_string(x = input$var1, y = input$var2, fill = "Prediction"), alpha = 0.3) +
      geom_point(data = train_smote, aes_string(x = input$var1, y = input$var2, color = "diabetes"), size = 2) +
      scale_fill_manual(values = c("blue", "red")) +
      scale_color_manual(values = c("blue", "red")) +
      labs(title = paste(input$model_type, "- Dynamic Classification Boundary"), x = input$var1, y = input$var2) +
      theme_minimal()
  })
  
}

# Run Shiny App
shinyApp(ui = ui, server = server)