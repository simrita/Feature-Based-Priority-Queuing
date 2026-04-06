list.of.packages <- c('zoo','ggplot2','magrittr','dplyr','corrplot','stats','reshape2','tidyr','knitr','readxl','lubridate','tinytex','writexl','gtools')
print(list.of.packages)
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
invisible(if(length(new.packages)) install.packages(new.packages))
options(error=utils::recover)
suppressWarnings(suppressMessages(lapply(list.of.packages,function(x) library(x, character.only=TRUE,quietly = TRUE))))