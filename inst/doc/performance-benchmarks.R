## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----eval=FALSE---------------------------------------------------------------
# library(riemtan)
# library(Matrix)
# library(microbenchmark)  # For benchmarking
# 
# # Load AIRM metric
# data(airm)
# 
# # Create example dataset (100 matrices of size 50x50)
# set.seed(42)
# create_spd_matrix <- function(p) {
#   mat <- diag(p) + matrix(rnorm(p*p, 0, 0.1), p, p)
#   mat <- (mat + t(mat)) / 2
#   mat <- mat + diag(p) * 0.5
#   Matrix::pack(Matrix::Matrix(mat, sparse = FALSE))
# }
# 
# connectomes <- lapply(1:100, function(i) create_spd_matrix(50))

## ----eval=FALSE---------------------------------------------------------------
# # Create sample
# sample <- CSample$new(conns = connectomes, metric_obj = airm)
# 
# # Sequential baseline
# set_parallel_plan("sequential")
# time_seq <- system.time(sample$compute_tangents())
# print(paste("Sequential:", round(time_seq[3], 2), "seconds"))
# 
# # Parallel with 4 workers
# set_parallel_plan("multisession", workers = 4)
# time_par4 <- system.time(sample$compute_tangents())
# print(paste("Parallel (4 workers):", round(time_par4[3], 2), "seconds"))
# print(paste("Speedup:", round(time_seq[3] / time_par4[3], 2), "x"))
# 
# # Parallel with 8 workers
# set_parallel_plan("multisession", workers = 8)
# time_par8 <- system.time(sample$compute_tangents())
# print(paste("Parallel (8 workers):", round(time_par8[3], 2), "seconds"))
# print(paste("Speedup:", round(time_seq[3] / time_par8[3], 2), "x"))
# 
# # Reset
# reset_parallel_plan()

## ----eval=FALSE---------------------------------------------------------------
# # Function to benchmark Frechet mean
# benchmark_fmean <- function(n, workers = 1) {
#   conns_subset <- connectomes[1:n]
#   sample <- CSample$new(conns = conns_subset, metric_obj = airm)
# 
#   if (workers == 1) {
#     set_parallel_plan("sequential")
#   } else {
#     set_parallel_plan("multisession", workers = workers)
#   }
# 
#   time <- system.time(sample$compute_fmean(tol = 0.01, max_iter = 50))
#   reset_parallel_plan()
# 
#   time[3]
# }
# 
# # Benchmark different sample sizes
# sample_sizes <- c(20, 50, 100, 200)
# results <- data.frame(
#   n = sample_sizes,
#   sequential = sapply(sample_sizes, benchmark_fmean, workers = 1),
#   parallel_4 = sapply(sample_sizes, benchmark_fmean, workers = 4),
#   parallel_8 = sapply(sample_sizes, benchmark_fmean, workers = 8)
# )
# 
# # Calculate speedups
# results$speedup_4 = results$sequential / results$parallel_4
# results$speedup_8 = results$sequential / results$parallel_8
# 
# print(results)

## ----eval=FALSE---------------------------------------------------------------
# # Create Parquet dataset
# write_connectomes_to_parquet(
#   connectomes,
#   output_dir = "benchmark_data",
#   subject_ids = paste0("subj_", 1:100)
# )
# 
# # Sequential loading
# backend_seq <- create_parquet_backend("benchmark_data")
# set_parallel_plan("sequential")
# time_load_seq <- system.time({
#   conns <- backend_seq$get_all_matrices()
# })
# 
# # Parallel loading
# backend_par <- create_parquet_backend("benchmark_data")
# set_parallel_plan("multisession", workers = 4)
# time_load_par <- system.time({
#   conns <- backend_par$get_all_matrices(parallel = TRUE)
# })
# 
# print(paste("Sequential load:", round(time_load_seq[3], 2), "seconds"))
# print(paste("Parallel load:", round(time_load_par[3], 2), "seconds"))
# print(paste("Speedup:", round(time_load_seq[3] / time_load_par[3], 2), "x"))
# 
# # Cleanup
# reset_parallel_plan()
# unlink("benchmark_data", recursive = TRUE)

## ----eval=FALSE---------------------------------------------------------------
# # Function to measure scaling
# measure_scaling <- function(workers_list) {
#   sample <- CSample$new(conns = connectomes, metric_obj = airm)
# 
#   times <- sapply(workers_list, function(w) {
#     if (w == 1) {
#       set_parallel_plan("sequential")
#     } else {
#       set_parallel_plan("multisession", workers = w)
#     }
# 
#     time <- system.time(sample$compute_tangents())[3]
#     reset_parallel_plan()
#     time
#   })
# 
#   data.frame(
#     workers = workers_list,
#     time = times,
#     speedup = times[1] / times,
#     efficiency = (times[1] / times) / workers_list * 100
#   )
# }
# 
# # Test with different worker counts
# workers <- c(1, 2, 4, 8)
# scaling_results <- measure_scaling(workers)
# 
# print(scaling_results)
# 
# # Plot scaling (if plotting package available)
# if (requireNamespace("ggplot2", quietly = TRUE)) {
#   library(ggplot2)
# 
#   p <- ggplot(scaling_results, aes(x = workers, y = speedup)) +
#     geom_line() +
#     geom_point(size = 3) +
#     geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
#     labs(
#       title = "Parallel Scaling Performance",
#       x = "Number of Workers",
#       y = "Speedup",
#       subtitle = "Dashed line = ideal linear scaling"
#     ) +
#     theme_minimal()
# 
#   print(p)
# }

## ----eval=FALSE---------------------------------------------------------------
# # Test different dataset sizes
# test_sizes <- c(10, 25, 50, 100, 200)
# 
# size_benchmark <- function(n) {
#   conns_subset <- lapply(1:n, function(i) create_spd_matrix(50))
#   sample <- CSample$new(conns = conns_subset, metric_obj = airm)
# 
#   # Sequential
#   set_parallel_plan("sequential")
#   time_seq <- system.time(sample$compute_tangents())[3]
# 
#   # Parallel
#   set_parallel_plan("multisession", workers = 4)
#   time_par <- system.time(sample$compute_tangents())[3]
# 
#   reset_parallel_plan()
# 
#   c(sequential = time_seq, parallel = time_par, speedup = time_seq / time_par)
# }
# 
# results_by_size <- t(sapply(test_sizes, size_benchmark))
# results_by_size <- data.frame(n = test_sizes, results_by_size)
# 
# print(results_by_size)

## ----eval=FALSE---------------------------------------------------------------
# # Conservative (recommended for most users)
# n_workers <- parallel::detectCores() - 1
# set_parallel_plan("multisession", workers = n_workers)
# 
# # Aggressive (maximum performance, may slow system)
# n_workers <- parallel::detectCores()
# set_parallel_plan("multisession", workers = n_workers)
# 
# # Custom (based on benchmarking)
# # Test different worker counts and choose optimal

## ----eval=FALSE---------------------------------------------------------------
# # For memory-constrained environments:
# 
# # 1. Use Parquet backend with small cache
# backend <- create_parquet_backend("large_dataset", cache_size = 5)
# 
# # 2. Use moderate worker count
# set_parallel_plan("multisession", workers = 2)
# 
# # 3. Use batch loading for very large datasets
# sample <- CSample$new(backend = backend, metric_obj = airm)
# conns_batch <- sample$load_connectomes_batched(
#   indices = 1:500,
#   batch_size = 50,   # Small batches
#   progress = TRUE
# )
# 
# # 4. Clear cache frequently
# backend$clear_cache()

## ----eval=FALSE---------------------------------------------------------------
# # With progress (slightly slower)
# set_parallel_plan("multisession", workers = 4)
# time_with_progress <- system.time({
#   sample$compute_tangents(progress = TRUE)
# })[3]
# 
# # Without progress (slightly faster)
# time_no_progress <- system.time({
#   sample$compute_tangents(progress = FALSE)
# })[3]
# 
# overhead <- (time_with_progress - time_no_progress) / time_no_progress * 100
# print(paste("Progress overhead:", round(overhead, 1), "%"))

## ----eval=FALSE---------------------------------------------------------------
# library(microbenchmark)
# 
# # Run multiple times to get stable estimates
# mb_result <- microbenchmark(
#   sequential = {
#     set_parallel_plan("sequential")
#     sample$compute_tangents()
#   },
#   parallel_4 = {
#     set_parallel_plan("multisession", workers = 4)
#     sample$compute_tangents()
#   },
#   times = 10  # Run 10 times each
# )
# 
# print(mb_result)
# plot(mb_result)

## ----eval=FALSE---------------------------------------------------------------
# # Clear any caches between runs
# sample <- CSample$new(conns = connectomes, metric_obj = airm)
# 
# # Warm up (first run may be slower)
# sample$compute_tangents()
# 
# # Now benchmark
# time <- system.time(sample$compute_tangents())[3]

## ----eval=FALSE---------------------------------------------------------------
# # Record system specs with benchmarks
# system_info <- list(
#   cores = parallel::detectCores(),
#   memory = as.numeric(system("wmic ComputerSystem get TotalPhysicalMemory", intern = TRUE)[2]) / 1e9,
#   r_version = R.version.string,
#   riemtan_version = packageVersion("riemtan"),
#   os = Sys.info()["sysname"]
# )
# 
# print(system_info)

## ----eval=FALSE---------------------------------------------------------------
# # Expect near-linear scaling up to physical cores
# set_parallel_plan("multisession", workers = 4)
# sample$compute_tangents()  # 3-4x speedup
# sample$compute_vecs()       # 2-4x speedup
# sample$compute_conns()      # 3-4x speedup

## ----eval=FALSE---------------------------------------------------------------
# # Expect 2-3x speedup (less than compute-bound)
# set_parallel_plan("multisession", workers = 4)
# sample$compute_fmean()  # 2-3x speedup

## ----eval=FALSE---------------------------------------------------------------
# # May see >4x speedup with 4 workers (parallel disk I/O)
# backend <- create_parquet_backend("dataset")
# set_parallel_plan("multisession", workers = 4)
# conns <- backend$get_all_matrices(parallel = TRUE)  # 5-10x speedup

## ----eval=FALSE---------------------------------------------------------------
# # Use multisession (multicore not available)
# set_parallel_plan("multisession", workers = 4)
# 
# # Expect slightly higher overhead than Unix
# # Typical speedup: 70-80% of Unix performance

## ----eval=FALSE---------------------------------------------------------------
# # Can use multicore for lower overhead
# set_parallel_plan("multicore", workers = 4)
# 
# # Or multisession for better stability
# set_parallel_plan("multisession", workers = 4)

## ----eval=FALSE---------------------------------------------------------------
# # Use cluster strategy for distributed computing
# library(future)
# plan(cluster, workers = c("node1", "node2", "node3", "node4"))
# 
# # Or use batchtools for SLURM integration
# library(future.batchtools)
# plan(batchtools_slurm, workers = 16)

## ----eval=FALSE---------------------------------------------------------------
# # Setup
# library(riemtan)
# set_parallel_plan("multisession", workers = parallel::detectCores() - 1)
# 
# # Load data
# backend <- create_parquet_backend("large_dataset")
# sample <- CSample$new(backend = backend, metric_obj = airm)
# 
# # Compute with progress
# sample$compute_tangents(progress = TRUE)
# sample$compute_fmean(progress = TRUE)
# 
# # Cleanup
# reset_parallel_plan()

