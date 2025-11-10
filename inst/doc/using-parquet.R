## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----eval=FALSE---------------------------------------------------------------
# install.packages("arrow")

## ----eval=FALSE---------------------------------------------------------------
# library(riemtan)
# library(Matrix)
# 
# # Create example connectomes (4x4 SPD matrices)
# set.seed(42)
# connectomes <- lapply(1:50, function(i) {
#   mat <- diag(4) + matrix(rnorm(16, 0, 0.1), 4, 4)
#   mat <- (mat + t(mat)) / 2  # Make symmetric
#   mat <- mat + diag(4) * 0.5  # Ensure positive definite
#   Matrix::pack(Matrix::Matrix(mat, sparse = FALSE))
# })
# 
# # Write to Parquet format
# write_connectomes_to_parquet(
#   connectomes,
#   output_dir = "my_connectomes",
#   subject_ids = paste0("subject_", 1:50),
#   provenance = list(
#     study = "Example Study",
#     acquisition_date = "2024-01-01",
#     preprocessing = "Standard pipeline v1.0"
#   )
# )

## ----eval=FALSE---------------------------------------------------------------
# # Detailed validation with verbose output
# validate_parquet_directory("my_connectomes", verbose = TRUE)

## ----eval=FALSE---------------------------------------------------------------
# # Load AIRM metric
# data(airm)
# 
# # Create Parquet backend (default cache size: 10 matrices)
# backend <- create_parquet_backend(
#   "my_connectomes",
#   cache_size = 10
# )
# 
# # Create CSample with the backend
# sample <- CSample$new(
#   backend = backend,
#   metric_obj = airm
# )
# 
# # Sample info
# print(paste("Sample size:", sample$sample_size))
# print(paste("Matrix dimension:", sample$matrix_size))

## ----eval=FALSE---------------------------------------------------------------
# # Compute tangent images
# sample$compute_tangents()
# 
# # Compute vectorized images
# sample$compute_vecs()
# 
# # Compute Frechet mean
# sample$compute_fmean(tol = 0.01, max_iter = 50)
# 
# # Center the sample
# sample$center()
# 
# # Compute variation
# sample$compute_variation()
# print(paste("Variation:", sample$variation))

## ----eval=FALSE---------------------------------------------------------------
# # Access specific connectome (loads from disk if not cached)
# conn_1 <- sample$connectomes[[1]]
# 
# # Access all connectomes (loads all from disk)
# all_conns <- sample$connectomes
# 
# # Cache management
# backend$get_cache_size()  # Check current cache usage
# backend$clear_cache()     # Clear cache to free memory

## ----eval=FALSE---------------------------------------------------------------
# # Create two samples with different backends
# backend1 <- create_parquet_backend("study1_connectomes")
# backend2 <- create_parquet_backend("study2_connectomes")
# 
# sample1 <- CSample$new(backend = backend1, metric_obj = airm)
# sample2 <- CSample$new(backend = backend2, metric_obj = airm)
# 
# # Create super sample
# super_sample <- CSuperSample$new(list(sample1, sample2))
# 
# # Gather all connectomes
# super_sample$gather()
# 
# # Compute statistics
# super_sample$compute_fmean()
# super_sample$compute_variation()

## ----eval=FALSE---------------------------------------------------------------
# # Traditional approach (all in memory)
# sample_memory <- CSample$new(
#   conns = connectomes,
#   metric_obj = airm
# )
# 
# # Parquet approach (lazy loading)
# backend <- create_parquet_backend("my_connectomes")
# sample_parquet <- CSample$new(
#   backend = backend,
#   metric_obj = airm
# )
# 
# # Both work identically
# sample_memory$compute_fmean()
# sample_parquet$compute_fmean()

## ----eval=FALSE---------------------------------------------------------------
# # Small cache (memory-constrained environments)
# backend_small <- ParquetBackend$new("my_connectomes", cache_size = 5)
# 
# # Large cache (memory-rich environments)
# backend_large <- ParquetBackend$new("my_connectomes", cache_size = 50)

## ----eval=FALSE---------------------------------------------------------------
# # Process in chunks
# n <- sample$sample_size
# batch_size <- 10
# 
# for (start in seq(1, n, by = batch_size)) {
#   end <- min(start + batch_size - 1, n)
# 
#   # Load batch
#   batch <- lapply(start:end, function(i) backend$get_matrix(i))
# 
#   # Process batch...
# 
#   # Clear cache to free memory
#   backend$clear_cache()
# }

## ----eval=FALSE---------------------------------------------------------------
# library(riemtan)
# 
# # Enable parallel processing (works on all platforms including Windows!)
# set_parallel_plan("multisession", workers = 4)
# 
# # Check status
# is_parallel_enabled()  # TRUE
# get_n_workers()        # 4
# 
# # Create Parquet-backed sample
# backend <- create_parquet_backend("large_dataset", cache_size = 20)
# sample <- CSample$new(backend = backend, metric_obj = airm)

## ----eval=FALSE---------------------------------------------------------------
# # Parallel tangent computations with progress bar
# sample$compute_tangents(progress = TRUE)   # 3-8x faster
# 
# # Parallel vectorization
# sample$compute_vecs(progress = TRUE)       # 2-4x speedup
# 
# # Parallel Frechet mean computation
# sample$compute_fmean(progress = TRUE)      # 2-5x faster for large samples

## ----eval=FALSE---------------------------------------------------------------
# # Load specific subset in batches
# subset_conns <- sample$load_connectomes_batched(
#   indices = 1:500,        # Load first 500 matrices
#   batch_size = 50,        # 50 matrices per batch
#   progress = TRUE         # Show progress
# )
# 
# # This loads 500 matrices in 10 batches, clearing cache between batches
# # Each batch is loaded in parallel for 5-10x speedup

## ----eval=FALSE---------------------------------------------------------------
# # Sequential (default if not configured)
# set_parallel_plan("sequential")
# system.time(sample$compute_tangents())  # Baseline
# 
# # Parallel with 4 workers
# set_parallel_plan("multisession", workers = 4)
# system.time(sample$compute_tangents())  # 3-4x faster
# 
# # Parallel with 8 workers
# set_parallel_plan("multisession", workers = 8)
# system.time(sample$compute_tangents())  # 6-8x faster

## ----eval=FALSE---------------------------------------------------------------
# # Install progressr for progress bars (optional)
# install.packages("progressr")
# 
# # Enable parallel processing
# set_parallel_plan("multisession", workers = 4)
# 
# # All operations support progress parameter
# sample$compute_tangents(progress = TRUE)
# sample$compute_vecs(progress = TRUE)
# sample$compute_fmean(progress = TRUE)
# 
# # Batch loading with progress
# conns <- sample$load_connectomes_batched(
#   indices = 1:1000,
#   batch_size = 100,
#   progress = TRUE  # Shows "Batch 1/10: loading matrices 1-100"
# )

## ----eval=FALSE---------------------------------------------------------------
# # Conservative (leave cores for system)
# set_parallel_plan("multisession", workers = parallel::detectCores() - 1)
# 
# # Maximum performance (use all cores)
# set_parallel_plan("multisession", workers = parallel::detectCores())

## ----eval=FALSE---------------------------------------------------------------
# # Each worker loads its own data copy
# # For 4 workers with 100 matrices of 200x200, expect:
# # Memory = 4 workers × cache_size × matrix_size ≈ 4 × 20 × 320 KB ≈ 25 MB
# 
# # Use smaller cache with more workers
# backend <- create_parquet_backend("dataset", cache_size = 10)
# set_parallel_plan("multisession", workers = 8)

## ----eval=FALSE---------------------------------------------------------------
# # Compute with parallelization
# set_parallel_plan("multisession", workers = 4)
# sample$compute_tangents(progress = TRUE)
# sample$compute_fmean(progress = TRUE)
# 
# # Reset to free worker resources
# reset_parallel_plan()

## ----eval=FALSE---------------------------------------------------------------
# # Small dataset (n < 10): Uses sequential processing automatically
# small_sample <- CSample$new(conns = small_connectomes[1:5], metric_obj = airm)
# small_sample$compute_tangents()  # Sequential (no overhead)
# 
# # Large dataset (n >= 10): Uses parallel processing automatically
# large_sample <- CSample$new(backend = backend, metric_obj = airm)
# large_sample$compute_tangents()  # Parallel (if plan is active)

## ----eval=FALSE---------------------------------------------------------------
# # multisession: Works on all platforms (Windows, Mac, Linux)
# set_parallel_plan("multisession", workers = 4)
# 
# # multicore: Unix only, lower overhead
# set_parallel_plan("multicore", workers = 4)  # Auto-fallback to multisession on Windows
# 
# # cluster: For remote/distributed computing
# set_parallel_plan("cluster", workers = c("node1", "node2"))

## ----eval=FALSE---------------------------------------------------------------
# # Get metadata
# metadata <- backend$get_metadata()
# 
# # Access subject IDs
# subject_ids <- metadata$subject_ids
# 
# # Access provenance information
# provenance <- metadata$provenance
# print(provenance$study)
# print(provenance$preprocessing)

## ----eval=FALSE---------------------------------------------------------------
# write_connectomes_to_parquet(
#   connectomes,
#   output_dir = "custom_naming",
#   file_pattern = "conn_%03d.parquet"
# )
# 
# # Files will be named: conn_001.parquet, conn_002.parquet, ...

## ----eval=FALSE---------------------------------------------------------------
# # Use minimal cache
# backend <- ParquetBackend$new("large_dataset", cache_size = 3)
# 
# # Compute statistics without loading all matrices at once
# sample <- CSample$new(backend = backend, metric_obj = airm)
# sample$compute_tangents()
# sample$compute_vecs()
# 
# # Operations that don't need all matrices in memory
# sample$compute_fmean(batch_size = 32)  # Uses batching

## ----eval=FALSE---------------------------------------------------------------
# # Check cache usage
# cache_size <- backend$get_cache_size()
# print(paste("Cached matrices:", cache_size))
# 
# # Free memory when needed
# backend$clear_cache()

