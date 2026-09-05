#!/usr/bin/env Rscript
# Visualise what the two irace races did, using the iraceplot package.
#
# iraceplot is separate from irace and is installed on demand. The exact
# signatures of its plotting functions vary between versions, so each call is
# guarded and the script prints the arguments each function actually takes on
# this installation before trying it. That way a version mismatch produces a
# usable message rather than a stack trace, and the correct call is visible.
#
# Run from basic_project/:  Rscript plots/irace_report.R
#
# Writes into figures/:
#   irace-<algo>-report.html   the full report, which is the thing worth reading
#   irace-<algo>-parcoord.pdf  elite configurations in parallel coordinates:
#                              which parameters the surviving configurations
#                              agree on, and which they scatter across, is the
#                              question the elite table cannot answer
#   irace-<algo>-frequency.pdf how often each parameter value was sampled

suppressWarnings(suppressMessages({
  if (!requireNamespace("iraceplot", quietly = TRUE)) {
    cat("iraceplot is not installed, and this script will NOT install it.\n")
    cat("Attempting to on this machine pulls ggplot2, plotly and a tidyverse\n")
    cat("dependency tree that compiles from source, fails part way, and ends\n")
    cat("in the same OMP Error #15 that broke data.table.\n\n")
    cat("Use the lightweight path instead, which needs no new packages:\n")
    cat("    Rscript plots/irace_export.R\n")
    cat("    python  plots/make_sec4_figures.py --irace\n")
    quit(status = 1)
  }
  library(irace)
  library(iraceplot)
}))

outdir <- "figures"
dir.create(outdir, showWarnings = FALSE)

show_args <- function(fname) {
  if (exists(fname)) {
    cat("    ", fname, ": ", paste(deparse(args(get(fname))), collapse = " "),
        "\n", sep = "")
  } else {
    cat("    ", fname, ": not available in this iraceplot version\n", sep = "")
  }
}

try_plot <- function(label, expr) {
  ok <- TRUE
  tryCatch(force(expr),
           error = function(e) {
             ok <<- FALSE
             cat("    ", label, " failed: ", conditionMessage(e), "\n", sep = "")
           })
  if (ok) cat("    ", label, " ok\n", sep = "")
  invisible(ok)
}

for (algo in c("mdls", "nsga2")) {
  log <- file.path("outputs", paste0("irace-", algo), "irace.Rdata")
  cat("== ", algo, " ==\n", sep = "")
  if (!file.exists(log)) {
    cat("    no log at ", log, "; run run_irace.sh first\n", sep = "")
    next
  }

  res <- read_logfile(log)
  elites <- res$iterationElites
  cat("    iterations: ", length(unique(res$state$indexIteration)), "   ",
      "experiments: ", nrow(res$experimentLog), "   ",
      "final elites: ", length(res$allElites[[length(res$allElites)]]), "\n",
      sep = "")

  cat("  signatures on this installation:\n")
  for (f in c("report", "parallel_coord", "sampling_frequency",
              "parameter_frequency", "boxplot_training", "plot_experiments_matrix"))
    show_args(f)

  cat("  plots:\n")
  try_plot("report",
           report(log, filename = file.path(outdir, paste0("irace-", algo, "-report"))))
  try_plot("parallel_coord",
           parallel_coord(res, filename = file.path(outdir, paste0("irace-", algo, "-parcoord"))))
  try_plot("sampling_frequency",
           sampling_frequency(res, filename = file.path(outdir, paste0("irace-", algo, "-frequency"))))
  cat("\n")
}

cat("Anything that failed above prints the signature it has, a line up.\n")
cat("The parallel-coordinates plot is the one to look at: a parameter whose\n")
cat("elite lines converge is identified by the tuning, one whose lines fan\n")
cat("across the range is not, and the paper should only claim the former.\n")
