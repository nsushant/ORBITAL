#!/usr/bin/env Rscript
# Export what irace did, as CSV. No plotting, no new packages.
#
# iraceplot would draw these directly, but it pulls in ggplot2, plotly and a
# large dependency tree that compiles from source on a Homebrew R -- tens of
# minutes, and exposed to the same OpenMP failure that broke data.table. It
# would also produce figures in a different visual language from the rest of
# the paper, which is matplotlib. So this script uses only `irace`, which is
# already installed, and writes tables that plots/make_sec4_figures.py draws.
#
# Run from basic_project/:  Rscript plots/irace_export.R
#
# Writes, per algorithm:
#   outputs/irace-<algo>-configurations.csv   every configuration irace tried,
#                                             with its iteration, its parent,
#                                             and whether it survived as an
#                                             elite
#   outputs/irace-<algo>-summary.txt          iterations, experiments, elites

suppressWarnings(suppressMessages(library(irace)))

for (algo in c("mdls", "nsga2")) {
  log <- file.path("outputs", paste0("irace-", algo), "irace.Rdata")
  cat("== ", algo, " ==\n", sep = "")
  if (!file.exists(log)) {
    cat("   no log at ", log, "\n", sep = "")
    next
  }

  res <- read_logfile(log)

  # The names carried by the result object differ a little between irace
  # versions, so report them rather than assume: if a field this script wants
  # is missing, the listing below says what is there instead.
  cat("   fields in the log: ", paste(names(res), collapse = ", "), "\n", sep = "")

  cfg <- res$allConfigurations
  # Which configurations were elite, and in which iteration. allElites is a
  # list with one integer vector of configuration IDs per iteration.
  elite_iter <- rep(NA_integer_, nrow(cfg))
  final_elite <- rep(FALSE, nrow(cfg))
  ok <- tryCatch({
    for (i in seq_along(res$allElites)) {
      ids <- res$allElites[[i]]
      elite_iter[cfg$.ID. %in% ids] <- i
    }
    last <- res$allElites[[length(res$allElites)]]
    final_elite <- cfg$.ID. %in% last
    TRUE
  }, error = function(e) {
    cat("   could not read allElites: ", conditionMessage(e), "\n", sep = "")
    FALSE
  })

  cfg$elite_iteration <- elite_iter
  cfg$final_elite <- final_elite

  out <- file.path("outputs", paste0("irace-", algo, "-configurations.csv"))
  write.csv(cfg, out, row.names = FALSE)
  cat("   wrote ", out, ": ", nrow(cfg), " configurations, ",
      sum(final_elite), " final elites\n", sep = "")

  sm <- file.path("outputs", paste0("irace-", algo, "-summary.txt"))
  con <- file(sm, "w")
  writeLines(c(
    paste("algorithm:", algo),
    paste("configurations tried:", nrow(cfg)),
    paste("iterations:", length(res$allElites)),
    paste("experiments:", tryCatch(nrow(res$experimentLog), error = function(e) NA)),
    paste("final elites:", sum(final_elite)),
    "",
    "final elite configurations:",
    paste(capture.output(print(cfg[final_elite, ])), collapse = "\n")
  ), con)
  close(con)
  cat("   wrote ", sm, "\n", sep = "")
}

cat("\nNow draw them:  python plots/make_sec4_figures.py --irace\n")
