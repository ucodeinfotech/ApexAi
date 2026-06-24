"""
Master Analysis Pipeline — runs Phases 10-13: Statistics, Mining, Clustering, Regimes.
"""
import time


def run_all(timeframes=["1day"]):
    t0 = time.time()

    print("=" * 60)
    print("Phase 10: Statistical Analysis")
    print("=" * 60)
    from src.analysis.statistical import run_all as run_stats
    run_stats(timeframes)

    print("\n" + "=" * 60)
    print("Phase 11: Pattern Mining")
    print("=" * 60)
    from src.analysis.mining import run_mining
    run_mining(timeframes)

    print("\n" + "=" * 60)
    print("Phase 12: Clustering")
    print("=" * 60)
    from src.analysis.clustering import run_clustering
    run_clustering(timeframes)

    print("\n" + "=" * 60)
    print("Phase 13: Regime Detection")
    print("=" * 60)
    from src.analysis.regime import run_all as run_regime
    run_regime(timeframes)

    print(f"\nAll analysis complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run_all(timeframes=["1day"])
