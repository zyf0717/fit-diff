"""
Benchmark script comparing Garmin FIT SDK vs fitparse for processing FIT files.
"""

import statistics
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def benchmark_fitparse(file_path: str) -> Dict[str, Any]:
    """Benchmark fitparse library."""
    try:
        import fitparse
    except ImportError:
        return {"error": "fitparse not installed"}

    results = {
        "library": "fitparse",
        "file_path": file_path,
        "success": False,
        "parse_time": None,
        "memory_usage": None,
        "record_count": 0,
        "message_types": [],
        "error": None,
    }

    try:
        start_time = time.perf_counter()

        # Parse the FIT file
        fitfile = fitparse.FitFile(file_path)

        # Extract all messages
        messages = list(fitfile.get_messages())
        message_types = set()
        record_count = 0

        for message in messages:
            message_types.add(message.name)
            record_count += 1

        end_time = time.perf_counter()

        results.update(
            {
                "success": True,
                "parse_time": end_time - start_time,
                "record_count": record_count,
                "message_types": sorted(list(message_types)),
            }
        )

    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()

    return results


def benchmark_garmin_fit_sdk(file_path: str) -> Dict[str, Any]:
    """Benchmark Garmin FIT SDK."""
    results = {
        "library": "garmin_fit_sdk",
        "file_path": file_path,
        "success": False,
        "parse_time": None,
        "memory_usage": None,
        "record_count": 0,
        "message_types": [],
        "error": None,
    }

    try:
        from garmin_fit_sdk import Decoder, Stream
    except ImportError:
        results["error"] = "garmin_fit_sdk not installed"
        return results

    try:
        start_time = time.perf_counter()

        # Parse the FIT file
        stream = Stream.from_file(file_path)
        decoder = Decoder(stream)

        # Decode all messages
        messages, errors = decoder.read()

        end_time = time.perf_counter()

        # Process results
        message_types = set()
        record_count = 0

        # messages is a dictionary with message type keys
        for message_type, message_list in messages.items():
            message_types.add(message_type)
            record_count += len(message_list)

        results.update(
            {
                "success": True,
                "parse_time": end_time - start_time,
                "record_count": record_count,
                "message_types": sorted(list(message_types)),
                "decode_errors": len(errors) if errors else 0,
            }
        )

    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()

    return results


def run_multiple_iterations(
    file_path: str, iterations: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """Run multiple iterations of each parser for statistical analysis."""
    fitparse_results = []
    garmin_sdk_results = []

    print(f"Running {iterations} iterations for file: {Path(file_path).name}")

    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}")

        # Test fitparse
        fitparse_result = benchmark_fitparse(file_path)
        fitparse_results.append(fitparse_result)

        # Test Garmin SDK
        garmin_result = benchmark_garmin_fit_sdk(file_path)
        garmin_sdk_results.append(garmin_result)

    return {"fitparse": fitparse_results, "garmin_sdk": garmin_sdk_results}


def analyze_results(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Analyze benchmark results and compute statistics."""
    analysis = {}

    for library, runs in results.items():
        successful_runs = [r for r in runs if r["success"]]

        if not successful_runs:
            analysis[library] = {
                "success_rate": 0,
                "error": runs[0].get("error", "Unknown error") if runs else "No runs",
            }
            continue

        parse_times = [r["parse_time"] for r in successful_runs]
        record_counts = [r["record_count"] for r in successful_runs]

        analysis[library] = {
            "success_rate": len(successful_runs) / len(runs),
            "avg_parse_time": statistics.mean(parse_times),
            "min_parse_time": min(parse_times),
            "max_parse_time": max(parse_times),
            "std_parse_time": (
                statistics.stdev(parse_times) if len(parse_times) > 1 else 0
            ),
            "avg_record_count": statistics.mean(record_counts),
            "message_types": successful_runs[0]["message_types"],
            "total_iterations": len(runs),
            "successful_iterations": len(successful_runs),
        }

        if library == "garmin_sdk" and "decode_errors" in successful_runs[0]:
            analysis[library]["avg_decode_errors"] = statistics.mean(
                [r.get("decode_errors", 0) for r in successful_runs]
            )

    return analysis


def print_comparison_report(analysis: Dict[str, Any], file_name: str):
    """Print a detailed comparison report."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK REPORT: {file_name}")
    print(f"{'='*60}")

    for library, stats in analysis.items():
        print(f"\n{library.upper()}:")
        print(f"  Success Rate: {stats.get('success_rate', 0):.1%}")

        if stats.get("success_rate", 0) > 0:
            print(f"  Average Parse Time: {stats['avg_parse_time']:.4f}s")
            print(f"  Min Parse Time: {stats['min_parse_time']:.4f}s")
            print(f"  Max Parse Time: {stats['max_parse_time']:.4f}s")
            print(f"  Std Dev Parse Time: {stats['std_parse_time']:.4f}s")
            print(f"  Average Record Count: {stats['avg_record_count']:.0f}")
            print(f"  Message Types: {len(stats['message_types'])}")

            if library == "garmin_sdk" and "avg_decode_errors" in stats:
                print(f"  Average Decode Errors: {stats['avg_decode_errors']:.1f}")
        else:
            print(f"  Error: {stats.get('error', 'Unknown')}")

    # Performance comparison
    if all(analysis[lib].get("success_rate", 0) > 0 for lib in analysis):
        fitparse_time = analysis["fitparse"]["avg_parse_time"]
        garmin_time = analysis["garmin_sdk"]["avg_parse_time"]

        print(f"\nPERFORMANCE COMPARISON:")
        if fitparse_time < garmin_time:
            speedup = garmin_time / fitparse_time
            print(f"  fitparse is {speedup:.2f}x faster than Garmin SDK")
        else:
            speedup = fitparse_time / garmin_time
            print(f"  Garmin SDK is {speedup:.2f}x faster than fitparse")

        print(f"  Time difference: {abs(fitparse_time - garmin_time):.4f}s")


def main():
    """Main benchmark execution."""
    print("FIT Parser Benchmark: Garmin FIT SDK vs fitparse")
    print("=" * 50)

    # Find FIT files in fixtures
    fixtures_dir = Path("tests/fixtures")
    fit_files = list(fixtures_dir.glob("*.fit"))

    if not fit_files:
        print("No FIT files found in tests/fixtures/")
        print("Please add some .fit files to run the benchmark.")
        return

    all_results = {}
    iterations = 5

    for fit_file in fit_files:
        print(f"\nProcessing: {fit_file.name}")
        results = run_multiple_iterations(str(fit_file), iterations)
        analysis = analyze_results(results)
        all_results[fit_file.name] = analysis
        print_comparison_report(analysis, fit_file.name)

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    summary_df_data = []
    for file_name, analysis in all_results.items():
        for library, stats in analysis.items():
            if stats.get("success_rate", 0) > 0:
                summary_df_data.append(
                    {
                        "File": file_name,
                        "Library": library,
                        "Avg Parse Time (s)": f"{stats['avg_parse_time']:.4f}",
                        "Record Count": f"{stats['avg_record_count']:.0f}",
                        "Message Types": len(stats["message_types"]),
                        "Success Rate": f"{stats['success_rate']:.1%}",
                    }
                )

    if summary_df_data:
        summary_df = pd.DataFrame(summary_df_data)
        print("\nSummary Table:")
        print(summary_df.to_string(index=False))

        # Save results to CSV
        summary_df.to_csv("benchmark_results.csv", index=False)
        print(f"\nResults saved to: benchmark_results.csv")


if __name__ == "__main__":
    main()
