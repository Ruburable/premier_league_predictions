#!/usr/bin/env python3
"""
run_all.py

Complete Premier League predictions pipeline runner.

Executes all steps in order:
1. Update data (historical + upcoming fixtures)
2. Download team logos
3. Predict match scores
4. Generate visualization dashboard

Usage:
    python run_all.py              # Run full pipeline
    python run_all.py --skip-logos # Skip logo downloads
    python run_all.py --help       # Show help
"""

import sys
import subprocess
from pathlib import Path
import argparse

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
SCRIPTS = {
    "update_data": "update_data.py",
    "predict": "predict_scores.py",
    "visualize": "visualise.py",
    "readme": "generate_readme.py"
}

OUTPUT_DASHBOARD = Path("output/predictions_dashboard.html")


# ------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")


def print_step(step_num, total_steps, description):
    """Print a step indicator."""
    print(
        f"\n{Colors.BOLD}{Colors.CYAN}[Step {step_num}/{total_steps}]{Colors.END} {Colors.BOLD}{description}{Colors.END}")
    print(f"{Colors.CYAN}{'-' * 80}{Colors.END}")


def print_success(message):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_error(message):
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


def run_script(script_path, description):
    """
    Run a Python script and handle errors.

    Args:
        script_path: Path to the script
        description: Description for error messages

    Returns:
        bool: True if successful, False otherwise
    """
    if not Path(script_path).exists():
        print_error(f"Script not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print_error(f"Error running {description}: {e}")
        return False


# ------------------------------------------------------------------
# PIPELINE STEPS
# ------------------------------------------------------------------
def step_update_data():
    """Step 1: Update match data."""
    print_step(1, 5, "Updating Match Data")
    print("Downloading historical matches and identifying upcoming fixtures...\n")
    return run_script(SCRIPTS["update_data"], "Data update")


def step_predict():
    """Step 2: Predict match scores."""
    print_step(2, 5, "Predicting Match Scores")
    print("Training model and generating predictions...\n")
    return run_script(SCRIPTS["predict"], "Score prediction")


def step_table():
    """Step 3: Project final table."""
    print_step(3, 5, "Projecting Final Table")
    print("Calculating projected league standings...\n")
    return run_script(SCRIPTS["table"], "Table projection")


def step_visualize():
    """Step 4: Generate visualization."""
    print_step(4, 5, "Generating Dashboard")
    print("Creating HTML visualization...\n")
    return run_script(SCRIPTS["visualize"], "Visualization")


def step_readme():
    """Step 5: Generate README."""
    print_step(5, 5, "Generating README")
    print("Creating README with predictions...\n")
    return run_script(SCRIPTS["readme"], "README generation")


# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------
def run_pipeline(skip_logos=False):
    """Run the complete pipeline."""

    print_header("PREMIER LEAGUE PREDICTIONS PIPELINE")

    # Track success
    steps_completed = 0
    total_steps = 3  # Changed from 4 to 3

    # Step 1: Update data
    if step_update_data():
        steps_completed += 1
        print_success("Data update completed")
    else:
        print_error("Data update failed - aborting pipeline")
        return False

    # Step 2: Predict scores (was step 3)
    if step_predict():
        steps_completed += 1
        print_success("Score prediction completed")
    else:
        print_error("Score prediction failed - aborting pipeline")
        return False

    # Step 3: Visualize (was step 4)
    if step_visualize():
        steps_completed += 1
        print_success("Dashboard generation completed")
    else:
        print_error("Dashboard generation failed")
        return False

    # Final summary
    print_header("PIPELINE COMPLETE")

    print(f"{Colors.GREEN}{Colors.BOLD}✓ All steps completed successfully!{Colors.END}\n")
    print(f"Steps completed: {Colors.BOLD}{steps_completed}/{total_steps}{Colors.END}\n")

    if OUTPUT_DASHBOARD.exists():
        print(f"{Colors.BOLD}📊 Dashboard available at:{Colors.END}")
        print(f"   {Colors.CYAN}{OUTPUT_DASHBOARD.resolve()}{Colors.END}\n")
        print(f"{Colors.BOLD}🌐 Open in browser:{Colors.END}")
        print(f"   file://{OUTPUT_DASHBOARD.resolve()}\n")

    if Path("README.md").exists():
        print(f"{Colors.BOLD}📖 README updated with predictions:{Colors.END}")
        print(f"   {Colors.CYAN}README.md{Colors.END}\n")

    if Path("output/projected_table.csv").exists():
        print(f"{Colors.BOLD}📊 Projected table:{Colors.END}")
        print(f"   {Colors.CYAN}output/projected_table.csv{Colors.END}\n")

    print(f"{Colors.BOLD}📁 Output files:{Colors.END}")
    output_dir = Path("output")
    if output_dir.exists():
        for file in sorted(output_dir.glob("*")):
            if file.is_file():
                size = file.stat().st_size / 1024  # KB
                print(f"   • {file.name} ({size:.1f} KB)")

    print(f"\n{Colors.GREEN}{'=' * 80}{Colors.END}\n")
    return True


# ------------------------------------------------------------------
# COMMAND LINE INTERFACE
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run the complete Premier League predictions pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                # Run full pipeline
  python run_all.py --help-logos   # Show info about logo downloads

Output:
  - data/matches_master.csv         (historical matches)
  - data/upcoming_fixtures.csv      (upcoming fixtures)
  - output/predictions_upcoming.csv (predictions)
  - output/predictions_dashboard.html (visualization)

Note: To download team logos, run 'python download_logos.py' separately
        """
    )

    parser.add_argument(
        "--help-logos",
        action="store_true",
        help="Show information about downloading team logos separately"
    )

    args = parser.parse_args()

    if args.help_logos:
        print("\n" + "=" * 80)
        print("TEAM LOGOS INFORMATION")
        print("=" * 80)
        print("\nTo download team logos for the dashboard, run:")
        print("  python download_logos.py")
        print("\nThis is a separate step because logo downloads are optional")
        print("and the dashboard works fine without them.")
        print("=" * 80 + "\n")
        return 0

    # Check if required scripts exist
    missing_scripts = []
    for name, script in SCRIPTS.items():
        if not Path(script).exists():
            missing_scripts.append(script)

    if missing_scripts:
        print_error("Missing required scripts:")
        for script in missing_scripts:
            print(f"   • {script}")
        print("\nPlease ensure all scripts are in the current directory.")
        return 1

    # Run pipeline
    try:
        success = run_pipeline()
        return 0 if success else 1
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Pipeline interrupted by user{Colors.END}")
        return 130
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())