import typer
import time
import os
from typing import Optional
from rich.console import Console
from rich.spinner import Spinner
from rich.table import Table
from rich.progress import Progress, TaskID, BarColumn
import typer
from pathlib import Path
import subprocess
import pandas as pd
import questionary
import shutil
import omneer_cli.processing.main as processing
from omneer_cli.processing.main import predict
from omneer_cli.processing.preprocess.preprocess import Data, preprocess_data
from omneer_cli.processing.visualization.raw.vis import load_and_preprocess_data, calculate_feature_importance, plot_feature_importance
from omneer_cli.processing.visualization.raw.eda import load_and_clean_data, perform_eda
from omneer_cli.processing.preprocess.features import process_data


main = typer.Typer(
    name="omneer",
    help="Welcome to Omneer SDK!",
    add_completion=False
)

console = Console()
progress = Progress(console=console)

CYAN = "[bold cyan]"

version_info = "Omneer SDK CLI v1.12.0"  # Update with the appropriate version information

disclaimer = """
© 2023 Omneer Corporation. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

# Define the tips/help text
tips = [
    "Tip 1: Use the --help flag to get a list of available commands and options.",
    "Tip 2: Make sure to update the SDK CLI regularly for the latest features and bug fixes.",
    "Tip 3: Check out our documentation for detailed usage instructions and examples.",
]

@main.command(name="start")
def intro_command():
    """Display the introduction to Omneer SDK."""
    welcome_message = f"""
{CYAN}
╭─────────────────────────────────────────────────────╮
│                [b]Welcome to Omneer SDK![/b]               │
│    Personalized Medicine AI SDK for Neuroscience    │
╰─────────────────────────────────────────────────────╯
"""
    console.print(welcome_message, style="bold", justify="center")
    console.print(version_info, style="bold", justify="center")
    console.print(disclaimer, style="bold", justify="center")

    with console.status("[bold green]Loading...[/bold green]", spinner="arrow3") as status:
        task1: TaskID = progress.add_task("[bold cyan]Task 1[/bold cyan]", total=100)
        task2: TaskID = progress.add_task("[bold cyan]Task 2[/bold cyan]", total=50)
        task3: TaskID = progress.add_task("[bold cyan]Task 3[/bold cyan]", total=75)
        task4: TaskID = progress.add_task("[bold cyan]Task 4[/bold cyan]", total=80)

        while not progress.finished:
            progress.update(task1, completed=progress.tasks[task1].completed + 1)
            time.sleep(0.02)
            if progress.tasks[task1].completed >= 50:
                progress.update(task2, completed=progress.tasks[task2].completed + 1)
            if progress.tasks[task1].completed >= 75:
                progress.update(task3, completed=progress.tasks[task3].completed + 1)
            if progress.tasks[task1].completed >= 80:
                progress.update(task4, completed=progress.tasks[task4].completed + 1)

    time.sleep(1)
    os.system('cls' if os.name == 'nt' else 'clear')

    logo = """
  ██████╗ ███╗   ███╗███╗   ██╗███████╗███████╗██████╗ 
██╔═══██╗████╗ ████║████╗  ██║██╔════╝██╔════╝██╔══██╗
██║   ██║██╔████╔██║██╔██╗ ██║█████╗  █████╗  ██████╔╝
██║   ██║██║╚██╔╝██║██║╚██╗██║██╔══╝  ██╔══╝  ██╔══██╗
╚██████╔╝██║ ╚═╝ ██║██║ ╚████║███████╗███████╗██║  ██║
 ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═╝
    """

    # Display tips/help text
    console.print("\n[b]Tips and Help:[/]")
    for tip in tips:
        console.print(f"• {tip}")

    console.print(logo, style="#2f74d4")

@main.command(name="options", help="Show the list of available commands")
def help_command():
    """Display a list of available commands."""
    commands = main.registered_commands
    command_list = "\n".join([command.name for command in commands if command.name is not None])
    
    typer.echo(f"Available Commands:\n{command_list}")

@main.command(name="init", help="Initialize directory structure")
def init_command(file_path: str):
    """Initialize the directory structure for Omneer SDK."""
    home_dir = Path.home()
    omneer_files_dir = home_dir / "omneer_files"
    data_dir = omneer_files_dir / "data"
    raw_dir = data_dir / "raw"
    preprocessed_dir = data_dir / "preprocessed"
    features_dir = data_dir / "features"
    storage_dir = data_dir / "storage"

    # Create omneer_files directory
    omneer_files_dir.mkdir(exist_ok=True)

    # Create data directory
    data_dir.mkdir(exist_ok=True)

    # Create raw directory
    raw_dir.mkdir(exist_ok=True)

    # Create preprocessed directory
    preprocessed_dir.mkdir(exist_ok=True)

    # Create features directory
    features_dir.mkdir(exist_ok=True)

    # Create storage directory
    storage_dir.mkdir(exist_ok=True)

    # Move the specified file to data/raw
    file = Path(file_path)
    if file.is_file():
        shutil.move(file, raw_dir / file.name)
        console.print(f"[bold green]File '{file.name}' moved to data/raw.[/bold green]")
    else:
        console.print(f"[bold red]File '{file.name}' not found.[/bold red]")

    # Change to home/omneer_files directory
    os.chdir(omneer_files_dir)
    console.print(f"[bold green]Current directory: {os.getcwd()}[/bold green]")

@main.command(name="predict", help="Perform analysis on a CSV file")
def predict_command(
    save_result: bool = typer.Option(False, "--save", "-s", help="Whether to save the analysis result to a file")
):
    """Perform analysis on a CSV file using the specified model and number of features."""
    csvfile = typer.prompt("Enter the name of the CSV file to analyze")
    model_name = typer.prompt("Enter the name of the model to use for the analysis")
    num_features = typer.prompt("Enter the number of features to use in the analysis", type=int)
    
    home_dir = Path.home()
    print(home_dir)
    omneer_files_dir = home_dir / "omneer_files"
    data_dir = omneer_files_dir / "data"
    raw_dir = data_dir / "raw"
    preprocessed_dir = data_dir / "preprocessed"
    features_dir = data_dir / "features"
    storage_dir = data_dir / "storage"

    # Find the input file in data/raw
    input_file = raw_dir / csvfile

    # Check if the input file exists
    if not input_file.is_file():
        console.print(f"[bold red]Input file '{input_file}' not found.[/bold red]")
        raise typer.Exit()

    # Confirm with the user before starting the analysis
    console.print(f"Going to analyze the file '{input_file}' using the model '{model_name}' with {num_features} features.")
    if typer.confirm("Do you want to continue?"):
        output_dir = preprocessed_dir / csvfile  # Use the csvfile directly as the output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name_str = str(model_name).strip('"\'')
        print("Model Name:", model_name_str)
        model_name_normalized = model_name_str if model_name_str != "None" else None

        print("Input File:", input_file)
        print("Model Name:", model_name_normalized)
        print("Number of Features:", num_features)

        try:
            # Show a progress bar while the analysis is being performed
            with console.status("[bold green]Performing the analysis...[/bold green]", spinner="dots"):
                result = predict(str(input_file), model_name_normalized, int(num_features))
            console.print("[bold green]Analysis completed![/bold green]")

            # Save the analysis result to a file if the user chose to do so
            if save_result:
                result_file = storage_dir / (csvfile + "_result.txt")
                with open(result_file, "w") as f:
                    f.write(str(result))
                console.print(f"[bold green]The analysis result has been saved to '{result_file}'.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]An error occurred during analysis: {e}[/bold red]")
    else:
        console.print("[bold yellow]Analysis cancelled by user.[/bold yellow]")

@main.command(name="preprocess", help="Preprocess a CSV file")
def preprocess_command():
    """Preprocess a CSV file."""

    csvfile = typer.prompt("Enter the name of the CSV file to preprocess")
    impute_method = typer.prompt("Enter the imputation method (iterative, knn, mean, median, most_frequent, constant)")
    scale_method = typer.prompt("Enter the scaling method (robust, quantile, standard, minmax, yeo-johnson)")
    transform_method = typer.prompt("Enter the transformation method (log, sqrt, None for no transformation)")
    feature_selection = typer.prompt("Enter the feature selection method (pca, linear, automl, None for no feature selection)")
    outlier_detection = typer.confirm("Would you like to perform outlier detection and removal?")
    augment_data = typer.confirm("Would you like to perform data augmentation using SMOTE?")
    handle_categorical = typer.prompt("Enter the method to handle categorical variables (onehot, ordinal)")

    # Change 'None' to None
    if transform_method.lower() == 'none':
        transform_method = None
    if feature_selection.lower() == 'none':
        feature_selection = None

    home_dir = Path.home()
    omneer_files_dir = home_dir / "omneer_files"
    data_dir = omneer_files_dir / "data"
    raw_dir = data_dir / "raw"

    input_file = raw_dir / csvfile

    if not input_file.is_file():
        console.print(f"[bold red]Input file '{input_file}' not found.[/bold red]")
        raise typer.Exit()

    # Determine label_name and features_count from the CSV file
    df = pd.read_csv(input_file, encoding='latin1')
    label_name = df.columns[1]  # Second column is the label
    features_count = len(df.columns) - 2  # Minus patient_name and label_name columns

    console.print(f"About to preprocess the file '{input_file}' using '{label_name}' as label and {features_count} as the number of features.")
    if typer.confirm("Do you want to continue?"):
        try:
            with console.status("[bold green]Running the preprocessing...[/bold green]", spinner="dots"):
                preprocessed_data = preprocess_data(
                    input_file, 
                    label_name, 
                    features_count, 
                    home_dir, 
                    impute_method=impute_method, 
                    scale_method=scale_method,
                    outlier_detection=outlier_detection,
                    feature_selection=feature_selection,
                    transform_method=transform_method,
                    augment_data=augment_data,
                    handle_categorical=handle_categorical
                )
            console.print(f"[bold green]Preprocessing completed! Preprocessed data is saved in {preprocessed_data}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]An error occurred during preprocessing: {e}[/bold red]")
    else:
        console.print("[bold yellow]Preprocessing cancelled by user.[/bold yellow]")

@main.command(name="features", help="Create new features CSV file")
def features_command():
    """Create new features CSV file."""

    csvfile = typer.prompt("Enter the name of the CSV file:")

    home_dir = Path.home()
    omneer_files_dir = home_dir / "omneer_files"
    data_dir = omneer_files_dir / "data"
    raw_dir = data_dir / "raw"
    features_dir = data_dir / "features"

    input_file = raw_dir / csvfile

    if not input_file.is_file():
        console.print(f"[bold red]Input file '{input_file}' not found.[/bold red]")
        raise typer.Exit()

    num_features = typer.prompt("Enter the number of features to select:", type=int)

    try:
        features_dir.mkdir(parents=True, exist_ok=True)  # Create the features directory if it doesn't exist
        with console.status("[bold green]Running feature selection...", spinner="dots"):
            process_data(input_file, num_features, features_dir)
        console.print("[bold green]Feature selection completed![/bold green]")
    except Exception as e:
        console.print(f"[bold red]An error occurred during feature selection: {e}[/bold red]")


@main.command(name="eda", help="Provide exploratory analysis")
def eda_command():
    # Prompt for CSV file name
    csvfile = typer.prompt("Enter the name of the CSV file")

    home_dir = Path.home()
    omneer_files_dir = home_dir / "omneer_files"
    data_dir = omneer_files_dir / "data"
    raw_dir = data_dir / "raw"
    
    # Load and clean data
    df = load_and_clean_data(raw_dir / csvfile)

    # Perform EDA
    perform_eda(df)

@main.command("plot features")
def analyze():
    """Analyze a CSV file."""

    # Directories to be created
    home_dir = Path.home()
    omneer_files_dir = home_dir / "omneer_files"
    data_dir = omneer_files_dir / "data"
    raw_dir = data_dir / "raw"
    # Create directories if they don't exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Fetch all CSV files in the raw directory
    csv_files = [f.name for f in raw_dir.glob('*.csv')]

    if not csv_files:
        typer.echo("[bold red]No CSV files found in the directory[/bold red]")
        raise typer.Exit()

    # Prompt user to select a CSV file
    file_name = typer.prompt("Select a CSV file to analyze", default=csv_files[0])
    input_file = raw_dir / file_name

    typer.echo(f"Analyzing the file '{input_file}'")

    # Load and preprocess data
    try:
        df = load_and_preprocess_data(input_file)
    except Exception as e:
        typer.echo(f"[bold red]Failed to load and preprocess data: {e}[/bold red]")
        raise typer.Exit()

    # Calculate feature importance
    try:
        importance_df = calculate_feature_importance(df)
    except Exception as e:
        typer.echo(f"[bold red]Failed to calculate feature importance: {e}[/bold red]")
        raise typer.Exit()

    # Plot feature importance
    try:
        plot_feature_importance(importance_df)
    except Exception as e:
        typer.echo(f"[bold red]Failed to plot feature importance: {e}[/bold red]")
        raise typer.Exit()

    typer.echo("[bold green]Analysis completed![/bold green]")

if __name__ == "__main__":
    main()

