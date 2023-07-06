import typer
import time
import os
from rich.console import Console
from rich.progress import Progress, TaskID
import subprocess

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
    """Display the introduction message with an exciting loading animation."""
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
    commands = app.registered_commands
    command_list = "\n".join([command.name for command in commands])
    
    typer.echo(f"Available Commands:\n{command_list}")


@main.command(name="predict", help="Run prediction on a CSV file using a specific model")
def predict_command(csvfile: str, model_name: str, num_features: int = typer.Argument(None)):
    """Run the prediction on a CSV file using a specific model."""
    main_script = os.path.join(os.path.dirname(__file__), "..", "omneer", "processing", "main.py")
    if not os.path.isfile(main_script):
        typer.echo("main.py file not found")
        raise typer.Exit(code=1)

    args = ["python", main_script, csvfile, model_name]

    if num_features:
        args.append(str(num_features))

    subprocess.run(args)


if __name__ == "__main__":
    main()