import os
import re
import random
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import time
from collections import defaultdict
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich import box
from rich.align import Align

class ArgoFloatScanner:
    def __init__(self, base_url="https://www.ncei.noaa.gov/data/oceans/argo/gadr/data/indian/"):
        self.base_url = base_url
        self.years = ['2017', '2018', '2019', '2020']
        self.months = [f"{i:02d}" for i in range(1, 13)]  # 01 to 12
        self.selected_floats = []
        self.float_locations = defaultdict(list)  # float_id -> [(year, month, filename), ...]
        self.console = Console()
        self.download_dir = Path("data/raw")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_float_id(self, filename):
        """Extract float ID from filename like 'nodc_R1901342_217.nc'"""
        match = re.search(r'R(\d+)', filename)
        return match.group(1) if match else None
    
    def get_directory_listing(self, url):
        """Get list of files/folders from a directory URL"""
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                links = []
                for link in soup.find_all('a'):
                    if hasattr(link, 'get'):
                        href = link.get('href') # type: ignore
                        if href and isinstance(href, str) and not href.startswith('../') and href != '/':
                            links.append(href.rstrip('/'))
                return links
            else:
                return []
        except requests.RequestException:
            return []
    
    def collect_all_float_ids(self):
        """First pass: collect all unique float IDs from all accessible folders"""
        all_float_ids = set()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="bright_green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            # Create main task for years
            year_task = progress.add_task("[cyan]Scanning years...", total=len(self.years))
            
            for year in self.years:
                progress.update(year_task, description=f"[cyan]Scanning year {year}")
                year_url = urljoin(self.base_url, f"{year}/")
                
                # Check if year directory exists
                year_folders = self.get_directory_listing(year_url)
                if not year_folders:
                    self.console.print(f"  [red]✗[/red] Year {year} directory not accessible")
                    progress.advance(year_task)
                    continue
                
                # Create month task
                month_task = progress.add_task(f"[green]  Months in {year}", total=len(self.months))
                
                year_float_count = 0
                for month in self.months:
                    progress.update(month_task, description=f"[green]  Processing {year}/{month}")
                    
                    if month not in year_folders:
                        progress.advance(month_task)
                        continue
                        
                    month_url = urljoin(year_url, f"{month}/")
                    files = self.get_directory_listing(month_url)
                    month_float_count = 0
                    
                    for file in files:
                        if file.endswith('.nc'):
                            float_id = self.extract_float_id(file)
                            if float_id:
                                all_float_ids.add(float_id)
                                month_float_count += 1
                    
                    if month_float_count > 0:
                        year_float_count += month_float_count
                        
                    progress.advance(month_task)
                    time.sleep(0.05)
                
                progress.remove_task(month_task)
                self.console.print(f"  [green]✓[/green] Year {year}: {year_float_count} float files found")
                progress.advance(year_task)
        
        # Display summary
        panel = Panel(
            f"[bold green]Found {len(all_float_ids)} unique float IDs[/bold green]",
            title="[bold cyan]First Pass Complete[/bold cyan]",
            border_style="green"
        )
        self.console.print(panel)
        
        return list(all_float_ids)
    
    def select_random_floats(self, all_float_ids, count=50):
        """Randomly select specified number of float IDs"""
        if len(all_float_ids) < count:
            actual_count = len(all_float_ids)
            self.console.print(f"[yellow]⚠[/yellow] Only {actual_count} float IDs available, selecting all")
            self.selected_floats = all_float_ids
        else:
            actual_count = count
            self.selected_floats = random.sample(all_float_ids, count)
        
        # Create a nice table showing selected floats
        table = Table(title=f"[bold cyan]Selected {actual_count} Random Float IDs[/bold cyan]", 
                      show_header=True, header_style="bold magenta", box=box.ROUNDED)
        
        # Add columns
        cols = 5
        for i in range(cols):
            table.add_column(f"Float {i+1}", style="cyan", justify="center")
        
        # Add rows
        selected_copy = self.selected_floats.copy()
        while selected_copy:
            row = []
            for _ in range(cols):
                if selected_copy:
                    row.append(selected_copy.pop(0))
                else:
                    row.append("")
            table.add_row(*row)
        
        self.console.print(table)
        return self.selected_floats
    
    def search_selected_floats(self):
        """Second pass: search for selected floats across all years and months"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]{task.description}"),
            BarColumn(complete_style="blue", finished_style="bright_blue"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            year_task = progress.add_task("[yellow]Searching for selected floats...", total=len(self.years))
            
            for year in self.years:
                progress.update(year_task, description=f"[yellow]Searching in year {year}")
                year_url = urljoin(self.base_url, f"{year}/")
                
                year_folders = self.get_directory_listing(year_url)
                if not year_folders:
                    progress.advance(year_task)
                    continue
                
                month_task = progress.add_task(f"[blue]  Months in {year}", total=len(self.months))
                year_found = 0
                
                for month in self.months:
                    progress.update(month_task, description=f"[blue]  Searching {year}/{month}")
                    
                    if month not in year_folders:
                        progress.advance(month_task)
                        continue
                        
                    month_url = urljoin(year_url, f"{month}/")
                    files = self.get_directory_listing(month_url)
                    found_count = 0
                    
                    for file in files:
                        if file.endswith('.nc'):
                            float_id = self.extract_float_id(file)
                            if float_id in self.selected_floats:
                                self.float_locations[float_id].append((year, month, file))
                                found_count += 1
                    
                    if found_count > 0:
                        year_found += found_count
                        
                    progress.advance(month_task)
                    time.sleep(0.05)
                
                progress.remove_task(month_task)
                if year_found > 0:
                    self.console.print(f"  [blue]✓[/blue] Year {year}: {year_found} files from selected floats")
                progress.advance(year_task)
    
    def download_files(self):
        """Download all found .nc files to data/raw/ folder"""
        # Collect all files to download
        files_to_download = []
        for float_id, locations in self.float_locations.items():
            for year, month, filename in locations:
                file_url = urljoin(self.base_url, f"{year}/{month}/{filename}")
                local_path = self.download_dir / f"{float_id}_{year}_{month}_{filename}"
                files_to_download.append((file_url, local_path, float_id, year, month))
        
        if not files_to_download:
            self.console.print("[red]No files to download![/red]")
            return
        
        # Create download progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(complete_style="green", finished_style="bright_green"),
            TaskProgressColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeRemainingColumn(),
            "•", 
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            download_task = progress.add_task(
                f"[green]Downloading {len(files_to_download)} files...", 
                total=len(files_to_download)
            )
            
            downloaded = 0
            failed = 0
            
            for file_url, local_path, float_id, year, month in files_to_download:
                progress.update(
                    download_task, 
                    description=f"[green]Downloading Float {float_id} ({year}/{month})"
                )
                
                try:
                    # Check if file already exists
                    if local_path.exists():
                        self.console.print(f"  [yellow]⚠[/yellow] File already exists: {local_path.name}")
                        downloaded += 1
                        progress.advance(download_task)
                        continue
                    
                    # Download file
                    response = requests.get(file_url, timeout=60, stream=True)
                    if response.status_code == 200:
                        with open(local_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        file_size = local_path.stat().st_size / (1024 * 1024)  # MB
                        self.console.print(f"  [green]✓[/green] Downloaded: {local_path.name} ({file_size:.1f} MB)")
                        downloaded += 1
                    else:
                        self.console.print(f"  [red]✗[/red] Failed to download: {file_url} (Status: {response.status_code})")
                        failed += 1
                        
                except Exception as e:
                    self.console.print(f"  [red]✗[/red] Error downloading {file_url}: {str(e)}")
                    failed += 1
                
                progress.advance(download_task)
                time.sleep(0.1)  # Be nice to the server
        
        # Download summary
        summary_panel = Panel(
            f"[bold green]✓ Successfully downloaded: {downloaded} files[/bold green]\n"
            f"[bold red]✗ Failed downloads: {failed} files[/bold red]\n"
            f"[bold cyan]📁 Location: {self.download_dir.absolute()}[/bold cyan]",
            title="[bold yellow]Download Summary[/bold yellow]",
            border_style="yellow"
        )
        self.console.print(summary_panel)
    
    def generate_report(self):
        """Generate a comprehensive report of findings"""
        # Main statistics
        found_floats = len(self.float_locations)
        missing_floats = len(self.selected_floats) - found_floats
        total_files = sum(len(locations) for locations in self.float_locations.values())
        
        # Create main summary table
        summary_table = Table(title="[bold cyan]Scan Summary[/bold cyan]", 
                              show_header=True, header_style="bold magenta", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan", justify="left")
        summary_table.add_column("Count", style="green bold", justify="center")
        
        summary_table.add_row("Selected Floats", str(len(self.selected_floats)))
        summary_table.add_row("Floats Found", str(found_floats))
        summary_table.add_row("Floats Missing", str(missing_floats))
        summary_table.add_row("Total Files Found", str(total_files))
        
        self.console.print(summary_table)
        
        # Detailed float table
        if self.float_locations:
            detail_table = Table(title="[bold green]Float Details[/bold green]", 
                                 show_header=True, header_style="bold blue", box=box.ROUNDED)
            detail_table.add_column("Float ID", style="cyan", justify="center")
            detail_table.add_column("Files", style="green", justify="center")
            detail_table.add_column("Years Present", style="yellow", justify="left")
            
            for float_id in sorted(self.float_locations.keys()):
                locations = self.float_locations[float_id]
                file_count = len(locations)
                years_present = sorted(set(loc[0] for loc in locations))
                years_str = ", ".join(years_present)
                
                detail_table.add_row(float_id, str(file_count), years_str)
            
            self.console.print(detail_table)
    
    def save_results_to_file(self, filename="argo_float_results.txt"):
        """Save results to a text file"""
        with open(filename, 'w') as f:
            f.write("Argo Float Search Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Base URL: {self.base_url}\n")
            f.write(f"Years searched: {', '.join(self.years)}\n")
            f.write(f"Total selected floats: {len(self.selected_floats)}\n")
            f.write(f"Floats with data found: {len(self.float_locations)}\n")
            f.write(f"Download directory: {self.download_dir.absolute()}\n\n")
            
            f.write("Selected Float IDs:\n")
            f.write("-" * 20 + "\n")
            for float_id in sorted(self.selected_floats):
                f.write(f"{float_id}\n")
            
            f.write(f"\nDetailed Results:\n")
            f.write("-" * 20 + "\n")
            for float_id in sorted(self.float_locations.keys()):
                f.write(f"\nFloat {float_id}:\n")
                locations = sorted(self.float_locations[float_id])
                for year, month, filename in locations:
                    file_url = urljoin(self.base_url, f"{year}/{month}/{filename}")
                    local_file = f"{float_id}_{year}_{month}_{filename}"
                    f.write(f"  {year}/{month}: {filename}\n")
                    f.write(f"    URL: {file_url}\n")
                    f.write(f"    Local file: {local_file}\n")
        
        self.console.print(f"[green]✓[/green] Results saved to [bold]{filename}[/bold]")
    
    def run_full_scan(self, num_floats=50):
        """Run the complete scanning and downloading process"""
        # Welcome banner
        welcome = Panel(
            Align.center(
                "[bold cyan]🌊 Argo Float Data Scanner 🌊[/bold cyan]\n\n"
                f"[green]Target:[/green] {num_floats} random floats\n"
                f"[green]Years:[/green] {' - '.join([self.years[0], self.years[-1]])}\n"
                f"[green]Base URL:[/green] {self.base_url}\n"
                f"[green]Download to:[/green] {self.download_dir.absolute()}"
            ),
            title="[bold yellow]Welcome[/bold yellow]",
            border_style="bright_blue",
            padding=(1, 2)
        )
        self.console.print(welcome)
        
        try:
            # Phase 1: Collect all float IDs
            phase1 = Panel("[bold cyan]Phase 1: Discovering all float IDs[/bold cyan]", 
                           border_style="cyan")
            self.console.print(phase1)
            all_float_ids = self.collect_all_float_ids()
            
            if not all_float_ids:
                self.console.print("[red]❌ No float IDs found. Exiting.[/red]")
                return
            
            # Phase 2: Select random floats
            phase2 = Panel("[bold magenta]Phase 2: Selecting random floats[/bold magenta]", 
                           border_style="magenta")
            self.console.print(phase2)
            self.select_random_floats(all_float_ids, num_floats)
            
            # Phase 3: Search for selected floats
            phase3 = Panel("[bold yellow]Phase 3: Searching for selected floats[/bold yellow]", 
                           border_style="yellow")
            self.console.print(phase3)
            self.search_selected_floats()
            
            # Phase 4: Verify and Download files
            phase4 = Panel("[bold green]Phase 4: Verifying and downloading files[/bold green]", 
                           border_style="green")
            self.console.print(phase4)
            self.download_files()
            
            # Final report
            final = Panel("[bold blue]Final Report[/bold blue]", border_style="blue")
            self.console.print(final)
            self.generate_report()
            self.save_results_to_file()
            
            # Success message
            success = Panel(
                Align.center("[bold green]🎉 Scan Complete! 🎉[/bold green]"),
                border_style="bright_green"
            )
            self.console.print(success)
            
        except KeyboardInterrupt:
            interrupt = Panel(
                "[bold red]⚠ Scan interrupted by user[/bold red]",
                border_style="red"
            )
            self.console.print(interrupt)
            if self.float_locations:
                self.console.print("[yellow]Generating partial report...[/yellow]")
                self.generate_report()
                self.save_results_to_file("partial_argo_results.txt")
        except Exception as e:
            error = Panel(
                f"[bold red]❌ An error occurred: {str(e)}[/bold red]",
                border_style="red"
            )
            self.console.print(error)


def main():
    # Create scanner instance
    scanner = ArgoFloatScanner()
    
    # Run the full scan with download
    scanner.run_full_scan(num_floats=20)


if __name__ == "__main__":
    main()