import os

def find_png_files(repo_path, output_file):
    """
    Finds all PNG files in the repository and writes their file paths to a text file.

    Args:
        repo_path (str): Path to the repository to search.
        output_file (str): Path to the output text file.
    """
    # List to store the file paths
    png_files = []

    # Walk through all directories and files in the repository
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.lower().endswith('.png'):  # Case-insensitive check for .png files
                full_path = os.path.join(root, file)
                png_files.append(full_path)

    # Write the file paths to the output file
    with open(output_file, 'w') as f:
        for file_path in png_files:
            f.write(f"{file_path}\n")

    print(f"Found {len(png_files)} PNG files. File paths written to '{output_file}'.")

# Example usage
if __name__ == "__main__":
    # Path to your repository
    repo_path = r"C:\Users\16476\Downloads\SPL-Open-Data-main"
    # Desired output file path
    output_file = r"C:\Users\16476\Downloads\SPL-Open-Data-main\png_file_locations.txt"

    find_png_files(repo_path, output_file)
