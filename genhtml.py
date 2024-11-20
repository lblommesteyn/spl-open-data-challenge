import os

def generate_html(repo_path, output_file):
    """
    Generate an HTML file listing all PNG files in the repository with links and previews.

    Args:
        repo_path (str): Path to the repository to search.
        output_file (str): Path to save the generated HTML file.
    """
    # List to store PNG file paths
    png_files = []

    # Walk through the repository to find all PNG files
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.lower().endswith('.png'):
                # Get relative path for the HTML file
                relative_path = os.path.relpath(os.path.join(root, file), repo_path)
                png_files.append(relative_path)

    # Start generating the HTML content
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PNG File List</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
      background-color: #f4f4f9;
      color: #333;
    }
    header {
      background-color: #333;
      color: #fff;
      padding: 1rem;
      text-align: center;
    }
    table {
      margin: 2rem auto;
      width: 90%;
      border-collapse: collapse;
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
      background-color: #fff;
    }
    th, td {
      padding: 1rem;
      border: 1px solid #ddd;
      text-align: left;
    }
    th {
      background-color: #555;
      color: #fff;
    }
    tr:nth-child(even) {
      background-color: #f9f9f9;
    }
    a {
      color: #007bff;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    img {
      max-height: 100px;
    }
  </style>
</head>
<body>
  <header>
    <h1>PNG File List</h1>
    <p>A comprehensive list of all .png files in the repository</p>
  </header>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>File Path</th>
        <th>Preview</th>
      </tr>
    </thead>
    <tbody>
"""

    # Add rows for each PNG file
    for idx, file_path in enumerate(png_files, start=1):
        html_content += f"""
      <tr>
        <td>{idx}</td>
        <td><a href="{file_path}" target="_blank">{file_path}</a></td>
        <td><img src="{file_path}" alt="Preview"></td>
      </tr>
"""

    # Close the HTML content
    html_content += """
    </tbody>
  </table>
</body>
</html>
"""

    # Write the HTML content to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML file generated successfully: {output_file}")

# Example usage
if __name__ == "__main__":
    # Specify the repository path and the output HTML file
    repo_path = r"C:\Users\16476\Downloads\SPL-Open-Data-main"
    output_file = os.path.join(repo_path, "index.html")

    # Generate the HTML file
    generate_html(repo_path, output_file)
