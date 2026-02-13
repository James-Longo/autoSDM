import ee
try:
    project_id = "927064910362"
    print(f"Attempting to initialize GEE with project: {project_id}")
    ee.Initialize(project=project_id)
    print("Success! GEE initialized.")
    # Try a simple operation to confirm access
    print(f"Default root: {ee.data.getAssetRoot()}")
except Exception as e:
    print(f"Failed to initialize: {e}")
