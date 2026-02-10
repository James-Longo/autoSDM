import ee
import sys
import json

def list_projects():
    try:
        # ee.data._get_cloud_projects() returns a list of dictionaries/objects
        # depending on the version, but usually has 'projectId' or 'project_id'
        projects = ee.data._get_cloud_projects()
        
        # Normalize to a simple list of IDs
        project_ids = []
        for p in projects:
            if isinstance(p, dict):
                pid = p.get('projectId') or p.get('project_id')
                if pid: project_ids.append(pid)
            elif hasattr(p, 'project_id'):
                project_ids.append(p.project_id)
            elif hasattr(p, 'projectId'):
                project_ids.append(p.projectId)
        
        return list(set(project_ids))
    except Exception as e:
        sys.stderr.write(f"Discovery error: {e}\n")
        return []

if __name__ == "__main__":
    # Ensure initialized (at least for discovery)
    try:
        ee.Initialize()
    except:
        pass
    
    p_list = list_projects()
    print(json.dumps(p_list))
