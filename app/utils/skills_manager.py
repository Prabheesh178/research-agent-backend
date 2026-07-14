import json
import httpx
import re
from typing import Optional
from app.database import (
    save_user_skill,
    delete_user_skill,
    toggle_skill_enabled,
    get_user_skills,
    save_user_plugin,
    delete_user_plugin,
    toggle_plugin_enabled,
    get_user_plugins
)

# Forbidden security phrases scanner
SECURITY_OVERRIDE_PATTERN = re.compile(
    r"(ignore\s+(?:previous\s+)?instructions|bypass|no\s+restrictions|system\s+override|override\s+security|ignore\s+security)", 
    re.IGNORECASE
)

async def handle_system_commands(prompt: str, user_id: str) -> Optional[dict]:
    """
    Checks if a prompt is a system slash command (e.g. /install, /skills).
    If it is, executes the command and returns a state dictionary to return directly.
    """
    clean_prompt = prompt.strip()
    if not clean_prompt.startswith("/"):
        return None

    cmd_parts = clean_prompt.split(" ", 1)
    cmd = cmd_parts[0].lower()
    arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

    if cmd == "/install":
        return await install_github_repo(user_id, arg)
    elif cmd == "/uninstall":
        return uninstall_repo(user_id, arg)
    elif cmd == "/skills":
        return list_skills(user_id)
    elif cmd == "/plugins":
        return list_plugins(user_id)
    elif cmd == "/enable":
        return toggle_enable_status(user_id, arg, enabled=True)
    elif cmd == "/disable":
        return toggle_enable_status(user_id, arg, enabled=False)

    return None

async def install_github_repo(user_id: str, repo: str) -> dict:
    """
    Installs a Skill or Plugin from GitHub.
    Shorthand: owner/repo
    Full URL: https://github.com/owner/repo
    Subdirectory: owner/repo/path/to/skill
    """
    if not repo:
        return make_command_response("❌ Please provide a repository (e.g., `/install owner/repo`).")

    # Clean shorthand/URL
    repo_clean = repo.replace("https://github.com/", "")
    parts = repo_clean.split("/")
    if len(parts) < 2:
        return make_command_response("❌ Invalid repository format. Use `owner/repo` or a GitHub URL.")

    owner = parts[0]
    repo_name = parts[1]
    sub_path = "/".join(parts[2:]) if len(parts) > 2 else ""

    # Build manifest raw URL paths
    branches = ["main", "master"]
    manifest = None
    manifest_url = ""
    active_branch = "main"

    async with httpx.AsyncClient(timeout=10.0) as client:
        for branch in branches:
            path_prefix = f"{sub_path}/" if sub_path else ""
            url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/{branch}/{path_prefix}manifest.json"
            try:
                res = await client.get(url)
                if res.status_code == 200:
                    manifest = res.json()
                    manifest_url = url
                    active_branch = branch
                    break
            except Exception:
                continue

        if not manifest:
            return make_command_response(f"❌ Could not install {repo}. Reason: manifest.json not found in main or master branch.")

        # Validate general manifest fields
        req_fields = ["id", "name", "type", "version"]
        for rf in req_fields:
            if rf not in manifest:
                return make_command_response(f"❌ Could not install {repo}. Reason: Missing manifest field '{rf}'.")

        item_type = manifest["type"].lower()

        # Skill Installation Flow
        if item_type == "skill":
            skill_fields = ["trigger_keywords", "intent_type", "system_prompt_extension"]
            for sf in skill_fields:
                if sf not in manifest:
                    return make_command_response(f"❌ Could not install skill {repo}. Reason: Missing manifest field '{sf}'.")

            # Fetch the system_prompt_extension file
            path_prefix = f"{sub_path}/" if sub_path else ""
            ext_url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/{active_branch}/{path_prefix}{manifest['system_prompt_extension']}"
            try:
                ext_res = await client.get(ext_url)
                if ext_res.status_code != 200:
                    return make_command_response(f"❌ Could not install skill {repo}. Reason: system_prompt_extension file not found.")
                
                prompt_content = ext_res.text
            except Exception as e:
                return make_command_response(f"❌ Could not install skill {repo}. Reason: Failed to load prompt extension: {str(e)}.")

            # Run Security Scan on prompt extensions
            if SECURITY_OVERRIDE_PATTERN.search(prompt_content):
                return make_command_response(f"❌ Could not install skill {repo}. Reason: Security scan failed (forbidden prompt override language detected).")

            # Store in SQLite
            manifest["system_prompt_extension_content"] = prompt_content
            manifest["enabled"] = True
            save_user_skill(user_id, manifest)
            
            return make_command_response(
                f"✅ **{manifest['name']}** v{manifest['version']} installed successfully.\n"
                f"- **ID**: `{manifest['id']}`\n"
                f"- **Trigger Keywords**: {', '.join([f'`{k}`' for k in manifest['trigger_keywords']])}\n"
                f"- **Intent Type**: `{manifest['intent_type']}`"
            )

        # Plugin Installation Flow
        elif item_type == "plugin":
            plugin_fields = ["tools_provided", "auth_type"]
            for pf in plugin_fields:
                if pf not in manifest:
                    return make_command_response(f"❌ Could not install plugin {repo}. Reason: Missing manifest field '{pf}'.")

            # Store in SQLite
            manifest["enabled"] = True
            save_user_plugin(user_id, manifest)
            
            return make_command_response(
                f"✅ Plugin **{manifest['name']}** v{manifest['version']} connected successfully.\n"
                f"- **ID**: `{manifest['id']}`\n"
                f"- **Tools Provided**: {', '.join([f'`{t}`' for t in manifest['tools_provided']])}\n"
                f"- **Auth Type**: `{manifest['auth_type']}`"
            )

        else:
            return make_command_response(f"❌ Could not install {repo}. Reason: Unknown type '{item_type}' (must be 'skill' or 'plugin').")

def uninstall_repo(user_id: str, item_id: str) -> dict:
    if not item_id:
        return make_command_response("❌ Please provide a skill/plugin ID (e.g., `/uninstall journal-formatter`).")

    # Try deleting skill
    delete_user_skill(user_id, item_id)
    # Try deleting plugin
    delete_user_plugin(user_id, item_id)

    return make_command_response(f"✅ Removed skill/plugin `{item_id}` from your workstation library.")

def list_skills(user_id: str) -> dict:
    skills = get_user_skills(user_id)
    if not skills:
        return make_command_response("🔧 **No skills currently installed.** Install one via `/install owner/repo`.")

    lines = ["🔧 **Installed Workstation Skills**:", ""]
    for s in skills:
        status = "🟢 Enabled" if s["enabled"] else "🔴 Disabled"
        lines.append(f"- **{s['name']}** (`{s['skill_id']}`) — {status}")
        lines.append(f"  *Keywords*: {', '.join([f'`{k}`' for k in s['trigger_keywords']])}")
        lines.append("")
    return make_command_response("\n".join(lines))

def list_plugins(user_id: str) -> dict:
    plugins = get_user_plugins(user_id)
    if not plugins:
        return make_command_response("🔌 **No plugins currently connected.** Connect one via `/install owner/repo`.")

    lines = ["🔌 **Connected Workstation Plugins**:", ""]
    for p in plugins:
        status = "🟢 Connected" if p["enabled"] else "🔴 Disconnected"
        lines.append(f"- **{p['name']}** (`{p['plugin_id']}`) — {status}")
        lines.append(f"  *Tools*: {', '.join([f'`{t}`' for t in p['tools_provided']])}")
        lines.append("")
    return make_command_response("\n".join(lines))

def toggle_enable_status(user_id: str, item_id: str, enabled: bool) -> dict:
    if not item_id:
        return make_command_response("❌ Please specify a skill/plugin ID.")

    # Try toggling skill
    toggle_skill_enabled(user_id, item_id, enabled)
    # Try toggling plugin
    toggle_plugin_enabled(user_id, item_id, enabled)

    status_str = "enabled" if enabled else "disabled"
    return make_command_response(f"✅ Skill/Plugin `{item_id}` has been {status_str}.")

def make_command_response(message: str) -> dict:
    """
    Builds a mock pipeline return state with skip_pipeline=True so the backend routes
    the message directly back to the UI chat history.
    """
    return {
        "final_output": message,
        "web_papers": [],
        "rag_chunks": [],
        "trace_logs": [{"agent": "System Installer", "status": "completed", "message": "Command executed successfully."}],
        "skip_pipeline": True
    }
