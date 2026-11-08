# Builds the deck; set SKILL_PPTX to the pptx skill folder (for applyTheme), or edit the path below.
$skill = $env:SKILL_PPTX
if (-not $skill) { $skill = "C:\Users\alisher\AppData\Local\Claude-Data\claude_pzs8sxrjxfjjc\local-agent-mode-sessions\skills-plugin\191cee66-c3d4-41c0-b192-b0bb1d9dfe09\29a4cce7-d757-4ee0-ac41-e7a939e87738\skills\pptx" }
$env:NODE_PATH = "$PSScriptRoot\node_modules"
$env:APPLY_THEME = "$skill\scripts\apply_theme.js"
node "$PSScriptRoot\make_slides.js"
