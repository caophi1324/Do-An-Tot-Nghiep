# ArduPilot Drone Kit - .skill archives

Generated: 2026-10-24 13:04

A `.skill` file is a zip archive. Extract it or point your skill installer at it.

## Bundle

| File | Size | SHA256 |
|---|---|---|
| `ardupilot-drone-kit.skill` | 157.0 KB | `8348200be800da04fd7e935fed8698d4fbc1bd66a126f250485a7fb8e042dfaf` |
| `airframe-profile.skill` | 15.4 KB | `8f000afd38a490df04442124fb753338e9cfefa6380250d442ebdb14e73cf915` |
| `bringup.skill` | 23.6 KB | `1f5ee9ff6be44b62f43fdb9cddb858165cffe6da4bab122eac95d9b35a687517` |
| `flight-safety.skill` | 26.1 KB | `916fd57c73987ec31d77bcc027693d72e8536cf44e61691893c5bbc42d0d27f6` |
| `log-review.skill` | 20.7 KB | `f76cc44e6207c0ffc26f522acc93b8d47770603e522df114453e5fbbd927d999` |
| `mavlink-review.skill` | 13.1 KB | `8a8bf36eacb09fb08c352db475bd48c48b102be7b25f5a93ff03c82ef5d7ad2b` |
| `sitl-sim.skill` | 16.2 KB | `a457c72f47c23e6c1db3ea12837354fc85771bc3a9ad988e3a13d1c61d0eb954` |
| `tuning.skill` | 18.9 KB | `306f84f25362fcdff27c9f0b5190d8e2212fb530c9c546edd6aad72131a4b032` |

## Install

```bash
# whole kit
unzip ardupilot-drone-kit.skill -d ardupilot-drone-kit
cp -r ardupilot-drone-kit/skills/* ~/.config/opencode/skills/

# one skill (SKILL.md is at the archive root)
unzip flight-safety.skill -d ~/.config/opencode/skills/flight-safety
```

Windows PowerShell:

```powershell
Copy-Item .\ardupilot-drone-kit.skill .\ardupilot-drone-kit.zip
Expand-Archive .\ardupilot-drone-kit.zip -DestinationPath .\ardupilot-drone-kit
Copy-Item -Recurse .\ardupilot-drone-kit\skills\* $env:USERPROFILE\.config\opencode\skills\
```
