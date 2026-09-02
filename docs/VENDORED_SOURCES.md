# Vendored source snapshots

RoboDojo is distributed as a single Git repository. The source trees below are
checked into RoboDojo as ordinary files, so cloning or pulling RoboDojo is
sufficient to obtain them; no `.gitmodules` file or submodule initialization is
required.

| RoboDojo path | Upstream repository | Imported commit |
| --- | --- | --- |
| `XPolicyLab/` | [`Aero-san/XPolicyLab`](https://github.com/Aero-san/XPolicyLab) | `930030a4c3dfd9f875f8bc460d4cb91a274343a4` |
| `third_party/IsaacLab/` | [`Aero-san/IsaacLab`](https://github.com/Aero-san/IsaacLab) | `15da9784f833969ce9631df960538fec11efec80` |
| `third_party/curobo/` | [`yuechen0614/curobo`](https://github.com/yuechen0614/curobo) | `895c6517243f8cb091c73c018c8167192d39599a` |
| `external_dependencies/WCM/` | [`Aero-san/WCM`](https://github.com/Aero-san/WCM) | `71a52e6701f4bbf7e1b02f006b4e30850cf13179` |
| `XPolicyLab/policy/G05/GalaxeaVLA/` | [`OpenGalaxea/GalaxeaVLA`](https://github.com/OpenGalaxea/GalaxeaVLA) | `89f2322b4ad016e192437adc1a2c253b05bab246` |

The imported files retain the upstream license and notice files. Changes to
these directories are now ordinary RoboDojo changes and must be committed and
synced from the RoboDojo repository. Do not run `git submodule update` or add a
new `.gitmodules` entry for them.

This conversion covers source code that was present in the working checkout. It
does not place machine-specific virtual environments, model checkpoints,
datasets, downloaded assets, build outputs, or evaluation results in Git. Those
remain local and are handled by the installation and migration instructions.
