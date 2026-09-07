# Ethics, Privacy, Data Governance, and Licensing

## 1. Intended use

This is an academic road-condition screening prototype. It may help identify locations for human review. It must not be presented as a replacement for certified pavement inspection, a guarantee of road safety, or a system that measures physical defect depth or repair cost.

## 2. Video permission and ownership

Before processing or publishing the teacher-provided video, record:

- who recorded/owns it;
- permission granted for coursework, model training, publication, and demonstration;
- whether redistribution is allowed;
- whether derived frames, labels, or annotated video may be public;
- any route/location restrictions.

Permission to view a video does not automatically mean permission to publish it or train a public model with it.

## 3. Privacy risks

Dashcam footage may contain:

- faces;
- vehicle number plates;
- homes and businesses;
- route patterns and location clues;
- timestamps or GPS overlays;
- accidents or sensitive personal activity.

Default controls:

- keep raw video private and access-limited;
- do not commit video or extracted frames to Git;
- remove audio unless it is necessary and authorized;
- blur faces/plates before public demonstrations when appropriate;
- strip unnecessary metadata from shared derivatives;
- avoid publishing exact GPS data;
- use short, permission-safe clips in documentation;
- delete temporary derivatives according to an agreed retention plan.

## 4. Dataset governance

For every public dataset, store:

- canonical source URL;
- version and download date;
- license/terms snapshot or citation;
- permitted research, modification, and redistribution uses;
- required attribution;
- any non-commercial or access restrictions.

Do not redistribute public images merely because they can be downloaded. Dataset licenses and paper access are separate from model/software licenses.

## 5. Annotation ethics and quality

- Annotators should see only the footage required for the task.
- Ambiguous defects must be flagged, not forced into a class.
- Test labels should receive stronger review than ordinary training examples.
- Report class imbalance, exclusions, and uncertain conditions.
- Do not remove hard conditions solely to improve metrics.

## 6. Bias and representativeness

A model trained mainly on one camera, route, season, or road type may not generalize. Potential biases include:

- daylight over night;
- dry roads over wet roads;
- asphalt over concrete/block paving;
- one camera height and field of view;
- urban roads over highways/rural roads;
- severe visible defects over fine cracks.

The final report must describe the actual training/evaluation distribution and restrict claims accordingly.

## 7. Safety and communication

Use careful terms:

- say **detected road-damage candidate**, not confirmed engineering defect;
- say **relative visual severity**, not physical severity;
- say **unique predicted event**, not proven unique real-world asset, unless manually verified;
- include confidence and limitations in exported summaries.

High-impact maintenance decisions should include human inspection and, where needed, engineering measurement.

## 8. Software licensing

Ultralytics currently provides AGPL-3.0 and enterprise licensing paths. Its licensing page identifies academic research and university coursework as examples for the open-source route when the complete project is released compatibly. The project should therefore remain fully open and license-compatible unless a different license has been obtained.

Before publication:

- include the chosen repository license;
- retain copyright and license notices;
- document major third-party packages;
- provide corresponding source as required;
- verify obligations for model weights and hosted/network use;
- re-check current terms rather than relying only on this summary.

References:

- [Ultralytics licensing overview](https://www.ultralytics.com/license)
- [Ultralytics AGPL-3.0 terms](https://www.ultralytics.com/legal/agpl-3-0-software-license)

This document is a project-governance checklist, not legal advice.

## 9. Model and artifact release checklist

Before releasing code, weights, frames, or results:

- [ ] Source-video permission covers the intended release.
- [ ] Faces/plates and metadata have been reviewed.
- [ ] Public-dataset terms allow the specific artifact.
- [ ] Software and weight licenses are compatible.
- [ ] Required citations and notices are present.
- [ ] Secrets, local paths, and personal information are absent.
- [ ] Limitations and intended use are visible.
- [ ] Model card or README states the training domain and known failure modes.

## 10. Incident handling

If private imagery or metadata is accidentally committed or shared:

1. stop further distribution;
2. revoke links/access when possible;
3. notify the project owner/teacher;
4. remove the material from current and historical publication channels using an appropriate safe process;
5. rotate any exposed credentials;
6. document the incident and prevention step.

## 11. Final report limitations to include

- monocular video cannot provide reliable physical depth/volume here;
- visual severity is camera- and zone-dependent;
- tracking IDs can fragment or merge;
- one-hour/local-route evaluation limits generalization;
- fine cracks and distant defects may be under-detected;
- shadows, water, repairs, markings, and manholes may cause false positives;
- performance depends on camera position, speed, weather, lighting, and road texture;
- the system requires human review for operational decisions.

