# A connectomics milestone: Mapping the complete male fruit fly brain

- **Source:** [Google Research Blog](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
- **Date:** September 3, 2026
- **Authors:** Michał Januszewski and Viren Jain, Research Scientists, Google Research
- **Saved:** September 4, 2026 (local markdown copy of the public page)

![A colorful, high-resolution 3D rendering mapping the neurons of a fruit fly's brain and nerve cord.](https://storage.googleapis.com/gweb-research2023-media/original_images/Male-Fruit-Fly-Brain-Map-hero.jpg)

We partnered with HHMI Janelia and collaborators to publish a complete map of the male fruit fly’s brain and central nervous system, creating the largest brain map to date. Together with ongoing research on other species, such as fish and mice, these wiring maps begin to reveal the mechanics of how all brains work.

## Quick links

- [Paper](https://doi.org/10.1016/j.cell.2026.08.015) — *Cell* DOI
- [Dataset](https://www.janelia.org/project-team/flyem/male-cns-connectome) — Janelia FlyEM Male CNS Connectome
- [Connectomics website](https://sites.research.google/gr/neural-mapping/)
- [Video](https://www.youtube.com/shorts/HD9cDLgSe-o)
- [News from Google post](https://blog.google/innovation-and-ai/technology/research/male-fruit-fly-brain-map/)

---

The common fruit fly, *Drosophila melanogaster*, has been central to scientific research leading to [multiple Nobel Prizes](https://www.theguardian.com/science/2017/oct/07/fruit-fly-fascination-nobel-prizes-genetics). Fruit flies have been a fundamental [model organism in genetics](https://hms.harvard.edu/news/why-fly), thanks to their stereotypical behavior and short life cycle, and promise to do the same for neuroscience. While the thoughts of this fruit-loving insect might seem far removed from human cognition, the brains of vastly different species share many similarities. Because mapping the 86 billion neurons in a human brain is not yet possible, scientists are using AI to map the brains of smaller organisms, like fruit flies. This will help us decipher how animal nervous systems perceive the world, react to stimuli, and how damaged neural pathways might one day be repaired.

Now, in a project led by [Howard Hughes Medical Institute (HHMI) Janelia Research Campus](https://www.hhmi.org/research/janelia), our team and collaborators have released a [complete wiring diagram of the male fruit fly’s brain and central nervous system](https://www.janelia.org/news/researchers-reveal-connectome-of-the-male-fruit-fly-central-nervous-system). Published in [*Cell*](https://www.cell.com/cell/home), “[Sexual dimorphism in the complete connectome of the Drosophila male central nervous system](https://doi.org/10.1016/j.cell.2026.08.015)”, is the result of a decade-long partnership that advances the field of connectomics using computing and AI to build cellular-scale maps of entire brains. With over 166,000 neurons and 125 million synaptic connections, this is the largest brain map by number of neurons to date, providing a fundamental resource for scientists to use fruit flies as a model organism for studying how the brain works.

![Two detailed 3D color-coded visualizations of a fruit fly's neural pathways and nerve cord.](https://storage.googleapis.com/gweb-research2023-media/original_images/Male-Fruit-Fly-Brain-Map-1.png)

*A small subset of cells in the male fruit fly’s brain and central nervous system, as viewed from in front at an angle (**left**) and from above (**right**).*

The [male fruit fly connectome](https://male-cns.janelia.org/) also includes the [ventral nerve cord](https://www.janelia.org/news/janelia-scientists-and-collaborators-unveil-fruit-fly-nerve-cord-connectome), analogous to the spinal cord, and so begins to expand from just the brain into how the brain controls the body. The male fly connectome has been annotated and verified, or proofread, by a team of human experts at HHMI Janelia. It can be viewed, explored and downloaded via [Neuroglancer](https://github.com/google/neuroglancer), the [open-source tool](https://connectomics.readthedocs.io/en/latest/external/neuroglancer.html) we created to enable researchers to [visualize](https://research.google/blog/an-interactive-automated-3d-reconstruction-of-a-fly-brain/) huge multidimensional datasets.

![Diagram of a male fruit fly nervous system mapping the central brain, optic lobes, and VNC.](https://storage.googleapis.com/gweb-research2023-media/original_images/Male-Fruit-Fly-Brain-Map-2.jpg)

*The new connectome contains neurons from the central brain (**green**), optic lobes (**purple**) and ventral (i.e., central) nerve cord (**blue**). The new connectome enables linking auditory, visual and olfactory inputs to motor outputs for this key model organism.*

This new brain map complements the [female fruit fly brain map](https://flywire.ai/) and [recently released complete female fruit fly brain and nerve cord map](https://www.nature.com/articles/s41586-026-10735-w). Having both male and female brains and central nervous systems mapped allows the two to be [compared](https://male-cns.janelia.org/build/dimorphism_overview/#__tabbed_2_1) in places where neurons differ, and used to study the biological mechanisms for fruit fly [courtship](https://hms.harvard.edu/news/love-lives-fruit-flies) and aggression. In parts of the brain that are similar in both sexes, having two complete fruit fly connectomes will allow researchers to begin to see the variability between individuals.

![Comparison of the dimorphic AOTU008 neural pathway, showing differences between male and female fruit flies.](https://storage.googleapis.com/gweb-research2023-media/original_images/Male-Fruit-Fly-Brain-Map-3.png)

*An example of a neuron that is different between the male (**green**) and previously mapped female (**magenta**) fruit fly brains. The male neuron has two additional projections. AI enables accurate 3D reconstructions to pinpoint these structural differences.*

## Steps toward mapping a full fruit fly brain

Brain mapping, or [connectomics](https://en.wikipedia.org/wiki/Connectomics), begins with sectioning a brain into millions of thin slices, taking an image of each section, and using computers and AI to stitch the images together. Our researchers build systems that leverage AI to turn flat electron microscope images into 3D reconstructions, using an evolving suite of techniques to generate accurate neural shapes.

Our [AI connectomics tools](https://sites.research.google/gr/neural-mapping/innovations/) include [flood-filling networks](https://research.google/blog/improving-connectomics-by-an-order-of-magnitude/), which use convolutional neural networks to start at a single pixel and identify all other pixels that are part of the same object. In 2019, our Connectomics team released an [initial, fully-automated reconstruction](https://research.google/blog/an-interactive-automated-3d-reconstruction-of-a-fly-brain/) of a female fruit fly brain. By 2020, our team and collaborators released a human-verified map of [half a female fruit fly brain](https://research.google/blog/releasing-the-drosophila-hemibrain-connectome-the-largest-synapse-resolution-map-of-brain-connectivity/) with 25,000 neurons and 21 million connections, a record at the time. Meanwhile, the team was already working on the full, verified brain map for a male fruit fly, which is now complete.

These methods continue to improve. A recent effort incorporated synthetic neurons into the training data, successfully improving the speed and accuracy of our state-of-the-art reconstruction system, [PATHFINDER](https://www.biorxiv.org/content/10.1101/2025.05.16.654254v1). We are also helping to develop new techniques for [labeling and annotating](https://www.nature.com/articles/s41467-026-72180-7) specific types of neurons. Currently, mapping the fruit fly brain requires years of human effort just to verify and annotate the neural shapes. By reducing this need for manual error correction, research groups can tackle even larger brain mapping projects within reasonable budgets and timelines.

## Looking ahead: Mapping entire fish brains

The field of connectomics is already advancing into vertebrates: organisms with a spinal cord. These are anatomically, evolutionarily and functionally more similar to humans. In a study led by Columbia University and published this week in [*Nature*](https://www.nature.com/), our team helped map a portion of the elephantnose fish’s hindbrain that is used in signal processing. This paper, “[Connectome analysis of a cerebellum-like circuit for sensory prediction](https://www.nature.com/articles/s41586-026-10690-6)”, shows for the first time how the connectome, a static resource, can be combined with other information to study [neural plasticity and learning](https://zuckermaninstitute.columbia.edu/electric-fish-keep-blinding-themselves-these-brain-cells), producing the most complete mechanistic model of learning in a vertebrate brain to date.

Larval zebrafish are one of the few vertebrates whose brains are small enough to be mapped from end to end using current techniques. Zebrafish also have the advantage of being transparent in their larval stage, allowing measurements of neural activity during experiments, as captured in the [ZAPBench dataset](https://research.google/blog/improving-brain-models-with-zapbench/). Our upcoming paper with Harvard, “[A connectomic resource for neural cataloguing and circuit dissection of the larval zebrafish brain](http://www.doi.org/10.1101/2025.06.10.658982)”, is the first whole-brain dataset for a vertebrate that includes the neural structure and molecular type spanning an entire vertebrate brain. Our team also released a preliminary version of a dataset that [combines neural activity and structure](https://www.janelia.org/fish-firewire) in the same larval zebrafish brain, as an open resource to the research community.

![Multi-colored 3D map of interconnected neurons inside a translucent gray outline of a fruit fly head.](https://storage.googleapis.com/gweb-research2023-media/original_images/Male-Fruit-Fly-Brain-Map-4.png)

*An image from the Fish Fire&Wire dataset, which combines whole-brain electrical activity and neural structure for the same larval zebrafish specimen.*

## Conclusion

The male fruit fly connectome is a foundational resource that will support a new era for experimental neuroscience, with future applications in biology, pharmacy and medicine. Three companion papers released today show how the male fruit fly connectome has already been used for research on the neuroscience of [visual systems](https://doi.org/10.1016/j.cell.2026.08.014), [taste](https://doi.org/10.1016/j.cell.2026.08.016) and [social behavior](https://doi.org/10.1016/j.cub.2026.08.013). Methods developed here will also help advance other projects, such as the upcoming fully proofread map of the zebrafish brain, and mapping a portion of the mouse brain. While modeling the 86 billion neurons in a human brain remains out of reach, we are moving toward revealing how brains function and understanding the processes underlying mental ailments, such as Alzheimer’s, depression or schizophrenia. Someday, we hope these efforts lead to new ways to treat cognitive ailments, improve brain health, and support brain repair.

## Acknowledgments

We thank our academic collaborators at HHMI Janelia and elsewhere, and acknowledge core contributions from the Connectomics Team at Google. We are grateful to Hannah Hickey and Elise Kleeman for their help. Thanks to Lizzie Dorfman, Michael Brenner, John Platt, and Yossi Matias for their support, coordination and leadership.

**Labels:** General Science · Health & Bioscience · Machine Intelligence · [Open Source Models & Datasets](https://research.google/blog/label/open-source-models-datasets)
