"""Real-time crime detection through video enhancement (SeoulTech capstone).

Pipeline: frame -> ESDNet (de-weathering) -> clip buffer -> X3D features -> anomaly classifier.
See ``capstone.pipeline.AnomalyPipeline`` for the inference path shared by the demo and scripts.
"""
