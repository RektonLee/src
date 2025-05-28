
import pymol
pymol.cmd.load("/home/lizihao/Work/enzyme_prediction/src/simple2/analysis_results/pocket_visualization/temp_km.pdb", "pocket")
pymol.cmd.spectrum("b", "blue_white_red", "pocket")
pymol.cmd.show("sticks")
pymol.cmd.set("stick_radius", "0.2")
pymol.cmd.select("ligand", "resn UNL or chain ''")
pymol.cmd.color("yellow", "ligand")
pymol.cmd.show("spheres", "ligand")
pymol.cmd.set("sphere_scale", "0.4", "ligand")
pymol.cmd.label("name CA", "%resi %resn")
pymol.cmd.set("label_size", "1.0")
pymol.cmd.bg_color("white")
pymol.cmd.set("ray_opaque_background", "off")
pymol.cmd.orient()
pymol.cmd.set("depth_cue", "0")
pymol.cmd.set("ray_shadows", "0")
pymol.cmd.png("km_contribution.png", width=1200, height=1000, dpi=300, ray=1)
    