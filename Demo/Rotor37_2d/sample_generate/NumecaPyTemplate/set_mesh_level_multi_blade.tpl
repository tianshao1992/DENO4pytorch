igg_script_version(2.1)
levelList = ***level***
a5_open_project(r"***trb_load***")
a5_import_and_replace_geometry_file(r"***geom***")
for ii in range(***num***):
	row(ii+1).select()
	row(ii+1).row_wizard().set_grid_level(levelList[ii])
	row(ii+1).row_wizard().generate()
	row(ii+1).unselect()
a5_save_project(r"***trb_save***")
select_all_rows() 
a5_start_3d_generation()
a5_save_project(r"***trb_save***")