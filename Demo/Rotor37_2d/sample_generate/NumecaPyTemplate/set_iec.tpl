script_version(2.2)
import FT

FT.open_project(r"***iec_load***")
FT.save_project_as(r"***iec_save_as***", 0)
FT.set_active_computations([***computation***])
FT.set_cyl_cart_configuration(1) 
FT.link_mesh_file(r"***igg***",0) 

FT.set_mathematical_model(***model***) 

FT.get_bc_group(FT.get_bc_patch(0,0,0)).set_bc_type(***bounarycode***) 
FT.get_bc_group(FT.get_bc_patch(0,0,0)).set_parameter_value("Static_Pressure",***spre***) 
FT.get_bc_group(FT.get_bc_patch(0,0,0)).set_parameter_value("mass-flow",***massflow***) 

# FT.set_restart_mode(1) 
FT.get_initial_solution(0).set_mode(r"***initial***") 
FT.get_initial_solution(0).set_restart_filename(r"***refile***") 

FT.set_MG_flag(1) 
FT.set_MG_number_of_cycles(400) 

FT.set_output_writing_frequency(***freq***) 
FT.set_convergence_criteria(***crit***) 
FT.set_nb_iter_max(***itermax***) 
FT.set_MG_number_of_cycles(***itersave***) 

FT.save_selected_computations() 
FT.save_project() 