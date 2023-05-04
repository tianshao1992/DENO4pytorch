script_version(2.2)
import FT

FT.open_project(r"G:/WQN/DEC_fan/off_line_stra/stage2/stage.iec")
#FT.save_project_as(r"***iec_save***", 0)

FT.set_active_computations([0])
#FT.new_computation(***new***)
FT.set_active_computations([62])
FT.set_computation_name(62,r"computation_100100") 

FT.set_cyl_cart_configuration(1) 
#FT.link_mesh_file(r"***igg***",0) 

#FT.set_mathematical_model(***model***) 

FT.get_bc_group(FT.get_bc_patch(0,0,0)).set_bc_type([27,40]) 
FT.get_bc_group(FT.get_bc_patch(0,0,0)).set_parameter_value("Static_Pressure",100100) 
#FT.get_bc_group(FT.get_bc_patch(0,0,0)).set_parameter_value("mass-flow",***massflow***) 

#FT.set_MG_number_of_cycles(***iterMG***) 
#FT.set_MG_flag(***boolMG***) 

# FT.set_restart_mode(1) 
FT.get_initial_solution(0).set_mode(r"file")
FT.get_initial_solution(0).set_restart_filename(r"G:/WQN/DEC_fan/off_line_stra/stage2/stage_computation_100000/stage_computation_100000.run") 

#FT.set_output_writing_frequency(***freq***) 
#FT.set_convergence_criteria(***crit***) 
FT.set_nb_iter_max(1000) 

FT.save_selected_computations() 
FT.save_project() 