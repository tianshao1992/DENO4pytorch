CFViewBackward(1210)
FileOpenProject("***run_load***")
vp1 = OpenDefaultPitchAveragedView(1)
for quanlity in ***quanlitylist***:
	if quanlity=="atan_V":
		QntFieldDerived(0 ,'atan_V' ,'180/pi*atan(Vxyz_Y/Vxyz_X)' ,'' ,'' ,'deg')
	elif quanlity=="atan_W":
		QntFieldDerived(0 ,'atan_W' ,'180/pi*atan(Wxyz_Y/Wxyz_X)' ,'' ,'' ,'deg') 
	elif quanlity=="Mag_V":
		QntFieldDerived(0 ,'Mag_V' ,'sqrt(Vxyz_X*Vxyz_X +Vxyz_Y*Vxyz_Y)' ,'' ,'' ,'m/s') 
	elif quanlity=="Mag_W":
		QntFieldDerived(0 ,'Mag_W' ,'sqrt(Wxyz_X*Wxyz_X +Wxyz_Y*Wxyz_Y)' ,'' ,'' ,'m/s')
	elif quanlity=="Wxyz_X":
		QntFieldDerived(0 ,'Wxyz_X' ,'Wxyz_X' ,'' ,'' ,'m/s')
	elif quanlity=="Wxyz_Y":
		QntFieldDerived(0 ,'Wxyz_Y' ,'Wxyz_Y' ,'' ,'' ,'m/s')
	elif quanlity=="Wxyz_Z":
		QntFieldDerived(0 ,'Wxyz_Z' ,'Wxyz_Z' ,'' ,'' ,'m/s')
	elif quanlity=="Vxyz_X":
		QntFieldDerived(0 ,'Vxyz_X' ,'Vxyz_X' ,'' ,'' ,'m/s')
	elif quanlity=="Vxyz_Y":
		QntFieldDerived(0 ,'Vxyz_Y' ,'Vxyz_Y' ,'' ,'' ,'m/s')
	elif quanlity=="Vxyz_Z":
		QntFieldDerived(0 ,'Vxyz_Z' ,'Vxyz_Z' ,'' ,'' ,'m/s')
	else:
		QntFieldScalar(quanlity)
	for positionZ in ***positionZlist***:
		ViewActivate(vp1)
		vp2 = RprSection(positionZ,***upper***,0,positionZ,***lower***,0,0,0,1 ,'',0 ,'',0)
		ViewActivate(vp2)
		PlotCurveOutput(r"***span_save***"+"_"+quanlity+"_"+str(positionZ)+'.dat')
		FileCloseProject(vp2)
	ViewActivate(vp1)
FileCloseProject(vp1)