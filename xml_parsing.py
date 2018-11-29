################################################################################

# common XML parsing funcyions

# Copyright (c) 2017/18 - Tiancheng Guo / Toby Breckon, Durham University, UK

# License : https://github.com/GTC7788/raindropDetection/blob/master/LICENSE

################################################################################

import xml.etree.ElementTree as ET


################################################################################

"""
Parse the xml file that stores the ground truth raindrop locations in the image

Args:
	fileName: the xml file name
Returns:
	list that each element contains the location of a ground truth raindrop

"""
def parse_xml_file(fileName):
	xml_file = ET.parse(fileName)
	# XML_path to retrieve the x, y coordinates
	xIndex = xml_file.findall('object/polygon/pt/x')
	yIndex = xml_file.findall('object/polygon/pt/y')
	xList = []
	yList = []
	for x in xIndex:
		xList.append(int(x.text))

	for y in yIndex:
		yList.append(int(y.text))

	combinedList = zip(xList,yList)

	subList = []
	finalList = []
	counter = 1
	for element in combinedList:
		switch = counter % 4
		if switch == 0:
			subList.append(element)
			finalList.append(subList)
			subList = []
		else:
			subList.append(element)
		counter += 1

	return finalList

################################################################################

"""
Retrieve the coordinates of each ground truth raindrop locations
Args:
	xml_golden: a list that each element contains the location of a ground truth raindrop
Returns:
	a list of coordinates for each ground truth raindrops that ready for drawing.
"""
def xml_transform(xml_golden):
	xml_result = []
	for element in xml_golden:
		sub_list = []
		sub_list = [element[0][0], element[0][1],
		element[2][0], element[2][1]]

		xml_result.append(sub_list)
	return xml_result

################################################################################
