************************************************************************************************
				**** Stanford Large-Scale Indoor Spaces 3D Dataset ****
************************************************************************************************


We collected and annotated five large-scale indoor parts in three buildings of mainly educational and office use. Each area covers approximately 1900, 450, 1700, 870 and 1100 square meters (total of 6020 square meters). Conference rooms, personal offices, auditoriums, restrooms, open spaces, lobbies, stairways and hallways are commonly found. The areas show diverse properties in architectural style and appearance. We fully annotated our dataset for 12 semantic elements which pertain in the categories of structural building elements (ceiling, floor, wall, beam, column, window and door) and commonly found furniture (table, chair, sofa, bookcase and board). A clutter class exists as well for all other elements. You can visualize the data and find out more details at: http://buildingparser.stanford.edu/dataset.html
(Area 1 and 6 form a two floor area)

Dataset Split:
 We split our dataset based on buildings, using the following mapping of Areas to buidlings: Buidling 1 (Area 1, Area 3, Area 6), Building 2 (Area 2, Area 4), Buidling 3 (Area 5). In our method we cross-validate by training on 2 of the buidlings and testing on the 3rd one.

If you use this dataset please cite:
@InProceedings{armeni_cvpr16,
title           ={3D Semantic Parsing of Large-Scale Indoor Spaces},
author        = {Iro Armeni and Ozan Sener and Amir R. Zamir and Helen Jiang and Ioannis Brilakis and Martin Fischer and Silvio Savarese},

booktitle    = {Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition},
  year         = {2016},
}

The folder contains per area the following data:
- Global alignment* angles of the disjoint spaces (Area_#number#_alignmentAngle.txt file)
- Disjoint spaces per area ("#disjoint_space_name_&_instance#" folders) -e.g. "conferenceRoom_1", "office_3", "WC_2"
	Per disjoint space folder:
	- Raw point cloud of the disjoint space (#disjoint_space_name_&_instance#.txt file, e.g. conferenceRoom_1.txt, office_3.txt, WC_2.txt)
	- Building elements Annotations ("Annotations" folder)  
	Per Annotation folder:
		- Raw point cloud per instance for all 12 building elements plus clutter (#bldg_element_name_&_instance#.txt file, e.g. ceiling_1.txt, table_3.txt, clutter_5.txt)

All raw point clouds have the same format: as many lines as the points in the point cloud and 6 columns. The three first columns correspond to the 3D coordinates of the point and the last three to the RGB color value of it (in order: X, Y, Z, R, G, B).

If you prefer using a MATLAB (.mat) file instead, the folder includes "Areas.mat" which contains the same exact data as the .txt files.

For those who want to use the globally aligned* point cloud versions directly, there exists a folder named "Aligned_Version", which contains the exact data as above (original version), but rotated according to the global alignment angles provided. Please note that the aligned version is only rotated; neither translated nor normalized. Similarly to the original version, there is the "Aligned_Areas.mat" file that you can use instead of the .txt files.

If you find that you'd like access to items we're not currently sharing, contact us at the link provided below.

If you have any questions or problems, contact us here: buildingparser.stanford.edu/contact.html

*aligned = corresponds to the alignment that takes place before the element parsing (see in paper Section 3.2. Canonical Coordinate System Among Spaces)

> Change Log Report:
	- 9/29/16. Added element files from the rooms that were missing.
	- 7/19/17. Fixed typos in .txt files version. Affected files (in both original and aligned folders):
						* Area_4/hallway10/Annotations/ceiiling_2.txt
						* Area_4/office_1/Annotations/foor_1.txt
						* Area_4/office_21/Annotations/floor.txt
						* Area_4/office_21/Annotations/ceiling.txt
						* Area_5/office_42/Annotations/door.txt
						* Area_5/office_42/Annotations/floor.txt
						* Area_5/office_42/Annotations/ceiling.txt
						* Area_5/office_42/Annotations/window.txt
	- 10/13/17. Fixed invalid characters in:
						* Area_1/WC_1/WC_1.txt, line 429613, character 57
						* Area_6/hallway_2/hallway_2.txt, line 842733, character 53
						* Area_5/office_19/Annotations/ceiling.txt, line 323474, character 56


							**** Last modified: 10/13/17 ****