from tblNet import tblNet, prepMarmot

marmot_dir="./Marmot_data/"
col_mask_dir="./mask_col/"
tbl_mask_dir="./mask_tbl/"

# prep data
marmotParser = prepMarmot(
    marmot_dir="/home/mnewman/Source/Marmot_data/",
    col_mask_out="/home/mnewman/Source/mask_col/",
    tbl_mask_out="/home/mnewman/Source/mask_tbl/"
)
marmotParser.prepData()


# TableNet training
tbl = tblNet(
    marmot_dir=marmot_dir,
    col_mask_dir=col_mask_dir,
    tbl_mask_dir=tbl_mask_dir
)
tbl.build()
tbl.compile()
tbl.train()

# Prediction Test
tbl = tblNet()
tbl.classify("08-90.jpg")

