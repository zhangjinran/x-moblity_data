# x-moblity_data
- 准备好x_mobility_isaac_sim_nav2_100k.zip数据集；
- 将文件data_reader.py转移到数据集中test、train和val文件夹的上一级目录，如果文件路径有问题，可以在脚本的第十八行中修改文件路径；
- 这个脚本只是用来读取一个pqt文件，将里面的内容转化为dataframe，并输出结果。
- 数据集中还包含有MP4视频，视频长度不长，大概在十几秒之内。
# 模型输出的数据结构
-包括 action、path、semantic、rgb、depth 等张量的形状、含义及值域
详见仓库根目录下的 `output_x_mobility.md` 文件。
