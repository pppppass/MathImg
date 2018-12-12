function [] = Isosurface(mat, out, azi, ele, lightx, lighty, lightz, width, height)
width = str2num(width);
height = str2num(height);
azi = str2num(azi);
ele = str2num(ele);
lightx = str2num(lightx);
lighty = str2num(lighty);
lightz = str2num(lightz);
load(mat, "u");
isosurface(u, 0.5);
axis equal off
view(azi, ele);
light("Position", [lightx, lighty, lightz]);
set(gcf, "Position", [0, 0, width, height]);
print(out, "-dpng");
