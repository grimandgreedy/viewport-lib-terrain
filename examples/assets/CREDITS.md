# Example asset credits

Everything here is CC0 or public domain, and none of it requires attribution. These notes record where each set came from.

## Ground materials (textures/)

1K PBR sets, CC0 (colour, normal, occlusion and roughness maps only), from ambientCG (https://ambientcg.com) and Poly Haven (https://polyhaven.com).

The examples use these eight, in splatmap channel order: Grass004, Ground103, Rock051, coast_sand_01 (Poly Haven), Gravel041, Ground037, Snow014, Rock030.

## Heightmap (heightmaps/)

`rainier_513.r16` holds 513 x 513 16-bit little-endian height samples of the Mount Rainier area, normalised to the full range. The real elevation runs from 497.9 m to 4388.1 m over about 26.7 km.

They were built from the AWS Terrain Tiles terrarium tiles at zoom 12. Within the United States those tiles come from the USGS 3D Elevation Program, which is public domain: "3DEP terrain data courtesy of the U.S. Geological Survey."

## Models (models/)

The Kenney Nature Kit 2.1 (https://kenney.nl/assets/nature-kit), CC0. Only the pine tree and the grass tuft are kept. The examples recolour wood and foliage materials when they load a model.
