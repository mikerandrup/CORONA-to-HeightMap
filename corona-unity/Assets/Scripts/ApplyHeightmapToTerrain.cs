using UnityEngine;

public class ApplyHeightmapToTerrain : MonoBehaviour
{
    public Texture2D heightmapTexture;

    void OnEnable()
    {
        ApplyHeightmap();
    }

    public void ApplyHeightmap()
    {
        if (heightmapTexture == null)
        {
            Debug.LogError("No heightmap texture assigned");
            return;
        }

        var terrain = GetComponent<Terrain>();
        if (terrain == null)
        {
            Debug.LogError("No Terrain component found");
            return;
        }

        var terrainData = terrain.terrainData;
        int resolution = terrainData.heightmapResolution;

        float[,] heights = new float[resolution, resolution];

        for (int y = 0; y < resolution; y++)
        {
            for (int x = 0; x < resolution; x++)
            {
                float u = (float)x / (resolution - 1);
                float v = (float)y / (resolution - 1);

                Color pixel = heightmapTexture.GetPixelBilinear(u, v);
                heights[y, x] = pixel.grayscale;
            }
        }

        terrainData.SetHeights(0, 0, heights);

        Debug.Log(
            $"Applied heightmap {heightmapTexture.name} " +
            $"({heightmapTexture.width}x{heightmapTexture.height}) " +
            $"to terrain resolution {resolution}"
        );
    }
}
