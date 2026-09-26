namespace Jevlet.Core;

/// <summary>
/// Whether a model was trained for the catalogue this app asks. A model trained on other skill
/// names or questions still returns probabilities, but they mean nothing, so the app refuses it
/// instead of running it.
/// </summary>
public static class ModelCompatibility
{
    /// <summary>Why the model cannot serve this catalogue, or null when it can.</summary>
    public static string? Problem(ModelManifest manifest, Catalogue catalogue)
    {
        if (manifest.CatalogueFingerprint is null)
        {
            return $"The model {manifest.SourceCheckpoint} does not say which skill catalogue it was trained for. "
                + "Export it again with scripts/export_onnx.py.";
        }
        if (!string.Equals(manifest.CatalogueFingerprint, catalogue.Fingerprint, StringComparison.Ordinal))
        {
            return $"The model {manifest.SourceCheckpoint} was trained for skill catalogue v{manifest.CatalogueVersion} "
                + $"({Short(manifest.CatalogueFingerprint)}), but this app uses v{catalogue.Version} ({Short(catalogue.Fingerprint)}).";
        }
        return null;
    }

    private static string Short(string fingerprint) => fingerprint.Length > 12 ? fingerprint[..12] : fingerprint;

    /// <summary>True when the folder holds a model this catalogue can use.</summary>
    public static bool Serves(string folder, Catalogue catalogue)
    {
        try
        {
            return Problem(ModelManifest.Load(Path.Combine(folder, "model.json")), catalogue) is null;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or System.Text.Json.JsonException
            or InvalidDataException or KeyNotFoundException or InvalidOperationException)
        {
            return false;
        }
    }
}
